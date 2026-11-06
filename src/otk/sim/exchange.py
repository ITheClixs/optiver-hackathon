"""In-memory exchange matching the public Optibook client surface and published rules.

Written from the public guide and client documentation; it contains no organiser code.
Rules enforced: price-time priority, IOC never rests, pre-emptive worst-case position and
outstanding checks, 25 updates per second, disconnect on breach (orders deleted), last login takes
over the session, soft delta limit (10% hacked at 2% worse than mid after 10 s), public ticks with
aggressor side but empty counterparties.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from otk.exchange_types import (
    ActionResponse,
    InsertOrderResponse,
    OrderStatus,
    PriceBook,
    PriceVolume,
    SimInstrument,
    SocialMediaFeed,
    Trade,
    TradeTick,
)
from otk.fair import bs_price, etf_fair, future_fair, index_value
from otk.instruments import Kind, parse_instrument_id, years_to_expiry

SESSION_START = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)


@dataclass
class TeamStats:
    writes: int = 0
    disconnects: int = 0
    rate_breaches: int = 0
    limit_breaches: int = 0
    delta_hacks: int = 0
    self_trades: int = 0
    write_times: deque[float] = field(default_factory=deque)
    peak_write_rate: int = 0


@dataclass
class _Order:
    order_id: int
    team: str
    instrument_id: str
    side: str
    tick_price: int
    volume: int


class SimMarket:
    """Shared state of one simulated session; several teams and bots trade on it."""

    def __init__(
        self,
        instruments: list[str],
        tick: float = 0.1,
        position_limit: int = 100,
        outstanding_limit: int = 200,
        updates_per_second: int = 25,
        delta_soft: float = 100.0,
        delta_grace_s: float = 10.0,
        rate: float = 0.03,
        index_weights: dict[str, float] | None = None,
        option_vol: float = 0.25,
        tick_sizes: Mapping[str, float] | None = None,
    ) -> None:
        self.instruments = list(instruments)
        self.tick = tick
        self.tick_sizes = dict(tick_sizes or {})
        self.tape: list[tuple] = []  # (now, iid, price, volume, buyer, seller, aggressor) for tests
        self.paused: set[str] = set()  # markets closed right now
        self.position_limit = position_limit
        self.outstanding_limit = outstanding_limit
        self.updates_per_second = updates_per_second
        self.delta_soft = delta_soft
        self.delta_grace_s = delta_grace_s
        self.rate = rate
        self.index_weights = index_weights or {}
        self.option_vol = option_vol
        self.now = 0.0
        self.fair: dict[str, float] = {}
        self.last_trade: dict[str, float] = {}
        self.actors: list = []
        self.news: list[SocialMediaFeed] = []
        self.stats: dict[str, TeamStats] = defaultdict(TeamStats)
        self.positions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cash: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.ticks: dict[str, list[TradeTick]] = defaultdict(list)
        self.trades: dict[str, dict[str, list[Trade]]] = defaultdict(lambda: defaultdict(list))
        self.sessions: dict[str, SimExchange] = {}
        self._bids: dict[str, dict[int, deque[_Order]]] = defaultdict(dict)
        self._asks: dict[str, dict[int, deque[_Order]]] = defaultdict(dict)
        self._orders: dict[int, _Order] = {}
        self._next_order = 1
        self._next_trade = 1
        self._delta_since: dict[tuple, float] = {}

    def tick_of(self, instrument_id: str) -> float:
        return self.tick_sizes.get(instrument_id, self.tick)

    # --- clock -------------------------------------------------------------------------------

    def timestamp(self) -> datetime:
        return SESSION_START + timedelta(seconds=self.now)

    def advance(self, dt: float) -> None:
        self.now += dt
        for actor in self.actors:
            actor.act(self, self.now, dt)
        self._apply_delta_limit()

    # --- theoretical values ------------------------------------------------------------------

    def value(self, instrument_id: str) -> float | None:
        info = parse_instrument_id(instrument_id)
        if info.kind in (Kind.STOCK, Kind.DUAL):
            return self.fair.get(info.base, self.fair.get(instrument_id))
        spot = self._underlying_value(info.base)
        if spot is None:
            return None
        if info.kind is Kind.ETF:
            return etf_fair(spot)
        tau = years_to_expiry(info.expiry, self.timestamp())
        if info.kind is Kind.FUTURE:
            return future_fair(spot, self.rate, tau)
        return bs_price(info.option_kind, spot, info.strike, tau, self.rate, self.option_vol)

    def _underlying_value(self, base: str) -> float | None:
        if base in self.fair:
            return self.fair[base]
        return index_value(self.fair, self.index_weights) if self.index_weights else None

    # --- book --------------------------------------------------------------------------------

    def _side(self, instrument_id: str, side: str) -> dict[int, deque[_Order]]:
        return self._bids[instrument_id] if side == "bid" else self._asks[instrument_id]

    def book_snapshot(self, instrument_id: str) -> PriceBook:
        tick = self.tick_of(instrument_id)

        def levels(side: str, reverse: bool) -> list[PriceVolume]:
            book = self._side(instrument_id, side)
            return [
                PriceVolume(round(price * tick, 10), sum(o.volume for o in book[price]))
                for price in sorted(book, reverse=reverse)
            ]

        return PriceBook(
            self.timestamp(), instrument_id, levels("bid", True), levels("ask", False)
        )

    def orders_of(self, team: str, instrument_id: str) -> dict[int, OrderStatus]:
        return {
            oid: OrderStatus(
                oid, o.instrument_id, round(o.tick_price * self.tick_of(o.instrument_id), 10), o.volume, o.side
            )
            for oid, o in self._orders.items()
            if o.team == team and o.instrument_id == instrument_id
        }

    def open_volume(self, team: str, instrument_id: str, side: str) -> int:
        return sum(
            o.volume
            for o in self._orders.values()
            if o.team == team and o.instrument_id == instrument_id and o.side == side
        )

    def _book_trade(
        self, taker: str, order: _Order, side: str, volume: int, tick_price: int, taker_id: int
    ) -> None:
        iid = order.instrument_id
        price = round(tick_price * self.tick_of(iid), 10)
        buyer, seller = (taker, order.team) if side == "bid" else (order.team, taker)
        trade_id = self._next_trade
        self._next_trade += 1
        stamp = self.timestamp()
        self.ticks[iid].append(TradeTick(stamp, iid, price, volume, side, "", "", trade_id))
        self.tape.append((self.now, iid, price, volume, buyer, seller, side))
        self.positions[buyer][iid] += volume
        self.positions[seller][iid] -= volume
        self.cash[buyer][iid] -= price * volume
        self.cash[seller][iid] += price * volume
        self.last_trade[iid] = price
        buy_order = taker_id if side == "bid" else order.order_id
        sell_order = order.order_id if side == "bid" else taker_id
        self.trades[buyer][iid].append(Trade(stamp, buy_order, trade_id, iid, price, volume, "bid"))
        self.trades[seller][iid].append(Trade(stamp, sell_order, trade_id, iid, price, volume, "ask"))
        if buyer == seller:
            self.stats[buyer].self_trades += 1

    def _match(
        self, team: str, instrument_id: str, side: str, tick_price: int, volume: int, taker_id: int
    ) -> int:
        opposite = self._side(instrument_id, "ask" if side == "bid" else "bid")
        crosses = (lambda p: p <= tick_price) if side == "bid" else (lambda p: p >= tick_price)
        while volume > 0 and opposite:
            best = min(opposite) if side == "bid" else max(opposite)
            if not crosses(best):
                break
            queue = opposite[best]
            while volume > 0 and queue:
                head = queue[0]
                traded = min(volume, head.volume)
                self._book_trade(team, head, side, traded, best, taker_id)
                volume -= traded
                head.volume -= traded
                if head.volume == 0:
                    queue.popleft()
                    del self._orders[head.order_id]
            if not queue:
                del opposite[best]
        return volume

    def _rest(
        self, order_id: int, team: str, instrument_id: str, side: str, tick_price: int, volume: int
    ) -> int:
        order = _Order(order_id, team, instrument_id, side, tick_price, volume)
        self._orders[order.order_id] = order
        self._side(instrument_id, side).setdefault(tick_price, deque()).append(order)
        return order.order_id

    def remove_order(self, order_id: int) -> bool:
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        book = self._side(order.instrument_id, order.side)
        queue = book[order.tick_price]
        queue.remove(order)
        if not queue:
            del book[order.tick_price]
        return True

    def insert(
        self, team: str, instrument_id: str, side: str, price: float, volume: int, order_type: str
    ) -> int:
        """Match, then rest a limit remainder. Returns the order id. No limit checks here."""
        tick_price = round(price / self.tick_of(instrument_id))
        oid = self._next_order
        self._next_order += 1
        remaining = self._match(team, instrument_id, side, tick_price, volume, oid)
        if remaining > 0 and order_type == "limit":
            self._rest(oid, team, instrument_id, side, tick_price, remaining)
        return oid

    def wipe_orders(self, team: str, instrument_id: str | None = None) -> None:
        for oid in [
            oid
            for oid, order in self._orders.items()
            if order.team == team and instrument_id in (None, order.instrument_id)
        ]:
            self.remove_order(oid)

    # --- risk ---------------------------------------------------------------------------------

    def force_disconnect(self, team: str, reason: str) -> None:
        session = self.sessions.pop(team, None)
        if session is not None:
            session.drop(reason)
        self.stats[team].disconnects += 1
        self.wipe_orders(team)

    def _mid(self, instrument_id: str) -> float | None:
        book = self.book_snapshot(instrument_id)
        if book.bids and book.asks:
            return (book.bids[0].price + book.asks[0].price) / 2.0
        return self.last_trade.get(instrument_id)

    def _group_delta(self, team: str) -> dict[str, dict[str, int]]:
        groups: dict[str, dict[str, int]] = defaultdict(dict)
        for iid, position in self.positions[team].items():
            if position:
                groups[parse_instrument_id(iid).base][iid] = position
        return groups

    def _apply_delta_limit(self) -> None:
        for team in list(self.sessions):
            for base, members in self._group_delta(team).items():
                delta = sum(members.values())
                key = (team, base)
                if abs(delta) < self.delta_soft:
                    self._delta_since.pop(key, None)
                    continue
                since = self._delta_since.setdefault(key, self.now)
                if self.now - since > self.delta_grace_s:
                    self._hack(team, members, delta)
                    self._delta_since[key] = self.now

    def _hack(self, team: str, members: dict[str, int], delta: int) -> None:
        sign = 1 if delta > 0 else -1
        target = max(
            (iid for iid in members if self._mid(iid) is not None),
            key=lambda iid: sign * members[iid],
            default=None,
        )
        if target is None:
            return
        lots = math.ceil(0.1 * abs(delta))
        price = self._mid(target) * (1.0 - sign * 0.02)
        self.positions[team][target] -= sign * lots
        self.cash[team][target] += sign * lots * price
        self.stats[team].delta_hacks += 1

    # --- settlement ---------------------------------------------------------------------------

    def settlement_price(self, instrument_id: str) -> float:
        """Stocks at last trade, duals at their primary's last trade, others at theoretical."""
        info = parse_instrument_id(instrument_id)
        if info.kind is Kind.STOCK:
            return self.last_trade.get(instrument_id, self.value(instrument_id) or 0.0)
        if info.kind is Kind.DUAL:
            return self.last_trade.get(info.base, self.value(instrument_id) or 0.0)
        return self.value(instrument_id) or 0.0

    def settled_pnl(self, team: str) -> float:
        cash = sum(self.cash[team].values())
        marks = sum(
            position * self.settlement_price(iid) for iid, position in self.positions[team].items()
        )
        return cash + marks

    def marked_pnl(self, team: str, valuations: dict[str, float] | None = None) -> float:
        overrides = valuations or {}
        marks = sum(
            position * overrides.get(iid, self.last_trade.get(iid, 0.0))
            for iid, position in self.positions[team].items()
        )
        return sum(self.cash[team].values()) + marks


class SimExchange:
    """Client session with the same call shapes as `optibook.synchronous_client.Exchange`."""

    def __init__(self, market: SimMarket, team: str, bot: bool = False) -> None:
        self._market = market
        self.team = team
        self._bot = bot
        self._connected = bot
        self._tick_cursor: dict[str, int] = {}
        self._trade_cursor: dict[str, int] = {}
        self._news_cursor = 0
        self.disconnect_reason = ""

    # --- session ------------------------------------------------------------------------------

    def connect(self) -> None:
        current = self._market.sessions.get(self.team)
        if current is not None and current is not self:
            current.drop("someone else logged in with the same credentials")
        self._market.sessions[self.team] = self
        self._connected = True
        self.disconnect_reason = ""
        self._tick_cursor = {iid: len(self._market.ticks[iid]) for iid in self._market.instruments}
        self._trade_cursor = {
            iid: len(self._market.trades[self.team][iid]) for iid in self._market.instruments
        }
        self._news_cursor = len(self._market.news)

    def drop(self, reason: str) -> None:
        self._connected = False
        self.disconnect_reason = reason

    def disconnect(self) -> None:
        if self._market.sessions.get(self.team) is self:
            self._market.sessions.pop(self.team)
            self._market.wipe_orders(self.team)
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _require_session(self) -> None:
        if not self._connected:
            raise ConnectionError(self.disconnect_reason or "not connected to the exchange")

    def _count_write(self) -> bool:
        """Register an update; returns False when it breaches the rate limit (disconnects)."""
        self._require_session()
        stats = self._market.stats[self.team]
        stats.writes += 1
        if self._bot:
            return True
        now = self._market.now
        while stats.write_times and now - stats.write_times[0] >= 1.0:
            stats.write_times.popleft()
        if len(stats.write_times) >= self._market.updates_per_second:
            stats.rate_breaches += 1
            self._market.force_disconnect(self.team, "update rate limit breached")
            return False
        stats.write_times.append(now)
        stats.peak_write_rate = max(stats.peak_write_rate, len(stats.write_times))
        return True

    def _would_self_trade(self, iid: str, side: str, price: float) -> bool:
        opposite = "ask" if side == "bid" else "bid"
        for order in self._market.orders_of(self.team, iid).values():
            if order.side == opposite and (
                price >= order.price if side == "bid" else price <= order.price
            ):
                return True
        return False

    def _breaches_limits(self, iid: str, side: str, volume: int) -> bool:
        market = self._market
        position = market.positions[self.team][iid]
        open_buy = market.open_volume(self.team, iid, "bid")
        open_sell = market.open_volume(self.team, iid, "ask")
        if open_buy + open_sell + volume > market.outstanding_limit:
            return True
        if side == "bid":
            return position + open_buy + volume > market.position_limit
        return position - open_sell - volume < -market.position_limit

    # --- writes -------------------------------------------------------------------------------

    def insert_order(
        self, instrument_id: str, *, price: float, volume: int, side: str, order_type: str = "limit"
    ) -> InsertOrderResponse:
        if not self._count_write():
            return InsertOrderResponse(False, 0, "update rate limit breached")
        if not self._bot and instrument_id in self._market.paused:
            return InsertOrderResponse(False, 0, "instrument is paused")
        if not self._bot and self._would_self_trade(instrument_id, side, price):
            self._market.stats[self.team].self_trades += 1
            self._market.force_disconnect(self.team, "self trade")
            return InsertOrderResponse(False, 0, "self trade")
        if not self._bot and self._breaches_limits(instrument_id, side, volume):
            self._market.stats[self.team].limit_breaches += 1
            self._market.force_disconnect(self.team, "order would breach a limit")
            return InsertOrderResponse(False, 0, "order would breach a limit")
        order_id = self._market.insert(self.team, instrument_id, side, price, volume, order_type)
        return InsertOrderResponse(True, order_id, "None")

    def amend_order(self, instrument_id: str, *, order_id: int, volume: int) -> ActionResponse:
        if not self._count_write():
            return ActionResponse(False, "update rate limit breached")
        order = self._market._orders.get(order_id)
        if order is None or order.team != self.team:
            return ActionResponse(False, "Could not find order id to amend")
        order.volume = volume
        return ActionResponse(True, "None")

    def delete_order(self, instrument_id: str, *, order_id: int) -> ActionResponse:
        if not self._count_write():
            return ActionResponse(False, "update rate limit breached")
        order = self._market._orders.get(order_id)
        if order is None or order.team != self.team or not self._market.remove_order(order_id):
            return ActionResponse(False, "Could not find order id to delete")
        return ActionResponse(True, "None")

    def delete_orders(self, instrument_id: str) -> None:
        if self._count_write():
            self._market.wipe_orders(self.team, instrument_id)

    # --- reads --------------------------------------------------------------------------------

    def get_instruments(self) -> dict[str, SimInstrument]:
        market = self._market
        return {
            iid: SimInstrument(iid, market.tick_of(iid), market.rate, iid in market.paused)
            for iid in market.instruments
        }

    def get_last_price_book(self, instrument_id: str) -> PriceBook:
        return self._market.book_snapshot(instrument_id)

    def get_outstanding_orders(self, instrument_id: str) -> dict[int, OrderStatus]:
        return self._market.orders_of(self.team, instrument_id)

    def get_positions(self) -> dict[str, int]:
        held = self._market.positions[self.team]
        return {iid: held[iid] for iid in self._market.instruments}

    def get_positions_and_cash(self) -> dict[str, dict]:
        market = self._market
        return {
            iid: {"volume": market.positions[self.team][iid], "cash": market.cash[self.team][iid]}
            for iid in market.instruments
        }

    def get_cash(self) -> float:
        return sum(self._market.cash[self.team].values())

    def get_pnl(self, valuations: dict[str, float] | None = None) -> float:
        return self._market.marked_pnl(self.team, valuations)

    def get_trade_tick_history(self, instrument_id: str) -> list[TradeTick]:
        return list(self._market.ticks[instrument_id])

    def poll_new_trade_ticks(self, instrument_id: str) -> list[TradeTick]:
        ticks = self._market.ticks[instrument_id]
        start = self._tick_cursor.get(instrument_id, 0)
        self._tick_cursor[instrument_id] = len(ticks)
        return ticks[start:]

    def get_trade_history(self, instrument_id: str) -> list[Trade]:
        return list(self._market.trades[self.team][instrument_id])

    def poll_new_trades(self, instrument_id: str) -> list[Trade]:
        trades = self._market.trades[self.team][instrument_id]
        start = self._trade_cursor.get(instrument_id, 0)
        self._trade_cursor[instrument_id] = len(trades)
        return trades[start:]

    def poll_new_social_media_feeds(self) -> list[SocialMediaFeed]:
        news = self._market.news
        start = self._news_cursor
        self._news_cursor = len(news)
        return news[start:]

    def get_social_media_feeds_history(self) -> list[SocialMediaFeed]:
        return list(self._market.news)


class LaggedExchange(SimExchange):
    """Orders and positions arrive late, as they do through the real client's pushed cache.

    The plain simulator updates them instantly, which hides every race between an order being sent
    and the local view catching up. Tests run the full stack through this class for that reason.
    """

    def __init__(self, market: SimMarket, team: str, lag_s: float = 0.2) -> None:
        super().__init__(market, team)
        self.lag_s = lag_s
        self._history: deque[tuple] = deque()

    def _lagged(self) -> tuple:
        now = self._market.now
        market = self._market
        snapshot = {iid: market.orders_of(self.team, iid) for iid in market.instruments}
        self._history.append((now, snapshot))
        while len(self._history) > 1 and now - self._history[1][0] >= self.lag_s:
            self._history.popleft()
        return self._history[0][1]

    def get_outstanding_orders(self, instrument_id: str) -> dict[int, OrderStatus]:
        return self._lagged().get(instrument_id, {})

    def get_positions(self) -> dict[str, int]:
        """Exactly what the delayed fills imply, as fills and positions share one channel."""
        cutoff = self._market.timestamp() - timedelta(seconds=self.lag_s)
        positions = dict.fromkeys(self._market.instruments, 0)
        for iid in self._market.instruments:
            for trade in self._market.trades[self.team][iid]:
                if trade.timestamp <= cutoff:
                    positions[iid] += trade.volume if trade.side == "bid" else -trade.volume
        return positions

    def poll_new_trades(self, instrument_id: str) -> list[Trade]:
        """Fills share the channel, so they arrive with the same delay as positions."""
        trades = self._market.trades[self.team][instrument_id]
        start = self._trade_cursor.get(instrument_id, 0)
        cutoff = self._market.timestamp() - timedelta(seconds=self.lag_s)
        ready = [trade for trade in trades[start:] if trade.timestamp <= cutoff]
        self._trade_cursor[instrument_id] = start + len(ready)
        return ready
