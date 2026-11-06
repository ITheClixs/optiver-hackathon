"""Conservative backtest: replay a live capture through the real engine and strategies.

`latency_s` delays every order and cancel, so what speed is worth can be measured on a real book. Our
orders have no market impact, and passive fills are deliberately pessimistic:
  * a new order joins behind all visible volume at its price;
  * prints at our price first consume the queue ahead of us;
  * a print through our price, or a book that crosses it, fills us at our price, which keeps the
    adverse selection a real resting order would suffer.
IOC fills are capped by the visible size at the touch and consume it for that snapshot.

A capture recorded while our own bot ran leaves our resting quotes out of its books, but still holds our impact. Prefer captures from
a passive `scripts/record_market.py` session, and treat results as a filter, not a forecast.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from otk.engine import Engine, EngineConfig
from otk.exchange_types import (
    ActionResponse,
    InsertOrderResponse,
    OrderStatus,
    PriceBook,
    PriceVolume,
    SimInstrument,
    Trade,
)
from otk.fair import snap
from otk.instruments import DEFAULT_TICK, Universe, infer_ticks
from otk.params import ParamStore
from otk.recorder import read_capture


def load_events(path: str | Path) -> list[dict]:
    return sorted(read_capture(path), key=lambda event: event["t"])


class _Resting(NamedTuple):
    order_id: int
    instrument_id: str
    side: str
    price: float
    volume: int
    queue_ahead: int


class ReplayExchange:
    """Same call surface as the live client, driven by recorded events."""

    def __init__(
        self,
        events: Sequence[Mapping],
        instruments: Iterable[str],
        tick: float | Mapping[str, float] | None = None,
        position_limit: int = 100,
        outstanding_limit: int = 200,
        latency_s: float = 0.0,
    ) -> None:
        self.latency_s = latency_s
        self._in_flight: list[tuple] = []
        self.events = events
        self.instruments = list(instruments)
        self._ticks = infer_ticks(events) if tick is None else tick
        # A dashboard capture has named trades only; a client capture has the same prints unnamed.
        self._print_kind = "trade" if any(e["k"] == "trade" for e in events) else "named_trade"
        self.position_limit = position_limit
        self.outstanding_limit = outstanding_limit
        self.now = events[0]["t"] if events else 0.0
        self.limit_breaches = 0
        self.writes = 0
        self.fills = 0
        self._idx = 0
        self._connected = False
        self._books: dict[str, tuple[list, list]] = {}
        self._orders: dict[int, _Resting] = {}
        self._next_id = 1
        self._positions: dict[str, int] = defaultdict(int)
        self._cash: dict[str, float] = defaultdict(float)
        self._mids: dict[str, float] = {}
        self._new_fills: dict[str, list] = defaultdict(list)

    # --- session ------------------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def tick_of(self, instrument_id: str) -> float:
        if isinstance(self._ticks, Mapping):
            return self._ticks.get(instrument_id, DEFAULT_TICK)
        return self._ticks

    def is_connected(self) -> bool:
        return self._connected

    # --- replay -------------------------------------------------------------------------------

    def advance_to(self, now: float) -> None:
        """Book events and arriving orders, merged in time order, so an order meets the book of its moment."""
        while True:
            arrival = self._in_flight[0][0] if self._in_flight else float("inf")
            has_event = self._idx < len(self.events) and self.events[self._idx]["t"] <= now
            event_time = self.events[self._idx]["t"] if has_event else float("inf")
            if arrival <= now and arrival <= event_time:
                self.now = arrival
                _, action, args = self._in_flight.pop(0)
                action(*args)
            elif has_event:
                event = self.events[self._idx]
                self._idx += 1
                self.now = event["t"]
                if event["k"] == "book":
                    self._on_book(event)
                elif event["k"] == self._print_kind:
                    self._on_print(event)
            else:
                break
        self.now = now

    def _send(self, action, *args) -> None:
        if self.latency_s > 0:
            self._in_flight.append((self.now + self.latency_s, action, args))
        else:
            action(*args)

    @property
    def finished(self) -> bool:
        return self._idx >= len(self.events)

    def _on_book(self, event: Mapping) -> None:
        iid = event["iid"]
        bids = [list(level) for level in event["bids"]]
        asks = [list(level) for level in event["asks"]]
        self._books[iid] = (bids, asks)
        if bids and asks:
            self._mids[iid] = (bids[0][0] + asks[0][0]) / 2.0
        for order in list(self._orders.values()):
            if order.instrument_id != iid:
                continue
            crossed = (
                (order.side == "bid" and asks and asks[0][0] <= order.price)
                or (order.side == "ask" and bids and bids[0][0] >= order.price)
            )
            if crossed:
                self._fill_resting(order, order.volume)

    def _on_print(self, event: Mapping) -> None:
        iid, price, remaining = event["iid"], event["px"], event["vol"]
        resting_side = "bid" if event.get("aggr") == "ask" else "ask"
        orders = sorted(
            (o for o in self._orders.values() if o.instrument_id == iid and o.side == resting_side),
            key=lambda o: o.order_id,
        )
        for order in orders:
            if remaining <= 0:
                break
            through = price < order.price if resting_side == "bid" else price > order.price
            tick = self.tick_of(iid)
            at_price = snap(price, tick) == snap(order.price, tick)
            if through:
                traded = min(order.volume, remaining)
                self._fill_resting(order, traded)
                remaining -= traded
            elif at_price:
                eaten = min(order.queue_ahead, remaining)
                left = remaining - eaten
                traded = min(order.volume, left)
                updated = order._replace(queue_ahead=order.queue_ahead - eaten)
                self._orders[order.order_id] = updated
                if traded > 0:
                    self._fill_resting(updated, traded)

    def _fill_resting(self, order: _Resting, volume: int) -> None:
        self._book_fill(order.instrument_id, order.side, order.price, volume, order.order_id)
        if volume >= order.volume:
            self._orders.pop(order.order_id, None)
        else:
            self._orders[order.order_id] = order._replace(volume=order.volume - volume)

    def _book_fill(self, iid: str, side: str, price: float, volume: int, order_id: int) -> None:
        sign = 1 if side == "bid" else -1
        self._positions[iid] += sign * volume
        self._cash[iid] -= sign * price * volume
        self.fills += 1
        self._new_fills[iid].append(Trade(self.now, order_id, 0, iid, price, volume, side))

    # --- writes -------------------------------------------------------------------------------

    def _open(self, iid: str, side: str) -> int:
        return sum(
            o.volume for o in self._orders.values() if o.instrument_id == iid and o.side == side
        )

    def _breaches(self, iid: str, side: str, volume: int, price: float) -> bool:
        opposite = "ask" if side == "bid" else "bid"
        for order in self._orders.values():
            crosses = price >= order.price if side == "bid" else price <= order.price
            if order.instrument_id == iid and order.side == opposite and crosses:
                return True  # a self-trade is reported to force a disconnect
        position = self._positions[iid]
        open_buy, open_sell = self._open(iid, "bid"), self._open(iid, "ask")
        if open_buy + open_sell + volume > self.outstanding_limit:
            return True
        if side == "bid":
            return position + open_buy + volume > self.position_limit
        return position - open_sell - volume < -self.position_limit

    def _visible_at(self, iid: str, side: str, price: float) -> int:
        levels = self._books.get(iid, ([], []))[0 if side == "bid" else 1]
        tick = self.tick_of(iid)
        target = snap(price, tick)
        return sum(level[1] for level in levels if snap(level[0], tick) == target)

    def _sweep(self, iid: str, side: str, price: float, volume: int, order_id: int) -> int:
        bids, asks = self._books.get(iid, ([], []))
        levels = asks if side == "bid" else bids
        crosses = (lambda p: p <= price) if side == "bid" else (lambda p: p >= price)
        remaining = volume
        for level in levels:
            if remaining <= 0 or not crosses(level[0]):
                break
            traded = min(remaining, level[1])
            if traded > 0:
                self._book_fill(iid, side, level[0], traded, order_id)
                level[1] -= traded
                remaining -= traded
        return remaining

    def insert_order(
        self, instrument_id: str, *, price: float, volume: int, side: str, order_type: str = "limit"
    ) -> InsertOrderResponse:
        self.writes += 1
        order_id = self._next_id
        self._next_id += 1
        self._send(self._arrive_insert, order_id, instrument_id, price, volume, side, order_type)
        return InsertOrderResponse(True, order_id, "None")

    def _arrive_insert(
        self, order_id: int, instrument_id: str, price: float, volume: int, side: str, order_type: str
    ) -> None:
        if self._breaches(instrument_id, side, volume, price):
            self.limit_breaches += 1
            self._orders = {
                oid: o for oid, o in self._orders.items() if o.instrument_id != instrument_id
            }
            return
        remaining = self._sweep(instrument_id, side, price, volume, order_id)
        if remaining > 0 and order_type == "limit":
            ahead = self._visible_at(instrument_id, side, price)
            self._orders[order_id] = _Resting(order_id, instrument_id, side, price, remaining, ahead)

    def amend_order(self, instrument_id: str, *, order_id: int, volume: int) -> ActionResponse:
        self.writes += 1
        self._send(self._arrive_amend, order_id, volume)
        return ActionResponse(True, "None")

    def _arrive_amend(self, order_id: int, volume: int) -> None:
        order = self._orders.get(order_id)
        if order is not None:
            self._orders[order_id] = order._replace(volume=volume)

    def delete_order(self, instrument_id: str, *, order_id: int) -> ActionResponse:
        self.writes += 1
        self._send(self._arrive_delete, order_id)
        return ActionResponse(True, "None")

    def _arrive_delete(self, order_id: int) -> None:
        self._orders.pop(order_id, None)

    def delete_orders(self, instrument_id: str) -> None:
        self.writes += 1
        self._send(self._arrive_delete_all, instrument_id)

    def _arrive_delete_all(self, instrument_id: str) -> None:
        self._orders = {
            oid: o for oid, o in self._orders.items() if o.instrument_id != instrument_id
        }

    # --- reads --------------------------------------------------------------------------------

    def get_instruments(self) -> dict[str, SimInstrument]:
        return {iid: SimInstrument(iid, self.tick_of(iid)) for iid in self.instruments}

    def get_last_price_book(self, instrument_id: str) -> PriceBook | None:
        bids, asks = self._books.get(instrument_id, ([], []))
        return PriceBook(
            None,
            instrument_id,
            [PriceVolume(p, v) for p, v in bids if v > 0],
            [PriceVolume(p, v) for p, v in asks if v > 0],
        )

    def get_outstanding_orders(self, instrument_id: str) -> dict[int, OrderStatus]:
        return {
            oid: OrderStatus(oid, o.instrument_id, o.price, o.volume, o.side)
            for oid, o in self._orders.items()
            if o.instrument_id == instrument_id
        }

    def get_positions(self) -> dict[str, int]:
        return {iid: self._positions[iid] for iid in self.instruments}

    def get_pnl(self, valuations: Mapping[str, float] | None = None) -> float:
        marks = valuations or {}
        total = sum(self._cash.values())
        for iid, position in self._positions.items():
            total += position * marks.get(iid, self._mids.get(iid, 0.0))
        return total

    def poll_new_trade_ticks(self, instrument_id: str) -> list:
        return []

    def poll_new_trades(self, instrument_id: str) -> list:
        fills = self._new_fills[instrument_id]
        self._new_fills[instrument_id] = []
        return fills


@dataclass(frozen=True)
class ReplayResult:
    pnl: float
    pnl_path: Sequence[float]
    writes: int
    fills: int
    errors: int
    limit_breaches: int
    final_positions: Mapping[str, int]


def run_replay(
    events: Sequence[Mapping],
    strategy_factories: Sequence[Callable[[list[str]], object]],
    instruments: Sequence[str],
    step_s: float = 0.1,
    config: EngineConfig | None = None,
    params: ParamStore | None = None,
    tick: float | Mapping[str, float] | None = None,
    recorder=None,
    latency_s: float = 0.0,
) -> ReplayResult:
    limits = (config or EngineConfig()).limits
    exchange = ReplayExchange(events, instruments, tick, limits.position, limits.outstanding, latency_s)
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    strategies = [factory(list(instruments)) for factory in strategy_factories]
    engine = Engine(exchange, universe, strategies, config, params, recorder, clock=lambda: exchange.now)
    path: list[float] = []
    now = events[0]["t"] if events else 0.0
    end = events[-1]["t"] if events else 0.0
    next_sample = now
    while now <= end:
        exchange.advance_to(now)
        engine.tick()
        if now >= next_sample:
            path.append(exchange.get_pnl())
            next_sample = now + 1.0
        now += step_s
    return ReplayResult(
        exchange.get_pnl(), tuple(path), exchange.writes, exchange.fills, engine.errors,
        exchange.limit_breaches, exchange.get_positions(),
    )
