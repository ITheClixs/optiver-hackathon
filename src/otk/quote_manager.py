"""Turns desired quotes into the fewest exchange updates, protecting queue priority.

Writes are the scarce resource (25 per second across all instruments) and price-time priority
makes an old order worth more than a new one at the same price, so an unchanged order is left
alone. Deletes always precede inserts because capacity counts resting orders.

The local view lags the exchange. Orders, fills and positions arrive through a client thread, so
this class keeps its own short memory of what it just did: inserts the cache has not listed yet,
deletes it still lists, and IOC volume whose fills have not reached the position. All of it counts
toward worst-case capacity, because a breach or a self-trade means a forced disconnect.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from collections.abc import Callable, Mapping
from itertools import islice
from types import SimpleNamespace

from otk.budget import Priority, WriteBudget
from otk.fair import snap
from otk.intents import LadderIntent, QuoteIntent, Take
from otk.risk import Limits, clip_order

logger = logging.getLogger("otk.orders")

UNLISTED_MEMORY_S = 1.0
IOC_MEMORY_S = 1.0
MISSING_QUOTE_URGENCY = 2.0  # in ticks: below a stale quote that can be picked off
TIGHT_BUDGET_SHARE = 0.25
TIGHT_BUDGET_EXTRA_TICKS = 1.0
WRITE_SAMPLES = 50
REJECT_BACKOFF_S = 2.0  # a price or instrument the exchange refuses will not improve in 20 ms
MAX_TAGS = 50_000  # about two hours of quoting at 7 writes a second


class QuoteManager:
    def __init__(
        self,
        exchange,
        budget: WriteBudget,
        limits: Limits,
        tick: float | Callable[[str], float] = 0.1,
        topup_lots: int = 3,
    ) -> None:
        self.exchange = exchange
        self.budget = budget
        self.limits = limits
        self.tick_of = tick if callable(tick) else (lambda _instrument_id: tick)
        self.topup_lots = topup_lots
        self.reprice_ticks = 0.0
        self.reprice_by: Mapping[str, float] = {}
        self._tight_budget = int(budget.available() * TIGHT_BUDGET_SHARE)
        self.rejects = 0
        self.sent = 0
        self.last_reject = ""
        self._unlisted: dict[int, SimpleNamespace] = {}
        self._deleted: dict[int, float] = {}
        self._ioc: dict[str, deque] = {}
        self._settling: list[tuple] = []
        self._sources: dict[int, str] = {}
        self._tags: dict[int, str] = {}
        self.source_positions: dict[tuple[str, str], int] = {}
        self._refused_until: dict[tuple[str, str], float] = {}
        self._write_ms: deque[float] = deque(maxlen=WRITE_SAMPLES)

    # --- local view ---------------------------------------------------------------------------

    def forget_orders(self) -> None:
        """After a disconnect the exchange has deleted every order we had."""
        self._unlisted.clear()
        self._deleted.clear()
        self._ioc.clear()
        self._settling = []
        self._sources.clear()
        self.source_positions.clear()  # fills may have gone missing: what is left is plain inventory
        self._refused_until.clear()

    def note_fills(self, fills: Mapping[str, list]) -> None:
        """A fully filled order may never be listed, so stop remembering it."""
        for settled in self._settling:
            self._settle_ioc(*settled)
        self._settling = []
        for instrument_id, instrument_fills in fills.items():
            for fill in instrument_fills:
                order_id = getattr(fill, "order_id", None)
                self._attribute(instrument_id, order_id, fill)
                # One tick later: the client appends a fill before it updates the position.
                self._settling.append((instrument_id, order_id, fill.volume))
                order = self._unlisted.get(order_id)
                if order is None:
                    continue
                order.volume -= fill.volume
                if order.volume <= 0:
                    del self._unlisted[order_id]

    def _attribute(self, instrument_id: str, order_id: int | None, fill) -> None:
        """A strategy that holds a view must judge only what it holds, not the whole inventory."""
        source = self._sources.get(order_id)
        if source:
            signed = fill.volume if fill.side == "bid" else -fill.volume
            key = (source, instrument_id)
            self.source_positions[key] = self.source_positions.get(key, 0) + signed

    def _tag(self, order_id: int | None, source: str) -> None:
        """Who sent it, only to report who earned what. Kept apart from `_sources`, which drives real decisions."""
        if order_id is None or not source:
            return
        self._tags[order_id] = source
        if len(self._tags) > MAX_TAGS:  # oldest first: an order that old will not fill now
            for old in list(islice(self._tags, MAX_TAGS // 2)):
                del self._tags[old]

    def strategy_of(self, order_id: int | None) -> str:
        return self._tags.get(order_id, "")

    def _settle_ioc(self, instrument_id: str, order_id: int | None, volume: int) -> None:
        """Fills and positions share a channel, so a filled IOC is now in the position."""
        for entry in self._ioc.get(instrument_id, ()):
            if entry[3] == order_id:
                entry[2] = max(0, entry[2] - volume)
                return

    def _expire(self, now: float) -> None:
        for order_id in [k for k, v in self._unlisted.items() if now - v.sent_at >= UNLISTED_MEMORY_S]:
            del self._unlisted[order_id]
        for order_id in [k for k, sent in self._deleted.items() if now - sent >= UNLISTED_MEMORY_S]:
            del self._deleted[order_id]

    def _orders(self, instrument_id: str) -> tuple[list, list]:
        self._expire(self.budget.now())
        listed = self.exchange.get_outstanding_orders(instrument_id)
        for order_id in listed:
            self._unlisted.pop(order_id, None)
        visible = [o for oid, o in listed.items() if oid not in self._deleted]
        visible += [o for o in self._unlisted.values() if o.instrument_id == instrument_id]
        bids = sorted((o for o in visible if o.side == "bid"), key=lambda o: o.order_id)
        asks = sorted((o for o in visible if o.side == "ask"), key=lambda o: o.order_id)
        return bids, asks

    def resting(self, instrument_id: str) -> list:
        """Our best knowledge of what rests now, including inserts the cache has not listed."""
        bids, asks = self._orders(instrument_id)
        return bids + asks

    def _pending_ioc(self, instrument_id: str) -> tuple[int, int]:
        ledger = self._ioc.get(instrument_id)
        if not ledger:
            return 0, 0
        now = self.budget.now()
        while ledger and now - ledger[0][0] >= IOC_MEMORY_S:
            ledger.popleft()
        buys = sum(entry[2] for entry in ledger if entry[1] == "bid")
        return buys, sum(entry[2] for entry in ledger if entry[1] == "ask")

    def _open_volume(self, instrument_id: str, bids: list, asks: list) -> tuple[int, int]:
        pending_buy, pending_sell = self._pending_ioc(instrument_id)
        return sum(o.volume for o in bids) + pending_buy, sum(o.volume for o in asks) + pending_sell

    def _note_response(self, response, what: str) -> None:
        if getattr(response, "success", True) is False:
            self.rejects += 1
            self.last_reject = f"{what}: {getattr(response, 'error_reason', '')}"
            if self.rejects <= 20:
                logger.warning("rejected %s", self.last_reject)

    # --- primitives ---------------------------------------------------------------------------

    def _delete(self, instrument_id: str, order_id: int) -> bool:
        if not self.budget.try_acquire(Priority.RISK):
            return False
        self.exchange.delete_order(instrument_id, order_id=order_id)
        self._unlisted.pop(order_id, None)
        self._deleted[order_id] = self.budget.now()
        return True

    def _close_enough(self, instrument_id: str, working: float, desired: float | None) -> bool:
        """Hysteresis: a one-tick wobble in fair value should not cost queue priority."""
        if desired is None:
            return False
        ticks = self.reprice_by.get(instrument_id, self.reprice_ticks)
        if self.budget.available() <= self._tight_budget:
            ticks = ticks * 2.0 + TIGHT_BUDGET_EXTRA_TICKS  # one tick is not worth a scarce write
        return abs(working - desired) <= ticks * self.tick_of(instrument_id) + 1e-9

    def _timed_insert(self, instrument_id: str, price: float, volume: int, side: str, order_type: str):
        """The client blocks for a round trip on each write, so this is how slow the connection really is."""
        self.sent += 1
        started = time.perf_counter()
        response = self.exchange.insert_order(instrument_id, price=price, volume=volume, side=side, order_type=order_type)
        self.note_write_ms((time.perf_counter() - started) * 1000.0)
        return response

    def note_write_ms(self, milliseconds: float) -> None:
        self._write_ms.append(milliseconds)

    def write_ms(self) -> float | None:
        """Median of recent writes, so one slow reply does not read as a slow connection."""
        return statistics.median(self._write_ms) if self._write_ms else None

    def write_samples(self) -> int:
        return len(self._write_ms)

    def _crosses_own(self, instrument_id: str, side: str, price: float, bids: list, asks: list) -> bool:
        """Participant reports say a self-trade forces a disconnect, so never send one. In ticks: orders go out snapped."""
        tick = self.tick_of(instrument_id)
        level = round(price / tick)
        if side == "bid":
            return any(round(order.price / tick) <= level for order in asks)
        return any(round(order.price / tick) >= level for order in bids)

    def _insert(
        self, instrument_id: str, side: str, price: float, shortfall: int, position: int, source: str = ""
    ) -> None:
        if self.budget.now() < self._refused_until.get((instrument_id, side), float("-inf")):
            return
        bids, asks = self._orders(instrument_id)
        if self._crosses_own(instrument_id, side, price, bids, asks):
            return
        open_buy, open_sell = self._open_volume(instrument_id, bids, asks)
        volume = clip_order(side, shortfall, position, open_buy, open_sell, self.limits)
        if volume <= 0 or not self.budget.try_acquire(Priority.QUOTE):
            return
        snapped = snap(price, self.tick_of(instrument_id))
        response = self._timed_insert(instrument_id, snapped, volume, side, "limit")
        self._note_response(response, f"insert {instrument_id} {side} {volume}@{snapped}")
        if getattr(response, "success", True) is False:
            self._refused_until[(instrument_id, side)] = self.budget.now() + REJECT_BACKOFF_S
        order_id = getattr(response, "order_id", None)
        if order_id and getattr(response, "success", True):
            self._unlisted[order_id] = SimpleNamespace(
                order_id=order_id, instrument_id=instrument_id, side=side, price=snapped,
                volume=volume, sent_at=self.budget.now(),
            )
            self._tag(order_id, source)

    # --- quoting ------------------------------------------------------------------------------

    def _match_level(
        self, instrument_id: str, price: float, levels: tuple, remaining: dict
    ) -> int | None:
        matches = [
            idx for idx, level in enumerate(levels)
            if self._close_enough(instrument_id, price, level[0]) and remaining[idx] > 0
        ]
        return min(matches, key=lambda idx: abs(levels[idx][0] - price)) if matches else None

    def _trim(self, instrument_id: str, working: list, levels: tuple) -> dict:
        """Delete off-level orders and amend over-size ones; return the lots each level still needs."""
        remaining = {idx: int(level[1]) for idx, level in enumerate(levels)}
        for order in working:
            idx = self._match_level(instrument_id, order.price, levels, remaining)
            if idx is None:
                self._delete(instrument_id, order.order_id)
                continue
            if order.volume > remaining[idx] and self.budget.try_acquire(Priority.RISK):
                # "Not found" is normal here: the order filled before the cache caught up.
                self.exchange.amend_order(
                    instrument_id, order_id=order.order_id, volume=remaining[idx]
                )
                remaining[idx] = 0
            else:
                remaining[idx] = max(0, remaining[idx] - order.volume)
        return remaining

    def _fill_levels(
        self, instrument_id: str, side: str, levels: tuple, remaining: dict, position: int, source: str = ""
    ) -> None:
        """Best rung first, so scarce capacity and write budget go to the most valuable level."""
        for idx, (price, size) in enumerate(levels):
            shortfall = remaining[idx]
            if shortfall > 0 and (shortfall == size or shortfall >= self.topup_lots):
                self._insert(instrument_id, side, price, shortfall, position, source)

    def sync(self, intent: QuoteIntent | LadderIntent, positions: Mapping[str, int]) -> None:
        iid = intent.instrument_id
        if isinstance(intent, LadderIntent):
            bid_levels, ask_levels = intent.bids, intent.asks
        else:
            bid_levels = ((intent.bid, intent.bid_size),) if intent.bid is not None else ()
            ask_levels = ((intent.ask, intent.ask_size),) if intent.ask is not None else ()
        position = positions.get(iid, 0)
        bids, asks = self._orders(iid)
        bid_gap = self._trim(iid, bids, bid_levels)
        ask_gap = self._trim(iid, asks, ask_levels)
        self._fill_levels(iid, "bid", bid_levels, bid_gap, position, intent.source)
        self._fill_levels(iid, "ask", ask_levels, ask_gap, position, intent.source)

    def urgency(self, intent: QuoteIntent | LadderIntent) -> float:
        """How wrong our resting quotes are, in ticks. The stalest are fixed first when budget is short."""
        iid = intent.instrument_id
        if isinstance(intent, LadderIntent):
            desired_bid = intent.bids[0][0] if intent.bids else None
            desired_ask = intent.asks[0][0] if intent.asks else None
        else:
            desired_bid, desired_ask = intent.bid, intent.ask
        bids, asks = self._orders(iid)
        tick = self.tick_of(iid)

        def side(working: list, desired: float | None) -> float:
            if desired is None:
                return MISSING_QUOTE_URGENCY if working else 0.0
            if not working:
                return MISSING_QUOTE_URGENCY
            return min(abs(order.price - desired) for order in working) / tick

        return max(side(bids, desired_bid), side(asks, desired_ask))

    def pull(self, instrument_id: str) -> bool:
        """Remove every resting order. True only when nothing is left to pull."""
        bids, asks = self._orders(instrument_id)
        orders = bids + asks
        if not orders:
            return True
        if len(orders) == 1:
            return self._delete(instrument_id, orders[0].order_id)
        if not self.budget.try_acquire(Priority.RISK):
            return False
        self.exchange.delete_orders(instrument_id)
        now = self.budget.now()
        for order in orders:
            self._unlisted.pop(order.order_id, None)
            self._deleted[order.order_id] = now
        return True

    # --- taking -------------------------------------------------------------------------------

    def _clear_crossing(self, intent: Take, bids: list, asks: list) -> bool:
        opposite = asks if intent.side == "bid" else bids
        tick = self.tick_of(intent.instrument_id)
        limit = round(intent.price / tick)  # in ticks: the order goes out snapped, so float noise must not hide a cross
        for order in opposite:
            own = round(order.price / tick)
            crosses = own <= limit if intent.side == "bid" else own >= limit
            if crosses and not self._delete(intent.instrument_id, order.order_id):
                return False
        return True

    def _take_volume(self, intent: Take, positions: Mapping[str, int]) -> int:
        iid = intent.instrument_id
        bids, asks = self._orders(iid)
        open_buy, open_sell = self._open_volume(iid, bids, asks)
        # Read now, as the open orders just were: a fill since the tick began lowers those and raises this.
        position = self.exchange.get_positions().get(iid, positions.get(iid, 0))
        return clip_order(intent.side, intent.volume, position, open_buy, open_sell, self.limits)

    def _reducible(self, intent: Take, position: int) -> int:
        """What is left to close, after exits already sent whose fills have not reported yet."""
        pending_buy, pending_sell = self._pending_ioc(intent.instrument_id)
        if intent.side == "ask":
            return max(0, position - pending_sell)
        return max(0, -position - pending_buy)

    def take(self, intent: Take, positions: Mapping[str, int]) -> int:
        """Send an IOC clipped to worst-case capacity. Fills arrive later through positions."""
        iid = intent.instrument_id
        bids, asks = self._orders(iid)
        if not self._clear_crossing(intent, bids, asks):
            return 0
        volume = self._take_volume(intent, positions)
        if intent.hedge and volume < intent.volume and (bids or asks):
            # Resting quotes must not stop a risk-reducing order: outstanding volume counts too.
            self.pull(iid)
            volume = self._take_volume(intent, positions)
        if intent.reduce_only:
            volume = min(volume, self._reducible(intent, positions.get(iid, 0)))
        priority = Priority.RISK if intent.hedge else Priority.TAKE
        if volume <= 0 or not self.budget.try_acquire(priority):
            return 0
        snapped = snap(intent.price, self.tick_of(iid))
        response = self._timed_insert(iid, snapped, volume, intent.side, "ioc")
        self._note_response(response, f"ioc {iid} {intent.side} {volume}@{snapped}")
        if getattr(response, "success", True):
            order_id = getattr(response, "order_id", None)
            if intent.source and order_id is not None:
                self._sources[order_id] = intent.source
            self._tag(order_id, intent.source)
            if intent.reduce_only:
                # An exit closes the whole position, whoever built it. Nothing is left for them to exit.
                for key in [key for key in self.source_positions if key[1] == iid]:
                    del self.source_positions[key]
            self._ioc.setdefault(iid, deque()).append(
                [self.budget.now(), intent.side, volume, order_id]
            )
        return volume


def group_intents(intents: list) -> dict[str, QuoteIntent | LadderIntent]:
    """First quote per instrument wins; strategy order is the priority order."""
    chosen: dict[str, QuoteIntent | LadderIntent] = {}
    for intent in intents:
        if isinstance(intent, (QuoteIntent, LadderIntent)) and intent.instrument_id not in chosen:
            chosen[intent.instrument_id] = intent
    return chosen
