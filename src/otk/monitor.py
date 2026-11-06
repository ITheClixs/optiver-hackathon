"""Live view of who is on the other side, and whether our fills are good.

Runs inside the trading process and is cheap per tick. Its snapshot goes into `run/status.json`,
which is what supervising agents read. It only observes: nothing here sends orders.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable

from otk.fair import mid
from otk.opponents import BookProfiler, FlowProfiler, classify_book, classify_flow, recommend
from otk.recorder import NullRecorder

logger = logging.getLogger("otk.monitor")

MARKOUT_HORIZON_S = 5.0
UNKNOWN_STRATEGY = "unknown"


class _Tally:
    """Cash and holdings of one strategy or one instrument, so its PnL can be read at any mid."""

    __slots__ = ("cash", "fills", "held", "last_price", "lots")

    def __init__(self) -> None:
        self.fills = 0
        self.lots = 0
        self.cash = 0.0
        self.held: dict[str, int] = {}
        self.last_price: dict[str, float] = {}

    def add(self, instrument_id: str, sign: int, price: float, volume: int) -> None:
        self.fills += 1
        self.lots += volume
        self.cash -= sign * price * volume
        self.held[instrument_id] = self.held.get(instrument_id, 0) + sign * volume
        self.last_price[instrument_id] = price

    def pnl(self, mids: dict[str, float]) -> float:
        return self.cash + sum(lots * mids.get(iid, self.last_price[iid]) for iid, lots in self.held.items())


class _OwnFills:
    """Signed markout of our fills: s * (mid after horizon - fill price). Negative means picked off."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.scored = 0
        self.pending: deque = deque()

    def add(self, now: float, sign: int, price: float, volume: int) -> None:
        self.count += 1
        self.pending.append((now + MARKOUT_HORIZON_S, sign, price, volume))

    def settle(self, now: float, current_mid: float) -> None:
        while self.pending and self.pending[0][0] <= now:
            _, sign, price, volume = self.pending.popleft()
            self.total += sign * (current_mid - price) * volume
            self.scored += volume


class LiveMonitor(NullRecorder):
    def __init__(
        self,
        tick: float | Callable[[str], float] = 0.1,
        group_of: Callable[[str], str] | None = None,
        strategy_of: Callable[[int | None], str] | None = None,
    ) -> None:
        self.tick_of = tick if callable(tick) else (lambda _instrument_id: tick)
        self.group_of = group_of or (lambda instrument_id: instrument_id)
        self.strategy_of = strategy_of or (lambda _order_id: "")
        self.flow = FlowProfiler()
        self.book = BookProfiler(self.tick_of)
        self._fills: dict[str, _OwnFills] = defaultdict(_OwnFills)
        self._seen: set[str] = set()
        self._mids: dict[str, float] = {}
        self._by_strategy: dict[str, _Tally] = defaultdict(_Tally)
        self._by_instrument: dict[str, _Tally] = defaultdict(_Tally)
        self._strategy_fills: dict[str, dict[str, _OwnFills]] = defaultdict(lambda: defaultdict(_OwnFills))

    def tick(self, state) -> None:
        now = state.now
        for iid in state.books:
            top = state.top(iid)
            current = mid(top)
            if current is None:
                continue
            self._seen.add(iid)
            self._mids[iid] = current
            self.flow.on_mid(iid, now, current)
            self.book.on_top(iid, now, top.bid, top.bid_vol, top.ask, top.ask_vol)
            self._fills[iid].settle(now, current)
            for own in self._strategy_fills[iid].values():
                own.settle(now, current)
        for iid, ticks in state.new_ticks.items():
            for trade in ticks:
                self.flow.on_trade(iid, now, trade.price, trade.volume, trade.aggressor_side)
        for iid, fills in state.new_fills.items():
            for fill in fills:
                self._note_fill(now, iid, fill)

    def _note_fill(self, now: float, instrument_id: str, fill) -> None:
        sign = 1 if fill.side == "bid" else -1
        strategy = self.strategy_of(getattr(fill, "order_id", None)) or UNKNOWN_STRATEGY
        self._fills[instrument_id].add(now, sign, fill.price, fill.volume)
        self._strategy_fills[instrument_id][strategy].add(now, sign, fill.price, fill.volume)
        self._by_strategy[strategy].add(instrument_id, sign, fill.price, fill.volume)
        self._by_instrument[instrument_id].add(instrument_id, sign, fill.price, fill.volume)

    def by_strategy(self) -> dict:
        """Who earned what since this process started. Inventory it inherited counts for nobody."""
        return {
            strategy: {
                "pnl": round(tally.pnl(self._mids), 2),
                "fills": tally.fills,
                "lots": tally.lots,
                "markout_5s_ticks": self._strategy_markout(strategy),
            }
            for strategy, tally in sorted(self._by_strategy.items())
        }

    def _strategy_markout(self, strategy: str) -> float | None:
        total, scored = 0.0, 0
        for iid, owners in self._strategy_fills.items():
            own = owners.get(strategy)
            if own and own.scored:
                total += own.total / self.tick_of(iid)
                scored += own.scored
        return round(total / scored, 3) if scored else None

    def by_instrument(self) -> dict:
        """With its group, because the legs of a hedged pair are only worth judging together."""
        return {
            iid: {"pnl": round(tally.pnl(self._mids), 2), "fills": tally.fills, "lots": tally.lots, "group": self.group_of(iid)}
            for iid, tally in sorted(self._by_instrument.items())
        }

    def labels(self, instrument_id: str) -> list[str]:
        prints = self.flow.fingerprints(instrument_id)
        tick = self.tick_of(instrument_id)
        labels = classify_flow(prints, self.flow.markouts(instrument_id), tick)
        return labels + classify_book(self.book.fingerprints(instrument_id))

    def snapshot(self) -> dict:
        report = {}
        for iid in sorted(self._seen):
            prints = self.flow.fingerprints(iid)
            book = self.book.fingerprints(iid)
            labels = self.labels(iid)
            fills = self._fills[iid]
            per_lot = fills.total / fills.scored / self.tick_of(iid) if fills.scored else None
            report[iid] = {
                "flow": prints,
                "book": book,
                "labels": labels,
                "suggested": [change._asdict() for change in recommend(labels)],
                "own_fills": {
                    "n": fills.count,
                    "markout_5s_ticks": round(per_lot, 3) if per_lot is not None else None,
                },
            }
        return report


class TeeRecorder(NullRecorder):
    """Fan out recorder calls, so a failure in one cannot silence the others."""

    def __init__(self, recorders: list) -> None:
        self.recorders = list(recorders)
        self.errors = 0

    def _each(self, call: Callable) -> None:
        for recorder in self.recorders:
            try:
                call(recorder)
            except Exception:
                self.errors += 1
                if self.errors <= 5:
                    logger.exception("recorder %s failed", type(recorder).__name__)

    def record(self, kind: str, payload) -> None:
        self._each(lambda recorder: recorder.record(kind, payload))

    def tick(self, state) -> None:
        self._each(lambda recorder: recorder.tick(state))

    def close(self) -> None:
        self._each(lambda recorder: recorder.close())
