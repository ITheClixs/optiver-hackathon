"""Who is on the other side? Online fingerprints from anonymous flow.

Public trade ticks expose price, size, time and aggressor side, but buyer and seller were empty in
May 2026 client output, so identity must be inferred from behaviour. These detectors work online
and offline with O(1) cost per event. The action they can trigger is deliberately limited to small
bounded parameter changes: a frozen opponent model is what lost the G-Research round-3 field.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from typing import NamedTuple

MIN_TRADES = 30
PERIODIC_CV = 0.25
MOMENTUM_CORR = 0.3
INFORMED_T = 3.0
MAX_PENDING = 20_000


class Welford:
    __slots__ = ("mean", "n", "s2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.s2 = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.s2 += delta * (value - self.mean)

    def stats(self) -> dict:
        var = self.s2 / (self.n - 1) if self.n > 1 else 0.0
        t = self.mean / math.sqrt(var / self.n) if var > 0 else 0.0
        return {"mean": round(self.mean, 5), "t": round(t, 2), "n": self.n}


class _Series:
    def __init__(self) -> None:
        self.sizes: Counter = Counter()
        self.signed = 0
        self.count = 0
        self.gaps: deque[float] = deque(maxlen=500)
        self.last_trade: float | None = None
        self.mid: float | None = None
        self.mid_at_last_trade: float | None = None
        self.pairs: deque[tuple[float, int]] = deque(maxlen=500)


class FlowProfiler:
    """Size signature, timing regularity, aggressor bias, momentum and signed markouts."""

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 5.0, 30.0)) -> None:
        self.horizons_s = horizons_s
        self._series: dict[str, _Series] = defaultdict(_Series)
        self._pending: dict[str, dict[float, deque]] = defaultdict(
            lambda: {horizon: deque() for horizon in self.horizons_s}
        )
        self._marks: dict[str, dict[int, dict[float, Welford]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(Welford))
        )

    def on_mid(self, instrument_id: str, now: float, mid: float) -> None:
        self._series[instrument_id].mid = mid
        for horizon, pending in self._pending[instrument_id].items():
            while pending and pending[0][0] <= now:
                _, size, sign, price = pending.popleft()
                self._marks[instrument_id][size][horizon].add(sign * (mid - price))

    def on_trade(self, instrument_id: str, now: float, price: float, volume: int, aggressor: str) -> None:
        series = self._series[instrument_id]
        series.sizes[volume] += 1
        series.count += 1
        if series.last_trade is not None:
            series.gaps.append(now - series.last_trade)
        series.last_trade = now
        if aggressor not in ("bid", "ask"):
            return
        sign = 1 if aggressor == "bid" else -1
        series.signed += sign
        if series.mid is not None and series.mid_at_last_trade is not None:
            series.pairs.append((series.mid - series.mid_at_last_trade, sign))
        series.mid_at_last_trade = series.mid
        for horizon, pending in self._pending[instrument_id].items():
            if len(pending) < MAX_PENDING:
                pending.append((now + horizon, volume, sign, price))

    def fingerprints(self, instrument_id: str) -> dict:
        series = self._series[instrument_id]
        if series.count == 0:
            return {"n": 0}
        modal_size, modal_count = series.sizes.most_common(1)[0]
        gaps = list(series.gaps)
        mean_gap = statistics.mean(gaps) if gaps else 0.0
        cv = statistics.pstdev(gaps) / mean_gap if len(gaps) > 5 and mean_gap > 0 else None
        return {
            "n": series.count,
            "modal_size": modal_size,
            "modal_share": round(modal_count / series.count, 3),
            "interarrival_mean_s": round(mean_gap, 3),
            "interarrival_cv": round(cv, 3) if cv is not None else None,
            "periodic": cv is not None and cv < PERIODIC_CV,
            "aggressor_imbalance": round(series.signed / series.count, 3),
            "momentum_corr": round(_corr(series.pairs), 3),
        }

    def markouts(self, instrument_id: str) -> dict[int, dict[float, dict]]:
        """Mean of sign * (future mid - trade price), per size and horizon, in price units.

        Positive means aggressors of that size are informed: passive quotes lose to them.
        Negative means they overpay: passive quotes earn more than the spread.
        """
        return {
            size: {horizon: stats.stats() for horizon, stats in horizons.items()}
            for size, horizons in self._marks[instrument_id].items()
        }


def _corr(pairs) -> float:
    if len(pairs) < 10:
        return 0.0
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    sd_x, sd_y = statistics.pstdev(xs), statistics.pstdev(ys)
    if sd_x <= 0.0 or sd_y <= 0.0:
        return 0.0
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mean_x) * (y - mean_y) for x, y in pairs) / (len(xs) * sd_x * sd_y)


class BookProfiler:
    """Touch dynamics: how often the best price changes, who pennies, how long quotes survive."""

    def __init__(self, tick: float | Callable[[str], float] = 0.1, small_size: int = 50) -> None:
        self.tick_of = tick if callable(tick) else (lambda _instrument_id: tick)
        self.small_size = small_size
        self._last: dict[str, tuple] = {}
        self._first: dict[str, float] = {}
        self._now: dict[str, float] = {}
        self._changes: dict[str, int] = defaultdict(int)
        self._pennies: dict[str, int] = defaultdict(int)
        self._lifetimes: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))
        self._since: dict[str, float] = {}

    def on_top(
        self, instrument_id: str, now: float, bid: float, bid_vol: int, ask: float, ask_vol: int
    ) -> None:
        self._first.setdefault(instrument_id, now)
        self._now[instrument_id] = now
        last = self._last.get(instrument_id)
        self._last[instrument_id] = (bid, ask)
        if last is None:
            self._since[instrument_id] = now
            return
        last_bid, last_ask = last
        if bid == last_bid and ask == last_ask:
            return
        self._changes[instrument_id] += 1
        self._lifetimes[instrument_id].append(now - self._since[instrument_id])
        self._since[instrument_id] = now
        one_tick = 1.5 * self.tick_of(instrument_id)
        pennied_bid = 0 < bid - last_bid < one_tick and bid_vol <= self.small_size
        pennied_ask = 0 < last_ask - ask < one_tick and ask_vol <= self.small_size
        if pennied_bid or pennied_ask:
            self._pennies[instrument_id] += 1

    def fingerprints(self, instrument_id: str) -> dict:
        changes = self._changes.get(instrument_id, 0)
        elapsed = self._now.get(instrument_id, 0.0) - self._first.get(instrument_id, 0.0)
        lifetimes = self._lifetimes.get(instrument_id)
        return {
            "touch_changes": changes,
            "touch_changes_per_min": round(60.0 * changes / elapsed, 2) if elapsed > 0 else 0.0,
            "penny_share": round(self._pennies.get(instrument_id, 0) / changes, 3) if changes else 0.0,
            "median_touch_lifetime_s": round(statistics.median(lifetimes), 3) if lifetimes else None,
        }


def classify_flow(prints: dict, markouts: dict, tick: float = 0.1) -> list[str]:
    """Conservative labels; each one needs enough trades and a significant statistic."""
    if prints.get("n", 0) < MIN_TRADES:
        return ["insufficient_data"]
    labels: list[str] = []
    if prints.get("periodic"):
        labels.append("scheduled_bot")
    if prints.get("momentum_corr", 0.0) >= MOMENTUM_CORR:
        labels.append("momentum_chaser")
    elif prints.get("momentum_corr", 0.0) <= -MOMENTUM_CORR:
        labels.append("contrarian_flow")
    informed = overpaying = quiet = False
    for horizons in markouts.values():
        for stats in horizons.values():
            if stats["n"] < MIN_TRADES:
                continue
            if stats["t"] >= INFORMED_T and stats["mean"] >= tick:
                informed = True
            elif stats["t"] <= -INFORMED_T and stats["mean"] <= -tick:
                overpaying = True
            elif abs(stats["t"]) < INFORMED_T:
                quiet = True
    if informed:
        labels.append("informed_flow")
    if overpaying:
        labels.append("overpaying_flow")
    if quiet and not informed and not overpaying:
        labels.append("uninformed_flow")
    return labels


def classify_book(prints: dict) -> list[str]:
    labels: list[str] = []
    if prints.get("touch_changes", 0) >= 30 and prints.get("penny_share", 0.0) >= 0.3:
        labels.append("penny_war")
    return labels


class ParamChange(NamedTuple):
    parameter: str
    factor: float
    reason: str


_PLAYBOOK: dict[str, tuple[ParamChange, ...]] = {
    "informed_flow": (
        ParamChange("base_spread_ticks", 1.5, "takers are informed: widen"),
        ParamChange("size", 0.7, "takers are informed: show less"),
    ),
    "uninformed_flow": (
        ParamChange("base_spread_ticks", 0.8, "flow is uninformed: tighten toward the queue"),
        ParamChange("size", 1.2, "flow is uninformed: show more"),
    ),
    "overpaying_flow": (
        ParamChange("size", 1.3, "takers overpay: quote more into them"),
    ),
    "momentum_chaser": (
        ParamChange("vol_factor", 1.3, "chaser bursts: widen with volatility"),
        ParamChange("jump_ticks", 0.8, "chaser bursts: step away sooner"),
    ),
    "penny_war": (
        ParamChange("size", 0.5, "touch is contested: risk less per queue spot"),
        ParamChange("skew_ticks", 1.2, "touch is contested: clear inventory sooner"),
    ),
}


def recommend(labels: list[str]) -> list[ParamChange]:
    """Multiplicative nudges. Autopilot applies them while the label holds, then lets them lapse."""
    return [change for label in labels for change in _PLAYBOOK.get(label, ())]
