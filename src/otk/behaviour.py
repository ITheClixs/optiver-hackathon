"""Offline order-book behaviour report from a capture. Stdlib only, so it also runs in the devenv.

This answers what the book does before anyone writes a strategy: spreads, depth, size signatures,
mean reversion versus momentum, which leg leads, how fast a pair converges, how big jumps are and
what happens after them. Every number is a prompt to investigate, not an instruction to trade.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

from otk.instruments import DEFAULT_TICK, infer_ticks, parse_instrument_id
from otk.opponents import BookProfiler, FlowProfiler, classify_book, classify_flow, recommend

GRID_S = 1.0
JUMP_SIGMAS = 6.0
POST_JUMP_HORIZONS_S = (1, 5, 15, 30)


def autocorr(series: Sequence[float], lag: int = 1) -> float:
    if len(series) <= lag + 1:
        return 0.0
    mean = sum(series) / len(series)
    denominator = sum((value - mean) ** 2 for value in series)
    if denominator <= 0.0:
        return 0.0
    numerator = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(len(series) - lag))
    return numerator / denominator


def variance_ratio(levels: Sequence[float], k: int = 10) -> float:
    """Lo-MacKinlay: 1 for a random walk, below 1 for mean reversion, above 1 for momentum."""
    if len(levels) <= k + 1:
        return 1.0
    one = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]
    many = [levels[i + k] - levels[i] for i in range(len(levels) - k)]
    var_one = statistics.pvariance(one)
    return statistics.pvariance(many) / (k * var_one) if var_one > 0 else 1.0


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def _modes(counter: Counter, top: int = 5) -> list[list[float]]:
    total = sum(counter.values()) or 1
    return [[value, round(count / total, 4)] for value, count in counter.most_common(top)]


def _to_grid(points: Sequence[tuple[float, float]], start: float, end: float) -> list[float | None]:
    """Last observation carried forward onto a fixed one-second grid."""
    grid: list[float | None] = []
    idx, current = 0, None
    steps = int((end - start) / GRID_S) + 1
    for step in range(steps):
        now = start + step * GRID_S
        while idx < len(points) and points[idx][0] <= now:
            current = points[idx][1]
            idx += 1
        grid.append(current)
    return grid


def _jump_report(grid: Sequence[float | None], tick: float) -> dict:
    values = [value for value in grid if value is not None]
    if len(values) < 30:
        return {"count": 0}
    changes = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    sigma = statistics.pstdev(changes)
    if sigma <= 0.0:
        return {"count": 0}
    jumps = [i for i, change in enumerate(changes) if abs(change) >= JUMP_SIGMAS * sigma]
    follow: dict[int, list[float]] = defaultdict(list)
    for index in jumps:
        sign = 1.0 if changes[index] > 0 else -1.0
        for horizon in POST_JUMP_HORIZONS_S:
            if index + 1 + horizon < len(values):
                follow[horizon].append(sign * (values[index + 1 + horizon] - values[index + 1]) / tick)
    return {
        "count": len(jumps),
        "median_size_ticks": round(statistics.median(abs(changes[i]) / tick for i in jumps), 2)
        if jumps
        else 0.0,
        "drift_after_ticks": {h: round(statistics.mean(v), 2) for h, v in follow.items() if v},
    }


def dominant_period(series: Sequence[float], min_lag: int = 10, min_corr: float = 0.3) -> int | None:
    """First autocorrelation peak, in grid seconds. A Sep 2026 event showed a 180 s dual gap cycle."""
    limit = min(len(series) // 2, 900)
    correlations = [autocorr(series, lag) for lag in range(min_lag, limit)]
    for idx in range(1, len(correlations) - 1):
        peak = correlations[idx - 1] < correlations[idx] >= correlations[idx + 1]
        if peak and correlations[idx] >= min_corr:
            return idx + min_lag
    return None


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _pair_report(a: str, b: str, grid_a: Sequence, grid_b: Sequence, tick: float) -> dict | None:
    """Error-correction regression: the leg that adjusts toward the other is the follower."""
    rows = [(x, y) for x, y in zip(grid_a, grid_b) if x is not None and y is not None]
    if len(rows) < 30:
        return None
    spread = [x - y for x, y in rows]
    change_a = [rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1)]
    change_b = [rows[i + 1][1] - rows[i][1] for i in range(len(rows) - 1)]
    adjust_a = -_ols_slope(spread[:-1], change_a)  # positive: a falls when a is rich
    adjust_b = _ols_slope(spread[:-1], change_b)  # positive: b rises when a is rich
    total = adjust_a + adjust_b
    half_life = math.log(2) / total if 0 < total < 2 else None
    return {
        "pair": [a, b],
        "leader": b if adjust_a > adjust_b else a,
        "adjust_a": round(adjust_a, 4),
        "adjust_b": round(adjust_b, 4),
        "spread_half_life_s": round(half_life, 2) if half_life else None,
        "spread_mean_ticks": round(statistics.mean(spread) / tick, 2),
        "spread_std_ticks": round(statistics.pstdev(spread) / tick, 2),
        "spread_period_s": dominant_period(spread),
    }


def _related_pairs(instruments: Iterable[str]) -> list[tuple[str, str]]:
    ids = set(instruments)
    pairs = []
    for iid in sorted(ids):
        info = parse_instrument_id(iid)
        if iid != info.base and info.base in ids and info.strike is None:
            pairs.append((info.base, iid))
    return pairs


class _Collector:
    def __init__(self, tick_of: Callable[[str], float]) -> None:
        self.tick_of = tick_of
        self.mids: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.spreads: dict[str, list[float]] = defaultdict(list)
        self.top_sizes: dict[str, Counter] = defaultdict(Counter)
        self.trade_sizes: dict[str, Counter] = defaultdict(Counter)
        self.trade_volume: dict[str, int] = defaultdict(int)
        self.flow = FlowProfiler()
        self.book = BookProfiler(tick_of)
        self.news: list[float] = []
        self.first: float | None = None
        self.last = 0.0

    def on_book(self, now: float, event: Mapping) -> None:
        iid, bids, asks = event["iid"], event["bids"], event["asks"]
        if not bids or not asks:
            return
        best_bid, bid_vol, best_ask, ask_vol = bids[0][0], bids[0][1], asks[0][0], asks[0][1]
        mid = (best_bid + best_ask) / 2.0
        self.mids[iid].append((now, mid))
        self.spreads[iid].append((best_ask - best_bid) / self.tick_of(iid))
        self.top_sizes[iid][bid_vol] += 1
        self.top_sizes[iid][ask_vol] += 1
        self.flow.on_mid(iid, now, mid)
        self.book.on_top(iid, now, best_bid, bid_vol, best_ask, ask_vol)

    def on_trade(self, now: float, event: Mapping) -> None:
        iid = event["iid"]
        self.trade_sizes[iid][event["vol"]] += 1
        self.trade_volume[iid] += event["vol"]
        self.flow.on_trade(iid, now, event["px"], event["vol"], event.get("aggr", ""))

    def consume(self, events: Iterable[Mapping]) -> None:
        for event in events:
            now = float(event["t"])
            self.first = now if self.first is None else self.first
            self.last = now
            kind = event["k"]
            if kind == "book":
                self.on_book(now, event)
            elif kind == "trade":
                self.on_trade(now, event)
            elif kind == "news":
                self.news.append(now)


def _instrument_report(col: _Collector, iid: str, grid: Sequence) -> dict:
    values = [value for value in grid if value is not None]
    changes = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    spreads = col.spreads.get(iid, [])
    flow_prints = col.flow.fingerprints(iid)
    book_prints = col.book.fingerprints(iid)
    tick = col.tick_of(iid)
    labels = classify_flow(flow_prints, col.flow.markouts(iid), tick) + classify_book(book_prints)
    return {
        "tick": tick,
        "updates": len(spreads),
        "spread_ticks_median": statistics.median(spreads) if spreads else None,
        "spread_ticks_p90": _percentile(spreads, 0.9) if spreads else None,
        "top_size_modes": _modes(col.top_sizes.get(iid, Counter())),
        "trades": sum(col.trade_sizes.get(iid, Counter()).values()),
        "trade_volume": col.trade_volume.get(iid, 0),
        "trade_size_modes": _modes(col.trade_sizes.get(iid, Counter())),
        "return_ac1": round(autocorr(changes, 1), 4),
        "variance_ratio_10s": round(variance_ratio(values, 10), 3),
        "sigma_per_s_ticks": round(statistics.pstdev(changes) / tick, 3) if len(changes) > 1 else 0.0,
        "jumps": _jump_report(grid, tick),
        "flow": flow_prints,
        "book": book_prints,
        "markouts_ticks": {
            str(size): {
                str(h): {**stats, "mean": round(stats["mean"] / tick, 3)}
                for h, stats in horizons.items()
            }
            for size, horizons in col.flow.markouts(iid).items()
        },
        "labels": labels,
        "suggested_changes": [change._asdict() for change in recommend(labels)],
    }


def _tick_source(events: Sequence[Mapping], tick: float | None) -> Callable[[str], float]:
    if tick is not None:
        return lambda _instrument_id: tick
    ticks = infer_ticks(events)
    return lambda instrument_id: ticks.get(instrument_id, DEFAULT_TICK)


def analyse_capture(events: Iterable[Mapping], tick: float | None = None) -> dict:
    """Full report as JSON-serialisable data. Without `tick`, each instrument's tick is inferred."""
    rows = list(events)
    tick_of = _tick_source(rows, tick)
    col = _Collector(tick_of)
    col.consume(rows)
    if col.first is None:
        return {"duration_s": 0.0, "instruments": {}, "lead_lag": [], "news_events": 0}
    grids = {iid: _to_grid(points, col.first, col.last) for iid, points in col.mids.items()}
    per = {iid: _instrument_report(col, iid, grid) for iid, grid in sorted(grids.items())}
    pairs = [
        report
        for a, b in _related_pairs(grids)
        if (report := _pair_report(a, b, grids[a], grids[b], tick_of(b)))
    ]
    return {
        "duration_s": round(col.last - col.first, 3),
        "instruments": per,
        "lead_lag": pairs,
        "news_events": len(col.news),
    }
