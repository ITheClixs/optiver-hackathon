"""The backtest lab: how this order book actually behaves, before anyone picks a strategy.

`behaviour.py` describes a capture. This answers the questions a strategy choice turns on:

  * structure: does the price revert, trend, or wander? Variance ratios per horizon. Fading a random
    walk pays the spread for nothing, and fading a trend is how a public team finished near the bottom.
  * imbalance: does the book predict the next move, and does the near touch beat the full book?
  * passive surface: what would a resting order earn at each distance from the touch, after adverse
    selection? This is the number that sets the spread to quote.
  * arb windows: how often the listings cross, how much, and for how long. Their lifetime is how fast we
    need to be: a window that lasts seconds needs no millisecond work.

Every section is a measurement with its sample size, and `reading` turns them into plain sentences. All
of it is stdlib and works on a capture from the bot, the dashboard feed, or the public quote files.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from otk.instruments import DEFAULT_TICK, infer_ticks

GRID_S = 1.0
VR_HORIZONS_S = (2, 5, 10, 30, 60)
SIGNAL_HORIZONS_S = (1, 5, 30)
MIN_POINTS = 100
REVERTING_BELOW = 0.7
TRENDING_ABOVE = 1.3
TREND_T = 5.0  # a variance ratio cannot see a steady drift, since it subtracts the mean
NEAR_TOUCH_TICKS = 4.0
PASSIVE_DISTANCES = (0, 1, 2, 3, 5)
PASSIVE_SAMPLE_S = 5.0
PASSIVE_REST_S = 5.0
MARKOUT_S = 5.0
MIN_FILLS = 10
MAX_LEVELS = 10


def read_quote_csv(path: str | Path, instrument_id: str) -> Iterator[dict]:
    """Rows of recv_time, bid_px_0, bid_vol_0, ..., ask_px_0, ... as book events. Bad rows are skipped."""
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                stamp = datetime.fromisoformat(row["recv_time"]).timestamp()
                yield {
                    "t": stamp,
                    "k": "book",
                    "iid": instrument_id,
                    "bids": _csv_levels(row, "bid"),
                    "asks": _csv_levels(row, "ask"),
                }
            except (KeyError, TypeError, ValueError):
                continue


def read_trade_csv(path: str | Path, instrument_id: str) -> Iterator[dict]:
    """Rows of recv_time, price, volume, aggressor_side, buyer, seller as named trades."""
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                yield {
                    "t": datetime.fromisoformat(row["recv_time"]).timestamp(),
                    "k": "named_trade",
                    "iid": instrument_id,
                    "px": float(row["price"]),
                    "vol": int(float(row["volume"])),
                    "aggr": row.get("aggressor_side", ""),
                    "buyer": row.get("buyer", ""),
                    "seller": row.get("seller", ""),
                }
            except (KeyError, TypeError, ValueError):
                continue


def _csv_levels(row: Mapping, side: str) -> list[list]:
    levels = []
    for level in range(MAX_LEVELS):
        price, volume = row.get(f"{side}_px_{level}"), row.get(f"{side}_vol_{level}")
        if price and volume:
            levels.append([float(price), int(float(volume))])
    return levels


def _books(events: Iterable[Mapping]) -> dict[str, list[Mapping]]:
    books: dict[str, list[Mapping]] = defaultdict(list)
    for event in events:
        if event.get("k") == "book" and event.get("bids") and event.get("asks"):
            books[event["iid"]].append(event)
    return books


def _mid(event: Mapping) -> float:
    return (event["bids"][0][0] + event["asks"][0][0]) / 2.0


def _grid(rows: Sequence[Mapping], step: float = GRID_S) -> list[Mapping]:
    """The last book of each step, carried forward, so busy moments do not get extra weight."""
    if not rows:
        return []
    grid = []
    now = rows[0]["t"]
    index = 0
    while now <= rows[-1]["t"]:
        while index + 1 < len(rows) and rows[index + 1]["t"] <= now:
            index += 1
        grid.append(rows[index])
        now += step
    return grid


# --- structure ------------------------------------------------------------------------------------


def _variance_ratio(mids: Sequence[float], k: int) -> float | None:
    if len(mids) <= k + 2:
        return None
    one = [mids[i + 1] - mids[i] for i in range(len(mids) - 1)]
    many = [mids[i + k] - mids[i] for i in range(len(mids) - k)]
    base = statistics.pvariance(one)
    return round(statistics.pvariance(many) / (k * base), 3) if base > 0 else None


def _trend_t(mids: Sequence[float]) -> float:
    steps = [mids[i + 1] - mids[i] for i in range(len(mids) - 1)]
    if len(steps) < 2:
        return 0.0
    spread = statistics.pstdev(steps)
    return round(statistics.fmean(steps) / (spread / math.sqrt(len(steps))), 2) if spread > 0 else 0.0


def _regime(ratios: Mapping[str, float | None], points: int, trend_t: float) -> str:
    """Measured against 5 s, not 1 s: bid-ask bounce alone makes a 1 s base read as reverting."""
    base = ratios.get("5")
    long_horizons = [ratios[str(h)] for h in VR_HORIZONS_S if h >= 10 and ratios.get(str(h)) is not None]
    if points < MIN_POINTS or not base or not long_horizons:
        return "unknown"
    known = [value / base for value in long_horizons]
    if abs(trend_t) >= TREND_T:
        return "trending"
    middle = statistics.median(known)
    if middle < REVERTING_BELOW:
        return "reverting"
    return "trending" if middle > TRENDING_ABOVE else "random_walk"


def return_structure(events: Iterable[Mapping]) -> dict:
    report = {}
    for iid, rows in _books(events).items():
        mids = [_mid(row) for row in _grid(rows)]
        ratios = {str(k): _variance_ratio(mids, k) for k in VR_HORIZONS_S}
        trend_t = _trend_t(mids)
        report[iid] = {
            "n": len(mids),
            "variance_ratio": ratios,
            "trend_t": trend_t,
            "regime": _regime(ratios, len(mids), trend_t),
        }
    return report


# --- does the book predict? -----------------------------------------------------------------------


def _top_imbalance(row: Mapping) -> float:
    bid, ask = row["bids"][0][1], row["asks"][0][1]
    return (bid - ask) / (bid + ask) if bid + ask > 0 else 0.0


def _band_imbalance(row: Mapping, band: float) -> float:
    center = _mid(row)
    bid = sum(level[1] for level in row["bids"] if center - level[0] <= band)
    ask = sum(level[1] for level in row["asks"] if level[0] - center <= band)
    return (bid - ask) / (bid + ask) if bid + ask > 0 else 0.0


def _full_imbalance(row: Mapping) -> float:
    bid, ask = sum(level[1] for level in row["bids"]), sum(level[1] for level in row["asks"])
    return (bid - ask) / (bid + ask) if bid + ask > 0 else 0.0


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> dict:
    n = len(xs)
    if n < MIN_POINTS:
        return {"corr": 0.0, "t": 0.0, "n": n}
    sd_x, sd_y = statistics.pstdev(xs), statistics.pstdev(ys)
    if sd_x <= 0 or sd_y <= 0:
        return {"corr": 0.0, "t": 0.0, "n": n}
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    corr = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (n * sd_x * sd_y)
    bounded = max(-0.999999, min(0.999999, corr))
    return {"corr": round(corr, 4), "t": round(bounded * math.sqrt((n - 2) / (1 - bounded**2)), 2), "n": n}


def imbalance_signal(events: Iterable[Mapping], tick: float | None = None) -> dict:
    """Correlation of each imbalance with the next move, at one-second sampling."""
    rows_list = list(events)
    ticks = infer_ticks(rows_list)
    report = {}
    for iid, rows in _books(rows_list).items():
        grid = _grid(rows)
        band = NEAR_TOUCH_TICKS * (tick or ticks.get(iid, DEFAULT_TICK))
        measures = {
            "top": [_top_imbalance(row) for row in grid],
            "near": [_band_imbalance(row, band) for row in grid],
            "full": [_full_imbalance(row) for row in grid],
        }
        mids = [_mid(row) for row in grid]
        report[iid] = {
            name: {
                str(h): _correlation(values[:-h], [mids[i + h] - mids[i] for i in range(len(mids) - h)])
                for h in SIGNAL_HORIZONS_S
                if len(values) > h
            }
            for name, values in measures.items()
        }
    return report


# --- what would a resting order earn? --------------------------------------------------------------


def _prints(events: Iterable[Mapping]) -> dict[str, list[Mapping]]:
    prints: dict[str, list[Mapping]] = defaultdict(list)
    for event in events:
        if event.get("k") in ("trade", "named_trade"):
            prints[event["iid"]].append(event)
    return prints


class _MidLookup:
    """The mid at or after a time, by walking forward, since queries arrive in time order."""

    def __init__(self, rows: Sequence[Mapping]) -> None:
        self.times = [row["t"] for row in rows]
        self.mids = [_mid(row) for row in rows]
        self.index = 0

    def at(self, moment: float) -> float | None:
        while self.index < len(self.times) and self.times[self.index] < moment:
            self.index += 1
        return self.mids[self.index] if self.index < len(self.times) else None


def _rest_order(
    side: str, price: float, ahead: float, started: float, prints: Sequence[Mapping], tick: float
) -> tuple[float, float] | None:
    """When and how a virtual order fills: through its price at once, at its price after the queue."""
    for event in prints:
        if event["t"] < started:
            continue
        if event["t"] > started + PASSIVE_REST_S:
            return None
        if side == "bid" and event.get("aggr") == "ask":
            if event["px"] < price - tick / 2.0:
                return event["t"], price
            if abs(event["px"] - price) < tick / 2.0:
                ahead -= event["vol"]
                if ahead < 0:
                    return event["t"], price
        elif side == "ask" and event.get("aggr") == "bid":
            if event["px"] > price + tick / 2.0:
                return event["t"], price
            if abs(event["px"] - price) < tick / 2.0:
                ahead -= event["vol"]
                if ahead < 0:
                    return event["t"], price
    return None


def _visible_at(levels: Sequence[Sequence], price: float, tick: float) -> float:
    return sum(level[1] for level in levels if abs(level[0] - price) < tick / 2.0)


def _surface_row(rows: Sequence[Mapping], prints: Sequence[Mapping], distance: int, tick: float) -> dict:
    lookup = _MidLookup(rows)
    scores = []
    next_sample = rows[0]["t"]
    cursor = 0
    for row in rows:
        if row["t"] < next_sample:
            continue
        next_sample = row["t"] + PASSIVE_SAMPLE_S
        while cursor < len(prints) and prints[cursor]["t"] < row["t"]:
            cursor += 1
        window = prints[cursor:]
        for side, levels, sign in (("bid", row["bids"], 1), ("ask", row["asks"], -1)):
            price = levels[0][0] - sign * distance * tick
            filled = _rest_order(side, price, _visible_at(levels, price, tick), row["t"], window, tick)
            if filled is None:
                continue
            later = lookup.at(filled[0] + MARKOUT_S)
            if later is not None:
                scores.append(sign * (later - filled[1]) / tick)
    return {
        "fills": len(scores),
        "net_ticks": round(statistics.fmean(scores), 3) if scores else 0.0,
    }


def passive_surface(events: Iterable[Mapping], tick: float | None = None) -> dict:
    """Mean markout of a resting order per distance from the touch, joining behind visible size."""
    rows_list = sorted(events, key=lambda event: event["t"])
    ticks = infer_ticks(rows_list)
    prints = _prints(rows_list)
    report = {}
    for iid, rows in _books(rows_list).items():
        unit = tick or ticks.get(iid, DEFAULT_TICK)
        surface: dict = {
            str(distance): _surface_row(rows, prints.get(iid, []), distance, unit)
            for distance in PASSIVE_DISTANCES
        }
        usable = {d: row for d, row in surface.items() if row["fills"] >= MIN_FILLS}
        # Earned per sample: a far quote can earn most per fill and still earn least in total.
        surface["best_distance_ticks"] = (
            int(max(usable, key=lambda d: usable[d]["net_ticks"] * usable[d]["fills"])) if usable else 0
        )
        report[iid] = surface
    return report


# --- how fast must we be? --------------------------------------------------------------------------


def _pairs(ids: Iterable[str]) -> list[tuple[str, str]]:
    present = set(ids)
    return [(iid[: -len("_DUAL")], iid) for iid in sorted(present) if iid.endswith("_DUAL") and iid[: -len("_DUAL")] in present]


def _window_stats(windows: Sequence[tuple[float, float, int]], unit: float) -> dict:
    if not windows:
        return {"count": 0}
    lives, edges, sizes = zip(*windows)
    return {
        "count": len(windows),
        "median_life_s": round(statistics.median(lives), 3),
        "p90_life_s": round(sorted(lives)[int(0.9 * len(lives))], 3),
        "median_edge_ticks": round(statistics.median(edges) / unit, 2),
        "median_size": statistics.median(sizes),
        "earnable_ticks_lots": round(sum(e * s for e, s in zip(edges, sizes)) / unit, 1),
    }


def arb_windows(events: Iterable[Mapping]) -> dict:
    """Each spell when one listing's bid is above the other's ask. Its lifetime is the speed needed."""
    rows_list = sorted(events, key=lambda event: event["t"])
    ticks = infer_ticks(rows_list)
    books = _books(rows_list)
    report = {}
    for primary, dual in _pairs(books):
        merged = sorted(books[primary] + books[dual], key=lambda event: event["t"])
        tops: dict[str, Mapping] = {}
        windows: list[tuple[float, float, int]] = []
        opened: tuple | None = None
        for row in merged:
            tops[row["iid"]] = row
            if len(tops) < 2:
                continue
            a, b = tops[primary], tops[dual]
            edge_up, edge_down = b["bids"][0][0] - a["asks"][0][0], a["bids"][0][0] - b["asks"][0][0]
            edge = max(edge_up, edge_down)
            if edge > 1e-9:
                if opened is None:
                    legs = (b["bids"][0][1], a["asks"][0][1]) if edge_up >= edge_down else (a["bids"][0][1], b["asks"][0][1])
                    opened = (row["t"], edge, min(legs))
            elif opened is not None:
                windows.append((row["t"] - opened[0], opened[1], opened[2]))
                opened = None
        report[dual] = _window_stats(windows, ticks.get(dual, DEFAULT_TICK))
    return report


# --- the report -----------------------------------------------------------------------------------


def _reading(structure: Mapping, signal: Mapping, passive: Mapping, windows: Mapping) -> list[str]:
    lines = []
    for iid, row in sorted(structure.items()):
        if row["regime"] == "reverting":
            lines.append(f"{iid}: reverts. Fading can pay here; check it clears the spread with reversion_scan.")
        elif row["regime"] == "trending":
            lines.append(f"{iid}: trends. Do not fade it. Quote wider and lean with the move.")
        elif row["regime"] == "random_walk":
            lines.append(f"{iid}: a random walk. Fading has no edge; earn the spread instead.")
    for iid, surface in sorted(passive.items()):
        touch = surface["0"]
        if touch["fills"] >= MIN_FILLS:
            verdict = "earns" if touch["net_ticks"] > 0 else "loses"
            lines.append(
                f"{iid}: a quote at the touch {verdict} {touch['net_ticks']:+.2f} ticks per fill. "
                f"Best distance by total: {surface['best_distance_ticks']} ticks."
            )
    for iid, row in sorted(signal.items()):
        near, full = row.get("near", {}).get("5", {}), row.get("full", {}).get("5", {})
        if abs(near.get("t", 0.0)) > 3:
            lines.append(f"{iid}: near-touch imbalance predicts 5 s moves, corr {near['corr']}, full book {full.get('corr')}.")
    for iid, window in sorted(windows.items()):
        if window.get("count"):
            lines.append(
                f"{iid}: {window['count']} arb windows, median life {window['median_life_s']} s, "
                f"edge {window['median_edge_ticks']} ticks, size {window['median_size']}."
            )
    return lines


def lab_report(events: Iterable[Mapping]) -> dict:
    rows = sorted(events, key=lambda event: event["t"])
    structure = return_structure(rows)
    signal = imbalance_signal(rows)
    passive = passive_surface(rows)
    windows = arb_windows(rows)
    return {
        "structure": structure,
        "imbalance": signal,
        "passive": passive,
        "arb_windows": windows,
        "reading": _reading(structure, signal, passive, windows),
    }


def summary_lines(report: Mapping) -> list[str]:
    lines = list(report["reading"])
    for iid, row in sorted(report["structure"].items()):
        lines.append(f"{iid}: regime {row['regime']}, variance ratios {row['variance_ratio']}, n={row['n']}")
    return lines
