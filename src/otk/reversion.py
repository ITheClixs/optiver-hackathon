"""Paper trading of reversion on the live book, at executable prices, after costs.

This is what lets the system switch to aggressive mean reversion on its own: each arm trades one
paper unit, buys at the ask, sells at the bid, pays a haircut per fill, and takes the same exits as the
live strategy. Only arms whose paper round trips paid get promoted. The same code scans a capture
offline, so the live decision and the offline backtest cannot disagree about what reversion earned.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from otk.features import TimedZ
from otk.instruments import DEFAULT_TICK, infer_ticks, parse_instrument_id

_NEVER = float("-inf")
MIN_STD_TICKS = 0.25


@dataclass(frozen=True)
class ArmConfig:
    half_lives_s: tuple[float, ...] = (30.0, 120.0, 600.0)
    entry_zs: tuple[float, ...] = (1.0, 2.0)
    exit_z: float = 0.25
    stop_z: float = 5.0
    cooldown_s: float = 60.0
    min_edge_ticks: float = 1.0
    haircut_ticks: float = 0.5  # per fill: an IOC can miss the touch it saw
    window: int = 40
    max_warmup_s: float = 120.0


class ReversionArm:
    """One half-life and entry threshold, holding at most one paper unit."""

    def __init__(self, half_life_s: float, entry_z: float, config: ArmConfig, tick: float) -> None:
        self.half_life_s = half_life_s
        self.entry_z = entry_z
        self.config = config
        self.tick = tick
        self.position = 0
        self.stops = 0
        self.total_trades = 0
        self.last_close = _NEVER
        self.results: deque[float] = deque(maxlen=config.window)
        self._stats = TimedZ()
        self._entry = 0.0
        self._paused_until = _NEVER

    def _close(self, now: float, price: float) -> None:
        pnl = self.position * (price - self._entry) - 2.0 * self.config.haircut_ticks * self.tick
        self.results.append(pnl)
        self.total_trades += 1
        self.last_close = now
        self.position = 0

    def _maybe_exit(self, now: float, z: float, buy_px: float, sell_px: float) -> None:
        price = sell_px if self.position > 0 else buy_px
        if self.position * z <= -self.config.stop_z:
            self._close(now, price)
            self.stops += 1
            self._paused_until = now + self.config.cooldown_s
        elif abs(z) <= self.config.exit_z or self.position * z >= self.entry_z:
            self._close(now, price)

    def _maybe_enter(self, now: float, z: float, buy_px: float, sell_px: float) -> None:
        if now < self._paused_until:
            return
        mean = self._stats.mean
        edge = self.config.min_edge_ticks * self.tick
        if z <= -self.entry_z and mean - buy_px >= edge:
            self.position, self._entry = 1, buy_px
        elif z >= self.entry_z and sell_px - mean >= edge:
            self.position, self._entry = -1, sell_px

    def update(self, now: float, value: float, buy_px: float, sell_px: float) -> None:
        warmup = min(self.half_life_s, self.config.max_warmup_s)
        z = self._stats.update_z(now, value, self.half_life_s, warmup)
        if z is None:
            return
        if self.position:
            self._maybe_exit(now, z, buy_px, sell_px)
        else:
            self._maybe_enter(now, z, buy_px, sell_px)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def mean(self) -> float:
        return sum(self.results) / self.n if self.n else 0.0

    @property
    def t(self) -> float:
        if self.n < 2:
            return 0.0
        mean = self.mean
        var = sum((value - mean) ** 2 for value in self.results) / (self.n - 1)
        # Prices move in ticks, so a few equal round trips have no variance. That is not certainty.
        floor = (MIN_STD_TICKS * self.tick) ** 2
        return mean / math.sqrt(max(var, floor) / self.n)

    def recent_mean(self, count: int = 10) -> float:
        recent = list(self.results)[-count:]
        return sum(recent) / len(recent) if recent else 0.0

    def summary(self) -> dict:
        return {
            "half_life_s": self.half_life_s,
            "entry_z": self.entry_z,
            "round_trips": self.total_trades,
            "stops": self.stops,
            "recent_n": self.n,
            "mean_ticks": round(self.mean / self.tick, 3),
            "t": round(self.t, 2),
        }


def make_arms(tick: float, config: ArmConfig | None = None) -> list[ReversionArm]:
    chosen = config or ArmConfig()
    return [
        ReversionArm(half_life, entry, chosen, tick)
        for half_life in chosen.half_lives_s
        for entry in chosen.entry_zs
    ]


def best_arm(arms: Sequence[ReversionArm]) -> ReversionArm:
    return max(arms, key=lambda arm: (arm.t, arm.n))


def _top(event: Mapping) -> tuple | None:
    bids, asks = event.get("bids"), event.get("asks")
    return (bids[0][0], asks[0][0]) if bids and asks else None


def _pairs(instruments: Iterable[str]) -> dict[str, str]:
    ids = set(instruments)
    pairs = {}
    for iid in ids:
        info = parse_instrument_id(iid)
        if iid.endswith("_DUAL") and info.base in ids:
            pairs[iid] = info.base
    return pairs


def points_from_events(events: Iterable[Mapping], sample_s: float = 1.0) -> dict[str, tuple]:
    """Per instrument and per dual gap: (tick, [(now, value, buy price, sell price), ...])."""
    rows = list(events)
    ticks = infer_ticks(rows)
    tops: dict[str, tuple] = {}
    last: dict[str, float] = defaultdict(lambda: _NEVER)
    pairs = _pairs(e["iid"] for e in rows if e.get("k") == "book")
    points: dict[str, list] = defaultdict(list)
    tick_by_key: dict[str, float] = {}
    for event in rows:
        if event.get("k") != "book":
            continue
        iid, now = event["iid"], float(event["t"])
        top = _top(event)
        if top is None or now - last[iid] < sample_s:
            continue
        last[iid] = now
        tops[iid] = top
        tick = ticks.get(iid, DEFAULT_TICK)
        middle = (top[0] + top[1]) / 2.0
        points[iid].append((now, middle, top[1], top[0]))
        tick_by_key[iid] = tick
        primary = pairs.get(iid)
        if primary and primary in tops:
            bid_p, ask_p = tops[primary]
            key = f"{iid}-gap"
            points[key].append((now, middle - (bid_p + ask_p) / 2.0, top[1] - bid_p, top[0] - ask_p))
            tick_by_key[key] = tick
    return {key: (tick_by_key[key], series) for key, series in points.items()}


def scan_events(
    events: Iterable[Mapping], config: ArmConfig | None = None, sample_s: float = 1.0
) -> dict:
    """Run every arm over a capture: one row per instrument and per dual gap."""
    report = {}
    for key, (tick, series) in sorted(points_from_events(events, sample_s).items()):
        arms = make_arms(tick, config)
        for point in series:
            for arm in arms:
                arm.update(*point)
        report[key] = {"best": best_arm(arms).summary(), "arms": [arm.summary() for arm in arms]}
    return report
