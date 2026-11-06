"""Pick from a fixed list by replaying a capture, with no model, no tokens and no login. Default is no change.

Replay has no market impact and no rival racing us, so what it picks goes to a half-size live test that checks for
harm before it runs at full size. The list is fixed before the event, because a list you can add to on the day is a
search.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from otk.autopilot import Autopilot
from otk.engine import EngineConfig
from otk.formats import detect_pairs
from otk.instruments import Kind, Universe, infer_ticks, parse_instrument_id
from otk.params import (
    INSTRUMENTS_KEY,
    ParamStore,
    fine_tick_overrides,
    merge_params,
    validate_params,
)
from otk.recorder import read_capture
from otk.replay import load_events, run_replay
from otk.strategies import default_stack

CANDIDATES: tuple[dict, ...] = (
    {"skew_ticks": 3},
    {"skew_ticks": 2},
    {"hedge_threshold": 85, "hedge_to": 60},
    {"arb_edge_ticks": 2},
    {"base_spread_ticks": 3},
)
MIN_CAPTURE_S = 1200.0
SLICE_S = 60
MIN_T = 1.73  # two-sided 10% at 19 degrees of freedom, a 20-minute capture
MIN_EDGE_SHARE = 0.10
MAX_DRAWDOWN_RATIO = 1.25
DRAWDOWN_ALLOWANCE = 20.0  # so that a tiny baseline drawdown does not rule out everything
ORDER_DELAY_S = 0.05


class Verdict(NamedTuple):
    passed: bool
    edge: float
    t: float
    reason: str


def _slices(path: Sequence[float]) -> list[float]:
    return [path[end] - path[end - SLICE_S] for end in range(SLICE_S, len(path), SLICE_S)]


def _drawdown(path: Sequence[float]) -> float:
    peak, worst = float("-inf"), 0.0
    for value in path:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def judge(base, candidate) -> Verdict:
    """Same market moments on both sides, so a slice-by-slice difference is a fair comparison."""
    size = min(len(base.pnl_path), len(candidate.pnl_path))
    differences = [b - a for a, b in zip(_slices(base.pnl_path[:size]), _slices(candidate.pnl_path[:size]))]
    edge = candidate.pnl - base.pnl
    if len(differences) < 4:
        return Verdict(False, edge, 0.0, "too few slices")
    spread = statistics.stdev(differences)
    t = statistics.mean(differences) / (spread / len(differences) ** 0.5) if spread else 0.0
    half = len(differences) // 2
    if candidate.limit_breaches or candidate.errors:
        return Verdict(False, edge, t, f"{candidate.limit_breaches} breach(es), {candidate.errors} error(s)")
    if edge < MIN_EDGE_SHARE * abs(base.pnl):
        return Verdict(False, edge, t, f"edge under {MIN_EDGE_SHARE:.0%} of base PnL")
    if t < MIN_T:
        return Verdict(False, edge, t, f"t {t:.2f} under {MIN_T}")
    if sum(differences[:half]) <= 0 or sum(differences[half:]) <= 0:
        return Verdict(False, edge, t, "did not win in both halves")
    if _drawdown(candidate.pnl_path) > MAX_DRAWDOWN_RATIO * _drawdown(base.pnl_path) + DRAWDOWN_ALLOWANCE:
        return Verdict(False, edge, t, "drawdown much larger than base")
    return Verdict(True, edge, t, "wins clearly in both halves")


def choose(base, results: Sequence[tuple[dict, object]]) -> tuple[dict | None, list[tuple[dict, Verdict]]]:
    """The passing candidate with the largest edge, or None, which means leave everything as it is."""
    verdicts = [(overrides, judge(base, result)) for overrides, result in results]
    passing = [item for item in verdicts if item[1].passed]
    return (max(passing, key=lambda item: item[1].edge)[0] if passing else None), verdicts


def capture_problem(events: Sequence[Mapping], min_seconds: float, allow_own: bool) -> str | None:
    if not events or events[-1]["t"] - events[0]["t"] < min_seconds:
        return f"capture too short: need {min_seconds:.0f} s"
    # A fill with a strategy on it comes from a bot that leaves its own quotes out of the books it records.
    if not allow_own and any(event.get("k") == "fill" and "src" not in event for event in events):
        return "an older capture, with our own quotes in its books. Use a newer or passive one, or --allow-own"
    return None


def format_problem(ids: Sequence[str]) -> str | None:
    """What the replay here can stand for: stocks, and pairs named X and X_DUAL. Anything else needs event settings."""
    if any(parse_instrument_id(iid).kind not in (Kind.STOCK, Kind.DUAL) for iid in ids):
        return "this format needs index weights or a volatility, which the replay here does not have"
    odd = [pair for pair in detect_pairs(ids) if not pair[1].endswith("_DUAL")]
    return f"pairs named {odd}, and the replay here only knows X and X_DUAL" if odd else None


def replay(events: Sequence[Mapping], ids: Sequence[str], ticks: Mapping[str, float], overrides: Mapping, base: Mapping):
    universe = Universe(parse_instrument_id(iid)._replace(tick=ticks[iid]) for iid in ids)
    store = ParamStore(base=merge_params(base, overrides, {INSTRUMENTS_KEY: fine_tick_overrides(universe)}))
    stack = default_stack(ids)
    return run_replay(
        events, [(lambda _names, strategy=item: strategy) for item in stack], ids, step_s=ORDER_DELAY_S,
        config=EngineConfig(max_writes_per_second=20, risk_reserve=5), params=store, tick=ticks,
        recorder=Autopilot(store, universe), latency_s=ORDER_DELAY_S,
    )


def evaluate(events: Sequence[Mapping], base_params: Mapping) -> tuple[dict | None, object, list]:
    ids = sorted({event["iid"] for event in events if event.get("k") == "book"})
    ticks = infer_ticks(events)
    base = replay(events, ids, ticks, {}, base_params)
    results = [(candidate, replay(events, ids, ticks, candidate, base_params)) for candidate in CANDIDATES]
    picked, verdicts = choose(base, results)
    return picked, base, verdicts


def half_size_variants(picked: Mapping, base: Mapping) -> dict:
    """Half size on both sides, with the picked keys named on both, so the two differ in one thing only."""
    in_force = validate_params(base)
    size = max(1, round(in_force["size"] / 2))
    return {"A": {**{key: in_force[key] for key in picked}, "size": size}, "B": {**picked, "size": size}}


def next_step(picked: Mapping, base: Mapping, test_id: str) -> str:
    """A live test that only checks for harm. What passes then goes in at full size."""
    variants = half_size_variants(picked, base)
    return f"python scripts/live_test.py start {test_id} '{json.dumps(variants)}' --no-harm --adopt '{json.dumps(picked)}' --why \"replay pick\""


def recent_events(path, seconds: float | None) -> list[dict]:
    """The end of a capture, never holding the rest: six hours of it must not take memory from the bot beside it."""
    if not seconds:
        return load_events(path)
    kept: deque[dict] = deque()
    newest = float("-inf")
    for event in read_capture(path):
        newest = max(newest, event["t"])
        kept.append(event)
        while kept[0]["t"] < newest - seconds:
            kept.popleft()
    return sorted(kept, key=lambda event: event["t"])


def no_pick(problem: str, **extra) -> dict:
    return {"picked": None, "ran": False, "line": f"DECISION: no change. {problem}", "next": "", "verdicts": [], **extra}


def pick_from_capture(path, base: Mapping, last_seconds: float | None, min_seconds: float, allow_own: bool = False) -> dict:
    """Always answers, and says whether anything was really replayed. `never` means trying again cannot help."""
    try:
        events = recent_events(path, last_seconds)
    except OSError as exc:
        return no_pick(f"cannot read the capture: {exc}")
    problem = capture_problem(events, min_seconds, allow_own)
    if problem:
        return no_pick(problem)
    ids = sorted({event["iid"] for event in events if event.get("k") == "book"})
    problem = format_problem(ids)
    if problem:
        return no_pick(problem, never=True)
    picked, baseline, verdicts = evaluate(events, base)
    rows = [{"change": overrides, "edge": verdict.edge, "t": verdict.t, "reason": verdict.reason} for overrides, verdict in verdicts]
    report = {
        "ran": True, "verdicts": rows,
        "base": f"base: PnL {baseline.pnl:.0f}, writes {baseline.writes}, fills {baseline.fills}, breaches {baseline.limit_breaches}",
    }
    if picked is None:
        return {**report, "picked": None, "line": "DECISION: no change. Nothing won clearly.", "next": ""}
    test_id = time.strftime("pick-%H%M", time.localtime(events[-1]["t"]))
    return {
        **report, "picked": picked, "line": f"DECISION: {json.dumps(picked)}. Replay only: no rivals and no impact.",
        "next": next_step(picked, base, test_id), "variants": half_size_variants(picked, base),
    }
