#!/usr/bin/env python3
"""Replay a capture across a small parameter grid, then apply the overfitting gates.

Selection uses the median of immediate neighbours (a plateau), never the single best point. Every
run's config count belongs in a trial ledger: DSR is only honest if trials are counted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.evaluation import block_bootstrap_ci, decide, pnl_stats
from otk.instruments import Universe, infer_ticks, parse_instrument_id
from otk.overfit import cscv_pbo, deflated_sharpe, min_backtest_length
from otk.params import INSTRUMENTS_KEY, ParamStore, fine_tick_overrides
from otk.replay import load_events, run_replay
from otk.strategies import default_stack

SLICE_S = 30
DEFAULT_GRID = {"base_spread_ticks": [2, 4, 6], "skew_ticks": [2, 4, 8], "size": [5, 10, 20]}


def instruments_in(events: list[dict]) -> list[str]:
    return sorted({event["iid"] for event in events if event["k"] == "book"})


def evaluate(events: list[dict], ids: list[str], overrides: dict, tick: float | None) -> dict:
    ticks = infer_ticks(events) if tick is None else dict.fromkeys(ids, tick)
    universe = Universe(parse_instrument_id(iid)._replace(tick=ticks[iid]) for iid in ids)
    store = ParamStore(base={**overrides, INSTRUMENTS_KEY: fine_tick_overrides(universe)})
    result = run_replay(
        events, [(lambda _names, strategy=stack: strategy) for stack in default_stack(ids)], ids, params=store, tick=ticks
    )
    stats = pnl_stats(result.pnl_path)
    return {
        "params": overrides,
        "pnl": result.pnl,
        "stats": stats,
        "steps": [
            result.pnl_path[i + 1] - result.pnl_path[i] for i in range(len(result.pnl_path) - 1)
        ],
        "breaches": result.limit_breaches,
        "errors": result.errors,
        "fills": result.fills,
    }


def plateau_scores(results: list[dict], grid: dict) -> list[float]:
    """Median PnL over each config and its immediate neighbours on the grid."""
    keys = list(grid)
    index_of = {
        tuple(grid[k].index(row["params"][k]) for k in keys): row["pnl"] for row in results
    }
    scores = []
    for row in results:
        position = tuple(grid[k].index(row["params"][k]) for k in keys)
        neighbours = []
        for axis in range(len(keys)):
            for step in (-1, 1):
                other = list(position)
                other[axis] += step
                if tuple(other) in index_of:
                    neighbours.append(index_of[tuple(other)])
        # A lone spike must not score well, nor may a valley next to one.
        scores.append(min(row["pnl"], statistics.median(neighbours)) if neighbours else row["pnl"])
    return scores


def to_slices(steps: list[float], seconds: int = SLICE_S) -> list[float]:
    """Per-second Sharpe overstates significance: simulated A/A false positives ran 21 to 66%."""
    return [sum(steps[i : i + seconds]) for i in range(0, len(steps) - seconds + 1, seconds)]


def gates(results: list[dict], chosen: dict, prior_trials: int) -> dict:
    sliced = [to_slices(row["steps"]) for row in results]
    length = min(len(row) for row in sliced)
    if length < 16:
        decision = decide(None, None, length, 0, None, chosen["breaches"])
        return {"pbo": None, "dsr": None, "decision": decision._asdict(), "note": "sample too short"}
    matrix = np.array([row[:length] for row in sliced]).T
    pbo = cscv_pbo(matrix, n_blocks=8 if length < 64 else 16).pbo
    series = np.array(to_slices(chosen["steps"])[:length])
    sharpes = matrix.mean(axis=0) / np.maximum(matrix.std(axis=0, ddof=1), 1e-12)
    trials = len(results) + prior_trials
    dsr = deflated_sharpe(series, trials, float(np.var(sharpes)))
    sr = float(series.mean() / max(series.std(ddof=1), 1e-12))
    btl = min_backtest_length(trials, sr)
    low, _ = block_bootstrap_ci(series.tolist(), block=1)
    decision = decide(pbo, dsr, length, btl, low, chosen["breaches"])
    return {"pbo": pbo, "dsr": dsr, "min_btl": btl, "t": length, "decision": decision._asdict()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--grid", help="JSON object of parameter name to value list")
    parser.add_argument("--prior-trials", type=int, default=0, help="configs already tried today")
    parser.add_argument("--tick", type=float, help="force one tick; default: per instrument")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    grid = json.loads(args.grid) if args.grid else DEFAULT_GRID
    events = load_events(args.capture)
    ids = instruments_in(events)
    results = [
        evaluate(events, ids, dict(zip(grid, combo)), args.tick)
        for combo in itertools.product(*grid.values())
    ]
    scores = plateau_scores(results, grid)
    best = max(range(len(results)), key=lambda i: scores[i])
    report = {
        "n_configs": len(results),
        "chosen": results[best]["params"],
        "chosen_pnl": results[best]["pnl"],
        "plateau_pnl": scores[best],
        "gates": gates(results, results[best], args.prior_trials),
        "table": [
            {**row["params"], "pnl": round(row["pnl"], 2), "plateau": round(score, 2),
             "drawdown": round(row["stats"]["max_drawdown"], 2), "fills": row["fills"]}
            for row, score in zip(results, scores)
        ],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "table"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
