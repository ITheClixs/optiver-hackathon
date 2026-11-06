#!/usr/bin/env python3
"""Regression and stress run of the default stack across every scenario and seed.

Use it to catch breaches, disconnects, crashes and sign errors after any code change. Do not choose
parameter values from it: on 27 Oct 2026 every opponent model beat a frozen replica field by 2x in
simulation and scored near zero live.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.sim import scenarios
from otk.sim.harness import run_seeds
from otk.strategies import default_stack

PRESETS = {
    "quiet": lambda seed: scenarios.quiet_market(seed),
    "dual": lambda seed: scenarios.dual_listing(seed),
    "dual_adverse": lambda seed: scenarios.adverse_field(scenarios.dual_listing(seed), seed),
    "news": lambda seed: scenarios.news_stocks(seed),
    "index": lambda seed: scenarios.index_complex(seed),
    "index_adverse": lambda seed: scenarios.adverse_field(scenarios.index_complex(seed), seed),
}


def run_preset(name: str, seeds: list[int], seconds: float, lag: float = 0.2) -> dict:
    probe = PRESETS[name](0)
    weights = probe.index_weights or None
    results = run_seeds(
        PRESETS[name],
        lambda: default_stack(probe.instruments, index_weights=weights),
        seeds,
        seconds,
        lag_s=lag,
    )
    return {
        "scenario": name,
        "median_pnl": round(statistics.median(r.settled_pnl for r in results), 2),
        "worst_pnl": round(min(r.settled_pnl for r in results), 2),
        "median_drawdown": round(statistics.median(r.max_drawdown for r in results), 2),
        "disconnects": sum(r.disconnects for r in results),
        "limit_breaches": sum(r.limit_breaches for r in results),
        "rate_breaches": sum(r.rate_breaches for r in results),
        "delta_hacks": sum(r.delta_hacks for r in results),
        "peak_write_rate": max(r.peak_write_rate for r in results),
        "errors": sum(r.errors for r in results),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=",".join(PRESETS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--lag", type=float, default=0.2, help="simulated order and position lag, s")
    args = parser.parse_args(argv)
    rows = [
        run_preset(name, list(range(args.seeds)), args.seconds, args.lag)
        for name in args.scenarios.split(",")
    ]
    print(json.dumps(rows, indent=2))
    failed = any(row["disconnects"] or row["limit_breaches"] or row["rate_breaches"] or row["errors"] for row in rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
