#!/usr/bin/env python3
"""Where does the time go in one tick? Prints milliseconds per tick for the full stack on the simulator.

    uv run python scripts/bench_tick.py --profile

The simulator's own cost is included, so read this as an upper bound and compare before and after a
change. The exchange round trip is not in it: `scripts/probe_exchange.py` measures that on the day.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.autopilot import Autopilot
from otk.monitor import LiveMonitor, TeeRecorder
from otk.params import ParamStore
from otk.sim import scenarios
from otk.sim.harness import make_engine
from otk.strategies import default_stack


def build(scenario: str):
    market = {"dual": scenarios.dual_listing, "index": scenarios.index_complex}[scenario](1)
    probe = make_engine(market, "probe", [])
    store = ParamStore()
    monitor = LiveMonitor(probe.universe.tick_of)
    recorder = TeeRecorder([monitor, Autopilot(store, probe.universe, monitor=monitor)])
    stack = default_stack(market.instruments, index_weights=market.index_weights or None)
    return market, make_engine(market, "us", stack, params=store, recorder=recorder)


def measure(scenario: str, ticks: int) -> list[float]:
    market, engine = build(scenario)
    samples = []
    for _ in range(ticks):
        market.advance(0.02)
        started = time.perf_counter()
        engine.tick()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="index", choices=["dual", "index"])
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    samples = measure(args.scenario, args.ticks)
    ordered = sorted(samples)
    print(
        f"{args.scenario}: median {statistics.median(samples):.2f} ms, p90 {ordered[int(0.9 * len(ordered))]:.2f} ms, "
        f"p99 {ordered[int(0.99 * len(ordered))]:.2f} ms, worst {ordered[-1]:.2f} ms"
    )
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        measure(args.scenario, args.ticks // 3)
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(22)
    return 0


if __name__ == "__main__":
    sys.exit(main())
