#!/usr/bin/env python3
"""Run the main algo. `--sim` runs it on the simulator; otherwise it connects to Optibook.

Only one process may hold the exchange session: a second login disconnects the first, so the live
path takes a lock file. The live path is `otk/main.py`, which is also what the single file runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.autopilot import Autopilot
from otk.formats import plan_event, render_plan
from otk.main import (
    build_classifier,  # noqa: F401 - used through this script by tests and agents
    engine_config,  # noqa: F401
    event_base,
    load_event_config,
    write_status,  # noqa: F401
)
from otk.main import (
    main as run_main,
)
from otk.monitor import TeeRecorder
from otk.params import ParamStore
from otk.recorder import JsonlRecorder
from otk.sim import scenarios
from otk.sim.harness import make_engine, run_session
from otk.strategies import default_stack

SCENARIOS = {
    "quiet": scenarios.quiet_market,
    "dual": scenarios.dual_listing,
    "news": scenarios.news_stocks,
    "index": scenarios.index_complex,
    "passive": scenarios.passive_field,
}


def run_sim(args: argparse.Namespace, event: dict) -> int:
    market = SCENARIOS[args.sim](seed=args.seed)
    if args.adverse:
        market = scenarios.adverse_field(market, args.seed)
    stack = default_stack(market.instruments, index_weights=market.index_weights or None)
    capture = JsonlRecorder(args.capture, clock=lambda: market.now) if args.capture else None
    probe = make_engine(market, "probe", [])
    print("\n".join(render_plan(plan_event(probe.universe, event))), file=sys.stderr)
    store = ParamStore(args.params, base=event_base(event, probe.universe))
    autopilot = None if args.no_autopilot else Autopilot(store, probe.universe)
    recorder = TeeRecorder([r for r in (capture, autopilot) if r])
    engine = make_engine(market, "us", stack, params=store, recorder=recorder, lag_s=args.lag)
    result = run_session(market, {"us": engine}, args.seconds, seed=args.seed)["us"]
    recorder.close()
    summary = {k: v for k, v in result.__dict__.items() if k != "pnl_path"}
    summary["autopilot"] = autopilot.snapshot() if autopilot else {}
    print(json.dumps(summary, indent=2))
    return 0 if result.disconnects == result.limit_breaches == result.rate_breaches == 0 else 1


def parse_sim_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--adverse", action="store_true", help="add competitors and informed flow")
    parser.add_argument("--lag", type=float, default=0.2, help="simulated order and position lag, s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--event", help="event config JSON: limits, index_weights, option_vol")
    parser.add_argument("--params", default="run/params.json")
    parser.add_argument("--capture", help="JSONL capture path, .gz supported")
    parser.add_argument("--no-autopilot", action="store_true", help="run parameters as given")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if "--sim" in arguments:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        args = parse_sim_args(arguments)
        return run_sim(args, load_event_config(args.event))
    return run_main(arguments, entry=__file__)


if __name__ == "__main__":
    sys.exit(main())
