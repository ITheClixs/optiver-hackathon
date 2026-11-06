#!/usr/bin/env python3
"""Passive capture: no strategies and no orders, so the data is free of our own impact.

Run in the devenv before quoting. It holds the team session, so stop it before starting the bot.
Move the capture to a laptop for analysis, or analyse it in place.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.adapter import LiveExchange, optibook_client_factory
from otk.engine import Engine, EngineConfig
from otk.instruments import Universe
from otk.recorder import JsonlRecorder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="captures/market.jsonl.gz")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--book-every", type=float, default=0.1)
    parser.add_argument("--run-dir", default="run")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    exchange = LiveExchange(optibook_client_factory())
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    recorder = JsonlRecorder(args.out, book_every_s=args.book_every)
    config = EngineConfig(live_lock=str(run_dir / "LIVE.lock"))
    engine = Engine(exchange, universe, [], config, recorder=recorder)
    deadline = time.monotonic() + args.seconds
    engine.acquire_lock()
    try:
        while time.monotonic() < deadline:
            engine.tick()
            time.sleep(0.01)
    finally:
        engine.release_lock()
        recorder.close()
    print(f"captured to {args.out}; dropped {recorder.dropped} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
