#!/usr/bin/env python3
"""Would aggressive reversion have paid on this capture? Same paper trader as Autopilot.

Prints the best half-life and entry for every instrument and dual gap, net of crossing the spread and a
haircut per fill. That best arm is picked after the fact, so it flatters. The last columns are the fair
test: what Autopilot's own rule would have earned per unit, deciding only on what it had seen so far.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.autopilot import walk_forward
from otk.recorder import read_capture
from otk.reversion import points_from_events, scan_events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--out", help="full JSON report path")
    args = parser.parse_args(argv)
    events = sorted(read_capture(args.capture), key=lambda event: event["t"])
    report = scan_events(events)
    forward = {key: walk_forward(series, tick) for key, (tick, series) in points_from_events(events).items()}
    report = {key: {**row, "walk_forward": forward[key]} for key, row in report.items()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"{'target':<24}{'half-life':>10}{'entry z':>9}{'trips':>7}{'stops':>7}{'ticks':>9}{'t':>7}  | rule: trips, ticks")
    for key, row in report.items():
        best = row["best"]
        print(
            f"{key:<24}{best['half_life_s']:>10}{best['entry_z']:>9}{best['round_trips']:>7}"
            f"{best['stops']:>7}{best['mean_ticks']:>9}{best['t']:>7}"
            f"  | {row['walk_forward']['round_trips']}, {row['walk_forward']['total_ticks']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
