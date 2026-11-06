#!/usr/bin/env python3
"""Order-book behaviour report from a capture. Stdlib only: runs on a laptop or in the devenv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.behaviour import analyse_capture
from otk.recorder import read_capture


def summary_lines(report: dict) -> list[str]:
    lines = [f"duration {report['duration_s']}s, news events {report['news_events']}"]
    header = f"{'instrument':<20}{'spread':>7}{'trades':>8}{'ac1':>8}{'vr10':>7}{'sig/s':>7}  labels"
    lines.append(header)
    for iid, row in report["instruments"].items():
        lines.append(
            f"{iid:<20}{row['spread_ticks_median']!s:>7}{row['trades']:>8}{row['return_ac1']:>8}"
            f"{row['variance_ratio_10s']:>7}{row['sigma_per_s_ticks']:>7}  {','.join(row['labels'])}"
        )
    for pair in report["lead_lag"]:
        lines.append(
            f"pair {pair['pair'][0]}/{pair['pair'][1]}: leader {pair['leader']}, "
            f"half-life {pair['spread_half_life_s']}s, spread std {pair['spread_std_ticks']} ticks"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--tick", type=float, help="force one tick; default: per instrument")
    parser.add_argument("--out", help="full JSON report path")
    args = parser.parse_args(argv)
    report = analyse_capture(read_capture(args.capture), args.tick)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n".join(summary_lines(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
