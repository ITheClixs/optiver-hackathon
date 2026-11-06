#!/usr/bin/env python3
"""Who is who, from a dashboard capture or a named-trades CSV. Stdlib only.

Output names real participants, so it belongs under reports/, which git ignores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.participants import read_named_trades_csv, report_from_events
from otk.recorder import read_capture


def load(paths: list[str]) -> list[dict]:
    events: list[dict] = []
    for path in paths:
        events.extend(read_named_trades_csv(path) if path.endswith(".csv") else read_capture(path))
    return sorted(events, key=lambda event: event["t"])


def summary_lines(report: dict) -> list[str]:
    lines = [f"duration {report['duration_s']}s, {len(report['participants'])} names"]
    lines.append(f"{'name':<26}{'kind':<9}{'role':<12}{'volume':>9}{'aggr':>7}{'size':>7}{'est pnl':>12}")
    rows = sorted(report["participants"].items(), key=lambda item: -item[1]["volume"])
    for name, row in rows[:30]:
        share = row["aggressor_share"]
        lines.append(
            f"{name[:25]:<26}{row['kind']:<9}{row['role']:<12}{row['volume']:>9}"
            f"{'-' if share is None else share:>7}{row['modal_size']!s:>7}{row['est_pnl']:>12}"
        )
    for label in ("overpayers", "informed"):
        lines.append(f"{label}:")
        for row in report["targets"][label]:
            marks = next(iter(row["markout_ticks"].values()))
            lines.append(
                f"  {row['name'][:25]:<26}{row['instrument']:<18}size {row['modal_size']:<6}"
                f"markout {marks['mean']:+.2f} ticks, n={marks['n']}"
            )
    for team in report["teams"][:10]:
        lines.append(
            f"team {team['team'][:25]:<26}pnl {team.get('pnl')}, per min {team.get('pnl_per_min')}, "
            f"passive {team.get('passive_share')}, on {team.get('top_instruments')}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", help="feed capture (.jsonl, .jsonl.gz) or named-trades .csv")
    parser.add_argument("--me", help="our team's name")
    parser.add_argument("--tick", type=float, help="force one tick; default: per instrument")
    parser.add_argument("--out", help="full JSON report path")
    args = parser.parse_args(argv)
    report = report_from_events(load(args.captures), me=args.me, tick=args.tick)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print("\n".join(summary_lines(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
