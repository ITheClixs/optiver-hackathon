#!/usr/bin/env python3
"""Choose parameters from a capture, with no model, no tokens and no login. Default is no change.

    python scripts/pick_params.py live/capture.jsonl.gz --status live/status.json --last-seconds 3600   # the running bot's
    python scripts/record_market.py --seconds 1800         # or, with no agents, in the devenv before the bot starts
    python scripts/pick_params.py captures/market.jsonl.gz --apply run/params.json

It replays a short list of single changes through the real engine, on the same market moments as the current settings,
and takes the best one only if it wins clearly, in both halves, with no breach and no much larger drawdown. Replay has
no market impact and no rival racing us, so what it picks goes to a half-size live test that checks for harm before it
runs at full size. The list is fixed before the event, because a list you can add to
on the day is a search, and the trial count stops meaning anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.params import write_params_atomic
from otk.pick import (  # noqa: F401 - the tests and the watcher read these from here
    CANDIDATES,
    MIN_CAPTURE_S,
    capture_problem,
    choose,
    evaluate,
    format_problem,
    judge,
    next_step,
    pick_from_capture,
)


def _read_params(path: Path | None) -> dict:
    try:
        loaded = json.loads(path.read_text()) if path else {}
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture")
    parser.add_argument("--apply", help="write the choice into this parameter file, usually run/params.json")
    parser.add_argument("--status", help="the bot's status.json: start from what it has in force")
    parser.add_argument("--last-seconds", type=float, help="replay only the end of the capture")
    parser.add_argument("--min-seconds", type=float, default=MIN_CAPTURE_S)
    parser.add_argument("--allow-own", action="store_true", help="accept an older capture taken while our bot traded")
    parser.add_argument("--json", action="store_true", help="end with one line of JSON, for the watcher")
    args = parser.parse_args(argv)
    target = Path(args.apply) if args.apply else None
    in_file = _read_params(target)
    baseline = _read_params(Path(args.status)).get("baseline") if args.status else None
    current = baseline if isinstance(baseline, dict) else in_file
    report = pick_from_capture(args.capture, current, args.last_seconds, args.min_seconds, args.allow_own)
    if report.get("base"):
        print(report["base"])
    for row in report["verdicts"]:
        print(f"  {json.dumps(row['change']):<46} edge {row['edge']:>8.0f}  t {row['t']:>6.2f}  {row['reason']}")
    if report["picked"] and target:
        write_params_atomic(target, {**in_file, **report["picked"]})
        print(f"written to {target}; the bot picks it up within a second")
    print(report["line"])
    if report["next"]:
        print(f"Next: {report['next']}")
    if args.json:
        print(json.dumps({key: report[key] for key in ("picked", "ran", "line", "next")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
