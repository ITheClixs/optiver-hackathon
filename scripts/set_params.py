#!/usr/bin/env python3
"""Change live parameters safely: validated, clamped to hard bounds and atomically replaced.

Merges onto the current file, so an agent or teammate changes only what it names. Paste this
command into the devenv terminal, or run it there through whichever channel the event allows:

    python scripts/set_params.py '{"base_spread_ticks": 4, "size": 5}' --why "informed flow on ASML"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.params import BOUNDS, merge_params, validate_params, write_params_atomic


def merged(path: Path, changes: dict) -> dict:
    current = json.loads(path.read_text()) if path.exists() else {}
    return merge_params(current, changes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("changes", nargs="?", default="{}", help="JSON object of parameter changes")
    parser.add_argument("--path", default="run/params.json")
    parser.add_argument("--why", default="", help="reason, appended to run/param_changes.log")
    parser.add_argument("--list", action="store_true", help="print every parameter and its bounds")
    args = parser.parse_args(argv)
    if args.list:
        for name, bound in sorted(BOUNDS.items()):
            print(f"{name:<26} default {bound.default:<8} range [{bound.low}, {bound.high}]")
        return 0
    if os.environ.get("OTK_ROLE"):  # set by a role's loop. On a laptop this file reaches no bot, yet it prints as applied
        print("refused: a job's laptop has no running bot. Ask A, who uses publish_params.py", file=sys.stderr)
        return 2
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        params = merged(path, json.loads(args.changes))
        applied = validate_params(params)
    except ValueError as exc:
        print(f"rejected, nothing changed: {exc}", file=sys.stderr)
        return 2
    write_params_atomic(path, params)
    with open(path.parent / "param_changes.log", "a") as handle:
        entry = {"time": time.strftime("%H:%M:%S"), "changes": args.changes, "why": args.why}
        handle.write(json.dumps(entry) + "\n")
    print(json.dumps(applied, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
