#!/usr/bin/env python3
"""Live A/B test on one account: alternate parameter sets in time slices and compare within blocks.

The bot keeps running; this process only rewrites `params.json` and reads `status.json`. It holds
no exchange session, so it is safe to run next to the bot. PnL during each slice's washout belongs
to the previous variant's inventory and is excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.allocator import SwitchbackSchedule
from otk.evaluation import compare_variants
from otk.params import validate_params, write_params_atomic


def read_pnl(status_path: Path) -> float | None:
    try:
        return json.loads(status_path.read_text()).get("pnl")
    except (OSError, ValueError):
        return None


def run_schedule(
    variants: dict[str, dict],
    schedule: SwitchbackSchedule,
    duration_s: float,
    params_path: Path,
    status_path: Path,
    clock=time.monotonic,
    sleep=time.sleep,
) -> list[dict]:
    """Drives the rotation. Returns one record per slice in time order."""
    slices: list[dict] = []
    end = schedule.start + duration_s
    current: str | None = None
    pnl_after_washout: float | None = None
    while clock() < end:
        now = clock()
        variant = schedule.variant_at(now)
        if variant != current:
            if current is not None and pnl_after_washout is not None:
                pnl = read_pnl(status_path)
                if pnl is not None:
                    slices.append({"variant": current, "pnl": pnl - pnl_after_washout})
            write_params_atomic(params_path, variants[variant])
            current = variant
            pnl_after_washout = None
        if pnl_after_washout is None and not schedule.in_washout(now):
            pnl_after_washout = read_pnl(status_path)
        sleep(0.25)
    return slices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", help="JSON file: name to parameter object")
    parser.add_argument("--slice", type=float, default=60.0)
    parser.add_argument("--washout", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=1200.0)
    parser.add_argument("--run-dir", default="run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    variants = json.loads(Path(args.variants).read_text())
    for params in variants.values():
        validate_params(params)
    run_dir = Path(args.run_dir)
    schedule = SwitchbackSchedule(
        sorted(variants), args.slice, args.washout, time.monotonic(), args.seed
    )
    slices = run_schedule(
        variants, schedule, args.duration, run_dir / "params.json", run_dir / "status.json"
    )
    names = sorted(variants)
    comparisons = {
        f"{a}-{b}": compare_variants(slices, a, b)
        for idx, a in enumerate(names)
        for b in names[idx + 1 :]
    }
    print(json.dumps({"slices": slices, "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
