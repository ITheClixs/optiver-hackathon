#!/usr/bin/env python3
"""How does this book behave, and what does latency cost? Run it on any capture. Stdlib only.

    python scripts/book_lab.py run/capture.jsonl.gz
    python scripts/book_lab.py captures/market.jsonl.gz --latency 0,0.05,0.25,1 --out reports/lab.json
    python scripts/book_lab.py --quotes data/public/ASML/quotes_2026-07-14T03.csv:ASML ...

`--latency` replays the real engine and strategies at each order delay, so the answer to "how fast must
we be" comes from the book, not from intuition. Fills stay conservative and the replay has no market
impact, so read the differences between delays, not the absolute PnL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.booklab import lab_report, read_quote_csv, read_trade_csv, summary_lines
from otk.instruments import infer_ticks
from otk.participants import read_named_trades_csv
from otk.recorder import read_capture
from otk.replay import run_replay
from otk.strategies import default_stack


def load(captures: list[str], quotes: list[str], trades: list[str] = ()) -> list[dict]:
    events: list[dict] = []
    for path in captures:
        events.extend(read_named_trades_csv(path) if path.endswith(".csv") else read_capture(path))
    for item in quotes:
        path, instrument_id = item.rsplit(":", 1)
        events.extend(read_quote_csv(path, instrument_id))
    for item in trades:
        path, instrument_id = item.rsplit(":", 1)
        events.extend(read_trade_csv(path, instrument_id))
    return sorted(events, key=lambda event: event["t"])


def latency_table(events: list[dict], delays: list[float]) -> list[dict]:
    ids = sorted({event["iid"] for event in events if event["k"] == "book"})
    ticks = infer_ticks(events)
    rows = []
    for delay in delays:
        factories = [(lambda _names, strategy=item: strategy) for item in default_stack(ids)]
        result = run_replay(events, factories, ids, tick=ticks, latency_s=delay)
        rows.append(
            {"latency_s": delay, "pnl": round(result.pnl, 1), "fills": result.fills, "writes": result.writes,
             "breaches": result.limit_breaches, "errors": result.errors}
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*")
    parser.add_argument("--quotes", action="append", default=[], help="public quote CSV as PATH:INSTRUMENT")
    parser.add_argument("--trades", action="append", default=[], help="public trades CSV as PATH:INSTRUMENT")
    parser.add_argument("--latency", help="comma-separated order delays in seconds to replay")
    parser.add_argument("--out", help="full JSON report path")
    args = parser.parse_args(argv)
    events = load(args.captures, args.quotes, args.trades)
    report = lab_report(events)
    if args.latency:
        report["latency"] = latency_table(events, [float(value) for value in args.latency.split(",")])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print("\n".join(summary_lines(report)))
    for row in report.get("latency", ()):
        print(f"latency {row['latency_s']:>6}s  pnl {row['pnl']:>10}  fills {row['fills']:>6}  writes {row['writes']:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
