#!/usr/bin/env python3
"""First thing to run on event day: measure the platform facts we could not confirm in advance.

Run it only on the practice exchange, before the bot starts, because it takes over the team's
session. It stays well inside every published limit and never tests a breach. It rests one small
order far from the market, cancels it, and reports API shapes, latencies, and whether a burst of
reads is tolerated. Results go to run/probe.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.adapter import optibook_client_factory
from otk.fair import top_of_book

READ_BURST = 200
WRITE_PAUSE_S = 0.2


def describe(value: object) -> str:
    return repr(value)[:400]


def probe_reads(exchange, instrument_id: str) -> dict:
    """If reads count as updates, a burst far above 25 per second would disconnect us."""
    started = time.perf_counter()
    samples = []
    for _ in range(READ_BURST):
        before = time.perf_counter()
        exchange.get_last_price_book(instrument_id)
        samples.append(time.perf_counter() - before)
    elapsed = time.perf_counter() - started
    return {
        "reads": READ_BURST,
        "reads_per_second": round(READ_BURST / elapsed, 1),
        "median_read_ms": round(statistics.median(samples) * 1000, 4),
        "still_connected": exchange.is_connected(),
    }


def median_call_ms(call, samples: int = 50) -> float:
    timings = []
    for _ in range(samples):
        before = time.perf_counter()
        call()
        timings.append(time.perf_counter() - before)
    return round(statistics.median(timings) * 1000, 4)


def probe_read_costs(exchange, instrument_id: str) -> dict:
    """A cached read takes microseconds and a remote one a round trip, which sets the loop budget."""
    return {
        "get_positions_ms": median_call_ms(exchange.get_positions),
        "get_outstanding_orders_ms": median_call_ms(
            lambda: exchange.get_outstanding_orders(instrument_id)
        ),
        "get_pnl_ms": median_call_ms(exchange.get_pnl),
    }


def probe_writes(exchange, instrument_id: str) -> dict:
    """Insert, amend and delete one order far from the touch; times each round trip."""
    book = exchange.get_last_price_book(instrument_id)
    top = top_of_book(book)
    if top is None or top.bid is None:
        return {"skipped": "no bid on the book to anchor a far-away order"}
    far_price = round(top.bid * 0.8, 1)
    timings = {}
    started = time.perf_counter()
    response = exchange.insert_order(
        instrument_id, price=far_price, volume=1, side="bid", order_type="limit"
    )
    timings["insert_ms"] = round((time.perf_counter() - started) * 1000, 3)
    report = {"insert_response": describe(response), "far_price": far_price}
    time.sleep(WRITE_PAUSE_S)
    report["outstanding"] = describe(exchange.get_outstanding_orders(instrument_id))
    after = exchange.get_last_price_book(instrument_id)
    report["book_includes_own_orders"] = any(
        abs(level.price - far_price) < 1e-6 for level in (after.bids or [])
    )
    order_id = getattr(response, "order_id", None)
    if order_id is not None:
        started = time.perf_counter()
        report["amend_response"] = describe(
            exchange.amend_order(instrument_id, order_id=order_id, volume=2)
        )
        timings["amend_ms"] = round((time.perf_counter() - started) * 1000, 3)
        time.sleep(WRITE_PAUSE_S)
        started = time.perf_counter()
        report["delete_response"] = describe(exchange.delete_order(instrument_id, order_id=order_id))
        timings["delete_ms"] = round((time.perf_counter() - started) * 1000, 3)
    report["timings"] = timings
    return report


def probe_shapes(exchange, instrument_id: str) -> dict:
    instruments = exchange.get_instruments()
    sample = instruments.get(instrument_id)
    ticks = exchange.get_trade_tick_history(instrument_id)
    feed = getattr(exchange, "poll_new_social_media_feeds", None)
    return {
        "instrument_ids": sorted(instruments),
        "instrument_sample": describe(sample),
        "book_sample": describe(exchange.get_last_price_book(instrument_id)),
        "tick_sample": describe(ticks[-1]) if ticks else None,
        "ticks_expose_counterparty": bool(
            ticks and (getattr(ticks[-1], "buyer", "") or getattr(ticks[-1], "seller", ""))
        ),
        "has_news_feed": feed is not None,
        "positions_sample": describe(exchange.get_positions()),
        "pnl": exchange.get_pnl(),
        "client_methods": sorted(name for name in dir(exchange) if not name.startswith("_")),
    }


def probe_market_speed(exchange, instrument_id: str, seconds: float = 3.0) -> dict:
    """How often the book changes sets how much a faster loop can help."""
    changes = 0
    last = None
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        top = top_of_book(exchange.get_last_price_book(instrument_id))
        if top != last:
            changes += 1
            last = top
        time.sleep(0.001)
    return {"top_of_book_changes_per_second": round(changes / seconds, 1)}


def probe_instruments(exchange) -> dict:
    """Tick sizes differ by instrument, and `paused` shows which markets are closed right now."""
    instruments = exchange.get_instruments()
    return {
        iid: {"tick_size": getattr(item, "tick_size", None), "paused": getattr(item, "paused", None),
              "group": getattr(item, "instrument_group", None)}
        for iid, item in sorted(instruments.items())
    }


def run_probe(exchange, instrument_id: str | None) -> dict:
    instruments = sorted(exchange.get_instruments())
    target = instrument_id or instruments[0]
    report = {"instrument": target, "shapes": probe_shapes(exchange, target)}
    report["reads"] = probe_reads(exchange, target)
    report["read_costs"] = probe_read_costs(exchange, target)
    report["instruments"] = probe_instruments(exchange)
    report["market_speed"] = probe_market_speed(exchange, target)
    report["writes"] = probe_writes(exchange, target)
    report["connected_at_end"] = exchange.is_connected()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument")
    parser.add_argument("--out", default="run/probe.json")
    args = parser.parse_args(argv)
    exchange = optibook_client_factory()()
    exchange.connect()
    report = run_probe(exchange, args.instrument)
    exchange.disconnect()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
