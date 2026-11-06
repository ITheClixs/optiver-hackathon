#!/usr/bin/env python3
"""Emergency exit: cancel everything and IOC out of every position at the touch.

This logs in, so it takes over the team session and disconnects a running bot, which is intended
in an emergency. It stays within the write limit and repeats until flat or out of attempts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.adapter import optibook_client_factory
from otk.fair import top_of_book

PAUSE_S = 0.1


def flatten_once(exchange) -> int:
    """One pass; returns how many instruments still hold a position."""
    held = 0
    for iid, position in exchange.get_positions().items():
        exchange.delete_orders(iid)
        time.sleep(PAUSE_S)
        if not position:
            continue
        held += 1
        top = top_of_book(exchange.get_last_price_book(iid))
        if top is None:
            continue
        if position > 0 and top.bid is not None:
            exchange.insert_order(
                iid, price=top.bid, volume=min(position, top.bid_vol), side="ask", order_type="ioc"
            )
        elif position < 0 and top.ask is not None:
            exchange.insert_order(
                iid, price=top.ask, volume=min(-position, top.ask_vol), side="bid", order_type="ioc"
            )
        time.sleep(PAUSE_S)
    return held


def flatten(exchange, attempts: int = 20) -> dict:
    for _ in range(attempts):
        if flatten_once(exchange) == 0:
            break
    return {iid: pos for iid, pos in exchange.get_positions().items() if pos}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=20)
    args = parser.parse_args(argv)
    exchange = optibook_client_factory()()
    exchange.connect()
    remaining = flatten(exchange, args.attempts)
    print(f"remaining positions: {remaining or 'none'}; PnL {exchange.get_pnl():.2f}")
    exchange.disconnect()
    return 0 if not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
