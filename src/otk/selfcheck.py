"""A made-up market for `--selfcheck`: the stack runs through it with no login and no real order.

A new file proves here that it starts, builds what the event asks for, and quotes without raising, before the
running bot is stopped for it. It says nothing about whether the new strategy makes money.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

CHECK_TICK = 0.1
CHECK_IDS = ("CHECK", "CHECK_DUAL")


class CheckExchange:
    """Answers what the engine asks. Orders rest and never fill; prices drift so that quotes move."""

    def __init__(self) -> None:
        self._orders: dict[str, dict[int, SimpleNamespace]] = {iid: {} for iid in CHECK_IDS}
        self._next_id = 1
        self._reads = 0

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def has_news_feed(self) -> bool:
        return False

    def set_read_ttl(self, _seconds: float | None) -> None:
        pass

    def get_instruments(self) -> dict:
        return {
            iid: SimpleNamespace(instrument_id=iid, tick_size=CHECK_TICK, paused=False, expired=False)
            for iid in CHECK_IDS
        }

    def get_last_price_book(self, instrument_id: str) -> SimpleNamespace:
        self._reads += 1
        mid = round(100.0 + math.sin(self._reads / 40.0), 1)
        width = 0.5 if instrument_id == CHECK_IDS[0] else 1.0
        return SimpleNamespace(
            instrument_id=instrument_id,
            bids=[SimpleNamespace(price=round(mid - width, 1), volume=50)],
            asks=[SimpleNamespace(price=round(mid + width, 1), volume=50)],
        )

    def get_outstanding_orders(self, instrument_id: str) -> dict:
        return dict(self._orders[instrument_id])

    def get_positions(self) -> dict:
        return dict.fromkeys(CHECK_IDS, 0)

    def get_pnl(self, _valuations=None) -> float:
        return 0.0

    def poll_new_trade_ticks(self, _instrument_id: str) -> list:
        return []

    def poll_new_trades(self, _instrument_id: str) -> list:
        return []

    def insert_order(self, instrument_id: str, *, price: float, volume: int, side: str, order_type: str = "limit") -> SimpleNamespace:
        order_id = self._next_id
        self._next_id += 1
        if order_type == "limit":
            self._orders[instrument_id][order_id] = SimpleNamespace(
                order_id=order_id, instrument_id=instrument_id, price=price, volume=volume, side=side,
            )
        return SimpleNamespace(success=True, order_id=order_id, reason="")

    def amend_order(self, instrument_id: str, *, order_id: int, volume: int) -> SimpleNamespace:
        order = self._orders[instrument_id].get(order_id)
        if order is not None:
            order.volume = volume
        return SimpleNamespace(success=True, reason="")

    def delete_order(self, instrument_id: str, *, order_id: int) -> SimpleNamespace:
        self._orders[instrument_id].pop(order_id, None)
        return SimpleNamespace(success=True, reason="")

    def delete_orders(self, instrument_id: str) -> None:
        self._orders[instrument_id].clear()
