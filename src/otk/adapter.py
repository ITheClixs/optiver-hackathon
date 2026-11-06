"""Wraps the organiser's client for the engine. The only module that may import `optibook`.

`optibook` is private and exists only in the event devenv, so it is imported lazily. Reconnecting
builds a new client, because participant reports show the old object can stay unusable after a
forced disconnect. If `scripts/probe_exchange.py` shows that reads count against the update limit,
set `read_budget` in the event config: `run_bot` then slows the tick to fit, and this class reads
orders and positions at most once per `read_ttl_s`, re-reading after any write to that instrument.
Risk checks never see data older than one tick.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger("otk.adapter")

CLIENT_ARGS_ENV = "OPTIBOOK_CLIENT_ARGS"  # JSON, for a devenv where Exchange() alone cannot connect. May hold a password


def client_args() -> dict:
    """From the environment and never a file, since a built file goes to GitHub and may hold a password."""
    raw = os.environ.get(CLIENT_ARGS_ENV, "")
    if not raw:
        return {}
    try:
        args = json.loads(raw)
    except ValueError:
        args = None
    if not isinstance(args, dict):
        raise SystemExit(f"{CLIENT_ARGS_ENV} must be a JSON object")
    return args


def optibook_client_factory(**kwargs: Any) -> Callable[[], Any]:
    def build():
        from optibook.synchronous_client import Exchange

        return Exchange(**{**client_args(), **kwargs})

    return build


class LiveExchange:
    """Same surface as the client, with None normalised away and optional read caching."""

    def __init__(
        self,
        factory: Callable[[], Any],
        read_ttl_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self._client: Any = None
        self._ttl = read_ttl_s
        self._clock = clock
        self._orders: dict[str, tuple[float, dict]] = {}
        self._positions: tuple[float, dict] | None = None
        self._missing: set[str] = set()

    def _optional(self, name: str) -> Callable | None:
        """A call the bot can work without. This client has never been run here, so its absence must not stop trading."""
        found = getattr(self._client, name, None)
        if found is None and name not in self._missing:
            self._missing.add(name)
            logger.error("the client has no %s: carrying on without it", name)
        return found

    def connect(self) -> None:
        """Always a new client, since an old one can stay unusable. Close it first: one session per team."""
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception as exc:  # noqa: BLE001 - it was probably dead already, which is why we are here
                logger.warning("closing the old client failed: %s", exc)
        self._client = self._factory()
        self._client.connect()
        self._orders.clear()
        self._positions = None

    def set_read_ttl(self, read_ttl_s: float | None) -> None:
        self._ttl = read_ttl_s

    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected()

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.disconnect()

    def get_instruments(self) -> Mapping[str, Any]:
        return self._client.get_instruments()

    def get_last_price_book(self, instrument_id: str) -> Any:
        return self._client.get_last_price_book(instrument_id)

    def get_outstanding_orders(self, instrument_id: str) -> dict:
        now = self._clock()
        cached = self._orders.get(instrument_id)
        if self._ttl and cached and now - cached[0] < self._ttl:
            return cached[1]
        orders = self._client.get_outstanding_orders(instrument_id) or {}
        self._orders[instrument_id] = (now, orders)
        return orders

    def get_positions(self) -> dict[str, int]:
        now = self._clock()
        if self._ttl and self._positions and now - self._positions[0] < self._ttl:
            return self._positions[1]
        positions = self._client.get_positions() or {}
        self._positions = (now, positions)
        return positions

    def poll_new_trade_ticks(self, instrument_id: str) -> list:
        poll = self._optional("poll_new_trade_ticks")
        return (poll(instrument_id) or []) if poll else []

    def poll_new_trades(self, instrument_id: str) -> list:
        poll = self._optional("poll_new_trades")
        return (poll(instrument_id) or []) if poll else []

    def has_news_feed(self) -> bool:
        return getattr(self._client, "poll_new_social_media_feeds", None) is not None

    def poll_new_social_media_feeds(self) -> list:
        poll = getattr(self._client, "poll_new_social_media_feeds", None)
        return (poll() or []) if poll is not None else []

    def get_pnl(self, valuations: Mapping[str, float] | None = None) -> float:
        get_pnl = self._optional("get_pnl")
        if get_pnl is None:
            return 0.0  # so the drawdown stops cannot trip: the error logged above says so
        return get_pnl(valuations) if valuations else get_pnl()

    def _wrote(self, instrument_id: str) -> None:
        self._orders.pop(instrument_id, None)
        self._positions = None

    def insert_order(self, instrument_id: str, **kwargs: Any) -> Any:
        self._wrote(instrument_id)
        return self._client.insert_order(instrument_id, **kwargs)

    def amend_order(self, instrument_id: str, **kwargs: Any) -> Any:
        self._wrote(instrument_id)
        amend = self._optional("amend_order")
        if amend is None:  # the quote manager places the right size on its next pass
            return self.delete_order(instrument_id, order_id=kwargs["order_id"])
        return amend(instrument_id, **kwargs)

    def delete_order(self, instrument_id: str, **kwargs: Any) -> Any:
        self._wrote(instrument_id)
        delete = self._optional("delete_order")
        if delete is None:  # costs queue priority on the other orders, and the next pass replaces them
            return self.delete_orders(instrument_id)
        return delete(instrument_id, **kwargs)

    def delete_orders(self, instrument_id: str) -> Any:
        clear = self._optional("delete_orders")
        if clear is not None:
            self._wrote(instrument_id)
            return clear(instrument_id)
        orders = list(self._client.get_outstanding_orders(instrument_id) or {})
        self._wrote(instrument_id)
        return [self._client.delete_order(instrument_id, order_id=order_id) for order_id in orders]
