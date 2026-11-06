"""JSONL capture of everything the loop sees, for local order-book analysis and replay.

A background thread owns the file, so a slow disk cannot stall the trading loop. When the queue is
full, events are dropped and counted rather than blocking.
"""

from __future__ import annotations

import gzip
import json
import logging
import queue
import threading
import time
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path

logger = logging.getLogger("otk.recorder")


class NullRecorder:
    def record(self, kind: str, payload: Mapping) -> None:
        return None

    def tick(self, state) -> None:
        return None

    def close(self) -> None:
        return None


def _levels(levels) -> list[list]:
    return [[level.price, level.volume] for level in levels or []]


def book_event(instrument_id: str, book) -> dict:
    return {
        "iid": instrument_id,
        "bids": _levels(getattr(book, "bids", None)),
        "asks": _levels(getattr(book, "asks", None)),
    }


def tick_event(tick) -> dict:
    return {
        "iid": tick.instrument_id,
        "px": tick.price,
        "vol": tick.volume,
        "aggr": tick.aggressor_side,
        "buyer": getattr(tick, "buyer", "") or "",
        "seller": getattr(tick, "seller", "") or "",
        "id": getattr(tick, "trade_id", None),
    }


class JsonlRecorder(NullRecorder):
    """Writes `{"t": seconds, "k": kind, ...}` lines. `book_every_s` throttles book snapshots."""

    def __init__(
        self,
        path: str | Path,
        book_every_s: float = 0.1,
        clock: Callable[[], float] = time.time,
        max_queue: int = 200_000,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.book_every_s = book_every_s
        self.clock = clock
        self.strategy_of: Callable[[int | None], str | None] = lambda _order_id: ""  # set by whoever knows
        self.dropped = 0
        self.failed = False
        self._last_book: dict[str, tuple] = {}
        self._last_write: dict[str, float] = {}
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        opener = gzip.open if self.path.suffix == ".gz" else open
        try:
            with opener(self.path, "at") as handle:
                while True:
                    line = self._queue.get()
                    if line is None:
                        return
                    handle.write(line + "\n")
                    if self._queue.empty():
                        handle.flush()
        except OSError:
            self.failed = True
            logger.exception("capture stopped: cannot write %s", self.path)

    def record(self, kind: str, payload: Mapping) -> None:
        line = json.dumps({"t": round(self.clock(), 6), "k": kind, **payload}, default=str)
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def _record_book(self, instrument_id: str, book, now: float) -> None:
        event = book_event(instrument_id, book)
        fingerprint = (tuple(map(tuple, event["bids"])), tuple(map(tuple, event["asks"])))
        if self._last_book.get(instrument_id) == fingerprint:
            return
        if now - self._last_write.get(instrument_id, 0.0) < self.book_every_s:
            return
        self._last_book[instrument_id] = fingerprint
        self._last_write[instrument_id] = now
        self.record("book", event)

    def tick(self, state) -> None:
        now = self.clock()
        # The market without our resting quotes, or a replay would see them as someone else's liquidity to take.
        external = getattr(state, "external", None)
        for instrument_id, book in state.books.items():
            self._record_book(instrument_id, external(instrument_id) if external else book, now)
        for ticks in state.new_ticks.values():
            for trade_tick in ticks:
                self.record("trade", tick_event(trade_tick))
        for instrument_id, fills in state.new_fills.items():
            for fill in fills:
                self.record(
                    "fill",
                    {
                        "iid": instrument_id, "px": fill.price, "vol": fill.volume, "side": fill.side,
                        "src": self.strategy_of(getattr(fill, "order_id", None)) or "",
                    },
                )

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self.dropped += 1
        self._thread.join(timeout=5.0)


def read_capture(path: str | Path):
    """Yield events from a capture, tolerating an end cut off by a kill or a mid-write download."""
    target = Path(path)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "rt") as handle:
        try:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return
        except (EOFError, OSError, zlib.error):
            return
