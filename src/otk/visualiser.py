"""Reads the Optibook dashboard feed. Read-only: it sends subscriptions, never orders.

The organisers confirmed by email that code may read this data, as it is public to participants. The
protocol is not documented. It comes from public participant repos and frames captured from a 2026
event: a websocket at `wss://{host}/ws`, binary LZ4 frames around MessagePack, and messages shaped
`{type, module, team, data}` with column-oriented data. `trade_ticker` resends a rolling window of up
to 500 rows, newest first, so rows must be deduplicated. Column names differ between events and can
change on the day, so they are matched through aliases, and every schema seen is written to the capture
for checking. `lz4` and `msgpack` load lazily because the trading loop never imports this module.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from otk.participants import normalise_side

logger = logging.getLogger("otk.visualiser")

LZ4_MAGIC = b"\x04\x22\x4d\x18"
DEFAULT_MODULES = (
    "trade_ticker", "news_feed", "leaderboard", "instruments_overview", "pnl", "instrument_prices",
)
ADMIN_EVENT = "ADMIN_EVENT"  # competition start, end and cancel, market halt and resume, announcements
_MEMORY = 20_000
_GAP_ROWS = 100

_TRADE_REQUIRED = {
    "iid": ("instrumentId", "instrument_id", "instrument", "ticker"),
    "px": ("price",),
    "vol": ("volume", "quantity", "size"),
    "buyer": ("buyer", "buyerName", "buyer_name"),
    "seller": ("seller", "sellerName", "seller_name"),
}
_TRADE_OPTIONAL = {
    "aggr": ("aggressorSide", "aggressor_side", "aggressor", "aggressorType", "takerSide", "side"),
    "id": ("tradeId", "trade_id", "id"),
    "time": ("timestamp", "time"),
}
_OVERVIEW_OPTIONAL = {
    "bid": ("bestBid",),
    "bid_vol": ("bestBidVolume",),
    "ask": ("bestAsk",),
    "ask_vol": ("bestAskVolume",),
    "ltp": ("lastTradedPrice",),
    "pnl": ("pnl",),
    "hourly": ("hourlyPnl", "hourlyPnL"),
    "position": ("position",),
    "delta": ("delta",),
}
_BOARD_REQUIRED = {"team": ("displayName", "teamName", "username", "team", "name", "teamId")}
_BOARD_OPTIONAL = {
    "pnl": ("pnl",),
    "hourly": ("hourlyPnl", "hourlyPnL", "hourly_pnl"),
    "delta": ("delta",),
    "cash_delta": ("cashDelta", "cash_delta"),
    "position": ("position",),
    "vega": ("vega",),
    "connected": ("connected", "isConnected", "status"),
}
_NEWS_REQUIRED = {"post": ("post", "text", "message")}
_NEWS_OPTIONAL = {"time": ("timestamp", "time")}


# --- wire format ----------------------------------------------------------------------------------


def encode_frame(message: Any) -> bytes:
    import lz4.frame
    import msgpack

    return lz4.frame.compress(msgpack.packb(message, use_bin_type=True))


def decode_frame(data: bytes | str) -> Any | None:
    """The decoded message, or None for anything that does not decode."""
    import lz4.frame
    import msgpack

    if isinstance(data, str):
        try:
            return json.loads(data)
        except ValueError:
            return None
    try:
        payload = lz4.frame.decompress(data) if data[:4] == LZ4_MAGIC else data
        return msgpack.unpackb(payload, raw=False, strict_map_key=False)
    except (RuntimeError, ValueError, msgpack.exceptions.UnpackException):
        return None


def feed_url(host: str, username: str | None = None) -> str:
    return f"wss://{host}/ws?username={username}" if username else f"wss://{host}/ws"


def select_message(
    team: str | None = None, instrument: str | None = None, modules: Sequence[str] = DEFAULT_MODULES
) -> dict:
    return {
        "type": "SELECT",
        "team": team,
        "data": {"instrument": instrument, "team": team, "modules": list(modules)},
    }


def subscribe_message(module: str, team: str | None = None) -> dict:
    return {"type": "SUBSCRIBE", "module": module, "team": team}


# --- normalising ----------------------------------------------------------------------------------


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _connected(value: object) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    return str(value).lower() in ("connected", "true", "online", "1")


def _column(data: Mapping, names: Sequence[str]) -> list:
    for name in names:
        column = data.get(name)
        if isinstance(column, list):
            return column
    return []


def _rows(data: Mapping, required: Mapping, optional: Mapping | None = None) -> list[dict]:
    """Columns to rows. The shortest required column decides the count, so a short one drops no row."""
    columns = {key: _column(data, names) for key, names in required.items()}
    count = min((len(column) for column in columns.values()), default=0)
    extras = {key: _column(data, names) for key, names in (optional or {}).items()}
    rows = []
    for index in range(count):
        row = {key: column[index] for key, column in columns.items()}
        for key, column in extras.items():
            row[key] = column[index] if index < len(column) else None
        rows.append(row)
    return rows


def _chronological(rows: list[dict]) -> list[dict]:
    """The feed sends newest first; events should be in the order they happened."""
    if len(rows) > 1:
        first, last = _number(rows[0].get("time")), _number(rows[-1].get("time"))
        if first is not None and last is not None and first > last:
            return rows[::-1]
    return rows


def _aggressor(raw: object, buyer: str, seller: str) -> str:
    """The side, whether the column holds a side or the aggressive party's name."""
    if raw == buyer:
        return "bid"
    if raw == seller:
        return "ask"
    return normalise_side(raw)


def _trade_event(row: Mapping) -> dict | None:
    price, volume = _number(row["px"]), _number(row["vol"])
    if price is None or volume is None:
        return None
    buyer, seller = str(row["buyer"]), str(row["seller"])
    event = {
        "k": "named_trade", "iid": row["iid"], "px": price, "vol": int(volume), "buyer": buyer,
        "seller": seller, "aggr": _aggressor(row.get("aggr"), buyer, seller),
    }
    if row.get("id") is not None:
        event["id"] = row["id"]
    if row.get("time") is not None:
        event["time"] = row["time"]
    return event


def _levels(price: object, volume: object) -> list[list]:
    px, vol = _number(price), _number(volume)
    return [[px, int(vol)]] if px is not None and vol is not None else []


class FeedNormaliser:
    """Raw messages to capture events: named_trade, book, leaderboard, news, history, schema.

    Holds only what deduplication needs, in bounded memory. Never raises on a strange message.
    """

    def __init__(self, memory: int = _MEMORY) -> None:
        self.memory = memory
        self.errors = 0
        self._schemas: set[tuple] = set()
        self._ids: OrderedDict = OrderedDict()
        self._rows_seen: OrderedDict = OrderedDict()
        self._news: OrderedDict = OrderedDict()
        self._books: dict[str, tuple] = {}
        self._board: list | None = None
        self._portfolio: dict | None = None
        self._saw_trades = False

    def normalise(self, message: Any) -> list[dict]:
        if not isinstance(message, Mapping):
            return []
        module, data, kind = message.get("module"), message.get("data"), message.get("type")
        if not isinstance(module, str) or not isinstance(data, Mapping):
            return []
        events = self._schema(kind, module, data)
        try:
            events.extend(self._parse(kind, module, data, message))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.errors += 1
            events.append({"k": "feed_error", "module": module, "error": f"{type(exc).__name__}: {exc}"})
        return events

    def _parse(self, kind: object, module: str, data: Mapping, message: Mapping) -> list[dict]:
        if kind == "historical_data":
            return self._history(module, data)
        if module == "trade_ticker":
            return self._trades(data)
        if module == "instruments_overview":
            return self._overview(data, message.get("team"))
        if module == "leaderboard":
            return self._leaderboard(data)
        if module == "news_feed":
            return self._news_posts(data)
        return []

    def _schema(self, kind: object, module: str, data: Mapping) -> list[dict]:
        if kind == "historical_data":
            first = next(iter(data.values()), {})
            keys = sorted(map(str, first)) if isinstance(first, Mapping) else []
        else:
            keys = sorted(map(str, data))
        fingerprint = (kind, module, tuple(keys))
        if fingerprint in self._schemas:
            return []
        self._schemas.add(fingerprint)
        return [{"k": "schema", "module": module, "type": kind, "keys": keys}]

    def _remember(self, store: OrderedDict, key: object, value: int = 1) -> None:
        store[key] = value
        store.move_to_end(key)
        while len(store) > self.memory:
            store.popitem(last=False)

    def _is_new_trade(self, event: Mapping, counts: Counter) -> bool:
        trade_id = event.get("id")
        if trade_id is not None:
            if trade_id in self._ids:
                return False
            self._remember(self._ids, trade_id)
            return True
        # No id: identical rows are real if a frame holds more of them than any frame before it.
        key = (event["iid"], event["px"], event["vol"], event["buyer"], event["seller"], event.get("time"))
        counts[key] += 1
        if counts[key] <= self._rows_seen.get(key, 0):
            return False
        self._remember(self._rows_seen, key, counts[key])
        return True

    def _trades(self, data: Mapping) -> list[dict]:
        rows = _chronological(_rows(data, _TRADE_REQUIRED, _TRADE_OPTIONAL))
        counts: Counter = Counter()
        events = []
        overlapped = False
        for row in rows:
            event = _trade_event(row)
            if event is None:
                continue
            if self._is_new_trade(event, counts):
                events.append(event)
            else:
                overlapped = True
        if self._saw_trades and not overlapped and len(rows) >= _GAP_ROWS:
            events.insert(0, {"k": "feed_gap", "rows": len(rows)})  # trades may have been missed
        self._saw_trades = self._saw_trades or bool(rows)
        return events

    def _overview(self, data: Mapping, team: object) -> list[dict]:
        events = []
        portfolio = {}
        for row in _rows(data, {"iid": ("instrumentId",)}, _OVERVIEW_OPTIONAL):
            iid = row["iid"]
            event = {
                "k": "book", "iid": iid, "bids": _levels(row["bid"], row["bid_vol"]),
                "asks": _levels(row["ask"], row["ask_vol"]),
            }
            ltp = _number(row["ltp"])
            if ltp is not None:
                event["ltp"] = ltp
            fingerprint = (tuple(map(tuple, event["bids"])), tuple(map(tuple, event["asks"])), ltp)
            if self._books.get(iid) != fingerprint:
                self._books[iid] = fingerprint
                events.append(event)
            if row["pnl"] is not None or row["position"] is not None:
                portfolio[iid] = {k: _number(row[k]) for k in ("pnl", "hourly", "position", "delta")}
        if portfolio and portfolio != self._portfolio:
            self._portfolio = portfolio
            events.append({"k": "portfolio", "team": team, "rows": portfolio})
        return events

    def _leaderboard(self, data: Mapping) -> list[dict]:
        rows = []
        for row in _rows(data, _BOARD_REQUIRED, _BOARD_OPTIONAL):
            parsed = {
                key: _number(row[key])
                for key in ("pnl", "hourly", "delta", "cash_delta", "position", "vega")
            }
            rows.append({"team": str(row["team"]), **parsed, "connected": _connected(row["connected"])})
        if not rows or rows == self._board:
            return []
        self._board = rows
        return [{"k": "leaderboard", "rows": rows}]

    def _news_posts(self, data: Mapping) -> list[dict]:
        events = []
        for row in _chronological(_rows(data, _NEWS_REQUIRED, _NEWS_OPTIONAL)):
            key = (row.get("time"), row["post"])
            if key in self._news:
                continue
            self._remember(self._news, key)
            event = {"k": "news", "post": row["post"]}
            if row.get("time") is not None:
                event["time"] = row["time"]
            events.append(event)
        return events

    @staticmethod
    def _history(module: str, data: Mapping) -> list[dict]:
        events = []
        for key, series in data.items():
            if not isinstance(series, Mapping):
                continue
            stamps = series.get("timestamps")
            values = next(
                (v for k, v in series.items() if k != "timestamps" and isinstance(v, list)), None
            )
            if isinstance(stamps, list) and values is not None:
                events.append(
                    {"k": "history", "module": module, "key": str(key), "points": list(zip(stamps, values))}
                )
        return events


# --- connection -----------------------------------------------------------------------------------


def hello_messages(
    team: str | None = None, modules: Sequence[str] = DEFAULT_MODULES, subscribe: bool = True
) -> list[dict]:
    messages: list[dict] = [select_message(team, modules=modules)]
    if subscribe:
        messages.extend(subscribe_message(module, team) for module in modules)
    return messages


def websocket_frames(
    url: str,
    hello: Iterable[Mapping],
    cookie: str | None = None,
    timeout_s: float = 30.0,
    subprotocol: str | None = None,
) -> Iterator[bytes | str]:
    """Frames until the connection ends. A silent feed times out, which forces a reconnect."""
    import websocket

    subprotocols = [subprotocol] if subprotocol else None
    connection = websocket.create_connection(
        url, timeout=timeout_s, cookie=cookie, subprotocols=subprotocols
    )
    try:
        for message in hello:
            connection.send_binary(encode_frame(message))
        while True:
            frame = connection.recv()
            if frame in ("", b""):
                return
            yield frame
    finally:
        connection.close()


def run_feed(
    open_frames: Callable[[], Iterable[bytes | str]],
    on_frame: Callable[[bytes | str], None],
    should_stop: Callable[[], bool],
    sleep: Callable[[float], None] = time.sleep,
    min_backoff_s: float = 1.0,
    max_backoff_s: float = 30.0,
) -> None:
    """Keep reading, reconnecting with backoff. Any network error just means try again."""
    backoff = min_backoff_s
    while not should_stop():
        try:
            for frame in open_frames():
                backoff = min_backoff_s
                on_frame(frame)
                if should_stop():
                    return
        except Exception as exc:  # noqa: BLE001 - network libraries raise many unrelated types
            logger.warning("feed connection failed: %s: %s", type(exc).__name__, exc)
        if should_stop():
            return
        sleep(backoff)
        backoff = min(backoff * 2.0, max_backoff_s)


class FeedSink:
    """Decode a frame, write its events, update live participant analytics, keep a few raw samples."""

    def __init__(
        self,
        record: Callable[[str, Mapping], None],
        book=None,
        normaliser: FeedNormaliser | None = None,
        raw_per_module: int = 3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.record = record
        self.book = book
        self.normaliser = normaliser or FeedNormaliser()
        self.raw_per_module = raw_per_module
        self.clock = clock
        self.frames = 0
        self.undecoded = 0
        self._raw_counts: Counter = Counter()

    def _keep_raw(self, message: Any) -> None:
        if not isinstance(message, Mapping):
            return
        key = (message.get("type"), message.get("module"))
        if self._raw_counts[key] < self.raw_per_module:
            self._raw_counts[key] += 1
            self.record("raw", {"message": json.loads(json.dumps(message, default=str))})

    def on_frame(self, frame: bytes | str) -> None:
        from otk.participants import apply_event

        self.frames += 1
        message = decode_frame(frame)
        if message is None:
            self.undecoded += 1
            return
        self._keep_raw(message)
        if isinstance(message, Mapping) and message.get("type") == ADMIN_EVENT:
            logger.info("organiser event: %s", message.get("data"))
            self.record("admin", {"data": json.loads(json.dumps(message.get("data"), default=str))})
        now = self.clock()
        for event in self.normaliser.normalise(message):
            self.record(event["k"], {k: v for k, v in event.items() if k != "k"})
            if self.book is not None:
                apply_event(self.book, now, event)
