"""The dashboard feed: LZ4-framed MessagePack with column-oriented data, per public participant repos."""

import base64
import hashlib
import socket
import struct
import threading

import pytest

lz4_frame = pytest.importorskip("lz4.frame")
msgpack = pytest.importorskip("msgpack")

from otk.participants import ParticipantBook
from otk.visualiser import (
    FeedNormaliser,
    FeedSink,
    decode_frame,
    encode_frame,
    feed_url,
    hello_messages,
    run_feed,
    select_message,
    websocket_frames,
)


def frame(message):
    return lz4_frame.compress(msgpack.packb(message, use_bin_type=True))


def ticker(**columns):
    return {"type": "module_data", "module": "trade_ticker", "team": None, "data": columns}


def trades(events):
    return [e for e in events if e["k"] == "named_trade"]


def test_decodes_lz4_framed_messagepack():
    message = {"type": "module_data", "module": "news_feed", "data": {"post": ["hello"]}}
    assert decode_frame(frame(message)) == message


def test_decodes_plain_messagepack_too():
    message = {"type": "module_data", "module": "x"}
    assert decode_frame(msgpack.packb(message, use_bin_type=True)) == message


@pytest.mark.parametrize("junk", [b"", b"not a frame", b"\x04\x22\x4d\x18 truncated"])
def test_garbage_frames_decode_to_none(junk):
    assert decode_frame(junk) is None


def test_encode_round_trips_for_outgoing_messages():
    message = select_message("MyTeam", instrument="ASML")
    assert decode_frame(encode_frame(message)) == message
    assert message["type"] == "SELECT"
    assert message["data"]["instrument"] == "ASML"


def test_ticker_columns_become_named_trades():
    norm = FeedNormaliser()
    events = norm.normalise(
        ticker(
            instrumentId=["ASML", "MU"],
            price=[70.94, 41.99],
            volume=[20, 90],
            buyer=["TeamOne", "BotOne"],
            seller=["BotTwo", "TeamTwo"],
            aggressorSide=["bid", "ask"],
            tradeId=[1, 2],
            timestamp=["16:43:09", "16:43:09"],
        )
    )
    first, second = trades(events)
    assert first == {
        "k": "named_trade", "iid": "ASML", "px": 70.94, "vol": 20, "buyer": "TeamOne",
        "seller": "BotTwo", "aggr": "bid", "id": 1, "time": "16:43:09",
    }
    assert second["aggr"] == "ask"


@pytest.mark.parametrize(
    "raw,expected",
    [("bid", "bid"), ("BUY", "bid"), ("buy", "bid"), ("ask", "ask"), ("SELL", "ask"), ("sell", "ask"), ("", ""), (None, "")],
)
def test_aggressor_spellings_normalise(raw, expected):
    norm = FeedNormaliser()
    events = norm.normalise(
        ticker(instrumentId=["A"], price=[1.0], volume=[1], buyer=["x"], seller=["y"], aggressor=[raw])
    )
    assert trades(events)[0]["aggr"] == expected


def test_a_rolling_window_without_ids_emits_each_trade_once():
    norm = FeedNormaliser()
    base = {"price": [1.0, 2.0], "volume": [5, 6], "buyer": ["a", "b"], "seller": ["c", "d"], "timestamp": ["t1", "t2"]}
    first = norm.normalise(ticker(instrumentId=["A", "B"], **base))
    assert len(trades(first)) == 2
    second = norm.normalise(
        ticker(
            instrumentId=["A", "B", "C"], price=[1.0, 2.0, 3.0], volume=[5, 6, 7],
            buyer=["a", "b", "e"], seller=["c", "d", "f"], timestamp=["t1", "t2", "t3"],
        )
    )
    assert [t["iid"] for t in trades(second)] == ["C"]


def test_identical_trades_in_one_frame_both_count():
    norm = FeedNormaliser()
    events = norm.normalise(
        ticker(instrumentId=["A", "A"], price=[1.0, 1.0], volume=[5, 5], buyer=["a", "a"], seller=["c", "c"], timestamp=["t", "t"])
    )
    assert len(trades(events)) == 2


def test_trade_ids_dedupe_across_frames():
    norm = FeedNormaliser()
    columns = {"instrumentId": ["A"], "price": [1.0], "volume": [5], "buyer": ["a"], "seller": ["c"], "tradeId": [7]}
    assert len(trades(norm.normalise(ticker(**columns)))) == 1
    assert trades(norm.normalise(ticker(**columns))) == []


def test_overview_gives_top_of_book_for_every_instrument():
    norm = FeedNormaliser()
    events = norm.normalise(
        {
            "type": "module_data",
            "module": "instruments_overview",
            "data": {
                "indexMapping": {},
                "instrumentId": ["ASML", "THIN"],
                "status": ["OPEN", "OPEN"],
                "bestBid": [70.88, None],
                "bestBidVolume": [435.0, None],
                "bestAsk": [70.94, 5.0],
                "bestAskVolume": [124.0, 3.0],
                "lastTradedPrice": [70.94, 4.9],
            },
        }
    )
    books = [e for e in events if e["k"] == "book"]
    assert books[0] == {"k": "book", "iid": "ASML", "bids": [[70.88, 435]], "asks": [[70.94, 124]], "ltp": 70.94}
    assert books[1]["bids"] == []


def test_unchanged_overview_rows_are_not_repeated():
    norm = FeedNormaliser()
    message = {
        "type": "module_data", "module": "instruments_overview",
        "data": {"instrumentId": ["A"], "bestBid": [1.0], "bestBidVolume": [1], "bestAsk": [2.0], "bestAskVolume": [1]},
    }
    assert norm.normalise(message)
    assert not [e for e in norm.normalise(message) if e["k"] == "book"]


@pytest.mark.parametrize(
    "columns",
    [
        {"team": ["TeamThree", "TeamFour"], "pnl": [70754.66, 5.0], "hourlyPnl": [70753.76, 5.0], "delta": [63.0, 0.0], "cashDelta": [3371.66, 0.0], "position": [63, 0], "vega": [0.0, 0.0], "connected": [True, False]},
        {"username": ["TeamThree", "TeamFour"], "pnl": [70754.66, 5.0], "hourlyPnl": [70753.76, 5.0], "delta": [63.0, 0.0], "cashDelta": [3371.66, 0.0], "position": [63, 0], "vega": [0.0, 0.0], "connected": ["connected", "disconnected"]},
        {"displayName": ["TeamThree", "TeamFour"], "teamId": ["team-1", "team-2"], "pnl": [70754.66, 5.0], "hourlyPnL": [70753.76, 5.0], "delta": [63.0, 0.0], "cash_delta": [3371.66, 0.0], "position": [63, 0], "vega": [0.0, 0.0], "isConnected": [True, False]},
    ],
)
def test_leaderboard_columns_with_alternative_spellings(columns):
    norm = FeedNormaliser()
    events = norm.normalise({"type": "module_data", "module": "leaderboard", "data": columns})
    board = next(e for e in events if e["k"] == "leaderboard")
    top = board["rows"][0]
    assert top["team"] == "TeamThree"
    assert top["pnl"] == pytest.approx(70754.66)
    assert top["hourly"] == pytest.approx(70753.76)
    assert top["delta"] == pytest.approx(63.0)
    assert top["cash_delta"] == pytest.approx(3371.66)
    assert top["position"] == 63
    assert top["connected"] is True
    assert board["rows"][1]["connected"] is False


def test_news_posts_are_captured():
    norm = FeedNormaliser()
    events = norm.normalise(
        {"type": "module_data", "module": "news_feed", "data": {"timestamp": ["t"], "post": ["NVDA beats"]}}
    )
    assert [e["post"] for e in events if e["k"] == "news"] == ["NVDA beats"]


def test_unknown_modules_report_their_schema_once_for_day_one_discovery():
    norm = FeedNormaliser()
    message = {"type": "module_data", "module": "new_thing", "data": {"alpha": [1], "beta": [2]}}
    schema = [e for e in norm.normalise(message) if e["k"] == "schema"]
    assert schema == [{"k": "schema", "module": "new_thing", "type": "module_data", "keys": ["alpha", "beta"]}]
    assert norm.normalise(message) == []


def test_known_modules_also_report_their_columns_once():
    norm = FeedNormaliser()
    columns = {"instrumentId": ["A"], "price": [1.0], "volume": [5], "buyer": ["a"], "seller": ["c"], "tradeId": [7], "surprise": [1]}
    schema = [e for e in norm.normalise(ticker(**columns)) if e["k"] == "schema"]
    assert "surprise" in schema[0]["keys"]


@pytest.mark.parametrize(
    "message",
    [None, 5, "x", {}, {"module": "trade_ticker"}, {"module": "trade_ticker", "data": 5}, {"type": "historical_data", "module": "pnl", "data": {"team-1": {"timestamps": [1]}}}, ticker(instrumentId=["A"], price=["bad"], volume=[1], buyer=["a"], seller=["b"])],
)
def test_odd_messages_never_raise(message):
    FeedNormaliser().normalise(message)


def test_short_columns_do_not_lose_the_rows_that_are_complete():
    norm = FeedNormaliser()
    events = norm.normalise(ticker(instrumentId=["A", "B"], price=[1.0], volume=[5], buyer=["a"], seller=["c"]))
    assert [t["iid"] for t in trades(events)] == ["A"]


def test_the_real_ticker_shape_arrives_newest_first_with_epoch_seconds():
    """Observed in frames from a 2026 event: a 500-row window, newest first, resent on every update."""
    norm = FeedNormaliser()
    events = norm.normalise(
        ticker(
            timestamp=[1783018733, 1783018732], instrumentId=["SAP_DUAL", "AAPL"], price=[51.5, 160.8],
            volume=[44, 20], buyer=["a", "b"], seller=["c", "d"], aggressor=["sell", "buy"],
        )
    )
    assert [t["iid"] for t in trades(events)] == ["AAPL", "SAP_DUAL"]
    assert trades(events)[0]["aggr"] == "bid"


def test_an_aggressor_column_holding_a_name_gives_the_side():
    norm = FeedNormaliser()
    events = norm.normalise(
        ticker(instrumentId=["A"], price=[1.0], volume=[1], buyer=["Bidders"], seller=["Y"], aggressor=["Y"])
    )
    assert trades(events)[0]["aggr"] == "ask"


def test_a_window_with_no_overlap_reports_a_possible_gap():
    norm = FeedNormaliser()
    base = {"price": [1.0] * 100, "volume": [1] * 100, "buyer": ["a"] * 100, "seller": ["b"] * 100}
    norm.normalise(ticker(instrumentId=["A"] * 100, tradeId=list(range(100)), **base))
    events = norm.normalise(ticker(instrumentId=["A"] * 100, tradeId=list(range(1000, 1100)), **base))
    assert [e for e in events if e["k"] == "feed_gap"]


def test_memory_stays_bounded():
    norm = FeedNormaliser(memory=10)
    for trade_id in range(100):
        norm.normalise(ticker(instrumentId=["A"], price=[1.0], volume=[1], buyer=["a"], seller=["b"], tradeId=[trade_id]))
    assert len(norm._ids) == 10


def test_history_becomes_series():
    norm = FeedNormaliser()
    events = norm.normalise(
        {"type": "historical_data", "module": "pnl", "data": {"team-1": {"timestamps": [1, 2], "records": [5.0, 6.0]}}}
    )
    history = [e for e in events if e["k"] == "history"]
    assert history == [{"k": "history", "module": "pnl", "key": "team-1", "points": [(1, 5.0), (2, 6.0)]}]


def test_portfolio_columns_are_kept_when_the_feed_fills_them():
    norm = FeedNormaliser()
    columns = {"instrumentId": ["A"], "bestBid": [1.0], "bestBidVolume": [1], "bestAsk": [2.0], "bestAskVolume": [1], "pnl": [5.0], "hourlyPnl": [1.0], "position": [3.0], "delta": [3.0]}
    events = norm.normalise({"type": "module_data", "module": "instruments_overview", "team": "us", "data": columns})
    portfolio = next(e for e in events if e["k"] == "portfolio")
    assert portfolio["team"] == "us"
    assert portfolio["rows"]["A"]["position"] == 3.0


def test_run_feed_reconnects_with_backoff_and_stops():
    frames = []
    sleeps = []
    calls = []

    def open_frames():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("refused")
        return [b"one", b"two"]

    run_feed(open_frames, frames.append, lambda: len(frames) >= 2, sleep=sleeps.append)
    assert frames == [b"one", b"two"]
    assert sleeps == [1.0, 2.0]


def test_run_feed_survives_a_connection_that_just_ends():
    calls = []
    sleeps = []

    def open_frames():
        calls.append(1)
        return []

    run_feed(open_frames, lambda _f: None, lambda: len(calls) >= 2, sleep=sleeps.append)
    assert sleeps == [1.0]


def test_sink_writes_events_keeps_raw_samples_and_feeds_the_book():
    records = []
    book = ParticipantBook(horizons_s=(5.0,))
    sink = FeedSink(lambda kind, payload: records.append((kind, payload)), book, raw_per_module=1, clock=lambda: 10.0)
    message = ticker(instrumentId=["A"], price=[1.0], volume=[2], buyer=["a"], seller=["b"], tradeId=[1], aggressor=["buy"])
    sink.on_frame(frame(message))
    sink.on_frame(frame({**message, "data": {**message["data"], "tradeId": [2]}}))
    sink.on_frame(b"garbage")
    assert [kind for kind, _ in records].count("raw") == 1
    assert [kind for kind, _ in records].count("named_trade") == 2
    assert "k" not in dict(records)["named_trade"]
    assert book.report()["participants"]["a"]["volume"] == 4
    assert sink.undecoded == 1


def test_every_organiser_event_is_kept():
    """The site's own feed announces a competition's start and length, market halts and announcements. Each is rare and decisive."""
    records = []
    sink = FeedSink(lambda kind, payload: records.append((kind, payload)), raw_per_module=1, clock=lambda: 10.0)
    for number in range(3):
        sink.on_frame(frame({"type": "ADMIN_EVENT", "data": {"type": "market_halt", "number": number}}))
    kept = [payload for kind, payload in records if kind == "admin"]
    assert [item["data"]["number"] for item in kept] == [0, 1, 2]


def test_feed_url():
    assert feed_url("x.optibook.net") == "wss://x.optibook.net/ws"
    assert feed_url("x.optibook.net", "me") == "wss://x.optibook.net/ws?username=me"


def test_hello_can_be_select_only():
    assert [m["type"] for m in hello_messages(subscribe=False)] == ["SELECT"]
    assert {m["type"] for m in hello_messages()} == {"SELECT", "SUBSCRIBE"}


def _serve_once(server, frames, received):
    connection, _ = server.accept()
    with connection:
        request = b""
        while b"\r\n\r\n" not in request:
            request += connection.recv(4096)
        received["request"] = request.decode()
        key = next(
            line.split(":", 1)[1].strip()
            for line in received["request"].split("\r\n")
            if line.lower().startswith("sec-websocket-key")
        )
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        connection.sendall(
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + accept + b"\r\n\r\n"
        )
        received["hello"] = _read_ws_frame(connection)
        for payload in frames:
            connection.sendall(b"\x82\x7e" + struct.pack(">H", len(payload)) + payload)
        connection.sendall(b"\x88\x00")


def _read_ws_frame(connection):
    head = connection.recv(2)
    length = head[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", connection.recv(2))[0]
    mask = connection.recv(4)
    data = b""
    while len(data) < length:
        data += connection.recv(length - len(data))
    return bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))


def test_websocket_client_path_against_a_local_server():
    """The real client library, the handshake and the frame path, without any external host."""
    pytest.importorskip("websocket")
    payload = frame(ticker(instrumentId=["A"], price=[1.0], volume=[2], buyer=["a"], seller=["b"], tradeId=[1]))
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    received = {}
    thread = threading.Thread(target=_serve_once, args=(server, [payload], received), daemon=True)
    thread.start()
    url = f"ws://127.0.0.1:{server.getsockname()[1]}/ws"
    try:
        frames = list(websocket_frames(url, [select_message("us")], timeout_s=5.0))
    finally:
        thread.join(timeout=5.0)
        server.close()
    assert "GET /ws " in received["request"]
    assert decode_frame(received["hello"])["type"] == "SELECT"
    assert frames == [payload]
    assert decode_frame(frames[0])["module"] == "trade_ticker"
