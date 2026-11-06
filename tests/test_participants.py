"""Who is who, who pays, who hurts us, and which teams are worth copying."""

import random

import pytest

from otk.participants import ParticipantBook, infer_aggressor, report_from_events


def book_with_mid(horizons=(5.0,), me=None, tick=0.1):
    book = ParticipantBook(horizons_s=horizons, me=me, tick_of=lambda _iid: tick)
    book.on_mid("A", 0.0, 100.0)
    return book


def trade(book, now, buyer, seller, price=100.0, volume=10, aggressor="bid", iid="A"):
    book.on_trade(now, iid, price, volume, buyer, seller, aggressor)


def test_estimated_pnl_follows_each_name():
    book = book_with_mid()
    trade(book, 1.0, "Bob", "Sue", price=100.0, volume=10)
    book.on_mid("A", 2.0, 101.0)
    report = book.report()["participants"]
    assert report["Bob"]["est_pnl"] == pytest.approx(10.0)
    assert report["Sue"]["est_pnl"] == pytest.approx(-10.0)
    assert report["Bob"]["positions"] == {"A": 10}
    assert report["Sue"]["positions"] == {"A": -10}


def test_roles_come_from_aggressor_share():
    book = book_with_mid()
    for step in range(20):
        trade(book, step, "TakerBot", "MakerBot", aggressor="bid")
        trade(book, step, "TwoWay", "Other", aggressor="bid" if step % 2 else "ask")
    report = book.report()["participants"]
    assert report["TakerBot"]["role"] == "pure_taker"
    assert report["MakerBot"]["role"] == "pure_maker"
    assert report["TwoWay"]["role"] == "two_way"


def test_leaderboard_names_are_teams_and_the_rest_are_bots():
    book = book_with_mid()
    book.on_leaderboard(1.0, [{"team": "Sue", "pnl": 5.0, "hourly": 5.0, "delta": 1.0, "cash_delta": 0.0, "position": 1, "vega": 0.0, "connected": True}])
    trade(book, 1.0, "Bot", "Sue")
    report = book.report()["participants"]
    assert report["Sue"]["kind"] == "team"
    assert report["Bot"]["kind"] == "bot"


def overpaying_market(book, name="Payer", size=25, drop=0.5, count=60):
    """Each buy prints above fair and the mid then falls back: the buyer overpaid."""
    now = 1.0
    for _ in range(count):
        book.on_mid("A", now, 100.0)
        trade(book, now, name, "Maker", price=100.0 + drop, volume=size, aggressor="bid")
        book.on_mid("A", now + 6.0, 100.0)
        now += 10.0
    return book


def test_a_name_that_overpays_is_listed_as_a_target():
    report = overpaying_market(book_with_mid()).report()
    target = report["targets"]["overpayers"][0]
    assert target["name"] == "Payer"
    assert target["instrument"] == "A"
    assert target["modal_size"] == 25
    assert target["markout_ticks"]["5.0"]["mean"] == pytest.approx(-5.0)
    assert target["markout_ticks"]["5.0"]["n"] == 60


def test_a_name_that_moves_the_market_is_listed_as_informed():
    book = book_with_mid()
    now = 1.0
    mid = 100.0
    rng = random.Random(0)
    for _ in range(60):
        book.on_mid("A", now, mid)
        step = rng.choice((1.0, 0.8))
        trade(book, now, "Whale", "Maker", price=mid, volume=500, aggressor="bid")
        mid += step
        book.on_mid("A", now + 6.0, mid)
        now += 10.0
    informed = book.report()["targets"]["informed"][0]
    assert informed["name"] == "Whale"
    assert informed["modal_size"] == 500


def test_too_few_trades_is_never_a_target():
    report = overpaying_market(book_with_mid(), count=5).report()
    assert report["targets"]["overpayers"] == []


def test_noise_is_neither_target():
    book = book_with_mid()
    rng = random.Random(1)
    now = 1.0
    for _ in range(200):
        book.on_mid("A", now, 100.0)
        trade(book, now, "Noise", "Maker", aggressor=rng.choice(("bid", "ask")))
        book.on_mid("A", now + 6.0, 100.0 + rng.gauss(0, 0.05))
        now += 10.0
    targets = book.report()["targets"]
    assert not [t for t in targets["overpayers"] + targets["informed"] if t["name"] == "Noise"]


def test_our_own_counterparties_show_who_pays_us_and_who_hurts_us():
    book = book_with_mid(me="Us")
    now = 1.0
    for _ in range(40):
        book.on_mid("A", now, 100.0)
        # We sell above fair to Payer; the mid stays: we earned.
        trade(book, now, "Payer", "Us", price=100.5, aggressor="bid")
        # We buy from Whale and the mid then falls: we were hit.
        book.on_mid("A", now + 1.0, 100.0)
        trade(book, now + 1.0, "Us", "Whale", price=100.0, aggressor="ask")
        book.on_mid("A", now + 7.0, 99.0)
        book.on_mid("A", now + 8.0, 100.0)
        now += 20.0
    us = book.report()["me"]
    assert us["pays_us"][0]["name"] == "Payer"
    assert us["pays_us"][0]["our_markout_ticks"] > 0
    assert us["hurts_us"][0]["name"] == "Whale"
    assert us["hurts_us"][0]["our_markout_ticks"] < 0


def test_me_is_empty_without_a_name():
    assert book_with_mid().report()["me"] == {}


def test_top_teams_are_ranked_with_a_style_to_copy():
    book = book_with_mid()
    book.on_leaderboard(0.0, [
        {"team": "Best", "pnl": 0.0, "hourly": 0.0, "delta": 0.0, "cash_delta": 0.0, "position": 0, "vega": 0.0, "connected": True},
        {"team": "Next", "pnl": 0.0, "hourly": 0.0, "delta": 0.0, "cash_delta": 0.0, "position": 0, "vega": 0.0, "connected": True},
    ])
    for step in range(30):
        trade(book, step + 1.0, "Noise", "Best", volume=7, aggressor="bid")
    book.on_leaderboard(60.0, [
        {"team": "Best", "pnl": 600.0, "hourly": 600.0, "delta": 5.0, "cash_delta": 0.0, "position": 30, "vega": 0.0, "connected": True},
        {"team": "Next", "pnl": 60.0, "hourly": 60.0, "delta": -9.0, "cash_delta": 0.0, "position": 0, "vega": 0.0, "connected": False},
    ])
    teams = book.report()["teams"]
    assert [t["team"] for t in teams] == ["Best", "Next"]
    best = teams[0]
    assert best["pnl_per_min"] == pytest.approx(600.0)
    assert best["passive_share"] == pytest.approx(1.0)
    assert best["modal_size"] == 7
    assert best["top_instruments"] == ["A"]


def test_a_falling_leaderboard_pnl_shows_a_negative_rate():
    book = book_with_mid()
    row = {"team": "T", "hourly": 0.0, "delta": 0.0, "cash_delta": 0.0, "position": 0, "vega": 0.0, "connected": True}
    book.on_leaderboard(0.0, [{**row, "pnl": 1000.0}])
    book.on_leaderboard(60.0, [{**row, "pnl": 0.0}])
    assert book.report()["teams"][0]["pnl_per_min"] == pytest.approx(-1000.0)


def test_missing_aggressor_is_inferred_from_the_touch():
    assert infer_aggressor(100.2, 99.9, 100.2) == "bid"
    assert infer_aggressor(99.9, 99.9, 100.2) == "ask"
    assert infer_aggressor(100.0, 99.9, 100.2) == ""
    assert infer_aggressor(100.0, None, None) == ""


def test_a_trade_with_no_aggressor_uses_the_last_top_of_book():
    book = ParticipantBook(horizons_s=(5.0,), tick_of=lambda _i: 0.1)
    book.on_top("A", 0.0, 99.9, 100.1)
    book.on_trade(1.0, "A", 100.1, 10, "Buyer", "Seller", "")
    report = book.report()["participants"]
    assert report["Buyer"]["aggressive_volume"] == 10
    assert report["Seller"]["passive_volume"] == 10


def test_report_from_events_reads_a_feed_capture():
    events = [
        {"t": 0.0, "k": "book", "iid": "A", "bids": [[99.9, 100]], "asks": [[100.1, 100]]},
        {"t": 0.5, "k": "leaderboard", "rows": [{"team": "T", "pnl": 1.0, "hourly": 1.0, "delta": 0.0, "cash_delta": 0.0, "position": 0, "vega": 0.0, "connected": True}]},
        {"t": 1.0, "k": "named_trade", "iid": "A", "px": 100.1, "vol": 10, "buyer": "T", "seller": "Bot", "aggr": "bid"},
        {"t": 2.0, "k": "trade", "iid": "A", "px": 100.1, "vol": 10, "aggr": "bid"},
        {"t": 3.0, "k": "news", "post": "ignored"},
    ]
    report = report_from_events(events, me="T", tick=0.1)
    assert report["participants"]["T"]["volume"] == 10
    assert report["participants"]["T"]["kind"] == "team"
    assert report["duration_s"] == 3.0


def test_report_is_json_serialisable():
    import json

    report = overpaying_market(book_with_mid(me="Maker")).report()
    json.dumps(report)


def test_the_backlog_replayed_at_connect_is_not_scored_as_live():
    """The feed resends about 500 old trades when it connects. They happened before we watched."""
    connect = 1_000.0
    events = [
        {"t": connect, "k": "book", "iid": "A", "bids": [[99.9, 100]], "asks": [[100.1, 100]]},
        {"t": connect, "k": "named_trade", "iid": "A", "px": 100.1, "vol": 20, "buyer": "Old", "seller": "Maker",
         "aggr": "bid", "time": connect - 120},
        {"t": connect + 2, "k": "named_trade", "iid": "A", "px": 100.1, "vol": 20, "buyer": "Live", "seller": "Maker",
         "aggr": "bid", "time": connect + 1},
        {"t": connect + 3, "k": "named_trade", "iid": "A", "px": 100.1, "vol": 5, "buyer": "NoClock", "seller": "Maker",
         "aggr": "bid"},
    ]
    names = report_from_events(events, tick=0.1)["participants"]
    assert "Old" not in names
    assert names["Live"]["volume"] == 20
    assert names["NoClock"]["volume"] == 5, "a trade with no server time is kept"
    assert names["Maker"]["volume"] == 25
