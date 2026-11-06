"""The backtest lab: what the book does, measured so a wrong conclusion would show up as a failed test."""

import json
import random

import pytest

from otk.booklab import (
    arb_windows,
    imbalance_signal,
    lab_report,
    passive_surface,
    read_quote_csv,
    read_trade_csv,
    return_structure,
    summary_lines,
)
from otk.replay import ReplayExchange, run_replay
from otk.strategies import MarketMaker


def snapshot(t, iid, bid, ask, bid_vol=100, ask_vol=100, deeper=None):
    bids = [[bid, bid_vol], *(deeper or {}).get("bids", [])]
    asks = [[ask, ask_vol], *(deeper or {}).get("asks", [])]
    return {"t": float(t), "k": "book", "iid": iid, "bids": bids, "asks": asks}


def trade(t, iid, price, volume, aggressor):
    return {"t": float(t), "k": "trade", "iid": iid, "px": price, "vol": volume, "aggr": aggressor}


# --- does the price revert or trend? ---------------------------------------------------------------


def random_walk(seconds=3000, sigma=0.1, seed=0):
    rng = random.Random(seed)
    mid = 100.0
    for second in range(seconds):
        mid += rng.gauss(0.0, sigma)
        yield snapshot(second, "A", round(mid - 0.1, 1), round(mid + 0.1, 1))


def reverting(seconds=3000, sigma=0.3, seed=0):
    rng = random.Random(seed)
    for second in range(seconds):
        mid = 100.0 + rng.gauss(0.0, sigma)
        yield snapshot(second, "A", round(mid - 0.1, 1), round(mid + 0.1, 1))


def trending(seconds=3000, drift=0.05, seed=0):
    rng = random.Random(seed)
    mid = 100.0
    for second in range(seconds):
        mid += drift + rng.gauss(0.0, 0.02)
        yield snapshot(second, "A", round(mid - 0.1, 1), round(mid + 0.1, 1))


def test_noise_around_a_level_reads_as_reverting():
    assert return_structure(list(reverting()))["A"]["regime"] == "reverting"


def test_a_random_walk_reads_as_a_random_walk():
    assert return_structure(list(random_walk()))["A"]["regime"] == "random_walk"


def test_a_steady_drift_reads_as_trending():
    assert return_structure(list(trending()))["A"]["regime"] == "trending"


def test_variance_ratios_are_reported_per_horizon():
    report = return_structure(list(reverting()))["A"]
    assert report["variance_ratio"]["10"] < 0.5
    assert report["n"] > 1000


def test_too_little_data_gives_no_verdict():
    assert return_structure(list(reverting(seconds=5)))["A"]["regime"] == "unknown"


# --- does the imbalance predict the next move? -----------------------------------------------------


def imbalance_market(seconds=2000, predictive=True, seed=2):
    rng = random.Random(seed)
    mid = 100.0
    events = []
    for second in range(seconds):
        lean = rng.choice((-1, 1))
        bid_vol, ask_vol = (300, 50) if lean > 0 else (50, 300)
        events.append(snapshot(second, "A", round(mid - 0.1, 1), round(mid + 0.1, 1), bid_vol, ask_vol))
        mid += 0.1 * (lean if predictive else rng.choice((-1, 1)))
    return events


def test_an_imbalance_that_predicts_is_found():
    report = imbalance_signal(imbalance_market())["A"]
    assert report["top"]["1"]["corr"] > 0.5
    assert report["top"]["1"]["t"] > 5


def test_an_imbalance_that_does_not_predict_is_not_claimed():
    report = imbalance_signal(imbalance_market(predictive=False))["A"]
    assert abs(report["top"]["1"]["corr"]) < 0.1


def test_near_touch_and_full_book_are_reported_separately():
    """A public study found near-touch predicts where full-book does not. Keep them apart to check it."""
    events = []
    rng = random.Random(4)
    mid = 100.0
    for second in range(2000):
        lean = rng.choice((-1, 1))
        near = (300, 50) if lean > 0 else (50, 300)
        far = {"bids": [[round(mid - 5.0, 1), rng.randint(0, 9000)]], "asks": [[round(mid + 5.0, 1), rng.randint(0, 9000)]]}
        events.append(snapshot(second, "A", round(mid - 0.1, 1), round(mid + 0.1, 1), *near, far))
        mid += 0.1 * lean
    report = imbalance_signal(events)["A"]
    assert report["near"]["1"]["corr"] > 0.9
    assert report["full"]["1"]["corr"] < 0.5


# --- where should we rest? -------------------------------------------------------------------------


def maker_friendly(seconds=2400, seed=5):
    """Takers trade at the touch and the price does not follow them: resting orders earn the spread."""
    rng = random.Random(seed)
    events = []
    for second in range(seconds):
        events.append(snapshot(second, "A", 99.9, 100.1, 40, 40))
        if second % 2 == 0:
            side = rng.choice(("bid", "ask"))
            price = 100.1 if side == "bid" else 99.9
            events.append(trade(second + 0.5, "A", price, 60, side))
    return events


def toxic(seconds=2400, seed=6):
    """Every sell print is followed by the book falling, and every buy by it rising: fills lose."""
    rng = random.Random(seed)
    events = []
    mid = 100.0
    for second in range(seconds):
        events.append(snapshot(second, "A", round(mid - 0.1, 1), round(mid + 0.1, 1), 40, 40))
        if second % 2 == 0:
            side = rng.choice(("bid", "ask"))
            price = round(mid + 0.1, 1) if side == "bid" else round(mid - 0.1, 1)
            events.append(trade(second + 0.5, "A", price, 60, side))
            mid += 0.5 if side == "bid" else -0.5
    return events


def test_resting_at_the_touch_earns_when_takers_are_uninformed():
    surface = passive_surface(maker_friendly())["A"]
    row = surface["0"]
    assert row["fills"] > 20
    assert row["net_ticks"] > 0


def test_resting_at_the_touch_loses_when_takers_are_informed():
    surface = passive_surface(toxic())["A"]
    assert surface["0"]["net_ticks"] < 0


def test_the_surface_finds_a_better_distance_when_there_is_one():
    surface = passive_surface(toxic())["A"]
    assert surface["best_distance_ticks"] != 0 or surface["0"]["fills"] == 0
    assert set(surface) >= {"0", "1", "2", "best_distance_ticks"}


def test_a_farther_quote_fills_less_often():
    surface = passive_surface(maker_friendly())["A"]
    assert surface["0"]["fills"] >= surface["2"]["fills"]


# --- how fast must we be? --------------------------------------------------------------------------


def crossing_market():
    events = []
    for second in range(100):
        crossed = second % 10 == 0
        events.append(snapshot(second, "A", 99.9, 100.1))
        events.append(snapshot(second + 0.1, "A_DUAL", 100.3 if crossed else 99.9, 100.6 if crossed else 100.1, 25, 25))
        if crossed:
            events.append(snapshot(second + 0.4, "A_DUAL", 99.9, 100.1))
    return events


def test_arb_windows_report_how_often_how_long_and_how_much():
    report = arb_windows(crossing_market())["A_DUAL"]
    assert report["count"] == 10
    assert report["median_life_s"] == pytest.approx(0.3, abs=0.05)
    assert report["median_edge_ticks"] == pytest.approx(2.0, abs=0.01)
    assert report["median_size"] == 25


def test_no_crossing_means_no_windows():
    events = [snapshot(1, "A", 99.9, 100.1), snapshot(1, "A_DUAL", 99.9, 100.1)]
    assert arb_windows(events)["A_DUAL"]["count"] == 0


# --- reading real data -----------------------------------------------------------------------------


def test_public_quote_csvs_load(tmp_path):
    path = tmp_path / "quotes.csv"
    path.write_text(
        "recv_time,mid,spread,bid_px_0,bid_vol_0,bid_px_1,bid_vol_1,ask_px_0,ask_vol_0,ask_px_1,ask_vol_1\n"
        "2026-07-14T03:00:00.095672+00:00,80.0,0.2,79.9,828,79.8,5,80.1,830,,\n"
        "not-a-time,1,1,1,1,1,1,1,1,1,1\n"
    )
    events = list(read_quote_csv(path, "ASML"))
    assert len(events) == 1
    assert events[0]["bids"] == [[79.9, 828], [79.8, 5]]
    assert events[0]["asks"] == [[80.1, 830]]
    assert events[0]["k"] == "book"


def test_public_trade_csvs_load(tmp_path):
    path = tmp_path / "trades.csv"
    path.write_text(
        "timestamp,recv_time,price,volume,aggressor_side,buyer,seller,trade_id\n"
        "2026-07-13T07:11:37.646570,2026-07-13T07:11:37.658912+00:00,70.5,2,bid,x,y,576910\n"
        "bad,bad,bad,bad,bad,bad,bad,bad\n"
    )
    events = list(read_trade_csv(path, "ASML"))
    assert len(events) == 1
    assert (events[0]["px"], events[0]["vol"], events[0]["aggr"], events[0]["k"]) == (70.5, 2, "bid", "named_trade")


def test_the_full_report_runs_on_a_capture_and_serialises():
    events = list(random_walk(seconds=600)) + [trade(10.5, "A", 100.1, 5, "bid")]
    events.sort(key=lambda event: event["t"])
    report = lab_report(events)
    json.dumps(report)
    assert set(report) >= {"structure", "imbalance", "passive", "arb_windows", "reading"}
    assert any("A" in line for line in summary_lines(report))


def test_the_report_says_in_words_what_to_do():
    report = lab_report(list(reverting()))
    assert any("revert" in line for line in report["reading"])


# --- latency in the replay -------------------------------------------------------------------------


def moving_events():
    events = [snapshot(0, "A", 99.9, 100.1)]
    events.append(snapshot(1.0, "A", 100.9, 101.1))
    return events


def test_an_order_arrives_after_the_latency_and_meets_the_book_of_that_moment():
    exchange = ReplayExchange(moving_events(), ["A"], latency_s=0.5)
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("A", price=100.1, volume=5, side="bid", order_type="ioc")
    assert exchange.get_positions()["A"] == 0, "still in flight"
    exchange.advance_to(0.4)
    assert exchange.get_positions()["A"] == 0
    exchange.advance_to(0.6)
    assert exchange.get_positions()["A"] == 5


def test_a_slow_ioc_misses_a_price_that_moved_away():
    exchange = ReplayExchange(moving_events(), ["A"], latency_s=2.0)
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("A", price=100.1, volume=5, side="bid", order_type="ioc")
    exchange.advance_to(3.0)
    assert exchange.get_positions()["A"] == 0


@pytest.mark.parametrize("latency,picked_off", [(1.0, 5), (0.1, 0)])
def test_a_slow_cancel_can_be_picked_off(latency, picked_off):
    events = [snapshot(0, "A", 99.9, 100.1), snapshot(2.5, "A", 99.0, 99.2)]
    exchange = ReplayExchange(events, ["A"], latency_s=latency)
    exchange.connect()
    exchange.advance_to(0.0)
    response = exchange.insert_order("A", price=99.5, volume=5, side="bid", order_type="limit")
    exchange.advance_to(2.0)
    assert exchange.get_outstanding_orders("A")
    exchange.delete_order("A", order_id=response.order_id)
    exchange.advance_to(3.5)
    assert exchange.get_positions()["A"] == picked_off
    assert not exchange.get_outstanding_orders("A")


def test_zero_latency_behaves_as_before():
    exchange = ReplayExchange(moving_events(), ["A"])
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("A", price=100.1, volume=5, side="bid", order_type="ioc")
    assert exchange.get_positions()["A"] == 5


def test_more_latency_never_earns_more_on_a_market_that_punishes_it():
    """A quote that cannot leave in time gets run over. If this fails, the latency model is wrong."""
    events = []
    level = 100.0
    rng = random.Random(9)
    for step in range(3000):
        t = step * 0.1
        if step % 40 == 0:
            level += rng.choice((-1.0, 1.0))
        events.append(snapshot(t, "A", round(level - 0.3, 1), round(level + 0.3, 1), 50, 50))
        if step % 5 == 0:
            side = rng.choice(("bid", "ask"))
            events.append(trade(t + 0.01, "A", round(level + (0.3 if side == "bid" else -0.3), 1), 80, side))
    results = {}
    for latency in (0.0, 1.0):
        replayed = run_replay(events, [lambda names: MarketMaker(names)], ["A"], latency_s=latency)
        results[latency] = replayed.pnl
        assert replayed.errors == 0
        assert replayed.limit_breaches == 0
    assert results[0.0] > results[1.0]
