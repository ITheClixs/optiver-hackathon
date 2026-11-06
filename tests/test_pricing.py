"""Fair values, instrument parsing and the pure quoting function."""

import math
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from otk.fair import (
    Level,
    Top,
    bs_delta,
    bs_price,
    etf_fair,
    external_book,
    future_fair,
    imbalance,
    implied_vol,
    index_value,
    microprice,
    round_ask,
    round_bid,
    snap,
    top_of_book,
    wall_mid,
)
from otk.instruments import Kind, delta_per_lot, parse_instrument_id, third_friday
from otk.quoter import QuoteParams, compute_quotes


def book(bids, asks):
    return SimpleNamespace(
        bids=[SimpleNamespace(price=p, volume=v) for p, v in bids] if bids else None,
        asks=[SimpleNamespace(price=p, volume=v) for p, v in asks] if asks else None,
    )


def test_top_of_book_handles_none_sides():
    assert top_of_book(None) is None
    assert top_of_book(book(None, None)) is None
    top = top_of_book(book([(10.0, 5)], None))
    assert top.bid == 10.0 and top.ask is None


def test_microprice_leans_to_the_thin_side():
    top = Top(99.0, 900, 101.0, 100)
    assert microprice(top) == pytest.approx(100.8)
    assert imbalance(top) == pytest.approx(0.8)


def test_microprice_one_sided_book_is_none():
    assert microprice(Top(99.0, 10, None, None)) is None
    assert imbalance(Top(None, None, 101.0, 5)) == 0.0


def test_wall_mid_uses_largest_level_per_side():
    levels = book([(153.7, 625), (153.6, 200)], [(153.8, 5), (154.0, 625)])
    assert wall_mid(levels) == pytest.approx((153.7 + 154.0) / 2)


def test_external_book_removes_only_our_volume():
    own = [SimpleNamespace(price=100.0, volume=5, side="bid")]
    levels = book([(100.0, 5), (99.9, 200)], [(100.5, 100)])
    ext = external_book(levels, own)
    assert ext.bids[0] == Level(99.9, 200)
    assert ext.asks[0] == Level(100.5, 100)

    own = [SimpleNamespace(price=100.0, volume=5, side="bid")]
    levels = book([(100.0, 25)], [(100.5, 100)])
    assert external_book(levels, own).bids[0] == Level(100.0, 20)


def test_rounding_is_on_tick_and_conservative():
    assert snap(100.30000000000001, 0.1) == pytest.approx(100.3)
    assert round_bid(100.27, 0.1) == pytest.approx(100.2)
    assert round_ask(100.21, 0.1) == pytest.approx(100.3)
    assert round_bid(100.3, 0.1) == pytest.approx(100.3)
    assert round_ask(100.30000000000001, 0.1) == pytest.approx(100.3)


def test_index_etf_and_future_formulas_match_the_guide():
    weights = {"A": 1000.0, "B": 2000.0}
    index = index_value({"A": 10.0, "B": 20.0}, weights, 1000.0)
    assert index == pytest.approx(50.0)
    assert etf_fair(index, multiplier=0.25, cash=2.5) == pytest.approx(15.0)
    assert future_fair(100.0, 0.03, 0.5) == pytest.approx(100.0 * math.exp(0.015))
    assert index_value({"A": 10.0}, weights, 1000.0) is None


def test_black_scholes_put_call_parity():
    spot, strike, tau, rate, vol = 100.0, 95.0, 0.5, 0.03, 0.25
    call = bs_price("call", spot, strike, tau, rate, vol)
    put = bs_price("put", spot, strike, tau, rate, vol)
    assert call - put == pytest.approx(spot - strike * math.exp(-rate * tau))


def test_black_scholes_expiry_and_delta_bounds():
    assert bs_price("call", 110.0, 100.0, 0.0, 0.03, 0.2) == pytest.approx(10.0)
    assert bs_price("put", 110.0, 100.0, 0.0, 0.03, 0.2) == 0.0
    assert 0.0 < bs_delta("call", 100.0, 100.0, 0.5, 0.03, 0.2) < 1.0
    assert -1.0 < bs_delta("put", 100.0, 100.0, 0.5, 0.03, 0.2) < 0.0


def test_implied_vol_round_trip():
    price = bs_price("call", 100.0, 105.0, 0.25, 0.03, 0.31)
    assert implied_vol("call", price, 100.0, 105.0, 0.25, 0.03) == pytest.approx(0.31, abs=1e-4)
    assert implied_vol("call", 0.0, 100.0, 105.0, 0.25, 0.03) is None


@pytest.mark.parametrize(
    "iid,kind,base,strike,option",
    [
        ("ASML", Kind.STOCK, "ASML", None, None),
        ("ASML_DUAL", Kind.DUAL, "ASML", None, None),
        ("OB5X_ETF", Kind.ETF, "OB5X", None, None),
        ("OB5X_202609_F", Kind.FUTURE, "OB5X", None, None),
        ("ASML_202703_F", Kind.FUTURE, "ASML", None, None),
        ("OB5X_202609_080C", Kind.OPTION, "OB5X", 80.0, "call"),
        ("ASML_202609_075P", Kind.OPTION, "ASML", 75.0, "put"),
    ],
)
def test_parse_instrument_id(iid, kind, base, strike, option):
    info = parse_instrument_id(iid)
    assert (info.kind, info.base, info.strike, info.option_kind) == (kind, base, strike, option)


def test_expiry_is_third_friday_at_noon_utc():
    assert third_friday(2026, 9) == datetime(2026, 9, 18, 12, tzinfo=timezone.utc)
    assert parse_instrument_id("OB5X_202609_F").expiry == third_friday(2026, 9)


def test_delta_per_lot():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert delta_per_lot(parse_instrument_id("ASML_DUAL"), 100.0, 0.2, 0.03, now) == 1.0
    assert delta_per_lot(parse_instrument_id("ASML_202609_F"), 100.0, 0.2, 0.03, now) == 1.0
    call = delta_per_lot(parse_instrument_id("ASML_202609_100C"), 100.0, 0.2, 0.03, now)
    assert 0.4 < call < 0.8


TICK = 0.1


def quotes(fair=100.0, position=0, vol=0.0, imb=0.0, top=None, **overrides):
    params = QuoteParams(**{"base_spread_ticks": 4.0, "size": 10, **overrides})
    return compute_quotes(fair, position, 100, vol, imb, params, TICK, top)


def test_flat_quotes_are_symmetric_and_on_tick():
    q = quotes()
    assert q.bid == pytest.approx(99.8) and q.ask == pytest.approx(100.2)
    assert q.bid_size == q.ask_size == 10


def test_inventory_skew_lowers_both_quotes_when_long():
    flat, long_ = quotes(), quotes(position=80)
    assert long_.bid < flat.bid and long_.ask < flat.ask


def test_inventory_skew_is_bounded_by_parameter():
    q = quotes(position=100, skew_ticks=3.0, skew_curve=10.0)
    assert q.ask == pytest.approx(99.9, abs=0.051)


def test_inventory_shrinks_the_side_that_adds_risk():
    q = quotes(position=50)
    assert q.bid_size < q.ask_size
    q = quotes(position=-50)
    assert q.ask_size < q.bid_size


def test_volatility_widens_up_to_a_cap():
    calm = quotes()
    wild = quotes(vol=5.0, vol_factor=1.0, max_vol_ticks=3.0)
    assert wild.bid == pytest.approx(calm.bid - 0.3)
    assert wild.ask == pytest.approx(calm.ask + 0.3)


def test_imbalance_shift_is_bounded_in_ticks():
    q = quotes(imb=1.0, imbalance_ticks=1.0)
    assert q.bid == pytest.approx(99.9) and q.ask == pytest.approx(100.3)


def test_quotes_never_cross_the_external_book():
    top = Top(99.0, 100, 100.1, 100)
    q = quotes(fair=100.5, top=top)
    assert q.ask > q.bid
    assert q.bid <= 100.0


def test_pennying_does_not_give_away_more_than_one_tick():
    top = Top(99.0, 100, 101.0, 100)
    q = quotes(top=top, penny_only=True)
    assert q.bid == pytest.approx(99.1) and q.ask == pytest.approx(100.9)


def test_pennying_never_violates_the_edge_floor():
    top = Top(99.9, 100, 100.1, 100)
    q = quotes(top=top, penny_only=True)
    assert q.bid <= 99.8 and q.ask >= 100.2


def test_spread_never_collapses_after_skew_and_rounding():
    q = quotes(base_spread_ticks=0.0, position=7)
    assert q.ask - q.bid >= TICK - 1e-9
