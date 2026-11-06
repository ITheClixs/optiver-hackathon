"""The simulator must enforce Optibook's published rules, because tests lean on it."""

import pytest

from otk.sim.exchange import SimExchange, SimMarket
from otk.sim.scenarios import quiet_market


def market_with(*ids, **kwargs):
    return SimMarket(instruments=list(ids), **kwargs)


def connected(market, team="us"):
    exchange = SimExchange(market, team)
    exchange.connect()
    return exchange


def rest(market, iid, side, price, volume, team="bot"):
    return SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def test_limit_order_rests_and_shows_in_book():
    market = market_with("A")
    us = connected(market)
    assert us.insert_order("A", price=10.0, volume=5, side="bid", order_type="limit").success
    book = us.get_last_price_book("A")
    assert book.bids[0].price == 10.0 and book.bids[0].volume == 5
    assert next(iter(us.get_outstanding_orders("A").values())).volume == 5


def test_ioc_never_rests_and_reports_fills_only_through_trades():
    market = market_with("A")
    rest(market, "A", "ask", 10.0, 3)
    us = connected(market)
    assert us.insert_order("A", price=10.0, volume=5, side="bid", order_type="ioc").success
    assert us.get_outstanding_orders("A") == {}
    assert sum(t.volume for t in us.poll_new_trades("A")) == 3
    assert us.get_positions()["A"] == 3


def test_price_time_priority_fills_earliest_order_first():
    market = market_with("A")
    rest(market, "A", "bid", 10.0, 5, team="first")
    us = connected(market)
    us.insert_order("A", price=10.0, volume=5, side="bid", order_type="limit")
    rest(market, "A", "ask", 10.0, 5, team="taker")
    assert us.get_positions()["A"] == 0
    assert market.positions["first"]["A"] == 5


def test_public_ticks_hide_counterparties_but_show_aggressor():
    market = market_with("A")
    rest(market, "A", "ask", 10.0, 3)
    us = connected(market)
    us.insert_order("A", price=10.0, volume=3, side="bid", order_type="ioc")
    tick = us.poll_new_trade_ticks("A")[0]
    assert (tick.buyer, tick.seller) == ("", "")
    assert tick.aggressor_side == "bid" and tick.volume == 3
    assert us.poll_new_trade_ticks("A") == []


def test_worst_case_position_breach_rejects_and_disconnects():
    market = market_with("A")
    us = connected(market)
    assert us.insert_order("A", price=10.0, volume=60, side="bid", order_type="limit").success
    response = us.insert_order("A", price=9.0, volume=60, side="bid", order_type="limit")
    assert not response.success
    assert not us.is_connected()
    assert market.stats["us"].disconnects == 1
    assert market.orders_of("us", "A") == {}


def test_outstanding_cap_counts_both_sides():
    market = market_with("A")
    us = connected(market)
    assert us.insert_order("A", price=10.0, volume=100, side="bid", order_type="limit").success
    assert us.insert_order("A", price=11.0, volume=100, side="ask", order_type="limit").success
    assert not us.insert_order("A", price=11.0, volume=1, side="ask", order_type="limit").success
    assert not us.is_connected()


def test_update_rate_breach_disconnects():
    market = market_with("A")
    us = connected(market)
    for _ in range(25):
        assert us.insert_order("A", price=10.0, volume=1, side="bid", order_type="limit").success
    assert not us.insert_order("A", price=10.0, volume=1, side="bid", order_type="limit").success
    assert not us.is_connected()
    assert market.stats["us"].rate_breaches == 1


def test_update_budget_recovers_after_a_second():
    market = market_with("A")
    us = connected(market)
    for _ in range(25):
        us.insert_order("A", price=10.0, volume=1, side="bid", order_type="limit")
    market.advance(1.01)
    assert us.insert_order("A", price=10.0, volume=1, side="bid", order_type="limit").success


def test_reads_do_not_count_against_the_rate_limit():
    market = market_with("A")
    us = connected(market)
    for _ in range(100):
        us.get_last_price_book("A")
        us.get_positions()
    assert us.is_connected()


def test_last_login_takes_over_the_connection():
    market = market_with("A")
    first, second = SimExchange(market, "us"), SimExchange(market, "us")
    first.connect()
    second.connect()
    assert second.is_connected() and not first.is_connected()


def test_calls_after_disconnect_raise_for_writes():
    market = market_with("A")
    us = SimExchange(market, "us")
    with pytest.raises(ConnectionError):
        us.insert_order("A", price=10.0, volume=1, side="bid", order_type="limit")


def test_delete_and_amend_semantics():
    market = market_with("A")
    us = connected(market)
    order_id = us.insert_order("A", price=10.0, volume=5, side="bid", order_type="limit").order_id
    assert us.amend_order("A", order_id=order_id, volume=2).success
    assert us.get_outstanding_orders("A")[order_id].volume == 2
    assert us.delete_order("A", order_id=order_id).success
    assert not us.delete_order("A", order_id=order_id).success


def test_delete_orders_removes_all_and_counts_as_one_update():
    market = market_with("A")
    us = connected(market)
    for _ in range(3):
        us.insert_order("A", price=10.0, volume=1, side="bid", order_type="limit")
    us.delete_orders("A")
    assert us.get_outstanding_orders("A") == {}
    assert market.stats["us"].writes == 4


def test_pnl_marks_positions_at_last_traded_price():
    market = market_with("A")
    rest(market, "A", "ask", 10.0, 5)
    us = connected(market)
    us.insert_order("A", price=10.0, volume=5, side="bid", order_type="ioc")
    assert us.get_pnl() == pytest.approx(0.0)
    rest(market, "A", "ask", 12.0, 1, team="b2")
    rest(market, "A", "bid", 12.0, 1, team="b3")
    assert us.get_pnl() == pytest.approx(10.0)
    assert us.get_pnl({"A": 11.0}) == pytest.approx(5.0)


def test_delta_above_soft_limit_is_hacked_after_grace():
    market = market_with("A", "A_DUAL")
    rest(market, "A", "ask", 10.0, 90)
    rest(market, "A_DUAL", "ask", 10.0, 90)
    rest(market, "A", "bid", 9.0, 500)
    rest(market, "A", "ask", 11.0, 500)
    us = connected(market)
    us.insert_order("A", price=10.0, volume=90, side="bid", order_type="ioc")
    us.insert_order("A_DUAL", price=10.0, volume=90, side="bid", order_type="ioc")
    assert us.get_positions()["A"] + us.get_positions()["A_DUAL"] == 180
    for _ in range(12):  # first breach is observed on the first advance, grace runs from there
        market.advance(1.0)
    positions = us.get_positions()
    assert positions["A"] + positions["A_DUAL"] < 180
    assert market.stats["us"].delta_hacks >= 1


def test_quiet_market_is_deterministic_for_a_seed():
    first = quiet_market(seed=7, instruments=("A",))
    second = quiet_market(seed=7, instruments=("A",))
    for market in (first, second):
        for _ in range(100):
            market.advance(0.1)
    assert first.last_trade == second.last_trade
    assert first.book_snapshot("A").bids[0].price == second.book_snapshot("A").bids[0].price
    assert first.stats["noise"].writes > 0
