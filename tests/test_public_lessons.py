"""Mechanisms taken from public Optibook write-ups, each with the behaviour it should produce.

Sources are public Optibook repos and write-ups. These test the mechanics only: whether they
earn more on the day is what the live evidence decides.
"""

import os

import pytest

from otk.engine import Engine, EngineConfig
from otk.instruments import Universe
from otk.intents import LadderIntent, QuoteIntent, Take
from otk.news import NewsSignal
from otk.params import ParamStore, write_params_atomic
from otk.sim.exchange import SimExchange, SimMarket, SocialMediaFeed
from otk.sim.harness import make_engine
from otk.strategies import (
    DualListingArb,
    GapReversion,
    Hedger,
    MarketMaker,
    MeanReversionTaker,
    NewsTrader,
    default_stack,
)


def rest(market, iid, side, price, volume, team="wall"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def book(market, iid, bid, ask, volume=500):
    rest(market, iid, "bid", bid, volume)
    rest(market, iid, "ask", ask, volume)


def pair_market(primary=(99.9, 100.1), dual=(99.0, 101.0), base="A"):
    market = SimMarket(instruments=[base, f"{base}_DUAL"])
    book(market, base, *primary)
    book(market, f"{base}_DUAL", *dual)
    return market


def store(tmp_path, **params):
    path = tmp_path / "params.json"
    write_params_atomic(path, params)
    return ParamStore(path)


def quotes(engine, iid):
    return {o.side: o.price for o in engine.exchange.get_outstanding_orders(iid).values()}


def buy(engine, iid, volume, price=101.0):
    engine.exchange.insert_order(iid, price=price, volume=volume, side="bid", order_type="ioc")


def advance(market, engine, steps=4, dt=0.3):
    for _ in range(steps):
        market.advance(dt)
        engine.tick()


# --- hedging: a spread paid is a cost, and small deltas often cancel (Parity, Imperial 2026 runner-up) --


def test_a_small_delta_is_left_to_cancel_on_its_own(tmp_path):
    market = pair_market(dual=(99.0, 100.1))
    engine = make_engine(market, "us", [Hedger({"A": "A"})])
    buy(engine, "A_DUAL", 50)
    advance(market, engine)
    assert engine.exchange.get_positions()["A"] == 0


def test_a_large_delta_is_hedged_only_part_of_the_way(tmp_path):
    market = pair_market(dual=(99.0, 100.1))
    engine = make_engine(market, "us", [Hedger({"A": "A"})])
    buy(engine, "A_DUAL", 90)
    advance(market, engine)
    assert engine.exchange.get_positions()["A"] == -60, "back to hedge_to, which defaults to 30"


def test_hedging_all_the_way_is_still_available(tmp_path):
    market = pair_market(dual=(99.0, 100.1))
    params = store(tmp_path, hedge_threshold=0, hedge_to=0)
    engine = make_engine(market, "us", [Hedger({"A": "A"})], params=params)
    buy(engine, "A_DUAL", 50)
    advance(market, engine)
    assert engine.exchange.get_positions()["A"] == -50


def test_the_urgent_hedge_can_be_given_up_to_accept_the_soft_penalty(tmp_path):
    def run_with(**params):
        market = SimMarket(instruments=["A", "A_DUAL"], position_limit=500)
        book(market, "A", 99.9, 100.1)
        book(market, "A_DUAL", 99.0, 100.1)
        engine = make_engine(market, "us", [Hedger({"A": "A"})], params=store(tmp_path, hedge_enabled=0, **params))
        buy(engine, "A_DUAL", 150)
        advance(market, engine, steps=30)
        return engine.exchange.get_positions()["A"]

    assert run_with(hedge_urgent=1) < 0, "near the hack-out it hedges despite standing down"
    assert run_with(hedge_urgent=0) == 0


# --- news: conviction belongs on both listings, and must not be hedged away (hackaTUM 2025, TUM) -------


class Always:
    """A classifier with a fixed verdict."""

    def __init__(self, instrument_id="A", direction=1, confidence=0.9):
        self.signal = NewsSignal(instrument_id, direction, confidence)

    def classify(self, _text):
        return self.signal


def publish(market, text="A beats expectations"):
    market.news.append(SocialMediaFeed(market.timestamp(), text, None))


def test_a_news_position_is_not_hedged_away():
    market = pair_market(dual=(99.9, 100.1))
    stack = default_stack(market.instruments, classifier=Always())
    engine = make_engine(market, "us", stack)
    engine.tick()
    publish(market)
    advance(market, engine, steps=6)
    positions = engine.exchange.get_positions()
    assert positions["A"] > 50, positions
    assert positions["A_DUAL"] > 50, positions


def test_news_trades_both_listings():
    market = pair_market(dual=(99.9, 100.1))
    engine = make_engine(market, "us", [NewsTrader(Always())])
    engine.tick()
    publish(market)
    advance(market, engine, steps=6)
    positions = engine.exchange.get_positions()
    assert positions["A"] > 50 and positions["A_DUAL"] > 50


def test_news_can_stay_on_the_named_listing(tmp_path):
    market = pair_market(dual=(99.9, 100.1))
    engine = make_engine(market, "us", [NewsTrader(Always())], params=store(tmp_path, news_both_listings=0))
    engine.tick()
    publish(market)
    advance(market, engine, steps=6)
    positions = engine.exchange.get_positions()
    assert positions["A"] > 50 and positions["A_DUAL"] == 0


def test_hedger_resumes_once_the_news_position_is_over(tmp_path):
    market = pair_market(dual=(99.9, 100.1))
    params = store(tmp_path, news_hold_max_s=1.0, news_cooldown_s=0.0, news_both_listings=0)
    stack = [NewsTrader(Always())]
    stack.insert(0, Hedger({"A": "A_DUAL"}, holding=stack[0].holding))
    engine = make_engine(market, "us", stack, params=params)
    engine.tick()
    publish(market)
    advance(market, engine, steps=3)
    assert engine.exchange.get_positions()["A_DUAL"] == 0, "standing down while the news runs"
    advance(market, engine, steps=12)
    assert engine.exchange.get_positions()["A_DUAL"] < 0, "hedging again afterwards"


# --- closed markets: `paused` flips live, as at IC Hack 2024 and hackaTUM 2024 ---------------------


def test_a_paused_instrument_gets_no_orders_and_reads_as_empty():
    market = pair_market(primary=(99.0, 101.0), dual=(99.0, 101.0))
    market.paused.add("A")
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    advance(market, engine, steps=2)
    assert not engine.exchange.get_outstanding_orders("A")
    assert engine.manager.rejects == 0
    assert engine.errors == 0


def test_quotes_come_back_when_the_market_reopens():
    market = pair_market(primary=(99.0, 101.0))
    market.paused.add("A")
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    advance(market, engine, steps=2)
    market.paused.discard("A")
    advance(market, engine, steps=2)
    assert engine.exchange.get_outstanding_orders("A")


def test_the_simulator_rejects_orders_on_a_paused_instrument_without_a_disconnect():
    market = pair_market()
    market.paused.add("A")
    us = SimExchange(market, "us")
    us.connect()
    response = us.insert_order("A", price=100.0, volume=1, side="bid", order_type="limit")
    assert response.success is False
    assert us.is_connected()
    assert us.get_instruments()["A"].paused is True


def test_the_dual_keeps_quoting_off_its_own_book_while_the_primary_is_closed():
    market = pair_market(dual=(99.0, 101.0))
    market.paused.add("A")
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")])])
    advance(market, engine, steps=2)
    assert set(quotes(engine, "A_DUAL")) == {"bid", "ask"}


def test_an_empty_primary_that_is_not_paused_still_stops_dual_quotes():
    market = SimMarket(instruments=["A", "A_DUAL"])
    book(market, "A_DUAL", 99.0, 101.0)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")])])
    advance(market, engine, steps=2)
    assert not quotes(engine, "A_DUAL")


def test_a_closed_dual_stops_trades_on_both_legs():
    market = pair_market(primary=(99.9, 100.1), dual=(100.5, 101.0))
    market.paused.add("A_DUAL")
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)])
    advance(market, engine, steps=2)
    assert engine.exchange.get_positions() == {"A": 0, "A_DUAL": 0}


def test_hedging_falls_back_on_the_listing_that_is_open():
    market = pair_market(dual=(99.9, 100.1))
    engine = make_engine(market, "us", [Hedger({"A": "A"})])
    buy(engine, "A", 90)
    market.paused.add("A")
    advance(market, engine)
    assert engine.exchange.get_positions()["A_DUAL"] == -60


# --- never quote tighter than the hedge allows (MAD-LAD, Parity) -----------------------------------


def test_dual_quotes_stay_where_a_fill_can_be_hedged_at_a_profit():
    market = pair_market(primary=(99.9, 100.1), dual=(99.9, 100.3))
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")])])
    advance(market, engine, steps=2)
    resting = quotes(engine, "A_DUAL")
    assert resting["bid"] == pytest.approx(99.8), "it would otherwise join at 99.9"
    assert resting["ask"] >= 100.2 - 1e-9


def test_the_side_that_cancels_our_delta_may_quote_tighter(tmp_path):
    def asks(long_lots):
        market = pair_market(primary=(99.9, 100.1), dual=(99.9, 100.1))
        params = store(tmp_path, size=10, base_spread_ticks=0, imbalance_ticks=0, skew_ticks=0, reprice_ticks=0)
        engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")])], params=params)
        if long_lots:
            buy(engine, "A", long_lots)
        advance(market, engine, steps=2)
        return quotes(engine, "A_DUAL")

    flat = asks(0)
    long_position = asks(30)
    assert flat["ask"] >= 100.2 - 1e-9
    assert long_position["ask"] < flat["ask"]


def test_the_bound_can_be_switched_off(tmp_path):
    market = pair_market(primary=(99.9, 100.1), dual=(99.9, 100.3))
    params = store(tmp_path, dual_pillow_ticks=0, imbalance_ticks=0, skew_ticks=0)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")])], params=params)
    advance(market, engine, steps=2)
    assert quotes(engine, "A_DUAL")["bid"] > 99.8


def test_the_bound_applies_to_every_ladder_rung():
    arb = DualListingArb([("A", "A_DUAL")])
    ladder = LadderIntent("A_DUAL", ((100.0, 5), (99.5, 3)), ((100.0, 5), (100.5, 3)), "x")
    bounded = arb._capped(ladder, 99.8, 100.2)
    assert bounded.bids == ((99.8, 5), (99.5, 3))
    assert bounded.asks == ((100.2, 5), (100.5, 3))
    assert arb._capped(QuoteIntent("A_DUAL", 100.0, 5, 100.0, 5), None, 100.2).bid == 100.0


# --- capacity is the scarce resource: small edges must not spend it (IC Hack 2024 winner) -----------


def arb_fill(tmp_path, edge_ticks, start_short=0, **params):
    """Dual bid above primary ask by `edge_ticks`: sell the dual, buy the primary."""
    market = SimMarket(instruments=["A", "A_DUAL"])
    book(market, "A", 99.9, 100.0)
    bid = round(100.0 + edge_ticks * 0.1, 1)
    book(market, "A_DUAL", bid, round(bid + 0.5, 1))
    limits = store(tmp_path, size=30, **params)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)], params=limits)
    if start_short:
        engine.exchange.insert_order("A_DUAL", price=bid, volume=start_short, side="ask", order_type="ioc")
    market.advance(0.1)
    before = engine.exchange.get_positions()["A_DUAL"]
    engine.tick()
    return before - engine.exchange.get_positions()["A_DUAL"]


def test_a_small_edge_uses_little_capacity(tmp_path):
    assert arb_fill(tmp_path, edge_ticks=1) == 10


def test_a_large_edge_uses_the_full_clip(tmp_path):
    assert arb_fill(tmp_path, edge_ticks=3) == 30


def test_less_capacity_left_means_a_smaller_clip(tmp_path):
    assert arb_fill(tmp_path, edge_ticks=3, start_short=50) == 15


def test_edge_scaling_can_be_switched_off(tmp_path):
    assert arb_fill(tmp_path, edge_ticks=1, arb_scale_ticks=0) == 30


def test_a_reducing_arb_is_not_scaled_down(tmp_path):
    market = SimMarket(instruments=["A", "A_DUAL"])
    book(market, "A", 99.9, 100.0)
    book(market, "A_DUAL", 100.1, 100.5)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)], params=store(tmp_path, size=30))
    buy(engine, "A_DUAL", 40, price=100.5)
    market.advance(0.1)
    engine.tick()
    assert engine.exchange.get_positions()["A_DUAL"] == 10


# --- kill switch: on a drawdown stop, flatten as well as pull ---------------------------------------


class Falling:
    """Moves the book down after we are long, so marked PnL falls through the stop."""

    def __init__(self):
        self.level = 100.0

    def act(self, market, now, dt):
        market.wipe_orders("wall", "A")
        self.level -= 0.2
        rest(market, "A", "bid", round(self.level - 0.1, 1), 500)
        rest(market, "A", "ask", round(self.level + 0.1, 1), 500)
        # Someone else trades, so the last price, and with it marked PnL, keeps up with the fall.
        SimExchange(market, "noise", bot=True).insert_order(
            "A", price=round(self.level - 0.1, 1), volume=1, side="ask", order_type="ioc"
        )


def kill_engine(market, **config):
    exchange = SimExchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    settings = EngineConfig(max_drawdown=5.0, **config)
    engine = Engine(exchange, universe, [], settings, ParamStore(), clock=lambda: market.now)
    return engine, exchange


def test_a_drawdown_stop_pulls_quotes_and_flattens_inventory():
    market = SimMarket(instruments=["A"])
    market.actors.append(Falling())
    market.advance(0.5)
    engine, exchange = kill_engine(market)
    exchange.insert_order("A", price=101.0, volume=30, side="bid", order_type="ioc")
    exchange.insert_order("A", price=90.0, volume=5, side="bid", order_type="limit")
    for _ in range(20):
        market.advance(0.5)
        engine.tick()
    assert engine.halted
    assert exchange.get_positions()["A"] == 0
    assert not exchange.get_outstanding_orders("A")
    assert market.stats["us"].limit_breaches == 0


def test_flattening_on_a_stop_can_be_disabled():
    market = SimMarket(instruments=["A"])
    market.actors.append(Falling())
    market.advance(0.5)
    engine, exchange = kill_engine(market, flatten_on_stop=False)
    exchange.insert_order("A", price=101.0, volume=30, side="bid", order_type="ioc")
    for _ in range(20):
        market.advance(0.5)
        engine.tick()
    assert engine.halted
    assert exchange.get_positions()["A"] == 30


def test_a_kill_file_only_pulls_because_an_operator_chose_that(tmp_path):
    kill = tmp_path / "KILL"
    kill.write_text("stop")
    market = pair_market()
    exchange = SimExchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    engine = Engine(
        exchange, universe, [], EngineConfig(kill_file=str(kill)), ParamStore(), clock=lambda: market.now
    )
    exchange.insert_order("A", price=100.1, volume=10, side="bid", order_type="ioc")
    advance(market, engine)
    assert engine.halted
    assert exchange.get_positions()["A"] == 10


@pytest.mark.parametrize("name", ["dual_pillow_ticks", "arb_scale_ticks", "hedge_to", "hedge_urgent", "news_both_listings", "dual_quote_when_closed"])
def test_new_controls_are_runtime_parameters(name):
    assert name in ParamStore().current


# --- directional strategies trade their own position, not everyone's (found on real books) ----------


def own_position_market():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.9, 100.1)
    return market


def test_fills_are_attributed_to_the_strategy_that_sent_them():
    market = own_position_market()
    engine = make_engine(market, "us", [])
    engine.manager.take(Take("A", "bid", 100.1, 7, False, "mean_reversion"), {"A": 0})
    engine.manager.take(Take("A", "bid", 100.1, 3, False, "other"), {"A": 0})
    market.advance(0.1)
    engine.tick()
    state = engine._snapshot()
    assert state.position_of("mean_reversion", "A") == 7
    assert state.position_of("other", "A") == 3
    assert state.position_of("hedger", "A") == 0
    assert state.position("A") == 10


class _Z:
    def __init__(self, z, mean=100.0):
        self.z = z
        self.mean = mean

    def update_z(self, *_args):
        return self.z


def test_fading_ignores_inventory_it_did_not_create(tmp_path):
    market = own_position_market()
    params = store(tmp_path, mr_enabled=1, mr_max_position=20, mr_clip=20, mr_min_edge_ticks=0)
    strategy = MeanReversionTaker(["A"])
    engine = make_engine(market, "us", [strategy], params=params)
    buy(engine, "A", 90)  # inventory from something else
    strategy._stats["A"] = _Z(z=0.1)
    market.advance(0.1)
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 90, "z near zero must not liquidate someone else's position"


def test_fading_closes_its_own_position_when_switched_off(tmp_path):
    market = own_position_market()
    path = tmp_path / "params.json"
    write_params_atomic(path, {"mr_enabled": 1, "mr_max_position": 20, "mr_clip": 20, "mr_min_edge_ticks": 0, "mr_interval_s": 0})
    strategy = MeanReversionTaker(["A"])
    engine = make_engine(market, "us", [strategy], params=ParamStore(path))
    buy(engine, "A", 30)  # not ours
    strategy._stats["A"] = _Z(z=-3.0, mean=101.0)
    market.advance(0.1)
    engine.tick()
    market.advance(0.1)
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 50
    write_params_atomic(path, {"mr_enabled": 0, "mr_clip": 20, "mr_interval_s": 0})
    os.utime(path, (1, 1))
    for _ in range(3):
        market.advance(0.1)
        engine.tick()
    assert engine.exchange.get_positions()["A"] == 30, "back to what was there before"


def test_the_hedger_leaves_a_fading_position_alone_but_hedges_the_rest(tmp_path):
    market = pair_market(dual=(99.9, 100.1))
    params = store(tmp_path, hedge_threshold=10, hedge_to=0)
    engine = make_engine(market, "us", [Hedger({"A": "A"})], params=params)
    engine.manager.take(Take("A_DUAL", "bid", 100.1, 40, False, "mean_reversion"), {"A_DUAL": 0})
    buy(engine, "A", 30)
    advance(market, engine, steps=3)
    positions = engine.exchange.get_positions()
    assert positions["A_DUAL"] == 40
    assert positions["A"] == 0, "the other 30 are hedged"


def test_gap_trades_only_its_own_legs(tmp_path):
    market = SimMarket(instruments=["A", "A_DUAL"])
    book(market, "A", 99.9, 100.1)
    book(market, "A_DUAL", 100.9, 101.1)
    params = store(tmp_path, gap_enabled=1, gap_max_position=20, gap_clip=20, gap_min_edge_ticks=0)
    strategy = GapReversion([("A", "A_DUAL")])
    engine = make_engine(market, "us", [strategy], params=params)
    buy(engine, "A", 40)
    engine.exchange.insert_order("A_DUAL", price=100.9, volume=80, side="ask", order_type="ioc")
    strategy._stats["A_DUAL"] = _Z(z=0.2, mean=1.0)
    market.advance(0.1)
    engine.tick()
    assert engine.exchange.get_positions() == {"A": 40, "A_DUAL": -80}, "inventory from elsewhere is left alone"
