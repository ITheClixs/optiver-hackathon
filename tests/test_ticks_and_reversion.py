"""Tick size differs by instrument: dashboards show stocks at 0.01, and ETF and options at 0.1.

Also the aggressive mean-reversion taker, for events where a passive strategy is suggested but not required.
"""

import math

import pytest

from otk.behaviour import analyse_capture
from otk.budget import WriteBudget
from otk.engine import EngineConfig
from otk.instruments import Universe, parse_instrument_id
from otk.intents import QuoteIntent
from otk.params import ParamStore, fine_tick_overrides, params_for, write_params_atomic
from otk.quote_manager import QuoteManager
from otk.replay import ReplayExchange, infer_ticks
from otk.risk import Limits
from otk.sim.exchange import SimExchange, SimMarket
from otk.sim.harness import make_engine
from otk.sim.scenarios import index_complex
from otk.strategies import Hedger, MarketMaker, MeanReversionTaker, default_stack

TICKS = {"S": 0.01, "E": 0.1}


def rest(market, iid, side, price, volume, team="bot"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def fine_market():
    return SimMarket(instruments=["S", "E"], tick_sizes=TICKS)


def test_simulator_rounds_each_instrument_to_its_own_tick():
    market = fine_market()
    rest(market, "S", "bid", 70.88, 10)
    rest(market, "E", "bid", 70.88, 10)
    assert market.book_snapshot("S").bids[0].price == pytest.approx(70.88)
    assert market.book_snapshot("E").bids[0].price == pytest.approx(70.9)


def test_instruments_report_their_own_tick():
    exchange = SimExchange(fine_market(), "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    assert universe.by_id["S"].tick == 0.01
    assert universe.by_id["E"].tick == 0.1


def test_quote_manager_snaps_per_instrument():
    market = fine_market()
    us = SimExchange(market, "us")
    us.connect()
    budget = WriteBudget(25, 0, clock=lambda: market.now)
    manager = QuoteManager(us, budget, Limits(margin=0), tick={"S": 0.01, "E": 0.1}.__getitem__)
    manager.sync(QuoteIntent("S", 70.88, 5, None, 0), {})
    manager.sync(QuoteIntent("E", 70.88, 5, None, 0), {})
    assert [o.price for o in us.get_outstanding_orders("S").values()] == [pytest.approx(70.88)]
    assert [o.price for o in us.get_outstanding_orders("E").values()] == [pytest.approx(70.9)]


def test_market_maker_quotes_on_the_fine_grid_and_improves_by_one_real_tick():
    market = fine_market()
    rest(market, "S", "bid", 70.80, 100)
    rest(market, "S", "ask", 71.00, 100)
    rest(market, "E", "bid", 45.0, 100)
    rest(market, "E", "ask", 46.0, 100)
    engine = make_engine(market, "us", [MarketMaker(["S", "E"])])
    engine.tick()
    fine = {o.side: o.price for o in engine.exchange.get_outstanding_orders("S").values()}
    assert fine["bid"] == pytest.approx(70.81)
    assert fine["ask"] == pytest.approx(70.99)
    coarse = {o.side: o.price for o in engine.exchange.get_outstanding_orders("E").values()}
    assert coarse["bid"] == pytest.approx(45.1)
    assert coarse["ask"] == pytest.approx(45.9)


def test_own_orders_are_removed_on_the_right_grid():
    market = fine_market()
    rest(market, "S", "bid", 70.80, 100)
    rest(market, "S", "ask", 71.00, 100)
    engine = make_engine(market, "us", [MarketMaker(["S"])])
    for _ in range(5):
        market.advance(0.1)
        engine.tick()
    bids = [o.price for o in engine.exchange.get_outstanding_orders("S").values() if o.side == "bid"]
    assert bids == [pytest.approx(70.81)], "must not penny itself up"


def test_safety_parameters_scale_for_fine_ticks_only():
    universe = Universe(
        [
            parse_instrument_id("S")._replace(tick=0.01),
            parse_instrument_id("E")._replace(tick=0.1),
        ]
    )
    overrides = fine_tick_overrides(universe)
    assert "E" not in overrides
    assert overrides["S"]["jump_ticks"] == 50.0
    assert overrides["S"]["max_spread_ticks"] == 500.0
    assert "base_spread_ticks" not in overrides["S"]


def test_base_and_file_instrument_overrides_merge(tmp_path):
    path = tmp_path / "params.json"
    write_params_atomic(path, {"instruments": {"S": {"size": 30}}})
    store = ParamStore(path, base={"instruments": {"S": {"jump_ticks": 50.0}}})
    params = params_for(store.refresh(), "S")
    assert params["size"] == 30
    assert params["jump_ticks"] == 50.0


def test_ticks_can_be_inferred_from_a_capture():
    events = [
        {"t": 0.0, "k": "book", "iid": "S", "bids": [[70.88, 1]], "asks": [[70.94, 1]]},
        {"t": 0.0, "k": "book", "iid": "E", "bids": [[45.1, 1]], "asks": [[45.2, 1]]},
        {"t": 0.0, "k": "book", "iid": "F", "bids": [[86.80000000000001, 1]], "asks": [[87.4, 1]]},
    ]
    assert infer_ticks(events) == {"S": 0.01, "E": 0.1, "F": 0.1}


def test_replay_keeps_fine_tick_levels_apart():
    events = [
        {"t": 0.0, "k": "book", "iid": "S", "bids": [[70.88, 100]], "asks": [[70.94, 100]]},
        {"t": 1.0, "k": "trade", "iid": "S", "px": 70.88, "vol": 50, "aggr": "ask"},
    ]
    exchange = ReplayExchange(events, ["S"])
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("S", price=70.87, volume=5, side="bid", order_type="limit")
    exchange.advance_to(1.0)
    assert exchange.get_positions()["S"] == 0, "a print at 70.88 must not fill a bid at 70.87"


def test_replay_accepts_named_trades_from_the_dashboard_feed():
    events = [
        {"t": 0.0, "k": "book", "iid": "S", "bids": [[70.88, 10]], "asks": [[70.94, 100]]},
        {"t": 1.0, "k": "named_trade", "iid": "S", "px": 70.88, "vol": 15, "aggr": "ask", "buyer": "a", "seller": "b"},
    ]
    exchange = ReplayExchange(events, ["S"])
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("S", price=70.88, volume=10, side="bid", order_type="limit")
    exchange.advance_to(1.0)
    assert exchange.get_positions()["S"] == 5


def test_behaviour_report_measures_spread_in_each_instruments_own_ticks():
    events = []
    for step in range(40):
        events.append({"t": float(step), "k": "book", "iid": "S", "bids": [[70.88, 10]], "asks": [[70.94, 10]]})
        events.append({"t": float(step), "k": "book", "iid": "E", "bids": [[45.1, 10]], "asks": [[45.2, 10]]})
    report = analyse_capture(events)
    assert report["instruments"]["S"]["spread_ticks_median"] == pytest.approx(6.0, abs=0.01)
    assert report["instruments"]["E"]["spread_ticks_median"] == pytest.approx(1.0, abs=0.01)
    assert report["instruments"]["S"]["tick"] == 0.01


def test_full_stack_runs_clean_with_mixed_ticks():
    market = index_complex(1, stock_tick=0.01)
    assert market.tick_of("ASML") == 0.01 and market.tick_of("OB5X_ETF") == 0.1
    engine = make_engine(
        market, "us", default_stack(market.instruments, index_weights=market.index_weights), lag_s=0.2
    )
    for _ in range(600):
        market.advance(0.1)
        engine.tick()
    stats = market.stats["us"]
    assert (stats.disconnects, stats.limit_breaches, stats.rate_breaches, engine.errors) == (0, 0, 0, 0)


# --- aggressive mean reversion ----------------------------------------------------------------------


class Oscillator:
    """A wall whose center mean-reverts on a sine: fading it pays even after crossing the spread."""

    def __init__(self, amplitude=1.0, period_s=60.0, half_spread=0.1):
        self.amplitude = amplitude
        self.period_s = period_s
        self.half_spread = half_spread

    def act(self, market, now, dt):
        market.wipe_orders("wall", "A")
        center = 100.0 + self.amplitude * math.sin(2 * math.pi * now / self.period_s)
        rest(market, "A", "bid", round(center - self.half_spread, 1), 500, team="wall")
        rest(market, "A", "ask", round(center + self.half_spread, 1), 500, team="wall")


def oscillating(**kwargs):
    market = SimMarket(instruments=["A"])
    market.actors.append(Oscillator(**kwargs))
    return market


def store(tmp_path, **params):
    path = tmp_path / "params.json"
    write_params_atomic(path, params)
    return ParamStore(path)


def run(market, engine, steps=1600, dt=0.5):
    peak = 0
    for _ in range(steps):
        market.advance(dt)
        engine.tick()
        peak = max(peak, abs(engine.exchange.get_positions()["A"]))
    return peak


def test_mean_reversion_is_off_by_default():
    market = oscillating()
    engine = make_engine(market, "us", [MeanReversionTaker(["A"])])
    assert run(market, engine, steps=400) == 0


def test_mean_reversion_harvests_an_oscillation(tmp_path):
    market = oscillating()
    params = store(tmp_path, mr_enabled=1, mr_entry_z=1.0, mr_half_life_s=120, mr_min_edge_ticks=2)
    engine = make_engine(market, "us", [MeanReversionTaker(["A"])], params=params)
    peak = run(market, engine)
    assert peak > 0
    assert engine.exchange.get_pnl({"A": 100.0}) > 0


def test_mean_reversion_can_be_enabled_for_one_instrument(tmp_path):
    market = SimMarket(instruments=["A", "B"])
    market.actors.append(Oscillator())
    rest(market, "B", "bid", 99.9, 500)
    rest(market, "B", "ask", 100.1, 500)
    params = store(tmp_path, instruments={"A": {"mr_enabled": 1, "mr_entry_z": 1.0, "mr_min_edge_ticks": 2}})
    engine = make_engine(market, "us", [MeanReversionTaker(["A", "B"])], params=params)
    run(market, engine, steps=400)
    assert engine.exchange.get_positions()["B"] == 0
    assert market.stats["us"].writes > 0


def test_mean_reversion_respects_its_cap_and_never_breaches(tmp_path):
    market = oscillating(amplitude=3.0)
    params = store(tmp_path, mr_enabled=1, mr_entry_z=0.5, mr_max_position=20, mr_clip=8, mr_min_edge_ticks=1)
    engine = make_engine(market, "us", [MeanReversionTaker(["A"])], params=params, lag_s=0.2)
    peak = run(market, engine, steps=800)
    assert 0 < peak <= 20 + 8, "the lagged position can overshoot by one clip"
    assert market.stats["us"].limit_breaches == 0


def test_mean_reversion_stays_out_when_the_edge_cannot_pay_the_spread(tmp_path):
    market = oscillating(amplitude=0.1, half_spread=1.0)
    params = store(tmp_path, mr_enabled=1, mr_entry_z=0.5, mr_min_edge_ticks=5)
    engine = make_engine(market, "us", [MeanReversionTaker(["A"])], params=params)
    run(market, engine, steps=400)
    assert market.stats["us"].writes == 0


def test_mean_reversion_stops_out_of_a_persistent_move_and_pauses(tmp_path):
    market = SimMarket(instruments=["A"])
    params = store(
        tmp_path, mr_enabled=1, mr_entry_z=1.0, mr_stop_z=4.0, mr_cooldown_s=1000, mr_min_edge_ticks=0,
        mr_half_life_s=300,
    )
    strategy = MeanReversionTaker(["A"])
    engine = make_engine(market, "us", [strategy], params=params)
    level = 100.0
    for step in range(600):
        market.wipe_orders("wall", "A")
        wobble = 0.1 * math.sin(step / 3)
        if step > 300:
            level -= 0.02  # the "cheap" price keeps falling: news, not noise
        rest(market, "A", "bid", round(level + wobble - 0.1, 1), 500, team="wall")
        rest(market, "A", "ask", round(level + wobble + 0.1, 1), 500, team="wall")
        market.advance(0.5)
        engine.tick()
    assert strategy.stopped_out("A", market.now)
    assert abs(engine.exchange.get_positions()["A"]) < 20


def test_hedger_stands_down_where_a_directional_strategy_is_enabled(tmp_path):
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.9, 500)
    rest(market, "A", "ask", 100.1, 500)
    rest(market, "A_DUAL", "ask", 100.1, 50)
    params = store(tmp_path, instruments={"A": {"mr_enabled": 1}})
    engine = make_engine(market, "us", [Hedger(hedge_on={"A": "A"})], params=params)
    engine.exchange.insert_order("A_DUAL", price=100.1, volume=50, side="bid", order_type="ioc")
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 0


def test_hedger_can_be_switched_off_per_underlying(tmp_path):
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.9, 500)
    rest(market, "A", "ask", 100.1, 500)
    rest(market, "A_DUAL", "ask", 100.1, 50)
    params = store(tmp_path, instruments={"A": {"hedge_enabled": 0}})
    engine = make_engine(market, "us", [Hedger(hedge_on={"A": "A"})], params=params)
    engine.exchange.insert_order("A_DUAL", price=100.1, volume=50, side="bid", order_type="ioc")
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 0


def test_default_stack_carries_mean_reversion_and_stays_clean():
    ids = ["ASML", "ASML_DUAL", "SAP"]
    stack = default_stack(ids)
    assert any(isinstance(item, MeanReversionTaker) for item in stack)
    market = SimMarket(instruments=ids)
    engine = make_engine(market, "us", stack, EngineConfig())
    engine.tick()
    assert engine.errors == 0

