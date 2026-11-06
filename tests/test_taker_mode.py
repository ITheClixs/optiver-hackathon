"""Taker mode: the backup for when writes run out or the connection is slow. Quoting stops, and the thin leg is taken."""

import json
import time

import pytest

from otk.budget import WriteBudget
from otk.intents import QuoteIntent
from otk.main import write_status
from otk.params import ParamStore, scaled_sizes, validate_params
from otk.quote_manager import QuoteManager
from otk.risk import Limits
from otk.sim.exchange import SimExchange, SimMarket
from otk.sim.harness import make_engine
from otk.strategies import ThinLegTaker, default_stack

IDS = ["A", "A_DUAL"]


def rest(market, iid, side, price, volume, team="wall"):
    SimExchange(market, team, bot=True).insert_order(iid, price=price, volume=volume, side=side, order_type="limit")


def book(market, iid, bid, ask, volume=500):
    rest(market, iid, "bid", bid, volume)
    rest(market, iid, "ask", ask, volume)


def engine_for(market, **params):
    return make_engine(market, "us", default_stack(IDS), params=ParamStore(base=params))


def position(market, iid="A_DUAL"):
    return market.positions["us"][iid]


def crossed_market():
    """The thin leg's ask sits two ticks under the liquid leg's mid: a stale quote."""
    market = SimMarket(instruments=IDS)
    book(market, "A", 99.9, 100.1)
    book(market, "A_DUAL", 99.5, 99.8)
    return market


def run(engine, market, ticks=5):
    for _ in range(ticks):
        market.advance(0.1)
        engine.tick()


def test_it_is_off_by_default_and_does_not_switch_itself_on():
    defaults = validate_params({})
    assert defaults["taker_mode"] == 0
    assert defaults["taker_auto_ms"] == 0, "replays with capped writes do not support switching on a slow connection"


def test_off_means_the_taker_sends_nothing():
    market = crossed_market()
    engine = engine_for(market)
    strategy = next(s for s in engine.strategies if isinstance(s, ThinLegTaker))
    assert strategy.on_tick(engine._snapshot()) == []


def test_on_it_takes_a_stale_thin_leg_quote():
    market = crossed_market()
    engine = engine_for(market, taker_mode=1, taker_clip=20)
    run(engine, market)
    assert position(market) > 0


def test_it_never_takes_the_liquid_leg():
    market = crossed_market()
    engine = engine_for(market, taker_mode=1)
    run(engine, market)
    assert position(market, "A") == 0


def test_it_leaves_a_thin_leg_within_a_tick_of_the_mid_alone():
    market = SimMarket(instruments=IDS)
    book(market, "A", 99.9, 100.1)
    book(market, "A_DUAL", 99.8, 100.2)
    engine = engine_for(market, taker_mode=1)
    run(engine, market)
    assert position(market) == 0


def test_a_rich_thin_leg_is_sold():
    market = SimMarket(instruments=IDS)
    book(market, "A", 99.9, 100.1)
    book(market, "A_DUAL", 100.3, 100.6)
    engine = engine_for(market, taker_mode=1)
    run(engine, market)
    assert position(market) < 0


def test_a_wider_edge_passes_on_a_smaller_mispricing():
    market = crossed_market()
    engine = engine_for(market, taker_mode=1, taker_edge_ticks=5)
    run(engine, market)
    assert position(market) == 0


def test_no_more_than_the_limit_however_long_the_mispricing_lasts():
    market = crossed_market()
    engine = engine_for(market, taker_mode=1, hedge_enabled=0)
    run(engine, market, ticks=100)
    assert 0 < position(market) <= 100


def test_quoting_stops_and_resting_quotes_are_pulled():
    market = SimMarket(instruments=IDS)
    book(market, "A", 99.0, 101.0)
    book(market, "A_DUAL", 98.0, 102.0)
    store = ParamStore(base={})
    engine = make_engine(market, "us", default_stack(IDS), params=store)
    run(engine, market)
    assert engine.exchange.get_outstanding_orders("A") or engine.exchange.get_outstanding_orders("A_DUAL")
    store.set_overlay({"taker_mode": 1})
    run(engine, market)
    assert not engine.exchange.get_outstanding_orders("A")
    assert not engine.exchange.get_outstanding_orders("A_DUAL")


def test_a_book_that_is_closed_or_one_sided_is_left_alone():
    market = SimMarket(instruments=IDS)
    rest(market, "A", "bid", 99.9, 100)
    book(market, "A_DUAL", 99.5, 99.8)
    engine = engine_for(market, taker_mode=1)
    run(engine, market)
    assert position(market) == 0


def test_the_hedger_stays_on_for_a_real_delta_breach_only():
    market = crossed_market()
    engine = engine_for(market, taker_mode=1, taker_clip=100, hedge_threshold=20, hedge_to=10)
    run(engine, market, ticks=3)
    assert position(market) > 60, "the hedge waits, so a low setting does not eat the taker's position"
    params = engine._current_params()
    assert params["hedge_threshold"] >= 98
    assert params["hedge_to"] >= 90


def test_a_stack_with_no_pairs_ignores_the_mode():
    market = SimMarket(instruments=["B"])
    book(market, "B", 99.0, 101.0)
    engine = make_engine(market, "us", default_stack(["B"]), params=ParamStore(base={"taker_mode": 1}))
    run(engine, market)
    assert engine.exchange.get_outstanding_orders("B"), "with nothing to take, it must keep quoting"


def test_a_kill_resume_at_half_size_also_halves_the_clip():
    params = validate_params({"taker_clip": 20})
    assert scaled_sizes(params, 0.5)["taker_clip"] == 10


# --- noticing a slow connection by itself ----------------------------------------------------------


class SlowExchange(SimExchange):
    delay = 0.0

    def insert_order(self, *args, **kwargs):
        time.sleep(self.delay)
        return super().insert_order(*args, **kwargs)


def make_manager(market, delay):
    us = SlowExchange(market, "us")
    us.delay = delay
    us.connect()
    return QuoteManager(us, WriteBudget(25, 0, clock=lambda: market.now), Limits(margin=0))


def test_the_time_a_write_takes_is_measured():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.0, 101.0)
    manager = make_manager(market, 0.02)
    assert manager.write_ms() is None
    manager.sync(QuoteIntent("A", 99.5, 5, 100.5, 5), {})
    assert manager.write_ms() >= 15.0


def test_one_slow_write_does_not_move_the_median():
    market = SimMarket(instruments=["A"])
    manager = make_manager(market, 0.0)
    for ms in (1.0, 1.0, 1.0, 5000.0, 1.0):
        manager.note_write_ms(ms)
    assert manager.write_ms() == 1.0


def fill(engine, ms, count=None):
    for _ in range(count or 20):
        engine.manager.note_write_ms(ms)


def test_a_slow_connection_switches_it_on_and_a_recovery_off():
    market = crossed_market()
    engine = engine_for(market, taker_auto_ms=1000)
    assert engine._current_params()["taker_mode"] == 0
    fill(engine, 1500.0)
    assert engine._current_params()["taker_mode"] == 1
    fill(engine, 700.0, 50)
    assert engine._current_params()["taker_mode"] == 1, "it does not flap around the line"
    fill(engine, 100.0, 50)
    assert engine._current_params()["taker_mode"] == 0


def test_too_few_writes_prove_nothing():
    market = crossed_market()
    engine = engine_for(market, taker_auto_ms=1000)
    fill(engine, 5000.0, 2)
    assert engine._current_params()["taker_mode"] == 0


def test_the_switch_can_be_turned_off():
    market = crossed_market()
    engine = engine_for(market, taker_auto_ms=0)
    fill(engine, 5000.0)
    assert engine._current_params()["taker_mode"] == 0


def test_a_person_can_force_it_on_whatever_the_connection_does():
    market = crossed_market()
    engine = engine_for(market, taker_mode=1)
    fill(engine, 1.0)
    assert engine._current_params()["taker_mode"] == 1


def test_no_pairs_means_no_switch():
    market = SimMarket(instruments=["B"])
    engine = make_engine(market, "us", default_stack(["B"]), params=ParamStore(base={"taker_auto_ms": 1000}))
    fill(engine, 5000.0)
    assert engine._current_params()["taker_mode"] == 0


def test_status_shows_write_time_and_mode(tmp_path):
    market = crossed_market()
    engine = engine_for(market)
    run(engine, market, ticks=2)
    path = tmp_path / "status.json"
    write_status(path, engine, time.monotonic())
    status = json.loads(path.read_text())
    assert status["taker_mode"] == 0
    assert "write_ms" in status


@pytest.mark.parametrize("key,bad", [("taker_mode", 2), ("taker_clip", 0), ("taker_edge_ticks", 0)])
def test_bounds_clamp_rather_than_trust(key, bad):
    assert validate_params({key: bad})[key] != bad
