"""The deterministic trading loop: state reads, intents, risk clipping, budgeted writes."""

import json

import pytest

from otk.budget import Priority, WriteBudget
from otk.engine import Engine, EngineConfig
from otk.instruments import Universe
from otk.intents import Pull, QuoteIntent, Take
from otk.params import ParamStore, validate_params
from otk.quote_manager import QuoteManager
from otk.risk import Limits
from otk.sim.exchange import SimExchange, SimMarket
from otk.sim.harness import make_engine


def market_with(*ids):
    return SimMarket(instruments=list(ids))


def connected(market, team="us"):
    exchange = SimExchange(market, team)
    exchange.connect()
    return exchange


def rest(market, iid, side, price, volume, team="bot"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def manager_for(exchange, market, budget=None, **kwargs):
    clock = lambda: market.now
    return QuoteManager(
        exchange, budget or WriteBudget(25, 0, clock=clock), Limits(margin=0), tick=market.tick, **kwargs
    )


def test_sync_places_two_sided_quote():
    market = market_with("A")
    us = connected(market)
    manager_for(us, market).sync(QuoteIntent("A", 99.0, 5, 101.0, 5), {})
    book = us.get_last_price_book("A")
    assert (book.bids[0].price, book.bids[0].volume) == (99.0, 5)
    assert (book.asks[0].price, book.asks[0].volume) == (101.0, 5)


def test_sync_is_idle_when_quote_is_unchanged_preserving_queue_priority():
    market = market_with("A")
    us = connected(market)
    manager = manager_for(us, market)
    intent = QuoteIntent("A", 99.0, 5, 101.0, 5)
    manager.sync(intent, {})
    before = market.stats["us"].writes
    ids = set(us.get_outstanding_orders("A"))
    manager.sync(intent, {})
    assert market.stats["us"].writes == before
    assert set(us.get_outstanding_orders("A")) == ids


def test_sync_reprices_by_delete_then_insert():
    market = market_with("A")
    us = connected(market)
    manager = manager_for(us, market)
    manager.sync(QuoteIntent("A", 99.0, 5, None, 0), {})
    manager.sync(QuoteIntent("A", 99.5, 5, None, 0), {})
    orders = list(us.get_outstanding_orders("A").values())
    assert [(o.price, o.volume) for o in orders] == [(99.5, 5)]


def test_sync_amends_down_rather_than_losing_priority():
    market = market_with("A")
    us = connected(market)
    manager = manager_for(us, market)
    manager.sync(QuoteIntent("A", 99.0, 10, None, 0), {})
    order_id = next(iter(us.get_outstanding_orders("A")))
    manager.sync(QuoteIntent("A", 99.0, 4, None, 0), {})
    assert us.get_outstanding_orders("A")[order_id].volume == 4


def test_sync_tops_up_with_a_second_order_keeping_the_first_in_queue():
    market = market_with("A")
    us = connected(market)
    manager = manager_for(us, market, topup_lots=2)
    manager.sync(QuoteIntent("A", 99.0, 5, None, 0), {})
    first = next(iter(us.get_outstanding_orders("A")))
    manager.sync(QuoteIntent("A", 99.0, 9, None, 0), {})
    orders = us.get_outstanding_orders("A")
    assert first in orders and sum(o.volume for o in orders.values()) == 9


def test_sync_clips_to_worst_case_capacity_and_never_breaches():
    market = market_with("A")
    us = connected(market)
    manager = manager_for(us, market)
    manager.sync(QuoteIntent("A", 99.0, 150, None, 0), {"A": 0})
    assert us.is_connected()
    assert sum(o.volume for o in us.get_outstanding_orders("A").values()) == 100


def test_sync_replacing_a_full_bid_deletes_before_inserting():
    market = market_with("A")
    us = connected(market)
    manager = manager_for(us, market)
    manager.sync(QuoteIntent("A", 99.0, 100, None, 0), {"A": 0})
    manager.sync(QuoteIntent("A", 98.0, 100, None, 0), {"A": 0})
    assert us.is_connected()
    orders = list(us.get_outstanding_orders("A").values())
    assert [(o.price, o.volume) for o in orders] == [(98.0, 100)]


def test_quotes_stop_at_budget_but_the_session_survives():
    market = market_with("A", "B")
    us = connected(market)
    budget = WriteBudget(3, 0, clock=lambda: market.now)
    manager = manager_for(us, market, budget=budget)
    manager.sync(QuoteIntent("A", 99.0, 5, 101.0, 5), {})
    manager.sync(QuoteIntent("B", 99.0, 5, 101.0, 5), {})
    assert market.stats["us"].writes == 3
    assert us.is_connected()


def test_cancel_uses_risk_priority_when_quote_budget_is_spent():
    market = market_with("A")
    us = connected(market)
    budget = WriteBudget(4, 3, clock=lambda: market.now)
    manager = manager_for(us, market, budget=budget)
    manager.sync(QuoteIntent("A", 99.0, 5, None, 0), {})
    assert budget.available(Priority.QUOTE) == 0
    manager.sync(QuoteIntent("A", None, 0, None, 0), {})
    assert us.get_outstanding_orders("A") == {}


def test_take_is_clipped_and_ioc():
    market = market_with("A")
    rest(market, "A", "ask", 100.0, 500)
    us = connected(market)
    manager = manager_for(us, market)
    manager.take(Take("A", "bid", 100.0, 150), {"A": 0})
    assert us.get_positions()["A"] == 100
    assert us.get_outstanding_orders("A") == {}


def test_validate_params_clamps_to_hard_bounds_and_rejects_unknown_keys():
    assert validate_params({"size": 10_000})["size"] == 100
    assert validate_params({"base_spread_ticks": -5})["base_spread_ticks"] == 0
    with pytest.raises(ValueError):
        validate_params({"not_a_param": 1})
    with pytest.raises(ValueError):
        validate_params({"size": "ten"})


def test_param_store_reloads_valid_and_keeps_last_good_on_bad_file(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"size": 7}))
    store = ParamStore(path)
    assert store.refresh()["size"] == 7
    path.write_text("{broken")
    assert store.refresh()["size"] == 7
    assert store.last_error
    path.write_text(json.dumps({"size": 8}))
    assert store.refresh()["size"] == 8
    assert store.last_error == ""


class Boom:
    name = "boom"

    def on_tick(self, state):
        raise RuntimeError("bug in a strategy")


class Fixed:
    name = "fixed"

    def __init__(self, intents):
        self.intents = intents

    def on_tick(self, state):
        return list(self.intents)


def engine_for(market, strategies, config=None):
    return make_engine(market, "us", strategies, config or EngineConfig())


def test_a_strategy_exception_pulls_its_quotes_and_never_escapes_the_loop():
    market = market_with("A")
    good = Fixed([QuoteIntent("A", 99.0, 5, 101.0, 5)])
    engine = engine_for(market, [good])
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A")
    engine.strategies = [Boom()]
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A") == {}
    assert engine.errors == 1


def test_kill_file_pulls_everything_and_stops_quoting(tmp_path):
    market = market_with("A")
    kill = tmp_path / "KILL"
    config = EngineConfig(kill_file=str(kill))
    engine = engine_for(market, [Fixed([QuoteIntent("A", 99.0, 5, 101.0, 5)])], config)
    engine.tick()
    kill.write_text("stop")
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A") == {}
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A") == {}


def test_drawdown_stop_pulls_quotes_and_latches():
    market = market_with("A")
    rest(market, "A", "ask", 100.0, 10)
    config = EngineConfig(max_drawdown=5.0)
    engine = engine_for(market, [Fixed([QuoteIntent("A", 90.0, 5, None, 0)])], config)
    engine.exchange.insert_order("A", price=100.0, volume=10, side="bid", order_type="ioc")
    engine.tick()
    rest(market, "A", "bid", 95.0, 1, team="b2")
    rest(market, "A", "ask", 95.0, 1, team="b3")
    engine.tick()
    assert engine.halted
    assert engine.exchange.get_outstanding_orders("A") == {}


def test_engine_reconnects_after_a_forced_disconnect_and_reconciles():
    market = market_with("A")
    engine = engine_for(market, [Fixed([QuoteIntent("A", 99.0, 5, 101.0, 5)])])
    engine.tick()
    market.force_disconnect("us", "test")
    assert not engine.exchange.is_connected()
    engine.tick()
    assert engine.exchange.is_connected()
    assert engine.reconnects == 1
    assert not engine.exchange.get_outstanding_orders("A")  # quiet just after a reconnect: it may be a reset
    engine._quiet_until = float("-inf")
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A")


class Remembers(Fixed):
    def __init__(self, intents):
        super().__init__(intents)
        self.resets = 0

    def on_reset(self):
        self.resets += 1


def test_a_reconnect_resets_every_strategy_and_trades_nothing_for_a_moment():
    market = market_with("A")
    rest(market, "A", "ask", 100.0, 100)
    strategy = Remembers([Take("A", "bid", 100.0, 5)])
    engine = engine_for(market, [strategy])
    engine.tick()
    bought = engine.exchange.get_positions()["A"]
    market.force_disconnect("us", "reset")
    engine.tick()
    engine.tick()
    assert strategy.resets == 1
    assert engine.exchange.get_positions()["A"] == bought  # no take while quiet


def test_takes_run_before_quotes_within_a_tick():
    market = market_with("A")
    rest(market, "A", "ask", 100.0, 100)
    engine = engine_for(
        market,
        [Fixed([QuoteIntent("A", 90.0, 95, None, 0), Take("A", "bid", 100.0, 95)])],
    )
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 95
    assert sum(o.volume for o in engine.exchange.get_outstanding_orders("A").values()) <= 5


def test_first_strategy_owns_an_instrument_quote():
    market = market_with("A")
    first = Fixed([QuoteIntent("A", 99.0, 5, None, 0)])
    second = Fixed([QuoteIntent("A", 80.0, 5, None, 0)])
    engine = engine_for(market, [first, second])
    engine.tick()
    assert [o.price for o in engine.exchange.get_outstanding_orders("A").values()] == [99.0]


def test_pull_overrides_quote_from_a_later_strategy():
    market = market_with("A")
    engine = engine_for(market, [Fixed([Pull("A")]), Fixed([QuoteIntent("A", 99.0, 5, None, 0)])])
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A") == {}


def test_state_exposes_external_book_without_our_orders():
    market = market_with("A")
    rest(market, "A", "bid", 98.0, 50)
    seen = {}

    class Spy:
        name = "spy"

        def on_tick(self, state):
            seen["top"] = state.top("A")
            return [QuoteIntent("A", 99.0, 5, None, 0)]

    engine = engine_for(market, [Spy()])
    engine.tick()
    engine.tick()
    assert seen["top"].bid == 98.0


def test_universe_from_exchange_lists_dual_pairs():
    market = market_with("ASML", "ASML_DUAL", "SAP")
    universe = Universe.from_instruments(connected(market).get_instruments())
    assert universe.dual_pairs() == [("ASML", "ASML_DUAL")]


def test_engine_rejects_a_second_process_with_the_live_lock(tmp_path):
    market = market_with("A")
    lock = str(tmp_path / "LIVE.lock")
    config = EngineConfig(live_lock=lock)
    first = engine_for(market, [], config)
    first.acquire_lock()
    second = engine_for(market, [], config)
    with pytest.raises(RuntimeError):
        second.acquire_lock()
    first.release_lock()
    second.acquire_lock()
    second.release_lock()


def test_config_type_is_exported():
    assert Engine is not None
