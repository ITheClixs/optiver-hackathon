"""Regression tests for the code review: races that only appear when the local view lags."""

import pytest

from otk.budget import WriteBudget
from otk.engine import MAX_TICK_ERRORS, EngineConfig
from otk.intents import QuoteIntent, Take
from otk.news import NewsSignal
from otk.params import ParamStore, write_params_atomic
from otk.quote_manager import QuoteManager
from otk.recorder import NullRecorder
from otk.risk import Limits
from otk.sim.exchange import LaggedExchange, SimExchange, SimMarket, SocialMediaFeed
from otk.sim.harness import make_engine, run_seeds
from otk.sim.scenarios import adverse_field, dual_listing, index_complex
from otk.strategies import DualListingArb, Flattener, MarketMaker, NewsTrader, default_stack


def rest(market, iid, side, price, volume, team="bot"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def two_sided(market, *ids, bid=99.9, ask=100.1, size=500):
    for iid in ids:
        rest(market, iid, "bid", bid, size)
        rest(market, iid, "ask", ask, size)


class Fixed:
    name = "fixed"

    def __init__(self, intents):
        self.intents = intents

    def on_tick(self, state):
        return list(self.intents)


def lagged_manager(market, lag=5.0, limit=100):
    exchange = LaggedExchange(market, "us", lag_s=lag)
    exchange.connect()
    budget = WriteBudget(25, 0, clock=lambda: market.now)
    return exchange, QuoteManager(exchange, budget, Limits(position=limit, margin=0), tick=market.tick)


# --- kill path ------------------------------------------------------------------------------------


def test_kill_keeps_pulling_until_every_order_is_gone_when_the_budget_runs_out(tmp_path):
    ids = [f"I{n}" for n in range(12)]
    market = SimMarket(instruments=ids)
    intents = [QuoteIntent(iid, 99.0, 5, 101.0, 5) for iid in ids]
    kill = tmp_path / "KILL"
    config = EngineConfig(kill_file=str(kill), max_writes_per_second=6, risk_reserve=2)
    engine = make_engine(market, "us", [Fixed(intents)], config)
    for _ in range(60):
        market.advance(0.5)
        engine.tick()
    assert sum(len(engine.exchange.get_outstanding_orders(i)) for i in ids) > 12
    kill.write_text("stop")
    engine.tick()
    assert sum(len(engine.exchange.get_outstanding_orders(i)) for i in ids) > 0  # budget ran out
    for _ in range(10):
        market.advance(1.1)
        engine.tick()
    assert sum(len(engine.exchange.get_outstanding_orders(i)) for i in ids) == 0


def test_shutdown_survives_a_dead_client_and_releases_the_lock(tmp_path):
    market = SimMarket(instruments=["A"])
    lock = tmp_path / "LIVE.lock"
    engine = make_engine(market, "us", [], EngineConfig(live_lock=str(lock)))
    engine.acquire_lock()

    def dead(_iid):
        raise OSError("client died")

    engine.exchange.get_outstanding_orders = dead
    engine.shutdown()
    assert not lock.exists()


# --- lag: capacity must count what the cache has not shown yet ---------------------------------


def test_ioc_volume_counts_until_the_position_catches_up():
    market = SimMarket(instruments=["A"])
    two_sided(market, "A")
    exchange, quote_manager = lagged_manager(market)
    quote_manager.take(Take("A", "bid", 100.1, 90), {"A": 0})
    assert quote_manager.take(Take("A", "bid", 100.1, 90), {"A": 0}) == 10
    assert exchange.is_connected()
    assert market.positions["us"]["A"] == 100


def test_unlisted_resting_volume_counts_toward_capacity():
    market = SimMarket(instruments=["A"])
    exchange, quote_manager = lagged_manager(market)
    quote_manager.sync(QuoteIntent("A", 100.0, 90, None, 0), {"A": 0})
    quote_manager.sync(QuoteIntent("A", 99.9, 90, None, 0), {"A": 0})
    assert exchange.is_connected()
    live = market.orders_of("us", "A")
    assert sum(o.volume for o in live.values()) <= 100
    assert {o.price for o in live.values()} == {99.9}, "the unlisted old order must be replaced"


def test_a_full_fill_clears_the_memory_so_the_side_can_requote():
    market = SimMarket(instruments=["A"])
    exchange, quote_manager = lagged_manager(market)
    quote_manager.sync(QuoteIntent("A", 99.9, 5, None, 0), {"A": 0})
    SimExchange(market, "taker", bot=True).insert_order(
        "A", price=99.9, volume=5, side="ask", order_type="ioc"
    )
    market.advance(6.0)  # past the simulated lag, so the fill is visible
    quote_manager.note_fills({"A": exchange.poll_new_trades("A")})
    assert quote_manager._unlisted == {}
    quote_manager.sync(QuoteIntent("A", 99.9, 5, None, 0), {"A": 5})
    assert sum(o.volume for o in market.orders_of("us", "A").values()) == 5


def test_a_deleted_order_the_cache_still_lists_is_not_deleted_twice():
    market = SimMarket(instruments=["A"])
    exchange = LaggedExchange(market, "us", lag_s=5.0)
    exchange.connect()
    budget = WriteBudget(25, 0, clock=lambda: market.now)
    quote_manager = QuoteManager(exchange, budget, Limits(margin=0), tick=market.tick)
    order_id = exchange.insert_order("A", price=99.0, volume=5, side="bid", order_type="limit").order_id
    exchange.get_outstanding_orders("A")
    market.advance(6.0)
    assert order_id in exchange.get_outstanding_orders("A")
    quote_manager.pull("A")
    writes = market.stats["us"].writes
    quote_manager.pull("A")
    assert market.stats["us"].writes == writes


@pytest.mark.parametrize(
    "scenario",
    [
        lambda seed: dual_listing(seed),
        lambda seed: adverse_field(dual_listing(seed), seed),
        lambda seed: index_complex(seed),
        lambda seed: adverse_field(index_complex(seed), seed),
    ],
)
@pytest.mark.parametrize("lag", [0.2, 0.5])
def test_full_stack_never_breaches_under_read_lag(scenario, lag):
    probe = scenario(0)
    weights = probe.index_weights or None
    results = run_seeds(
        scenario,
        lambda: default_stack(probe.instruments, index_weights=weights),
        seeds=range(2),
        seconds=60,
        lag_s=lag,
    )
    for result in results:
        assert result.disconnects == 0, result
        assert result.limit_breaches == 0, result
        assert result.rate_breaches == 0, result
        assert result.errors == 0, result


def test_news_chase_cannot_double_up_under_lag():
    market = SimMarket(instruments=["A"])
    two_sided(market, "A")

    class Certain:
        def classify(self, _text):
            return NewsSignal("A", 1, 0.95)

    engine = make_engine(market, "us", [NewsTrader(Certain())], lag_s=0.5)
    market.news.append(SocialMediaFeed(market.timestamp(), "A up", None))
    for _ in range(20):
        market.advance(0.1)
        engine.tick()
    assert engine.exchange.is_connected()
    assert market.stats["us"].limit_breaches == 0
    assert market.positions["us"]["A"] <= 100


# --- telemetry cannot hurt trading -----------------------------------------------------------------


class Exploding(NullRecorder):
    def tick(self, state):
        raise ValueError("odd field from the real client")

    def record(self, kind, payload):
        raise ValueError("cannot record")


def test_a_recorder_failure_neither_stops_quoting_nor_blocks_the_kill_file(tmp_path):
    market = SimMarket(instruments=["A"])
    kill = tmp_path / "KILL"
    engine = make_engine(
        market, "us", [Fixed([QuoteIntent("A", 99.0, 5, 101.0, 5)])],
        EngineConfig(kill_file=str(kill)), recorder=Exploding(),
    )
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A")
    assert engine.errors == 0
    assert engine.telemetry_errors == 1
    kill.write_text("stop")
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A") == {}


# --- arb legs ----------------------------------------------------------------------------------------


def crossed_market():
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.8, 200)
    rest(market, "A", "ask", 100.0, 200)
    rest(market, "A_DUAL", "bid", 100.5, 20)
    rest(market, "A_DUAL", "ask", 101.0, 20)
    return market


def test_arb_sends_the_thin_leg_first():
    market = crossed_market()
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)])
    sent = []
    original = engine.exchange.insert_order

    def spy(iid, **kwargs):
        sent.append(iid)
        return original(iid, **kwargs)

    engine.exchange.insert_order = spy
    engine.tick()
    assert sent[:2] == ["A_DUAL", "A"]


def test_arb_skips_the_pair_when_either_leg_has_no_capacity(tmp_path):
    market = crossed_market()
    rest(market, "A", "ask", 100.0, 100)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)])
    engine.exchange.insert_order("A", price=100.0, volume=97, side="bid", order_type="ioc")
    market.advance(1.1)
    writes = market.stats["us"].writes
    engine.tick()
    assert market.stats["us"].writes == writes
    assert engine.exchange.get_positions()["A_DUAL"] == 0


def test_a_reducing_arb_needs_no_edge_so_pinned_inventory_unwinds(tmp_path):
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.9, 200)
    rest(market, "A", "ask", 100.1, 200)
    rest(market, "A_DUAL", "ask", 100.2, 40)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)])
    engine.exchange.insert_order("A_DUAL", price=100.2, volume=40, side="bid", order_type="ioc")
    market.wipe_orders("bot", "A_DUAL")
    rest(market, "A_DUAL", "bid", 100.1, 40)  # sell dual at the primary's ask: zero edge
    rest(market, "A_DUAL", "ask", 100.5, 40)
    market.advance(1.1)
    engine.tick()
    assert engine.exchange.get_positions()["A_DUAL"] < 40


# --- end-game and hedge -------------------------------------------------------------------------


def test_flattener_stops_other_strategies_from_reopening():
    market = SimMarket(instruments=["A"])
    two_sided(market, "A")
    opener = Fixed([Take("A", "bid", 100.1, 10)])
    engine = make_engine(market, "us", [Flattener(flatten_at=5.0), opener])
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 10
    market.advance(6.0)
    for _ in range(3):
        engine.tick()
        market.advance(1.1)
    assert engine.exchange.get_positions()["A"] == 0


def test_hedge_pulls_resting_quotes_when_outstanding_volume_blocks_it():
    market = SimMarket(instruments=["A"])
    two_sided(market, "A")
    us = SimExchange(market, "us")
    us.connect()
    budget = WriteBudget(25, 0, clock=lambda: market.now)
    quote_manager = QuoteManager(us, budget, Limits(margin=0), tick=market.tick)
    us.insert_order("A", price=90.0, volume=100, side="bid", order_type="limit")
    us.insert_order("A", price=110.0, volume=100, side="ask", order_type="limit")
    assert quote_manager.take(Take("A", "ask", 99.9, 10, True), {"A": 0}) == 10
    assert us.is_connected()
    assert market.positions["us"]["A"] == -10


# --- errors ------------------------------------------------------------------------------------------


def test_persistent_tick_errors_force_a_fresh_session():
    market = SimMarket(instruments=["A"])
    engine = make_engine(market, "us", [])
    real_read = engine.exchange.get_last_price_book

    def broken(_iid):
        raise RuntimeError("session is dead but claims otherwise")

    engine.exchange.get_last_price_book = broken
    for _ in range(MAX_TICK_ERRORS):
        engine.tick()
    assert not engine.exchange.is_connected()
    engine.exchange.get_last_price_book = real_read
    market.advance(2.0)
    engine.tick()
    assert engine.exchange.is_connected()
    assert engine.reconnects == 1


def test_rejected_orders_are_counted_with_a_reason():
    market = SimMarket(instruments=["A"])
    us = SimExchange(market, "us")
    us.connect()
    budget = WriteBudget(25, 0, clock=lambda: market.now)
    quote_manager = QuoteManager(us, budget, Limits(position=500, outstanding=900, margin=0), tick=market.tick)
    quote_manager.sync(QuoteIntent("A", 99.0, 150, None, 0), {"A": 0})
    assert quote_manager.rejects == 1
    assert "breach" in quote_manager.last_reject


def test_event_position_limit_reaches_strategies_through_the_base(tmp_path):
    path = tmp_path / "params.json"
    write_params_atomic(path, {"size": 7})
    store = ParamStore(path, base={"position_limit": 50})
    assert store.current["position_limit"] == 50
    assert store.refresh()["position_limit"] == 50
    assert store.refresh()["size"] == 7


def test_market_maker_quotes_on_a_lagged_session_without_flip_flopping():
    market = SimMarket(instruments=["A"])
    two_sided(market, "A", bid=99.0, ask=101.0)
    engine = make_engine(market, "us", [MarketMaker(["A"])], lag_s=0.5)
    for _ in range(100):
        market.advance(0.1)
        engine.tick()
    assert market.stats["us"].writes <= 4, "a steady book must not churn orders under lag"
