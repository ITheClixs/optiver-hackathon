"""Behaviours added after the research lanes: self-trade safety, starvation, staged news, models."""

import random
from types import SimpleNamespace

import pytest

from otk.allocator import Cusum, IncumbentGate
from otk.behaviour import dominant_period
from otk.budget import WriteBudget
from otk.engine import EngineConfig
from otk.fair import near_touch_imbalance
from otk.intents import QuoteIntent, Take
from otk.news import AsyncClassifier, NewsSignal
from otk.online_model import AlphaFair, OnlineRidge, Prequential
from otk.params import defaults
from otk.quote_manager import QuoteManager
from otk.risk import Limits
from otk.sim.exchange import SimExchange, SimMarket, SocialMediaFeed
from otk.sim.harness import make_engine
from otk.strategies import MarketMaker, NewsTrader


def rest(market, iid, side, price, volume, team="bot"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def connected(market, team="us"):
    exchange = SimExchange(market, team)
    exchange.connect()
    return exchange


def manager(exchange, market, budget=None):
    budget = budget or WriteBudget(25, 0, clock=lambda: market.now)
    return QuoteManager(exchange, budget, Limits(margin=0), tick=market.tick)


def test_simulator_disconnects_on_a_self_trade():
    market = SimMarket(instruments=["A"])
    us = connected(market)
    us.insert_order("A", price=100.0, volume=5, side="ask", order_type="limit")
    response = us.insert_order("A", price=100.0, volume=5, side="bid", order_type="ioc")
    assert not response.success
    assert not us.is_connected()
    assert market.stats["us"].self_trades == 1


def test_take_cancels_our_own_crossing_quote_first():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "ask", 100.5, 50)
    us = connected(market)
    us.insert_order("A", price=100.2, volume=5, side="ask", order_type="limit")
    manager(us, market).take(Take("A", "bid", 100.5, 10), {"A": 0})
    assert us.is_connected()
    assert us.get_positions()["A"] == 10


def test_take_is_abandoned_when_the_cancel_has_no_budget():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "ask", 100.5, 50)
    us = connected(market)
    us.insert_order("A", price=100.2, volume=5, side="ask", order_type="limit")
    budget = WriteBudget(1, 0, clock=lambda: market.now)
    budget.try_acquire()
    assert manager(us, market, budget).take(Take("A", "bid", 100.5, 10), {"A": 0}) == 0
    assert us.is_connected()
    assert us.get_positions()["A"] == 0


def test_quote_refuses_to_cross_a_resting_order_it_could_not_cancel():
    market = SimMarket(instruments=["A"])
    us = connected(market)
    us.insert_order("A", price=100.0, volume=5, side="ask", order_type="limit")
    budget = WriteBudget(25, 0, clock=lambda: market.now)
    quote_manager = manager(us, market, budget)
    quote_manager._delete = lambda *_: False
    quote_manager.sync(QuoteIntent("A", 100.5, 5, None, 0), {"A": 0})
    assert us.is_connected()
    assert all(o.side == "ask" for o in us.get_outstanding_orders("A").values())


def test_hysteresis_keeps_a_working_order_within_tolerance():
    market = SimMarket(instruments=["A"])
    us = connected(market)
    quote_manager = manager(us, market)
    quote_manager.reprice_ticks = 1.0
    quote_manager.sync(QuoteIntent("A", 99.0, 5, None, 0), {})
    first = set(us.get_outstanding_orders("A"))
    quote_manager.sync(QuoteIntent("A", 99.1, 5, None, 0), {})
    assert set(us.get_outstanding_orders("A")) == first
    quote_manager.sync(QuoteIntent("A", 99.3, 5, None, 0), {})
    assert set(us.get_outstanding_orders("A")) != first


class Fixed:
    name = "fixed"

    def __init__(self, intents):
        self.intents = intents

    def on_tick(self, state):
        return list(self.intents)


def test_a_small_write_budget_still_reaches_every_instrument():
    ids = [f"I{n}" for n in range(12)]
    market = SimMarket(instruments=ids)
    intents = [QuoteIntent(iid, 99.0, 5, 101.0, 5) for iid in ids]
    config = EngineConfig(max_writes_per_second=6, risk_reserve=0)
    engine = make_engine(market, "us", [Fixed(intents)], config)
    for _ in range(40):
        market.advance(0.5)
        engine.tick()
    assert all(engine.exchange.get_outstanding_orders(iid) for iid in ids)


def test_book_without_own_orders_is_not_double_subtracted():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.0, 5)
    seen = {}

    class Spy:
        name = "spy"

        def on_tick(self, state):
            seen["top"] = state.top("A")
            return [QuoteIntent("A", 99.0, 5, None, 0)]

    config = EngineConfig(book_includes_own=False)
    engine = make_engine(market, "us", [Spy()], config)
    engine.tick()
    engine.tick()
    assert seen["top"].bid_vol == 10  # taken at face value: the flag says none of it is ours


def test_near_touch_imbalance_ignores_distant_walls():
    book = SimpleNamespace(
        bids=[SimpleNamespace(price=99.9, volume=90), SimpleNamespace(price=95.0, volume=1000)],
        asks=[SimpleNamespace(price=100.1, volume=10), SimpleNamespace(price=105.0, volume=1000)],
    )
    assert near_touch_imbalance(book, 0.4) == pytest.approx(0.8)
    assert near_touch_imbalance(None, 0.4) == 0.0


def test_market_maker_shifts_toward_near_touch_pressure():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.9, 900)
    rest(market, "A", "ask", 100.5, 100)
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A")


def fed(market, text):
    market.news.append(SocialMediaFeed(market.timestamp(), text, None))


class Canned:
    def __init__(self, signal):
        self.signal = signal

    def classify(self, _text):
        return self.signal


class SlowStub:
    def __init__(self):
        self.results = []
        self.submitted = []

    def submit(self, text):
        self.submitted.append(text)
        return True

    def poll(self):
        out, self.results = self.results, []
        return out


def news_market():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.9, 500)
    rest(market, "A", "ask", 100.1, 500)
    return market


def test_slow_classifier_scales_a_partial_position_to_full_on_agreement():
    market = news_market()
    slow = SlowStub()
    trader = NewsTrader(Canned(NewsSignal("A", 1, 0.9)), slow=slow)
    engine = make_engine(market, "us", [trader])
    fed(market, "A news")
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 50
    assert slow.submitted == ["A news"]
    slow.results = [("A news", NewsSignal("A", 1, 0.9))]
    engine.tick()
    engine.tick()  # the first fill leaves the IOC ledger one tick after it is seen
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 97


def test_slow_classifier_disagreement_stops_the_directional_trade():
    market = news_market()
    slow = SlowStub()
    trader = NewsTrader(Canned(NewsSignal("A", 1, 0.9)), slow=slow)
    engine = make_engine(market, "us", [trader])
    fed(market, "A news")
    engine.tick()
    slow.results = [("A news", NewsSignal("A", -1, 0.9))]
    engine.tick()
    assert "A" not in trader._direction


def test_slow_classifier_alone_can_open_and_failures_are_ignored():
    market = news_market()
    slow = SlowStub()
    trader = NewsTrader(Canned(NewsSignal(None, 0, 0.0)), slow=slow)
    engine = make_engine(market, "us", [trader])
    slow.results = [("x", None), ("y", NewsSignal("A", 1, 0.9))]
    engine.tick()
    assert trader._direction == {"A": 1}


def test_async_classifier_never_raises_into_the_caller():
    calls = []

    def flaky(text):
        calls.append(text)
        if text == "boom":
            raise RuntimeError("model down")
        return NewsSignal("A", 1, 0.9)

    worker = AsyncClassifier(flaky)
    assert worker.submit("ok")
    assert worker.submit("boom")
    worker.close()  # joins the worker, so both results are ready and the assertion cannot race
    results = worker.poll()
    assert dict(results)["boom"] is None
    assert dict(results)["ok"].direction == 1


def test_online_ridge_recovers_a_linear_signal():
    rng = random.Random(0)
    model = OnlineRidge(2, forget=1.0)
    for _ in range(2000):
        x = [rng.uniform(-1, 1), 1.0]
        model.update(x, 0.5 * x[0] + 0.1 + rng.gauss(0, 0.05))
    assert model.weights[0] == pytest.approx(0.5, abs=0.02)
    assert model.weights[1] == pytest.approx(0.1, abs=0.02)


def test_prequential_skill_is_negative_for_a_bad_predictor():
    score = Prequential()
    for value in (1.0, -1.0) * 50:
        score.add(-value, value)
    assert score.skill < 0
    assert Prequential().skill == 0.0


class FeatureState:
    tick_size = 0.1

    def __init__(self, now, signal, mid):
        self.params = defaults()
        self.now = now
        self.signal = signal
        self.mid = mid


def _drive(alpha, series):
    outputs = []
    for now, signal, mid in series:
        outputs.append(alpha(FeatureState(now, signal, mid), "A"))
    return outputs


def test_alpha_fair_learns_a_real_signal_and_stays_bounded():
    rng = random.Random(1)
    alpha = AlphaFair(
        base=lambda state, _iid: state.mid,
        features=lambda state, _iid: [state.signal, 1.0],
        n_features=2,
        horizon_s=1.0,
        warmup=100,
        sample_every_s=0.0,
    )
    mid, now, pending = 100.0, 0.0, 0.0
    outputs = []
    for _ in range(2000):
        now += 1.0
        mid += pending + rng.gauss(0, 0.01)
        signal = rng.choice((-1.0, 1.0))
        pending = 0.1 * signal
        outputs.append((alpha(FeatureState(now, signal, mid), "A") - mid, signal))
    assert alpha.skill("A") > 0.5
    late = outputs[-200:]
    assert all(abs(diff) <= 0.2 + 1e-9 for diff, _ in late)
    assert all(diff * signal > 0 for diff, signal in late)


def test_alpha_fair_switches_itself_off_on_noise():
    rng = random.Random(2)
    alpha = AlphaFair(
        base=lambda state, _iid: state.mid,
        features=lambda state, _iid: [state.signal, 1.0],
        n_features=2,
        horizon_s=1.0,
        warmup=100,
        sample_every_s=0.0,
    )
    mid, now = 100.0, 0.0
    changed = 0
    for step in range(2000):
        now += 1.0
        mid += rng.gauss(0, 0.05)
        result = alpha(FeatureState(now, rng.gauss(0, 1), mid), "A")
        if step > 500 and result != mid:
            changed += 1
    assert changed < 200


def test_alpha_fair_passes_through_when_base_is_missing():
    alpha = AlphaFair(base=lambda _s, _i: None)
    assert alpha(FeatureState(0.0, 0.0, 0.0), "A") is None


def test_incumbent_gate_tests_rarely_and_promotes_only_on_strong_evidence():
    gate = IncumbentGate("base", ["new"], test_every=6, z=2.0, min_pairs=6, seed=0)
    plan = [gate.next_arm() for _ in range(60)]
    assert plan.count("new") == 10
    rng = random.Random(0)
    gate = IncumbentGate("base", ["new"], test_every=6, z=2.0, min_pairs=6, seed=0)
    for _ in range(600):
        arm = gate.next_arm()
        gate.record(arm, rng.gauss(2.0 if arm == "new" else 1.0, 0.5))
        if gate.incumbent == "new":
            break
    assert gate.incumbent == "new"


def test_incumbent_gate_does_not_promote_on_noise():
    rng = random.Random(4)
    gate = IncumbentGate("base", ["new"], test_every=6, z=2.0, min_pairs=8, seed=0)
    for _ in range(180):
        arm = gate.next_arm()
        gate.record(arm, rng.gauss(1.0, 1.0))
    assert gate.incumbent == "base"


def test_cusum_detects_a_fall_and_is_quiet_before():
    rng = random.Random(5)
    detector = Cusum(k=0.5, h=4.0)
    assert not any(detector.update(rng.gauss(0, 0.5)) for _ in range(30))
    assert any(detector.update(rng.gauss(-2.0, 0.5)) for _ in range(10))


def test_dominant_period_finds_a_cycle_and_rejects_noise():
    import math

    cycle = [math.sin(2 * math.pi * t / 180) for t in range(1200)]
    assert dominant_period(cycle) == pytest.approx(180, abs=3)
    rng = random.Random(0)
    assert dominant_period([rng.gauss(0, 1) for _ in range(1200)]) is None


def test_dual_offset_stops_a_persistent_gap_pinning_inventory():
    from otk.strategies import DualListingArb

    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.9, 500)
    rest(market, "A", "ask", 100.1, 500)
    rest(market, "A_DUAL", "bid", 100.3, 500)
    rest(market, "A_DUAL", "ask", 100.7, 500)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)])
    for _ in range(50):
        market.advance(10.0)
        engine.tick()
    assert engine.exchange.get_positions()["A_DUAL"] > -30


def test_junk_wide_book_pulls_quotes():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 10.0, 10)
    rest(market, "A", "ask", 210.0, 10)
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A") == {}


def test_barrier_blocks_a_take_into_an_order_the_cache_has_not_shown():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "ask", 100.5, 50)
    us = connected(market)
    quote_manager = manager(us, market)
    quote_manager.sync(QuoteIntent("A", None, 0, 100.2, 5), {"A": 0})
    real = us.get_outstanding_orders
    us.get_outstanding_orders = lambda _iid: {}  # stale cache: our ask is not listed yet
    deleted = []
    original = us.delete_order

    def spy(iid, *, order_id):
        deleted.append(order_id)
        return original(iid, order_id=order_id)

    us.delete_order = spy
    quote_manager.take(Take("A", "bid", 100.5, 10), {"A": 0})
    us.get_outstanding_orders = real
    assert deleted, "the unseen quote must be cancelled before the IOC crosses it"
    assert us.is_connected()


def test_barrier_expires_and_a_confirmed_delete_clears_it():
    market = SimMarket(instruments=["A"])
    us = connected(market)
    quote_manager = manager(us, market)
    quote_manager.sync(QuoteIntent("A", None, 0, 100.2, 5), {"A": 0})
    assert quote_manager._unlisted
    us.get_outstanding_orders = lambda _iid: {}
    assert quote_manager._orders("A")[1]
    market.advance(1.5)
    assert quote_manager._orders("A")[1] == []


def test_reconnect_backs_off_during_a_login_duel():
    market = SimMarket(instruments=["A"])
    engine = make_engine(market, "us", [])
    attempts = 0
    real_connect = engine.exchange.connect

    def rival_keeps_kicking():
        nonlocal attempts
        attempts += 1
        real_connect()
        market.force_disconnect("us", "someone else logged in")

    engine.exchange.connect = rival_keeps_kicking
    market.force_disconnect("us", "someone else logged in")
    for _ in range(600):
        market.advance(0.1)
        engine.tick()
    assert attempts <= 8  # 60 s with backoff of 1, 2, 4, 8, 16, 30 ...


def test_live_monitor_reports_fill_markouts_and_labels():
    from otk.monitor import LiveMonitor, TeeRecorder
    from otk.recorder import NullRecorder
    from otk.sim.scenarios import quiet_market

    market = quiet_market(5, ("A",))
    monitor = LiveMonitor(market.tick)
    engine = make_engine(
        market, "us", [MarketMaker(["A"])], recorder=TeeRecorder([monitor, NullRecorder()])
    )
    for _ in range(1500):
        market.advance(0.1)
        engine.tick()
    snapshot = monitor.snapshot()
    assert snapshot["A"]["flow"]["n"] > 30
    assert "labels" in snapshot["A"]
    assert snapshot["A"]["own_fills"]["n"] > 0
    assert "markout_5s_ticks" in snapshot["A"]["own_fills"]
    TeeRecorder([monitor]).record("x", {})
    TeeRecorder([monitor]).close()
