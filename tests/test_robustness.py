"""Failures found in review that would cost the final: overshoots, phantom positions, start-up loops."""

import json
import math
import os
import stat
import sys
import time
from types import SimpleNamespace

import pytest

from otk import main as bot_main
from otk.adapter import LiveExchange
from otk.budget import WriteBudget
from otk.engine import Engine, EngineConfig
from otk.formats import apply_pairs
from otk.instruments import Universe, parse_instrument_id
from otk.intents import QuoteIntent, Take
from otk.kill import KillAction, KillSwitch
from otk.monitor import TeeRecorder
from otk.news import NewsSignal
from otk.params import ParamStore, validate_params
from otk.quote_manager import QuoteManager
from otk.replay import ReplayExchange
from otk.reversion import ArmConfig, ReversionArm
from otk.risk import Limits
from otk.sim.exchange import InsertOrderResponse, SimExchange, SimMarket, SocialMediaFeed
from otk.sim.harness import make_engine
from otk.strategies import (
    DualListingArb,
    Flattener,
    GapReversion,
    Hedger,
    MarketMaker,
    MeanReversionTaker,
    NewsTrader,
)


def rest(market, iid, side, price, volume, team="wall"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def book(market, iid, bid, ask, volume=1000):
    rest(market, iid, "bid", bid, volume)
    rest(market, iid, "ask", ask, volume)


def real_position(market, iid="A"):
    return market.positions["us"][iid]


# --- exits must never overshoot through late positions ---------------------------------------------


def build_position(market, engine, volume=40, iid="A"):
    engine.exchange.insert_order(iid, price=100.1, volume=volume, side="bid", order_type="ioc")
    for _ in range(30):
        market.advance(0.02)
        engine.tick()


@pytest.mark.parametrize("lag", [0.1, 0.3, 0.6])
def test_a_flattener_cannot_flip_the_position_when_fills_report_late(lag):
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.9, 100.1)
    engine = make_engine(market, "us", [Flattener(flatten_at=1.0)], lag_s=lag)
    build_position(market, engine)
    lowest = real_position(market)
    for _ in range(200):
        market.advance(0.02)
        engine.tick()
        lowest = min(lowest, real_position(market))
    assert lowest >= 0, "sold more than it held"
    assert real_position(market) == 0


def test_a_kill_flatten_cannot_flip_the_position_when_fills_report_late():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.9, 100.1)
    engine = make_engine(market, "us", [], lag_s=0.6)
    build_position(market, engine)
    for _ in range(40):
        market.advance(0.02)
        engine.tick()
    state = engine._snapshot()
    lowest = real_position(market)
    for _ in range(200):
        market.advance(0.02)
        state = engine._snapshot()
        engine.manager.note_fills(state.new_fills)
        engine._flatten_out(state)
        lowest = min(lowest, real_position(market))
    assert lowest >= 0
    assert real_position(market) == 0


def test_a_news_close_out_cannot_flip_the_position():
    class Always:
        def classify(self, _text):
            return NewsSignal("A", 1, 0.9)

    class Falling:
        def __init__(self):
            self.level = 100.0
            self.falling = False

        def act(self, market, now, dt):
            market.wipe_orders("wall", "A")
            if self.falling:
                self.level -= 0.05
            book(market, "A", round(self.level - 0.1, 1), round(self.level + 0.1, 1))

    market = SimMarket(instruments=["A"])
    actor = Falling()
    market.actors.append(actor)
    market.advance(0.02)
    engine = make_engine(market, "us", [NewsTrader(Always())], lag_s=0.3)
    engine.tick()
    market.news.append(SocialMediaFeed(market.timestamp(), "post", None))
    for _ in range(40):
        market.advance(0.02)
        engine.tick()
    assert real_position(market) > 50
    actor.falling = True
    lowest = real_position(market)
    for _ in range(300):
        market.advance(0.02)
        engine.tick()
        lowest = min(lowest, real_position(market))
    assert lowest >= 0, "closing out must not go short"
    assert real_position(market) == 0


def test_a_reduce_only_take_does_nothing_on_the_wrong_side():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.9, 100.1)
    us = SimExchange(market, "us")
    us.connect()
    manager = QuoteManager(us, WriteBudget(25, 0, clock=lambda: market.now), Limits(margin=0))
    assert manager.take(Take("A", "bid", 100.1, 5, True, "kill", True), {"A": 10}) == 0
    assert manager.take(Take("A", "ask", 99.9, 5, True, "kill", True), {"A": 0}) == 0
    assert manager.take(Take("A", "ask", 99.9, 50, True, "kill", True), {"A": 10}) == 10


# --- what one strategy built must not outlive it ---------------------------------------------------


def test_a_flatten_clears_what_strategies_think_they_hold():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.9, 100.1)
    strategy = MeanReversionTaker(["A"])
    engine = make_engine(market, "us", [strategy], params=ParamStore(base={"mr_interval_s": 0, "mr_clip": 30}))
    engine.manager.take(Take("A", "bid", 100.1, 30, False, "mean_reversion"), {"A": 0})
    market.advance(0.1)
    engine.tick()
    assert engine.manager.source_positions[("mean_reversion", "A")] == 30
    engine._flatten_out(engine._snapshot())
    for _ in range(5):
        market.advance(0.1)
        engine.tick()
    assert real_position(market) == 0, "no phantom exit into a real short"
    assert engine.manager.source_positions.get(("mean_reversion", "A"), 0) == 0


def test_a_reconnect_forgets_what_strategies_think_they_hold():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.9, 100.1)
    engine = make_engine(market, "us", [])
    engine.manager.take(Take("A", "bid", 100.1, 30, False, "mean_reversion"), {"A": 0})
    market.advance(0.1)
    engine.tick()
    engine.manager.forget_orders()
    assert engine.manager.source_positions == {}


def test_the_delta_watch_sees_the_true_delta_not_the_adjusted_one():
    market = SimMarket(instruments=["A", "A_DUAL"])
    book(market, "A", 99.9, 100.1)
    book(market, "A_DUAL", 99.9, 100.1)
    engine = make_engine(market, "us", [Hedger({"A": "A"})])
    engine.manager.take(Take("A", "bid", 100.1, 60, False, "mean_reversion"), {"A": 0})
    engine.exchange.insert_order("A_DUAL", price=100.1, volume=60, side="bid", order_type="ioc")
    for _ in range(12):
        market.advance(0.5)
        engine.tick()
    assert real_position(market, "A") < 60, "true delta of 120 must trigger the urgent hedge"


# --- start-up must not become a re-login loop ------------------------------------------------------


class StubClient:
    def __init__(self):
        self.connects = 0
        self.disconnects = 0
        self.listed = {"A": SimpleNamespace(instrument_id="A", tick_size=0.1, expired=False, paused=False, instrument_group="A")}

    def connect(self):
        self.connects += 1

    def disconnect(self):
        self.disconnects += 1

    def is_connected(self):
        return True

    def get_instruments(self):
        return self.listed

    def get_last_price_book(self, _iid):
        return SimpleNamespace(bids=[SimpleNamespace(price=99.0, volume=1)], asks=[SimpleNamespace(price=101.0, volume=1)])


def test_an_unknown_instrument_in_the_config_is_skipped_not_fatal():
    exchange = StubClient()
    universe = bot_main.prepare_universe(exchange, {"instruments": ["A", "NOPE"]})
    assert list(universe.by_id) == ["A"]


def test_a_config_that_lists_nothing_real_falls_back_to_everything():
    exchange = StubClient()
    universe = bot_main.prepare_universe(exchange, {"instruments": ["NOPE"]})
    assert list(universe.by_id) == ["A"]


def test_a_bad_parameter_in_the_event_block_is_dropped_not_fatal(caplog):
    universe = Universe([parse_instrument_id("A")])
    event = {"params": {"size": 7, "nonsense": 1, "base_spread_ticks": "wide"}}
    with caplog.at_level("ERROR"):
        base = bot_main.event_base(event, universe)
    assert base["size"] == 7
    assert "nonsense" not in base
    assert "base_spread_ticks" not in base
    assert validate_params(base)
    assert "nonsense" in caplog.text


def test_a_missing_training_file_falls_back_to_rules(tmp_path, caplog):
    universe = Universe([parse_instrument_id("NVDA")])
    with caplog.at_level("ERROR"):
        fast, slow = bot_main.news_parts({"news_training_csv": str(tmp_path / "absent.csv")}, universe)
    assert fast.classify("Nvidia beats").direction == 1
    assert slow is None
    assert "absent.csv" in caplog.text


def test_an_empty_or_dead_lock_is_cleared_and_a_live_one_is_not(tmp_path):
    lock = tmp_path / "LIVE.lock"
    lock.write_text("")
    assert bot_main.claim_session(lock, tmp_path / "heartbeat") is not None
    bot_main.release_session(lock)
    assert not lock.exists()
    lock.write_text("999999")
    assert bot_main.claim_session(lock, tmp_path / "heartbeat") is not None
    bot_main.release_session(lock)
    lock.write_text(str(os.getppid()))
    (tmp_path / "heartbeat").touch()
    assert bot_main.claim_session(lock, tmp_path / "heartbeat") is None, "someone is really running"


def test_a_lock_whose_owner_stopped_beating_is_cleared(tmp_path):
    lock = tmp_path / "LIVE.lock"
    lock.write_text(str(os.getppid()))  # a pid that exists, as after pid reuse in a new container
    heartbeat = tmp_path / "heartbeat"
    heartbeat.touch()
    old = time.time() - 3600
    os.utime(heartbeat, (old, old))
    assert bot_main.claim_session(lock, heartbeat) is not None


def test_releasing_a_lock_twice_is_harmless(tmp_path):
    bot_main.release_session(tmp_path / "absent.lock")


def test_a_second_copy_exits_without_restarting(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(f"import sys; sys.exit({bot_main.EXIT_ALREADY_RUNNING})")
    code = bot_main.supervise([sys.executable, str(script)], tmp_path / "heartbeat", poll_s=0.01)
    assert code == bot_main.EXIT_ALREADY_RUNNING


def test_fast_failures_back_off_so_a_broken_start_does_not_hammer_the_login(tmp_path):
    script = tmp_path / "child.py"
    script.write_text("import sys; sys.exit(9)")
    pauses = []
    code = bot_main.supervise(
        [sys.executable, str(script)], tmp_path / "heartbeat", poll_s=0.01, restart_pause_s=1.0,
        max_restarts=5, sleep=pauses.append,
    )
    assert code == 9
    restarts = [pause for pause in pauses if pause >= 1.0]
    assert restarts == [1.0, 2.0, 4.0, 8.0, 16.0] or restarts[:4] == [1.0, 2.0, 4.0, 8.0]


def test_if_no_child_process_can_start_it_runs_in_place(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bot_main, "supervise", lambda *a, **k: (_ for _ in ()).throw(OSError("no subprocess")))
    monkeypatch.setattr(bot_main, "run_live", lambda args, event: calls.append(event) or 0)
    code = bot_main.main(["--run-dir", str(tmp_path / "run")], event={"x": 1}, entry="bot.py")
    assert code == 0
    assert calls == [{"x": 1}]


# --- somewhere it cannot write ---------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_an_unwritable_run_dir_falls_back_to_a_temporary_one(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        chosen = bot_main.usable_run_dir(locked / "run")
        assert chosen != locked / "run"
        (chosen / "probe").write_text("ok")
    finally:
        locked.chmod(stat.S_IRWXU)


def test_a_writable_run_dir_is_used_as_given(tmp_path):
    assert bot_main.usable_run_dir(tmp_path / "run") == tmp_path / "run"


def test_a_heartbeat_that_cannot_be_written_does_not_stop_the_loop(tmp_path):
    bot_main.beat(tmp_path / "absent" / "heartbeat")


# --- a market that keeps rejecting -----------------------------------------------------------------


class RejectingExchange(SimExchange):
    def insert_order(self, instrument_id, **kwargs):
        if instrument_id == "BAD":
            self.bad_attempts = getattr(self, "bad_attempts", 0) + 1
            return InsertOrderResponse(False, 0, "price outside band")
        return super().insert_order(instrument_id, **kwargs)


def test_a_rejecting_instrument_backs_off_and_cannot_halt_the_rest():
    market = SimMarket(instruments=["BAD", "GOOD"])
    book(market, "BAD", 99.0, 101.0)
    book(market, "GOOD", 99.0, 101.0)
    exchange = RejectingExchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    store = ParamStore(base={"kill_reject_limit": 30, "kill_pause_s": 5})
    engine = Engine(exchange, universe, [MarketMaker(["BAD", "GOOD"])], EngineConfig(), store, clock=lambda: market.now)
    halted = 0
    for _ in range(500):
        market.advance(0.02)
        engine.tick()
        halted += engine.halted
    assert exchange.bad_attempts < 40, "about one attempt per side every two seconds"
    assert halted == 0
    assert exchange.get_outstanding_orders("GOOD")


# --- staleness must be measured all the time -------------------------------------------------------


class Moving:
    def __init__(self):
        self.step = 0

    def act(self, market, now, dt):
        self.step += 1
        market.wipe_orders("wall", "A")
        book(market, "A", 99.9, 100.1, volume=500 + self.step)


def test_a_long_manual_halt_does_not_look_like_stale_data_afterwards(tmp_path):
    kill = tmp_path / "KILL"
    kill.write_text("stop")
    market = SimMarket(instruments=["A"])
    market.actors.append(Moving())
    market.advance(0.5)
    exchange = SimExchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    store = ParamStore(base={"kill_stale_s": 5})
    engine = Engine(exchange, universe, [MarketMaker(["A"])], EngineConfig(kill_file=str(kill)), store, clock=lambda: market.now)
    for _ in range(60):
        market.advance(0.5)
        engine.tick()
    kill.unlink()
    market.advance(0.5)
    engine.tick()
    assert not engine.halted


def test_a_reconnect_restarts_the_staleness_clock():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.9, 100.1)
    engine = make_engine(market, "us", [], params=ParamStore(base={"kill_stale_s": 5}))
    market.advance(0.1)
    engine.tick()
    market.advance(4.0)
    engine.tick()
    engine.exchange.drop("test")
    market.advance(4.0)
    engine.tick()
    market.advance(1.1)
    engine.tick()
    assert not engine.halted


# --- evidence cannot be perfect --------------------------------------------------------------------


def test_identical_round_trips_do_not_read_as_infinite_confidence():
    arm = ReversionArm(60.0, 1.0, ArmConfig(), tick=0.1)
    arm.results.extend([0.1] * 8)
    assert math.isfinite(arm.t)
    assert arm.t > 0


def test_a_new_arm_starts_small_however_clean_its_record():
    from otk.autopilot import AutopilotConfig, _Target

    arm = ReversionArm(60.0, 1.0, ArmConfig(), tick=0.1)
    arm.results.extend([0.1] * 8)
    target = _Target("A", "mr", "A", 0.1, [arm])
    target.active = arm
    assert target.values(AutopilotConfig())["mr_max_position"] == 15
    arm.results.extend([0.1] * 24)
    assert target.values(AutopilotConfig())["mr_max_position"] == 30


# --- pairs -----------------------------------------------------------------------------------------


class _Z:
    def __init__(self, z, mean):
        self.z = z
        self.mean = mean

    def update_z(self, *_args):
        return self.z


def test_gap_legs_are_clipped_together_so_neither_is_stranded():
    market = SimMarket(instruments=["A", "A_DUAL"])
    book(market, "A", 99.9, 100.1)
    book(market, "A_DUAL", 100.9, 101.1)
    params = ParamStore(base={"gap_enabled": 1, "gap_max_position": 40, "gap_clip": 40, "gap_min_edge_ticks": 0})
    strategy = GapReversion([("A", "A_DUAL")])
    engine = make_engine(market, "us", [strategy], params=params)
    engine.exchange.insert_order("A", price=100.1, volume=95, side="bid", order_type="ioc")
    strategy._stats["A_DUAL"] = _Z(z=3.0, mean=0.0)
    market.advance(0.1)
    engine.tick()
    assert real_position(market, "A_DUAL") == -(real_position(market, "A") - 95)
    assert real_position(market, "A") <= 97


def test_an_arb_is_not_resent_against_the_same_stale_book():
    market = SimMarket(instruments=["A", "A_DUAL"])
    book(market, "A", 99.9, 100.0, volume=1000)
    book(market, "A_DUAL", 100.3, 100.6, volume=1000)

    class Frozen(SimExchange):
        """Serves the first book it saw, as when the update has not reached us yet."""

        def get_last_price_book(self, instrument_id):
            self.seen = getattr(self, "seen", {})
            return self.seen.setdefault(instrument_id, super().get_last_price_book(instrument_id))

    exchange = Frozen(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    universe = apply_pairs(universe, [("A", "A_DUAL")])
    store = ParamStore(base={"size": 10})
    engine = Engine(exchange, universe, [DualListingArb([("A", "A_DUAL")], quote=False)], EngineConfig(), store, clock=lambda: market.now)
    for _ in range(5):
        market.advance(0.02)
        engine.tick()
    assert abs(real_position(market, "A_DUAL")) == 10, "one clip until the book shows what happened"
    for _ in range(40):
        market.advance(0.02)
        engine.tick()
    assert abs(real_position(market, "A_DUAL")) > 10, "and it does not stop for good"


# --- a latch must survive a crash, not a new start -------------------------------------------------


def trip_twice(switch):
    params = validate_params({"kill_max_loss": 5, "kill_cooloff_s": 1, "kill_max_trips": 2})
    switch.check(0.0, 0.0, 0.0, 0, 0.0, params)
    switch.check(1.0, -10.0, 0.0, 0, 0.0, params)
    switch.check(3.0, -10.0, 0.0, 0, 0.0, params)
    switch.check(4.0, -30.0, 0.0, 0, 0.0, params)
    return switch


def test_a_restart_by_the_same_supervisor_keeps_the_latch(tmp_path):
    state = tmp_path / "kill_state.json"
    assert trip_twice(KillSwitch(state, owner=123)).latched
    again = KillSwitch(state, owner=123)
    assert again.latched
    assert again.trips == 2
    verdict = again.check(0.0, 0.0, 0.0, 0, 0.0, validate_params({}))
    assert verdict.action is KillAction.FLATTEN


def test_a_new_launch_starts_clean_so_practice_cannot_latch_the_final(tmp_path):
    state = tmp_path / "kill_state.json"
    trip_twice(KillSwitch(state, owner=123))
    assert not KillSwitch(state, owner=456).latched


@pytest.mark.parametrize("content", ["not json", "[]", '{"owner": 123, "trips": "many"}'])
def test_a_broken_state_file_is_ignored(tmp_path, content):
    state = tmp_path / "kill_state.json"
    state.write_text(content)
    assert KillSwitch(state, owner=123).trips == 0


def test_a_state_file_that_cannot_be_written_changes_nothing(tmp_path):
    assert trip_twice(KillSwitch(tmp_path / "absent" / "kill_state.json", owner=1)).latched


# --- smaller things --------------------------------------------------------------------------------


def test_a_second_connect_closes_the_old_client_first():
    clients = []

    def factory():
        clients.append(StubClient())
        return clients[-1]

    exchange = LiveExchange(factory)
    exchange.connect()
    exchange.connect()
    assert clients[0].disconnects == 1
    assert clients[1].connects == 1


def test_a_failing_old_client_does_not_stop_the_reconnect():
    class Dead(StubClient):
        def disconnect(self):
            raise RuntimeError("already gone")

    clients = [Dead(), StubClient()]
    exchange = LiveExchange(lambda: clients.pop(0))
    exchange.connect()
    exchange.connect()
    assert not clients


def test_the_read_cache_can_be_set_without_a_second_login():
    made = []

    def factory():
        made.append(StubClient())
        return made[-1]

    exchange = LiveExchange(factory)
    exchange.connect()
    exchange.set_read_ttl(0.5)
    assert len(made) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_value_that_is_not_finite_is_refused(value):
    with pytest.raises(ValueError):
        validate_params({"size": value})


def test_nan_in_the_params_file_keeps_the_last_good_set(tmp_path):
    path = tmp_path / "params.json"
    path.write_text('{"size": NaN}')
    store = ParamStore(path)
    assert store.refresh()["size"] == validate_params({})["size"]
    assert store.last_error


def test_an_unwanted_quote_still_counts_as_urgent():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.0, 101.0)
    us = SimExchange(market, "us")
    us.connect()
    manager = QuoteManager(us, WriteBudget(25, 0, clock=lambda: market.now), Limits(margin=0))
    manager.sync(QuoteIntent("A", 99.5, 5, 100.5, 5), {})
    assert manager.urgency(QuoteIntent("A", 99.5, 5, None, 0)) > 0


def test_one_failing_recorder_does_not_silence_the_others():
    seen = []

    class Bad:
        def tick(self, _state):
            raise RuntimeError("boom")

        def record(self, _kind, _payload):
            raise RuntimeError("boom")

        def close(self):
            raise RuntimeError("boom")

    class Good:
        def tick(self, state):
            seen.append(state)

        def record(self, kind, _payload):
            seen.append(kind)

        def close(self):
            seen.append("closed")

    tee = TeeRecorder([Bad(), Good()])
    tee.tick("state")
    tee.record("k", {})
    tee.close()
    assert seen == ["state", "k", "closed"]
    assert tee.errors == 3


def test_a_status_with_a_kill_section_is_valid_json(tmp_path):
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.0, 101.0)
    engine = make_engine(market, "us", [])
    engine.tick()
    path = tmp_path / "status.json"
    bot_main.write_status(path, engine, time.monotonic())
    assert json.loads(path.read_text())["kill"]["latched"] is False



# --- float noise must not hide a cross with our own order ------------------------------------------


@pytest.mark.parametrize("side,price", [("ask", 99.80000000000001), ("bid", 100.19999999999999)])
def test_float_noise_in_a_take_price_cannot_hide_a_cross_with_our_own_quote(side, price):
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.5, 100.5)
    us = SimExchange(market, "us")
    us.connect()
    manager = QuoteManager(us, WriteBudget(25, 0, clock=lambda: market.now), Limits(margin=0))
    manager.sync(QuoteIntent("A", 99.8, 5, 100.2, 5), {})
    manager.take(Take("A", side, price, 5, False, "x"), {"A": 0})
    opposite = "bid" if side == "ask" else "ask"
    assert us.is_connected(), "a self-trade disconnects"
    assert not [o for o in us.get_outstanding_orders("A").values() if o.side == opposite]


# --- a delayed cancel must still cancel ------------------------------------------------------------


def test_a_delayed_cancel_survives_a_cancel_all_on_another_instrument():
    events = []
    for iid in ("A", "B"):
        events.append({"t": 0.0, "k": "book", "iid": iid, "bids": [[99.0, 10]], "asks": [[101.0, 10]]})
    events.append({"t": 10.0, "k": "book", "iid": "A", "bids": [[99.0, 10]], "asks": [[101.0, 10]]})
    exchange = ReplayExchange(events, ["A", "B"], 0.1, 100, 200, latency_s=0.05)
    exchange.connect()
    exchange.advance_to(0.0)
    order_id = exchange.insert_order("A", price=99.5, volume=5, side="bid", order_type="limit").order_id
    exchange.advance_to(0.1)
    exchange.delete_orders("B")  # arrives first, and replaces the exchange's table of orders
    exchange.delete_order("A", order_id=order_id)
    exchange.advance_to(0.2)
    assert not exchange.get_outstanding_orders("A")
