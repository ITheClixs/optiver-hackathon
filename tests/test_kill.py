"""Kill switches: what trips, what it does, how trading comes back, and what cannot stop trading."""

import sys
from pathlib import Path

import pytest

from otk.engine import Engine, EngineConfig
from otk.instruments import Universe
from otk.kill import KillAction, KillSwitch
from otk.params import ParamStore, scaled_sizes, validate_params
from otk.sim.exchange import SimExchange, SimMarket
from otk.strategies import MarketMaker


def params(**values):
    return validate_params(values)


def check(switch, now=0.0, pnl=0.0, notional=0.0, rejects=0, data_age=0.0, **values):
    return switch.check(now, pnl, notional, rejects, data_age, params(**values))


# --- the switch alone ------------------------------------------------------------------------------


def test_nothing_trips_by_default():
    switch = KillSwitch()
    assert check(switch, now=1.0, pnl=-1e9, notional=1e9, rejects=10_000, data_age=1e6) is None


def test_switching_it_off_stops_every_trigger():
    switch = KillSwitch()
    assert check(switch, pnl=-100.0, kill_enabled=0, kill_max_loss=1) is None


def test_absolute_drawdown_flattens():
    switch = KillSwitch()
    assert check(switch, now=0.0, pnl=100.0, kill_drawdown=50) is None
    verdict = check(switch, now=1.0, pnl=40.0, kill_drawdown=50)
    assert verdict.action is KillAction.FLATTEN
    assert "drawdown" in verdict.reason


def test_drawdown_can_scale_with_what_the_book_could_lose():
    switch = KillSwitch()
    assert check(switch, now=0.0, pnl=0.0, notional=10_000.0, kill_drawdown_frac=0.25) is None
    assert check(switch, now=1.0, pnl=-2_000.0, notional=10_000.0, kill_drawdown_frac=0.25) is None
    assert check(switch, now=2.0, pnl=-2_600.0, notional=10_000.0, kill_drawdown_frac=0.25).action is KillAction.FLATTEN


def test_an_absolute_drawdown_outranks_the_scaled_one():
    switch = KillSwitch()
    check(switch, now=0.0, pnl=0.0)
    verdict = check(switch, now=1.0, pnl=-60.0, notional=10_000.0, kill_drawdown=50, kill_drawdown_frac=0.25)
    assert verdict.action is KillAction.FLATTEN


def test_max_loss_counts_from_the_session_start_not_the_peak():
    switch = KillSwitch()
    check(switch, now=0.0, pnl=10.0, kill_max_loss=100)
    assert check(switch, now=1.0, pnl=500.0, kill_max_loss=100) is None
    assert check(switch, now=2.0, pnl=-50.0, kill_max_loss=100) is None
    assert check(switch, now=3.0, pnl=-95.0, kill_max_loss=100).action is KillAction.FLATTEN


def test_loss_velocity_catches_a_fast_bleed():
    switch = KillSwitch()
    check(switch, now=0.0, pnl=0.0, kill_loss_per_min=100)
    assert check(switch, now=30.0, pnl=-40.0, kill_loss_per_min=100) is None
    assert check(switch, now=45.0, pnl=-120.0, kill_loss_per_min=100).action is KillAction.FLATTEN


def test_a_slow_bleed_is_not_a_velocity_kill():
    switch = KillSwitch()
    verdict = None
    for minute in range(10):
        verdict = check(switch, now=minute * 60.0, pnl=-50.0 * minute, kill_loss_per_min=100)
    assert verdict is None


def test_trading_comes_back_after_the_cool_off_at_reduced_size():
    switch = KillSwitch()
    check(switch, now=0.0, pnl=0.0, kill_drawdown=50, kill_cooloff_s=100, kill_resume_size=0.5)
    assert check(switch, now=1.0, pnl=-60.0, kill_drawdown=50, kill_cooloff_s=100, kill_resume_size=0.5)
    assert switch.size_factor(params(kill_resume_size=0.5)) == 0.5
    still = check(switch, now=50.0, pnl=-60.0, kill_drawdown=50, kill_cooloff_s=100)
    assert still.action is KillAction.FLATTEN, "keeps flattening during the cool-off"
    assert check(switch, now=102.0, pnl=-60.0, kill_drawdown=50, kill_cooloff_s=100) is None
    assert not switch.latched


def test_the_drawdown_is_measured_afresh_after_a_cool_off():
    switch = KillSwitch()
    check(switch, now=0.0, pnl=100.0, kill_drawdown=50, kill_cooloff_s=10)
    check(switch, now=1.0, pnl=40.0, kill_drawdown=50, kill_cooloff_s=10)
    assert check(switch, now=12.0, pnl=40.0, kill_drawdown=50, kill_cooloff_s=10) is None
    assert check(switch, now=13.0, pnl=20.0, kill_drawdown=50, kill_cooloff_s=10) is None


def test_repeated_trips_latch_until_restarted():
    switch = KillSwitch()
    settings = {"kill_drawdown": 50, "kill_cooloff_s": 10, "kill_max_trips": 2}
    check(switch, now=0.0, pnl=0.0, **settings)
    check(switch, now=1.0, pnl=-60.0, **settings)
    check(switch, now=12.0, pnl=-60.0, **settings)
    verdict = check(switch, now=13.0, pnl=-120.0, **settings)
    assert verdict.action is KillAction.FLATTEN
    assert switch.latched
    assert check(switch, now=10_000.0, pnl=1000.0, **settings).action is KillAction.FLATTEN


def test_a_reject_storm_pulls_without_flattening_and_recovers():
    switch = KillSwitch()
    assert check(switch, now=0.0, rejects=0, kill_reject_limit=5, kill_pause_s=3) is None
    verdict = check(switch, now=1.0, rejects=6, kill_reject_limit=5, kill_pause_s=3)
    assert verdict.action is KillAction.PULL
    assert "reject" in verdict.reason
    assert check(switch, now=2.0, rejects=6, kill_reject_limit=5, kill_pause_s=3).action is KillAction.PULL
    assert check(switch, now=20.0, rejects=6, kill_reject_limit=5, kill_pause_s=3) is None


def test_old_rejects_do_not_count():
    switch = KillSwitch()
    check(switch, now=0.0, rejects=0, kill_reject_limit=5)
    check(switch, now=5.0, rejects=4, kill_reject_limit=5)
    assert check(switch, now=30.0, rejects=8, kill_reject_limit=5) is None


def test_stale_market_data_pulls_quotes_until_it_flows_again():
    switch = KillSwitch()
    assert check(switch, data_age=2.0, kill_stale_s=5) is None
    verdict = check(switch, data_age=6.0, kill_stale_s=5)
    assert verdict.action is KillAction.PULL
    assert "stale" in verdict.reason
    assert check(switch, data_age=0.1, kill_stale_s=5) is None


def test_sizes_scale_globally_and_per_instrument():
    current = validate_params({"size": 20, "instruments": {"A": {"size": 40}}})
    scaled = scaled_sizes(current, 0.5)
    assert scaled["size"] == 10
    assert scaled["instruments"]["A"]["size"] == 20
    assert current["size"] == 20, "the original must not change"
    assert scaled_sizes(current, 1.0) is current


def test_every_switch_passes_its_drill():
    import importlib.util

    spec = importlib.util.spec_from_file_location("kill_drill", Path(__file__).parent.parent / "scripts" / "kill_drill.py")
    drill = importlib.util.module_from_spec(spec)
    sys.modules["kill_drill"] = drill
    spec.loader.exec_module(drill)
    assert drill.run_all() == dict.fromkeys(drill.DRILLS, True)


# --- wired into the engine --------------------------------------------------------------------------


def rest(market, iid, side, price, volume, team="wall"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


class Falling:
    def __init__(self, per_step=0.2, stop_after=None):
        self.level = 100.0
        self.per_step = per_step
        self.stop_after = stop_after

    def act(self, market, now, dt):
        market.wipe_orders("wall", "A")
        if self.stop_after is None or now < self.stop_after:
            self.level -= self.per_step
        rest(market, "A", "bid", round(self.level - 0.1, 1), 500)
        rest(market, "A", "ask", round(self.level + 0.1, 1), 500)
        SimExchange(market, "noise", bot=True).insert_order(
            "A", price=round(self.level - 0.1, 1), volume=1, side="ask", order_type="ioc"
        )


def engine_for(market, strategies=(), config=None, **values):
    exchange = SimExchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    store = ParamStore(base=values)
    engine = Engine(
        exchange, universe, list(strategies), config or EngineConfig(), store, clock=lambda: market.now
    )
    return engine, exchange


def run(market, engine, steps, dt=0.5):
    for _ in range(steps):
        market.advance(dt)
        engine.tick()


def test_a_loss_kill_flattens_then_trading_resumes_after_the_cool_off():
    market = SimMarket(instruments=["A"])
    market.actors.append(Falling(stop_after=5.0))
    market.advance(0.5)
    engine, exchange = engine_for(
        market, [MarketMaker(["A"])], kill_max_loss=5, kill_cooloff_s=10, kill_resume_size=0.5, size=10,
    )
    exchange.insert_order("A", price=101.0, volume=30, side="bid", order_type="ioc")
    run(market, engine, 8)
    assert engine.halted
    assert "loss" in engine.kill.last_reason
    assert exchange.get_positions()["A"] == 0
    assert not exchange.get_outstanding_orders("A")
    run(market, engine, 30)
    assert not engine.halted
    sizes = [o.volume for o in exchange.get_outstanding_orders("A").values()]
    assert sizes and max(sizes) <= 5, "back at half size"


def test_the_kill_is_recorded_for_the_post_mortem():
    records = []

    class Recorder:
        def record(self, kind, payload):
            records.append((kind, dict(payload)))

        def tick(self, _state):
            return None

    market = SimMarket(instruments=["A"])
    market.actors.append(Falling())
    market.advance(0.5)
    exchange = SimExchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    engine = Engine(
        exchange, universe, [], EngineConfig(), ParamStore(base={"kill_max_loss": 5}), Recorder(),
        clock=lambda: market.now,
    )
    exchange.insert_order("A", price=101.0, volume=30, side="bid", order_type="ioc")
    run(market, engine, 8)
    halts = [payload for kind, payload in records if kind == "halt"]
    assert halts and "loss" in halts[0]["reason"]


def test_a_flatten_file_flattens_and_a_kill_file_only_pulls(tmp_path):
    def run_with(name):
        kill = tmp_path / "KILL"
        flatten = tmp_path / "FLATTEN"
        for path in (kill, flatten):
            path.unlink(missing_ok=True)
        (tmp_path / name).write_text("stop")
        market = SimMarket(instruments=["A"])
        market.actors.append(Falling(per_step=0.0))
        market.advance(0.5)
        config = EngineConfig(kill_file=str(kill), flatten_file=str(flatten))
        engine, exchange = engine_for(market, config=config)
        exchange.insert_order("A", price=101.0, volume=30, side="bid", order_type="ioc")
        run(market, engine, 4)
        assert engine.halted
        return exchange.get_positions()["A"]

    assert run_with("KILL") == 30
    assert run_with("FLATTEN") == 0


def test_stale_data_pulls_quotes_live_and_they_return():
    market = SimMarket(instruments=["A"])
    actor = Falling(per_step=0.0)
    market.actors.append(actor)
    market.advance(0.5)
    engine, exchange = engine_for(market, [MarketMaker(["A"])], kill_stale_s=5)
    actor.per_step = 0.1
    run(market, engine, 6)
    assert exchange.get_outstanding_orders("A")
    market.actors.remove(actor)
    run(market, engine, 20)
    assert engine.halted
    assert not exchange.get_outstanding_orders("A")
    market.actors.append(actor)
    run(market, engine, 4)
    assert not engine.halted
    assert exchange.get_outstanding_orders("A")


def test_a_client_without_pnl_cannot_break_the_kill_check():
    market = SimMarket(instruments=["A"])
    market.actors.append(Falling(per_step=0.0))
    market.advance(0.5)
    engine, exchange = engine_for(market, kill_max_loss=5)

    def broken(_valuations=None):
        raise RuntimeError("client bug")

    exchange.get_pnl = broken
    run(market, engine, 2)
    assert not engine.halted
    assert engine.telemetry_errors >= 1


@pytest.mark.parametrize("key", ["kill_drawdown", "kill_drawdown_frac", "kill_max_loss", "kill_loss_per_min", "kill_stale_s", "kill_reject_limit"])
def test_kill_limits_are_live_parameters_and_off_until_set(key):
    assert validate_params({})[key] == 0


def reset_params(**values):
    return params(kill_enabled=1, kill_drawdown=0, kill_drawdown_frac=0.25, kill_max_loss=0, kill_loss_per_min=0, **values)


def test_a_reset_to_zero_pnl_is_not_a_drawdown():
    """+500 at 20:59, 0 at 21:00 and a small first position: 500 down against a limit of 250 would close us out."""
    kill = KillSwitch()
    assert kill.check(100.0, 500.0, 40_000.0, 0, 0.0, reset_params()) is None
    kill.on_reset()
    assert kill.check(200.0, 0.0, 1_000.0, 0, 0.0, reset_params()) is None


def test_without_the_reset_the_same_numbers_do_trip():
    kill = KillSwitch()
    kill.check(100.0, 500.0, 40_000.0, 0, 0.0, reset_params())
    assert kill.check(200.0, 0.0, 1_000.0, 0, 0.0, reset_params()) is not None


def test_a_real_loss_after_a_reset_still_trips():
    kill = KillSwitch()
    kill.check(100.0, 500.0, 40_000.0, 0, 0.0, reset_params())
    kill.on_reset()
    assert kill.check(200.0, 0.0, 1_000.0, 0, 0.0, reset_params()) is None
    assert kill.check(300.0, -900.0, 1_000.0, 0, 0.0, reset_params()) is not None


def test_a_latch_survives_a_reset():
    kill = KillSwitch()
    kill.latched = True
    kill.on_reset()
    assert kill.latched
