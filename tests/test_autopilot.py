"""The bot adapting itself: on evidence, within bounds, reversibly, and outranked by a person."""

import json
import math
import os
import random
import time

import pytest

from otk.autopilot import Autopilot, AutopilotConfig, ParticipantIntel, walk_forward
from otk.engine import MarketState
from otk.instruments import Universe, parse_instrument_id
from otk.params import ParamStore, params_for, validate_params, write_params_atomic
from otk.reversion import ArmConfig, ReversionArm, make_arms, scan_events
from otk.sim import scenarios
from otk.sim.exchange import PriceBook, PriceVolume, SimExchange, SimMarket
from otk.sim.harness import make_engine, run_session
from otk.strategies import MeanReversionTaker, default_stack


def rest(market, iid, side, price, volume, team="wall"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


class Wave:
    """A wall anchored on a slow sine, then optionally a persistent trend: reversion, then no reversion."""

    def __init__(self, amplitude=2.0, period_s=60.0, half_spread=0.1, trend_after=None, trend_per_s=0.1):
        self.amplitude = amplitude
        self.period_s = period_s
        self.half_spread = half_spread
        self.trend_after = trend_after
        self.trend_per_s = trend_per_s

    def act(self, market, now, dt):
        market.wipe_orders("wall", "A")
        center = 100.0 + self.amplitude * math.sin(2 * math.pi * now / self.period_s)
        if self.trend_after is not None and now > self.trend_after:
            center = 100.0 - self.trend_per_s * (now - self.trend_after)
        rest(market, "A", "bid", round(center - self.half_spread, 1), 500)
        rest(market, "A", "ask", round(center + self.half_spread, 1), 500)


def wave_market(**kwargs):
    market = SimMarket(instruments=["A"])
    market.actors.append(Wave(**kwargs))
    return market


def adaptive_engine(market, config=None, stack=None, path=None, **kwargs):
    probe = make_engine(market, "probe", [])
    store = ParamStore(path)
    pilot = Autopilot(store, probe.universe, config, **kwargs)
    engine = make_engine(market, "us", stack if stack is not None else [MeanReversionTaker(market.instruments)], params=store, recorder=pilot)
    return engine, pilot, store


def run(market, engine, seconds, dt=0.5):
    for _ in range(int(seconds / dt)):
        market.advance(dt)
        engine.tick()


# --- paper trading ----------------------------------------------------------------------------------


def sine_points(seconds, half_spread, amplitude=2.0, period=60.0):
    for second in range(seconds):
        mid = 100.0 + amplitude * math.sin(2 * math.pi * second / period)
        yield float(second), mid, mid + half_spread, mid - half_spread


def test_paper_arm_earns_on_oscillation_after_paying_the_spread():
    arm = ReversionArm(120.0, 1.0, ArmConfig(), tick=0.1)
    for point in sine_points(1200, half_spread=0.1):
        arm.update(*point)
    assert arm.total_trades >= 10
    assert arm.mean > 0.0


def test_paper_arm_pays_the_spread_and_a_haircut():
    """Same path, wider spread: per-trip profit must fall by the extra spread paid."""
    narrow = ReversionArm(120.0, 1.0, ArmConfig(min_edge_ticks=0), tick=0.1)
    wide = ReversionArm(120.0, 1.0, ArmConfig(min_edge_ticks=0), tick=0.1)
    for point in sine_points(1200, half_spread=0.1):
        narrow.update(*point)
    for point in sine_points(1200, half_spread=0.5):
        wide.update(*point)
    assert wide.mean == pytest.approx(narrow.mean - 0.8, abs=0.15)


def test_paper_arm_finds_no_edge_on_a_random_walk():
    for seed in range(5):
        rng = random.Random(seed)
        mid = 100.0
        arms = make_arms(0.1)
        for second in range(3000):
            mid += rng.gauss(0.0, 0.05)
            for arm in arms:
                arm.update(float(second), mid, mid + 0.1, mid - 0.1)
        for arm in arms:
            assert arm.mean <= 0.0 or arm.t < 2.5


def test_paper_arm_stops_out_of_a_running_move_and_pauses():
    arm = ReversionArm(300.0, 1.0, ArmConfig(stop_z=4.0, min_edge_ticks=0, cooldown_s=10_000), tick=0.1)
    rng = random.Random(0)
    for second in range(300):
        mid = 100.0 + rng.gauss(0.0, 0.1)
        arm.update(float(second), mid, mid + 0.1, mid - 0.1)
    for second in range(300, 500):
        mid = 100.0 - 0.1 * (second - 300)
        arm.update(float(second), mid, mid + 0.1, mid - 0.1)
    assert arm.stops >= 1
    assert arm.results[-1] < 0.0
    assert arm.position == 0


def test_walk_forward_counts_only_trips_made_after_the_rule_switched_on():
    points = [point for point in sine_points(1800, half_spread=0.1)]
    result = walk_forward(points, tick=0.1)
    assert result["promotions"] >= 1
    assert 0 < result["round_trips"] < 60, "the evidence-gathering trips must not count"
    assert result["total_ticks"] > 0


def test_walk_forward_stays_out_of_a_random_walk():
    rng = random.Random(3)
    mid = 100.0
    points = []
    for second in range(3000):
        mid += rng.gauss(0.0, 0.05)
        points.append((float(second), mid, mid + 0.1, mid - 0.1))
    result = walk_forward(points, tick=0.1)
    assert result["round_trips"] == 0 or result["total_ticks"] < 50


def test_scan_reports_books_and_gaps_from_a_capture():
    events = []
    for second, mid, ask, bid in sine_points(1200, half_spread=0.1):
        events.append({"t": second, "k": "book", "iid": "A", "bids": [[round(bid, 1), 5]], "asks": [[round(ask, 1), 5]]})
        events.append({"t": second, "k": "book", "iid": "A_DUAL", "bids": [[99.9, 5]], "asks": [[100.1, 5]]})
    report = scan_events(events)
    assert report["A"]["best"]["round_trips"] > 0
    assert "A_DUAL-gap" in report
    json.dumps(report)


# --- enabling and disabling on evidence -------------------------------------------------------------


def test_reversion_switches_itself_on_where_it_pays_and_earns():
    market = wave_market()
    engine, pilot, store = adaptive_engine(market)
    run(market, engine, 1200)
    assert params_for(store.current, "A")["mr_enabled"] == 1
    assert engine.exchange.get_pnl({"A": 100.0}) > 0
    assert any("paper round trips" in decision["why"] for decision in pilot.decisions)


def test_reversion_stays_off_on_a_random_walk():
    for seed in range(3):
        market = scenarios.quiet_market(seed, sigma=0.05)
        engine, pilot, store = adaptive_engine(market)
        run(market, engine, 900)
        assert not store.current.get("instruments", {}), pilot.snapshot()["best_paper"]


def test_reversion_switches_itself_off_when_the_regime_ends():
    market = wave_market(trend_after=900.0)
    engine, pilot, store = adaptive_engine(market)
    run(market, engine, 900)
    assert params_for(store.current, "A")["mr_enabled"] == 1
    run(market, engine, 300)
    assert params_for(store.current, "A")["mr_enabled"] == 0
    assert any("mr off" in decision["why"] for decision in pilot.decisions)


class Staircase:
    """Prices wobble, then step up and stay up: the shape that sank a fading maker at a live event.

    A public post-mortem (Delta Dash, Oct 2026) reports a mean-reverting maker near the bottom of 79
    players because value trended on events and it kept shorting into the rise.
    """

    def __init__(self, step=1.5, every_s=45.0, wobble=0.2):
        self.step = step
        self.every_s = every_s
        self.wobble = wobble

    def act(self, market, now, dt):
        market.wipe_orders("wall", "A")
        center = 100.0 + self.step * int(now / self.every_s)
        center += self.wobble * math.sin(2 * math.pi * now / 20.0)
        rest(market, "A", "bid", round(center - 0.1, 1), 500)
        rest(market, "A", "ask", round(center + 0.1, 1), 500)


def test_a_trending_market_does_not_turn_on_fading_or_loses_little_if_it_does():
    market = SimMarket(instruments=["A"])
    market.actors.append(Staircase())
    engine, pilot, store = adaptive_engine(market)
    run(market, engine, 1500)
    level = 100.0 + 1.5 * int(market.now / 45.0)
    assert params_for(store.current, "A")["mr_enabled"] == 0, pilot.snapshot()["best_paper"]
    assert engine.exchange.get_pnl({"A": level}) > -100.0


def test_only_one_directional_target_per_underlying():
    market = SimMarket(instruments=["A", "A_DUAL"])
    market.actors.append(Wave())

    class DualWave(Wave):
        def act(self, market, now, dt):
            market.wipe_orders("wall2", "A_DUAL")
            center = 100.0 + self.amplitude * math.sin(2 * math.pi * now / self.period_s)
            rest(market, "A_DUAL", "bid", round(center - 0.1, 1), 500, team="wall2")
            rest(market, "A_DUAL", "ask", round(center + 0.1, 1), 500, team="wall2")

    market.actors.append(DualWave())
    engine, pilot, _store = adaptive_engine(market)
    run(market, engine, 1200)
    active = [key for key, t in pilot._targets.items() if t.active is not None]
    assert len(active) == 1


# --- quoting on labels and fills --------------------------------------------------------------------


class StubMonitor:
    def __init__(self):
        self.current = []

    def labels(self, _iid):
        return list(self.current)


def universe():
    return Universe([parse_instrument_id("A")._replace(tick=0.1)])


def state_at(store, now, fills=None, bid=99.9, ask=100.1):
    book = PriceBook(None, "A", [PriceVolume(bid, 10)], [PriceVolume(ask, 10)])
    return MarketState(now, universe(), {"A": 0}, {"A": book}, {"A": []}, store.refresh(), {"A": []}, {"A": fills or []})


def step_to(pilot, store, seconds, start=0.0, fills=None, **prices):
    now = start
    pilot.tick(state_at(store, now, fills, **prices))
    while now < start + seconds:
        now += 1.0
        pilot.tick(state_at(store, now, None, **prices))
    return now


def quick():
    return AutopilotConfig(interval_s=5.0, hold_s=0.0, min_fills=5)


def test_flow_labels_scale_quotes_and_lapse_when_the_label_does():
    store = ParamStore()
    monitor = StubMonitor()
    pilot = Autopilot(store, universe(), quick(), monitor=monitor)
    monitor.current = ["informed_flow"]
    now = step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["base_spread_ticks"] == pytest.approx(3.0)
    assert params_for(store.current, "A")["size"] == pytest.approx(7.0)
    monitor.current = []
    step_to(pilot, store, 6.0, start=now)
    assert params_for(store.current, "A")["size"] == 10
    assert not pilot.overlay


def test_label_factors_do_not_ratchet_while_a_label_persists():
    store = ParamStore()
    monitor = StubMonitor()
    pilot = Autopilot(store, universe(), quick(), monitor=monitor)
    monitor.current = ["informed_flow"]
    step_to(pilot, store, 60.0)
    assert params_for(store.current, "A")["size"] == pytest.approx(7.0)


class Fill:
    def __init__(self, side, price, volume=1):
        self.side = side
        self.price = price
        self.volume = volume


def test_being_picked_off_widens_quotes():
    store = ParamStore()
    pilot = Autopilot(store, universe(), quick())
    now = 0.0
    for _ in range(12):
        # Each buy is followed by the mid sitting well below the price paid.
        pilot.tick(state_at(store, now, [Fill("bid", 101.0 + (now % 3) * 0.01)]))
        now += 1.0
    now = step_to(pilot, store, 12.0, start=now)
    effective = params_for(store.current, "A")
    assert effective["base_spread_ticks"] > 2.0
    assert effective["size"] < 10


def test_changes_hold_for_a_minimum_time_unless_they_reduce_risk():
    store = ParamStore()
    monitor = StubMonitor()
    config = AutopilotConfig(interval_s=5.0, hold_s=1000.0)
    pilot = Autopilot(store, universe(), config, monitor=monitor)
    monitor.current = ["uninformed_flow"]
    now = step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["size"] == pytest.approx(12.0)
    monitor.current = ["overpaying_flow", "uninformed_flow"]
    now = step_to(pilot, store, 6.0, start=now)
    assert params_for(store.current, "A")["size"] == pytest.approx(12.0), "held: this adds risk"
    monitor.current = ["informed_flow"]
    step_to(pilot, store, 6.0, start=now)
    assert params_for(store.current, "A")["size"] == pytest.approx(7.0), "reduces risk: not held"


def test_factors_are_bounded():
    store = ParamStore()
    monitor = StubMonitor()
    pilot = Autopilot(store, universe(), quick(), monitor=monitor)
    monitor.current = ["penny_war"] * 8
    step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["size"] == pytest.approx(2.5)
    assert validate_params(store.current)


def test_session_drawdown_cuts_size_and_directional_risk():
    store = ParamStore(base={"ap_drawdown": 5})
    pilot = Autopilot(store, universe(), quick())
    pilot.tick(state_at(store, 0.0, [Fill("bid", 100.0, 10)]))
    now = step_to(pilot, store, 6.0, start=0.0)
    now = step_to(pilot, store, 12.0, start=now, bid=98.9, ask=99.1)
    assert params_for(store.current, "A")["size"] == pytest.approx(5.0)
    assert any("risk off" in decision["why"] for decision in pilot.decisions)


# --- people and agents outrank it -------------------------------------------------------------------


def test_a_key_in_the_params_file_pins_it_against_autopilot(tmp_path):
    path = tmp_path / "params.json"
    write_params_atomic(path, {"mr_enabled": 0})
    market = wave_market()
    engine, pilot, store = adaptive_engine(market, path=path)
    run(market, engine, 1200)
    assert any(t.active is not None for t in pilot._targets.values()), "it wanted to"
    assert params_for(store.current, "A")["mr_enabled"] == 0
    assert market.stats["us"].writes == 0


def test_an_instrument_key_in_the_params_file_pins_it_too(tmp_path):
    path = tmp_path / "params.json"
    write_params_atomic(path, {"instruments": {"A": {"mr_enabled": 0}}})
    store = ParamStore(path)
    store.refresh()
    store.set_overlay({"instruments": {"A": {"mr_enabled": 1, "size": 7}}})
    assert params_for(store.current, "A")["mr_enabled"] == 0
    assert params_for(store.current, "A")["size"] == 7


def test_switching_it_off_returns_to_baseline(tmp_path):
    path = tmp_path / "params.json"
    store = ParamStore(path)
    monitor = StubMonitor()
    pilot = Autopilot(store, universe(), quick(), monitor=monitor)
    monitor.current = ["informed_flow"]
    now = step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["size"] == pytest.approx(7.0)
    write_params_atomic(path, {"ap_enabled": 0})
    step_to(pilot, store, 6.0, start=now)
    assert params_for(store.current, "A")["size"] == 10
    assert not pilot.overlay


def test_switching_quoting_off_leaves_quotes_alone():
    store = ParamStore(base={"ap_quoting": 0})
    monitor = StubMonitor()
    pilot = Autopilot(store, universe(), quick(), monitor=monitor)
    monitor.current = ["informed_flow"]
    step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["size"] == 10


def test_overlay_cannot_exceed_hard_bounds_or_invent_keys():
    store = ParamStore()
    store.set_overlay({"instruments": {"A": {"size": 100000}}})
    assert params_for(store.current, "A")["size"] == 100
    with pytest.raises(ValueError):
        store.set_overlay({"instruments": {"A": {"nonsense": 1}}})


def test_base_values_survive_an_overlay_and_a_bad_file(tmp_path):
    path = tmp_path / "params.json"
    store = ParamStore(path, base={"instruments": {"A": {"jump_ticks": 50.0}}})
    store.set_overlay({"instruments": {"A": {"size": 7}}})
    path.write_text("not json")
    effective = params_for(store.refresh(), "A")
    assert effective["jump_ticks"] == 50.0
    assert effective["size"] == 7
    assert store.last_error


# --- names from the dashboard -----------------------------------------------------------------------


def write_intel(path, **targets):
    path.write_text(json.dumps({"targets": {"overpayers": [], "informed": [], **targets}}))


def test_an_informed_size_signature_is_followed(tmp_path):
    path = tmp_path / "participants.json"
    write_intel(path, informed=[{"name": "x", "instrument": "A", "modal_size": 500, "modal_share": 0.9}])
    store = ParamStore()
    pilot = Autopilot(store, universe(), quick(), intel=ParticipantIntel(path))
    step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["follow_min_size"] == 500


def test_a_size_shared_with_normal_flow_is_not_a_signature(tmp_path):
    path = tmp_path / "participants.json"
    write_intel(path, informed=[
        {"name": "x", "instrument": "A", "modal_size": 500, "modal_share": 0.2},
        {"name": "y", "instrument": "A", "modal_size": 3, "modal_share": 0.9},
    ])
    store = ParamStore()
    pilot = Autopilot(store, universe(), quick(), intel=ParticipantIntel(path))
    step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["follow_min_size"] == 0


def test_overpayers_get_more_size(tmp_path):
    path = tmp_path / "participants.json"
    write_intel(path, overpayers=[{"name": "x", "kind": "bot", "instrument": "A", "modal_size": 25}])
    store = ParamStore()
    pilot = Autopilot(store, universe(), quick(), intel=ParticipantIntel(path))
    step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["size"] == pytest.approx(12.5)


def test_stale_intel_is_ignored(tmp_path):
    path = tmp_path / "participants.json"
    write_intel(path, overpayers=[{"name": "x", "instrument": "A", "modal_size": 25}])
    old = time.time() - 3600
    os.utime(path, (old, old))
    store = ParamStore()
    pilot = Autopilot(store, universe(), quick(), intel=ParticipantIntel(path))
    step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["size"] == 10


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"targets": 5}', '{"targets": {"informed": []}}'])
def test_broken_intel_never_stops_it(tmp_path, content):
    path = tmp_path / "participants.json"
    path.write_text(content)
    store = ParamStore()
    pilot = Autopilot(store, universe(), quick(), intel=ParticipantIntel(path))
    step_to(pilot, store, 6.0)
    assert params_for(store.current, "A")["size"] == 10


def test_missing_intel_is_normal(tmp_path):
    intel = ParticipantIntel(tmp_path / "absent.json")
    assert intel.load() == {}
    assert intel.errors == 0


# --- integration ------------------------------------------------------------------------------------


def test_decisions_reach_the_sink_and_the_snapshot_serialises():
    entries = []
    market = wave_market()
    engine, pilot, _store = adaptive_engine(market, sink=lambda kind, entry: entries.append((kind, entry)))
    run(market, engine, 1200)
    assert entries and entries[0][0] == "decision"
    json.dumps(pilot.snapshot())
    assert "directional" in pilot.snapshot()


def test_an_error_inside_autopilot_cannot_stop_trading():
    class Exploding(Autopilot):
        def tick(self, state):
            raise RuntimeError("boom")

    market = wave_market()
    probe = make_engine(market, "probe", [])
    store = ParamStore()
    engine = make_engine(market, "us", [], params=store, recorder=Exploding(store, probe.universe))
    run(market, engine, 5)
    assert engine.telemetry_errors > 0
    assert engine.errors == 0


@pytest.mark.parametrize(
    "scenario",
    [
        lambda seed: scenarios.passive_field(seed),
        lambda seed: scenarios.adverse_field(scenarios.dual_listing(seed), seed),
        lambda seed: scenarios.index_complex(seed, stock_tick=0.01),
    ],
    ids=["passive", "dual_adverse", "index_fine_ticks"],
)
def test_full_stack_with_autopilot_stays_clean_under_lag(scenario):
    market = scenario(1)
    probe = make_engine(market, "probe", [])
    store = ParamStore()
    pilot = Autopilot(store, probe.universe, AutopilotConfig(interval_s=10.0, hold_s=10.0))
    stack = default_stack(market.instruments, index_weights=market.index_weights or None)
    engine = make_engine(market, "us", stack, params=store, recorder=pilot, lag_s=0.2)
    result = run_session(market, {"us": engine}, 300.0, seed=1)["us"]
    assert (result.disconnects, result.limit_breaches, result.rate_breaches, result.errors) == (0, 0, 0, 0)
    assert engine.telemetry_errors == 0
    assert market.stats["us"].writes > 0
