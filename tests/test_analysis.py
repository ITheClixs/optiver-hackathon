"""Order-book behaviour analysis, opponent fingerprints, replay backtest and evaluation gates."""

import json
import random

import numpy as np
import pytest

from otk.allocator import DiscountedThompson, PageHinkley, SwitchbackSchedule
from otk.behaviour import analyse_capture, autocorr, variance_ratio
from otk.evaluation import block_bootstrap_ci, compare_variants, decide, pnl_stats
from otk.opponents import FlowProfiler, classify_flow, recommend
from otk.overfit import cscv_pbo, deflated_sharpe, min_backtest_length, probabilistic_sharpe
from otk.recorder import JsonlRecorder, read_capture
from otk.replay import ReplayExchange, load_events, run_replay
from otk.sim.harness import make_engine
from otk.sim.scenarios import dual_listing, quiet_market
from otk.strategies import MarketMaker


def capture_of(market, path, ids, seconds=60, dt=0.1):
    recorder = JsonlRecorder(path, book_every_s=0.0, clock=lambda: market.now)
    engine = make_engine(market, "observer", [], recorder=recorder)
    for _ in range(int(seconds / dt)):
        market.advance(dt)
        engine.tick()
    recorder.close()
    return path


def test_recorder_round_trips_and_survives_a_torn_line(tmp_path):
    path = tmp_path / "cap.jsonl"
    capture_of(quiet_market(1, ("A",)), path, ("A",), seconds=5)
    with open(path, "a") as handle:
        handle.write('{"t": 1, "k": "bo')
    events = list(read_capture(path))
    assert any(e["k"] == "book" for e in events)
    assert any(e["k"] == "trade" for e in events)


def test_recorder_gzip_round_trip(tmp_path):
    path = tmp_path / "cap.jsonl.gz"
    capture_of(quiet_market(1, ("A",)), path, ("A",), seconds=3)
    assert list(read_capture(path))


def test_read_capture_survives_a_truncated_gzip(tmp_path):
    path = tmp_path / "cap.jsonl.gz"
    capture_of(quiet_market(1, ("A",)), path, ("A",), seconds=10)
    whole = list(read_capture(path))
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    partial = list(read_capture(cut))
    assert len(partial) < len(whole)


def test_statistics_on_known_series():
    rng = random.Random(0)
    white = [rng.gauss(0, 1) for _ in range(5000)]
    assert abs(autocorr(white, 1)) < 0.05
    walk = np.cumsum(white).tolist()
    assert variance_ratio(walk, 10) == pytest.approx(1.0, abs=0.25)
    mean_reverting = white
    assert variance_ratio(mean_reverting, 10) < 0.3
    assert autocorr([1.0, 1.0, 1.0], 1) == 0.0


def test_capture_report_finds_dual_lead_lag_and_spreads(tmp_path):
    path = capture_of(dual_listing(3), tmp_path / "cap.jsonl", (), seconds=120)
    report = analyse_capture(load_events(path), tick=0.1)
    per = report["instruments"]
    assert per["ASML"]["spread_ticks_median"] < per["ASML_DUAL"]["spread_ticks_median"]
    assert per["ASML"]["trades"] > 0
    assert per["ASML"]["top_size_modes"]
    pairs = {tuple(item["pair"]): item for item in report["lead_lag"]}
    assert ("ASML", "ASML_DUAL") in pairs
    assert pairs[("ASML", "ASML_DUAL")]["leader"] == "ASML"
    assert report["duration_s"] > 100


def test_capture_report_handles_an_empty_capture():
    report = analyse_capture([], tick=0.1)
    assert report["instruments"] == {}
    assert report["duration_s"] == 0.0


def test_flow_profiler_detects_periodic_fixed_size_bot():
    profiler = FlowProfiler()
    rng = random.Random(1)
    mid = 100.0
    for step in range(300):
        now = step * 2.0
        profiler.on_mid("A", now, mid)
        profiler.on_trade("A", now, mid, 25, rng.choice(("bid", "ask")))
    prints = profiler.fingerprints("A")
    assert prints["modal_size"] == 25
    assert prints["periodic"] is True
    assert abs(prints["aggressor_imbalance"]) < 0.3


def test_flow_profiler_scores_informed_flow_by_size_bucket():
    profiler = FlowProfiler(horizons_s=(1.0,))
    rng = random.Random(2)
    mid = 100.0
    now = 0.0
    for _ in range(400):
        now += 2.0
        if rng.random() < 0.5:
            profiler.on_mid("A", now, mid)
            side = rng.choice(("bid", "ask"))
            profiler.on_trade("A", now, mid, 50, side)
            mid += 0.5 if side == "bid" else -0.5
        else:
            profiler.on_mid("A", now, mid)
            side = rng.choice(("bid", "ask"))
            profiler.on_trade("A", now, mid, 5, side)
        profiler.on_mid("A", now + 1.1, mid)
    markouts = profiler.markouts("A")
    assert markouts[50][1.0]["mean"] > 0.3
    assert abs(markouts[5][1.0]["mean"]) < 0.1


def test_classify_flow_labels_archetypes_conservatively():
    assert classify_flow({"n": 3}, {}) == ["insufficient_data"]
    noise = {"n": 200, "periodic": False, "aggressor_imbalance": 0.0, "momentum_corr": 0.0}
    markouts = {10: {5.0: {"mean": 0.0, "t": 0.1, "n": 200}}}
    assert "uninformed_flow" in classify_flow(noise, markouts, tick=0.1)
    informed = {10: {5.0: {"mean": 0.5, "t": 6.0, "n": 200}}}
    assert "informed_flow" in classify_flow(noise, informed, tick=0.1)
    chaser = {**noise, "momentum_corr": 0.4}
    assert "momentum_chaser" in classify_flow(chaser, markouts, tick=0.1)


def test_recommend_only_returns_bounded_known_parameters():
    for labels in (["informed_flow"], ["uninformed_flow"], ["penny_war"], ["momentum_chaser"], []):
        for change in recommend(labels):
            assert change.parameter in ("base_spread_ticks", "size", "skew_ticks", "jump_ticks", "vol_factor")
            assert change.factor > 0


def test_replay_exchange_fills_only_when_trades_print_through_our_price():
    events = [
        {"t": 0.0, "k": "book", "iid": "A", "bids": [[99.0, 100]], "asks": [[101.0, 100]]},
        {"t": 1.0, "k": "trade", "iid": "A", "px": 99.0, "vol": 50, "aggr": "ask"},
        {"t": 2.0, "k": "trade", "iid": "A", "px": 98.9, "vol": 50, "aggr": "ask"},
    ]
    exchange = ReplayExchange(events, ["A"])
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("A", price=99.0, volume=10, side="bid", order_type="limit")
    exchange.advance_to(1.0)
    assert exchange.get_positions()["A"] == 0  # behind 100 lots of queue; the print ate only 50
    exchange.advance_to(2.0)
    assert exchange.get_positions()["A"] == 10  # printed strictly through our price


def test_replay_queue_ahead_is_consumed_by_prints_at_our_price():
    events = [
        {"t": 0.0, "k": "book", "iid": "A", "bids": [[99.0, 10]], "asks": [[101.0, 100]]},
        {"t": 1.0, "k": "trade", "iid": "A", "px": 99.0, "vol": 15, "aggr": "ask"},
    ]
    exchange = ReplayExchange(events, ["A"])
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("A", price=99.0, volume=10, side="bid", order_type="limit")
    exchange.advance_to(1.0)
    assert exchange.get_positions()["A"] == 5


def test_replay_improved_quote_fills_on_any_opposing_print_it_would_catch():
    events = [
        {"t": 0.0, "k": "book", "iid": "A", "bids": [[99.0, 100]], "asks": [[101.0, 100]]},
        {"t": 1.0, "k": "trade", "iid": "A", "px": 99.0, "vol": 8, "aggr": "ask"},
    ]
    exchange = ReplayExchange(events, ["A"])
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("A", price=99.5, volume=5, side="bid", order_type="limit")
    exchange.advance_to(1.0)
    assert exchange.get_positions()["A"] == 5


def test_replay_ioc_is_capped_by_visible_size_and_touch():
    events = [{"t": 0.0, "k": "book", "iid": "A", "bids": [[99.0, 100]], "asks": [[101.0, 7]]}]
    exchange = ReplayExchange(events, ["A"])
    exchange.connect()
    exchange.advance_to(0.0)
    exchange.insert_order("A", price=101.0, volume=20, side="bid", order_type="ioc")
    assert exchange.get_positions()["A"] == 7
    exchange.insert_order("A", price=100.0, volume=20, side="bid", order_type="ioc")
    assert exchange.get_positions()["A"] == 7


def test_replay_runs_strategies_through_the_real_engine(tmp_path):
    path = capture_of(quiet_market(4, ("A",)), tmp_path / "cap.jsonl", (), seconds=60)
    result = run_replay(load_events(path), [lambda ids: MarketMaker(ids)], ["A"])
    assert result.errors == 0
    assert result.writes > 0
    assert len(result.pnl_path) > 10
    assert result.limit_breaches == 0


def test_pbo_flags_noise_and_passes_a_real_edge():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (960, 20))
    assert cscv_pbo(noise, n_blocks=8).pbo > 0.2
    edge = rng.normal(0, 1, (960, 20))
    edge[:, 0] += 0.5
    assert cscv_pbo(edge, n_blocks=8).pbo < 0.05


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(1)
    series = rng.normal(0.05, 1.0, 2000)
    assert deflated_sharpe(series, 1000, 0.002) < probabilistic_sharpe(series)
    assert min_backtest_length(1, 0.1) == 1
    assert min_backtest_length(100, 0.1) == 922


def test_block_bootstrap_ci_covers_the_mean_and_widens_with_noise():
    rng = random.Random(0)
    series = [rng.gauss(1.0, 1.0) for _ in range(500)]
    low, high = block_bootstrap_ci(series, block=10, seed=0)
    assert low < 1.0 < high
    assert high - low < 0.5


def test_pnl_stats_are_per_second_and_report_drawdown():
    stats = pnl_stats([0.0, 1.0, 2.0, 1.0, 3.0])
    assert stats["total"] == 3.0
    assert stats["max_drawdown"] == 1.0
    assert stats["n"] == 4
    assert pnl_stats([1.0])["n"] == 0


def test_compare_variants_uses_only_same_window_differences():
    slices = [
        {"variant": "a", "pnl": 1.0, "drawdown": 0.5},
        {"variant": "b", "pnl": 3.0, "drawdown": 0.5},
    ] * 20
    rng = random.Random(0)
    slices = [{**s, "pnl": s["pnl"] + rng.gauss(0, 0.2)} for s in slices]
    result = compare_variants(slices, "b", "a")
    assert result["diff_mean"] == pytest.approx(2.0, abs=0.3)
    assert result["ci_low"] > 0


def test_compare_variants_says_so_when_data_is_thin():
    result = compare_variants([{"variant": "a", "pnl": 1.0}], "b", "a")
    assert result["n_pairs"] == 0
    assert result["ci_low"] is None


def test_decide_is_go_only_with_all_gates():
    strong = {"pbo": 0.1, "dsr": 0.97, "t": 500, "min_btl": 100, "ci_low": 0.5, "breaches": 0}
    assert decide(**strong).verdict == "GO"
    assert decide(**strong).size_fraction == 1.0
    assert decide(**{**strong, "pbo": 0.4, "dsr": 0.6}).verdict == "GO-REDUCED"
    assert decide(**{**strong, "ci_low": -1.0}).verdict == "GO-REDUCED"
    assert decide(**{**strong, "pbo": 0.8}).verdict == "NO-GO"
    assert decide(**{**strong, "breaches": 1}).verdict == "NO-GO"


def test_decide_passes_a_mechanism_without_tuning_on_missing_stats():
    verdict = decide(pbo=None, dsr=None, t=0, min_btl=0, ci_low=None, breaches=0)
    assert verdict.verdict == "GO-REDUCED"


def test_thompson_prefers_the_better_arm_and_forgets_after_a_regime_change():
    rng = random.Random(0)
    bandit = DiscountedThompson(["a", "b"], discount=0.9, seed=1)
    for _ in range(200):
        arm = bandit.choose()
        bandit.update(arm, rng.gauss(1.0 if arm == "b" else 0.0, 0.2))
    picks = [bandit.choose() for _ in range(100)]
    assert picks.count("b") > 90
    for _ in range(200):
        arm = bandit.choose()
        bandit.update(arm, rng.gauss(1.0 if arm == "a" else 0.0, 0.2))
    picks = [bandit.choose() for _ in range(100)]
    assert picks.count("a") > 70


def test_thompson_rejects_an_invalid_discount_and_reports_state():
    with pytest.raises(ValueError):
        DiscountedThompson(["a"], discount=1.5)
    bandit = DiscountedThompson(["a", "b"], seed=0)
    bandit.update("a", 1.0)
    assert set(bandit.snapshot()) == {"a", "b"}


def test_page_hinkley_detects_a_mean_shift_and_stays_quiet_before_it():
    rng = random.Random(3)
    detector = PageHinkley(delta=0.05, threshold=5.0)
    assert not any(detector.update(rng.gauss(0, 0.2)) for _ in range(500))
    assert any(detector.update(rng.gauss(-1.0, 0.2)) for _ in range(100))


def test_switchback_cycles_variants_with_a_washout():
    schedule = SwitchbackSchedule(["a", "b"], slice_s=10.0, washout_s=2.0, start=0.0, seed=5)
    assert schedule.variant_at(0.0) == schedule.variant_at(9.0)
    assert schedule.in_washout(1.0) and not schedule.in_washout(5.0)
    seen = {schedule.variant_at(t * 10.0) for t in range(8)}
    assert seen == {"a", "b"}


def test_switchback_blocks_are_balanced_pairs():
    schedule = SwitchbackSchedule(["a", "b"], slice_s=10.0, washout_s=0.0, start=0.0, seed=9)
    for block in range(10):
        first = schedule.variant_at(block * 20.0)
        second = schedule.variant_at(block * 20.0 + 10.0)
        assert {first, second} == {"a", "b"}


def test_analysis_report_is_json_serialisable(tmp_path):
    path = capture_of(quiet_market(2, ("A",)), tmp_path / "cap.jsonl", (), seconds=30)
    json.dumps(analyse_capture(load_events(path), tick=0.1))
