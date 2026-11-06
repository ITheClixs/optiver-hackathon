"""Operator scripts, exercised against the simulator and a fake client."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from otk.adapter import LiveExchange
from otk.allocator import SwitchbackSchedule
from otk.engine import Engine, EngineConfig
from otk.instruments import Universe
from otk.params import write_params_atomic
from otk.recorder import JsonlRecorder
from otk.sim.exchange import SimExchange, SimMarket
from otk.sim.harness import make_engine
from otk.sim.scenarios import quiet_market

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rest(market, iid, side, price, volume, team="bot"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def capture(market, path, seconds=40.0):
    recorder = JsonlRecorder(path, book_every_s=0.0, clock=lambda: market.now)
    engine = make_engine(market, "observer", [], recorder=recorder)
    for _ in range(int(seconds / 0.1)):
        market.advance(0.1)
        engine.tick()
    recorder.close()


def test_run_bot_sim_reports_clean_session(capsys):
    run_bot = load_script("run_bot")
    assert run_bot.main(["--sim", "dual", "--seconds", "20", "--params", "/nonexistent.json"]) == 0
    assert json.loads(capsys.readouterr().out)["limit_breaches"] == 0


def test_run_bot_sim_writes_a_capture_that_analyses(tmp_path, capsys):
    run_bot = load_script("run_bot")
    analyze = load_script("analyze_capture")
    path = tmp_path / "cap.jsonl"
    run_bot.main(["--sim", "quiet", "--seconds", "20", "--capture", str(path), "--params", "/nope"])
    capsys.readouterr()
    report_path = tmp_path / "report.json"
    assert analyze.main([str(path), "--out", str(report_path)]) == 0
    assert "instrument" in capsys.readouterr().out
    assert json.loads(report_path.read_text())["instruments"]


def test_event_config_sets_engine_limits(tmp_path):
    run_bot = load_script("run_bot")
    config = tmp_path / "event.json"
    config.write_text(json.dumps({"position_limit": 50, "outstanding_limit": 80}))
    event = run_bot.load_event_config(str(config))
    limits = run_bot.engine_config(event, tmp_path, None).limits
    assert (limits.position, limits.outstanding) == (50, 80)
    assert run_bot.load_event_config(None) == {}


def test_status_file_is_valid_json(tmp_path):
    run_bot = load_script("run_bot")
    market = SimMarket(instruments=["A"])
    engine = make_engine(market, "us", [])
    path = tmp_path / "status.json"
    run_bot.write_status(path, engine, time.monotonic())
    assert json.loads(path.read_text())["connected"] is True


def test_probe_reports_shapes_reads_and_writes_without_leaving_orders():
    probe = load_script("probe_exchange")
    probe.WRITE_PAUSE_S = 0.0
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.0, 100)
    rest(market, "A", "ask", 101.0, 100)
    exchange = SimExchange(market, "us")
    exchange.connect()
    report = probe.run_probe(exchange, "A")
    assert report["connected_at_end"] is True
    assert report["reads"]["still_connected"] is True
    assert report["shapes"]["ticks_expose_counterparty"] is False
    assert exchange.get_outstanding_orders("A") == {}
    assert market.stats["us"].writes == 3


def test_probe_skips_writes_on_an_empty_book():
    probe = load_script("probe_exchange")
    market = SimMarket(instruments=["A"])
    exchange = SimExchange(market, "us")
    exchange.connect()
    assert "skipped" in probe.probe_writes(exchange, "A")


def test_flatten_closes_positions_within_limits():
    flatten = load_script("flatten")
    flatten.PAUSE_S = 0.0
    market = SimMarket(instruments=["A", "B"])
    for iid in ("A", "B"):
        rest(market, iid, "bid", 99.0, 500)
        rest(market, iid, "ask", 101.0, 500)
    exchange = SimExchange(market, "us")
    exchange.connect()
    exchange.insert_order("A", price=101.0, volume=30, side="bid", order_type="ioc")
    exchange.insert_order("B", price=99.0, volume=20, side="ask", order_type="ioc")
    exchange.insert_order("A", price=90.0, volume=5, side="bid", order_type="limit")
    assert flatten.flatten(exchange) == {}
    assert exchange.get_outstanding_orders("A") == {}


def test_flatten_reports_what_it_could_not_close():
    flatten = load_script("flatten")
    flatten.PAUSE_S = 0.0
    market = SimMarket(instruments=["A"])
    rest(market, "A", "ask", 101.0, 500)
    exchange = SimExchange(market, "us")
    exchange.connect()
    exchange.insert_order("A", price=101.0, volume=30, side="bid", order_type="ioc")
    assert flatten.flatten(exchange, attempts=2) == {"A": 30}


def test_backtest_grid_reports_plateau_and_gates(tmp_path, capsys):
    grid = load_script("backtest_grid")
    path = tmp_path / "cap.jsonl"
    capture(quiet_market(2, ("A",)), path, seconds=60.0)
    out = tmp_path / "report.json"
    code = grid.main(
        [str(path), "--grid", json.dumps({"base_spread_ticks": [2, 4], "size": [5, 10]}), "--out", str(out)]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["n_configs"] == 4
    assert report["gates"]["decision"]["verdict"] in ("GO", "GO-REDUCED", "NO-GO")
    assert report["chosen"]["size"] in (5, 10)


def test_plateau_prefers_a_broad_hill_over_a_lone_spike():
    grid = load_script("backtest_grid")
    table = {"a": [1, 2, 3, 4, 5]}
    pnl = {1: 10.0, 2: 10.0, 3: 10.0, 4: 0.0, 5: 100.0}
    rows = [{"params": {"a": v}, "pnl": pnl[v]} for v in table["a"]]
    scores = grid.plateau_scores(rows, table)
    assert scores[1] > scores[4]


def test_switchback_rotates_params_and_measures_each_slice(tmp_path):
    switchback = load_script("switchback")
    status = tmp_path / "status.json"
    params = tmp_path / "params.json"
    fake = {"t": 0.0, "pnl": 0.0}

    def clock():
        return fake["t"]

    def sleep(seconds):
        fake["t"] += seconds
        current = json.loads(params.read_text())["size"]
        fake["pnl"] += seconds * (2.0 if current == 20 else 1.0)
        write_status(fake["pnl"])

    def write_status(pnl):
        status.write_text(json.dumps({"pnl": pnl}))

    write_status(0.0)
    variants = {"big": {"size": 20}, "small": {"size": 5}}
    schedule = SwitchbackSchedule(sorted(variants), 10.0, 2.0, 0.0, seed=3)
    slices = switchback.run_schedule(variants, schedule, 120.0, params, status, clock, sleep)
    big = [s["pnl"] for s in slices if s["variant"] == "big"]
    small = [s["pnl"] for s in slices if s["variant"] == "small"]
    assert big and small
    assert min(big) > max(small)


def test_switchback_read_pnl_tolerates_a_missing_file(tmp_path):
    switchback = load_script("switchback")
    assert switchback.read_pnl(tmp_path / "absent.json") is None


def test_sim_tournament_is_clean(capsys):
    tournament = load_script("sim_tournament")
    assert tournament.main(["--scenarios", "dual,news", "--seeds", "1", "--seconds", "20"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["scenario"] for row in rows} == {"dual", "news"}


def test_bundle_runs_without_the_source_tree(tmp_path):
    bundle = load_script("make_bundle")
    output = bundle.build(tmp_path / "otk_bot.pyz")
    result = subprocess.run(
        [sys.executable, "-I", str(output), "--sim", "quiet", "--seconds", "5", "--params", "/nope"],
        capture_output=True, text=True, cwd=tmp_path, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "limit_breaches" in result.stdout


class FakeClient:
    instances = 0

    def __init__(self):
        type(self).instances += 1
        self.connected = False
        self.reads = 0

    def connect(self):
        self.connected = True

    def is_connected(self):
        return self.connected

    def disconnect(self):
        self.connected = False

    def get_instruments(self):
        return {"A": SimpleNamespace(instrument_id="A", tick_size=0.1, expired=False),
                "OLD": SimpleNamespace(instrument_id="OLD", tick_size=0.1, expired=True)}

    def get_last_price_book(self, _iid):
        self.reads += 1

    def get_outstanding_orders(self, _iid):
        self.reads += 1

    def get_positions(self):
        self.reads += 1
        return {"A": 1}

    def poll_new_trade_ticks(self, _iid):
        self.reads += 1
        return []

    def poll_new_trades(self, _iid):
        self.reads += 1
        return []

    def get_pnl(self, valuations=None):
        return 1.5 if valuations else 1.0

    def insert_order(self, _iid, **kwargs):
        return kwargs

    def amend_order(self, _iid, **kwargs):
        return kwargs

    def delete_order(self, _iid, **kwargs):
        return kwargs

    def delete_orders(self, iid):
        return iid


def test_live_exchange_reconnects_with_a_fresh_client_and_normalises_none():
    FakeClient.instances = 0
    live = LiveExchange(FakeClient)
    assert not live.is_connected()
    live.connect()
    assert live.is_connected()
    assert live.get_last_price_book("A") is None
    assert live.get_outstanding_orders("A") == {}
    assert live.poll_new_social_media_feeds() == []
    live.connect()
    assert FakeClient.instances == 2
    assert live.get_pnl() == 1.0
    assert live.get_pnl({"A": 1.0}) == 1.5
    assert live.insert_order("A", price=1.0) == {"price": 1.0}
    assert live.amend_order("A", order_id=1) == {"order_id": 1}
    assert live.delete_order("A", order_id=1) == {"order_id": 1}
    assert live.delete_orders("A") == "A"
    live.disconnect()
    assert not live.is_connected()


def test_live_exchange_read_ttl_caches_risk_reads_until_a_write():
    now = {"t": 0.0}
    live = LiveExchange(FakeClient, read_ttl_s=1.0, clock=lambda: now["t"])
    live.connect()
    client = live._client
    for _ in range(50):
        live.get_positions()
        live.get_outstanding_orders("A")
    assert client.reads == 2
    live.insert_order("A", price=1.0)
    live.get_positions()
    live.get_outstanding_orders("A")
    assert client.reads == 4
    now["t"] = 2.0
    live.get_positions()
    assert client.reads == 5


def test_live_exchange_never_serves_a_stale_book():
    class Emptying(FakeClient):
        def __init__(self):
            super().__init__()
            self.books = [SimpleNamespace(bids=[], asks=[]), None]

        def get_last_price_book(self, _iid):
            return self.books.pop(0)

    live = LiveExchange(Emptying)
    live.connect()
    assert live.get_last_price_book("A") is not None
    assert live.get_last_price_book("A") is None


def test_universe_drops_expired_instruments():
    live = LiveExchange(FakeClient)
    live.connect()
    assert set(Universe.from_instruments(live.get_instruments()).by_id) == {"A"}


def test_engine_survives_a_client_exception_mid_tick():
    market = SimMarket(instruments=["A"])
    engine = make_engine(market, "us", [])

    def broken(_iid):
        raise ValueError("bad data from the client")

    engine.exchange.get_last_price_book = broken
    engine.tick()
    assert engine.errors == 1


def test_stale_lock_from_a_dead_process_is_cleared(tmp_path):
    lock = tmp_path / "LIVE.lock"
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True)
    lock.write_text(dead.stdout.strip())
    market = SimMarket(instruments=["A"])
    engine = make_engine(market, "us", [], EngineConfig(live_lock=str(lock)))
    engine.acquire_lock()
    engine.release_lock()
    assert not lock.exists()


def test_engine_run_loop_stops_and_releases_lock(tmp_path):
    market = SimMarket(instruments=["A"])
    lock = tmp_path / "LIVE.lock"
    engine = make_engine(market, "us", [], EngineConfig(live_lock=str(lock)))
    calls = []
    engine.run(should_stop=lambda: calls.append(1) or len(calls) > 3)
    assert not lock.exists()


def test_params_written_by_an_agent_reach_the_running_engine(tmp_path):
    from otk.params import ParamStore
    from otk.strategies import MarketMaker

    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.0, 100)
    rest(market, "A", "ask", 101.0, 100)
    path = tmp_path / "params.json"
    write_params_atomic(path, {"size": 3})
    engine = make_engine(market, "us", [MarketMaker(["A"])], params=ParamStore(path))
    engine.tick()
    assert {o.volume for o in engine.exchange.get_outstanding_orders("A").values()} == {3}
    write_params_atomic(path, {"size": 7})
    time.sleep(0.01)
    engine.tick()
    assert sum(o.volume for o in engine.exchange.get_outstanding_orders("A").values()) == 14


def test_write_params_atomic_refuses_invalid_input(tmp_path):
    with pytest.raises(ValueError):
        write_params_atomic(tmp_path / "params.json", {"bogus": 1})
    assert not (tmp_path / "params.json").exists()


def test_engine_class_is_importable_from_the_scripts_path():
    assert Engine is not None


def test_event_config_builds_keyword_and_trained_classifiers(tmp_path):
    run_bot = load_script("run_bot")
    assert run_bot.build_classifier({}) is None
    keyword = run_bot.build_classifier({"news_aliases": {"NVDA": ["nvidia", "nvda"]}})
    assert keyword.classify("Nvidia beats estimates").direction == 1
    csv_path = tmp_path / "training.csv"
    csv_path.write_text(
        "SocialMediaFeed,NVDA\nchip surge record,0.03\nchip recall lawsuit,-0.03\nweekend books,0\n"
    )
    trained = run_bot.build_classifier({"news_training_csv": str(csv_path)})
    assert trained.classify("chip surge").instrument_id == "NVDA"


def test_event_config_passes_the_own_orders_flag(tmp_path):
    run_bot = load_script("run_bot")
    config = run_bot.engine_config({"book_includes_own": False}, tmp_path, None)
    assert config.book_includes_own is False


def test_set_params_merges_clamps_logs_and_rejects_bad_input(tmp_path, capsys):
    set_params = load_script("set_params")
    path = tmp_path / "params.json"
    assert set_params.main(['{"size": 7}', "--path", str(path), "--why", "test"]) == 0
    assert set_params.main(['{"skew_ticks": 9}', "--path", str(path)]) == 0
    assert json.loads(path.read_text()) == {"size": 7, "skew_ticks": 9}
    capsys.readouterr()
    assert set_params.main(['{"size": 100000}', "--path", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["size"] == 100
    assert set_params.main(['{"nope": 1}', "--path", str(path)]) == 2
    assert set_params.main(["{broken", "--path", str(path)]) == 2
    assert json.loads(path.read_text())["size"] == 100000
    assert (tmp_path / "param_changes.log").read_text().count("\n") == 3
    assert set_params.main(["--list"]) == 0
    assert "base_spread_ticks" in capsys.readouterr().out
