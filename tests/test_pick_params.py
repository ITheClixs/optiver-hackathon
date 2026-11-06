"""With no agents and no tokens, one command still reads a passive capture and picks parameters. Default is no change."""

import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import pick_params

import otk.pick
from otk.params import validate_params


def result(path, breaches=0, errors=0):
    return SimpleNamespace(pnl=path[-1], pnl_path=tuple(path), limit_breaches=breaches, errors=errors, writes=100, fills=100)


def walk(step, seconds=1800, noise=0.0, seed=1):
    rng = random.Random(seed)
    total, path = 0.0, []
    for _ in range(seconds):
        total += step + rng.gauss(0, noise)
        path.append(total)
    return path


BASE = result(walk(0.30, noise=0.5))


def test_every_candidate_is_a_valid_setting():
    for candidate in pick_params.CANDIDATES:
        validate_params(candidate)


def test_a_clear_and_steady_edge_passes():
    verdict = pick_params.judge(BASE, result(walk(0.50, noise=0.5, seed=2)))
    assert verdict.passed, verdict.reason


def test_a_small_edge_does_not():
    verdict = pick_params.judge(BASE, result(walk(0.305, noise=0.5, seed=1)))
    assert not verdict.passed


def test_noise_does_not_pass():
    verdict = pick_params.judge(BASE, result(walk(0.30, noise=0.5, seed=3)))
    assert not verdict.passed


def test_an_edge_from_one_half_only_does_not_pass():
    half = walk(0.30, seconds=900, noise=0.5, seed=1)
    second = [half[-1] + x for x in walk(0.90, seconds=900, noise=0.5, seed=4)]
    verdict = pick_params.judge(BASE, result(half + second))
    assert not verdict.passed
    assert "halves" in verdict.reason


def test_a_limit_breach_rules_it_out():
    verdict = pick_params.judge(BASE, result(walk(0.60, noise=0.5, seed=2), breaches=1))
    assert not verdict.passed and "breach" in verdict.reason


def test_a_much_bigger_drawdown_rules_it_out():
    wild = walk(0.60, noise=0.5, seed=2)
    path = wild[:900] + [x - 60.0 for x in wild[900:1000]] + wild[1000:]
    verdict = pick_params.judge(BASE, result(path))
    assert not verdict.passed and "drawdown" in verdict.reason


def test_the_best_passing_candidate_wins_and_nothing_passing_means_no_change():
    better = result(walk(0.50, noise=0.5, seed=2))
    best = result(walk(0.70, noise=0.5, seed=5))
    flat = result(walk(0.30, noise=0.5, seed=3))
    assert pick_params.choose(BASE, [({"a": 1}, better), ({"b": 2}, best), ({"c": 3}, flat)])[0] == {"b": 2}
    assert pick_params.choose(BASE, [({"c": 3}, flat)])[0] is None


def test_a_short_capture_is_refused():
    events = [{"t": 0.0, "k": "book"}, {"t": 100.0, "k": "book"}]
    assert "short" in pick_params.capture_problem(events, 1200.0, False)


def test_a_capture_taken_while_the_bot_traded_is_refused_unless_allowed():
    events = [{"t": 0.0, "k": "book"}, {"t": 50.0, "k": "fill"}, {"t": 2000.0, "k": "book"}]
    assert "own" in pick_params.capture_problem(events, 1200.0, False)
    assert pick_params.capture_problem(events, 1200.0, True) is None


def test_a_capture_from_the_running_bot_is_accepted_because_it_leaves_our_quotes_out():
    events = [{"t": 0.0, "k": "book"}, {"t": 50.0, "k": "fill", "src": "market_maker"}, {"t": 2000.0, "k": "book"}]
    assert pick_params.capture_problem(events, 1200.0, False) is None


@pytest.mark.parametrize(
    "ids,word",
    [(["ASML_202612_100C", "ASML"], "index weights"), (["ASML_A", "ASML_B"], "pairs named")],
)
def test_a_format_the_replay_cannot_stand_for_is_refused(ids, word):
    assert word in pick_params.format_problem(ids)


@pytest.mark.parametrize("ids", [["ASML", "ASML_DUAL"], ["ASML", "SAP"], ["ASML", "ASML_DUAL", "SAP", "SAP_DUAL"]])
def test_stocks_and_dual_listings_can_be_replayed(ids):
    assert pick_params.format_problem(ids) is None


def test_the_next_step_is_a_half_size_check_on_both_sides():
    command = pick_params.next_step({"skew_ticks": 2}, {"size": 10, "skew_ticks": 4}, "pick-1912")
    assert command.startswith("python scripts/live_test.py start pick-1912 ")
    assert '"A": {"skew_ticks": 4' in command
    assert '"B": {"skew_ticks": 2, "size": 5}' in command
    assert "--no-harm" in command
    assert """--adopt '{"skew_ticks": 2}'""" in command


def test_the_next_step_uses_the_defaults_for_what_was_never_set():
    defaults = validate_params({})
    command = pick_params.next_step({"arb_edge_ticks": 2}, {}, "pick-1")
    assert f'"arb_edge_ticks": {defaults["arb_edge_ticks"]}' in command
    assert f'"size": {round(defaults["size"] / 2)}' in command


def test_half_of_a_tiny_size_is_still_at_least_one_lot():
    assert '"size": 1' in pick_params.next_step({"skew_ticks": 2}, {"size": 1}, "pick-1")


def pick_stub(_events, _ids, _ticks, overrides, _base):
    return BASE if not overrides else result(walk(0.70 if overrides == {"skew_ticks": 2} else 0.30, noise=0.5, seed=7))


def test_it_starts_from_what_the_bot_has_in_force(capture, tmp_path, monkeypatch):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"baseline": {"size": 5, "skew_ticks": 3}}))
    seen = []

    def spy(events, ids, ticks, overrides, base):
        seen.append(base)
        return pick_stub(events, ids, ticks, overrides, base)

    monkeypatch.setattr(otk.pick, "replay", spy)
    assert pick_params.main([str(capture), "--min-seconds", "60", "--status", str(status)]) == 0
    assert seen
    assert all(base == {"size": 5, "skew_ticks": 3} for base in seen)


def test_only_the_last_stretch_is_replayed_when_asked(capture, monkeypatch):
    spans = []

    def spy(events, ids, ticks, overrides, base):
        spans.append(events[-1]["t"] - events[0]["t"])
        return pick_stub(events, ids, ticks, overrides, base)

    monkeypatch.setattr(otk.pick, "replay", spy)
    assert pick_params.main([str(capture), "--min-seconds", "60", "--last-seconds", "100"]) == 0
    assert spans
    assert all(span <= 100.0 for span in spans)


def test_apply_with_a_status_writes_only_the_pick_over_the_file(capture, tmp_path, monkeypatch):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"baseline": {"size": 5, "position_limit": 100}}))
    target = tmp_path / "params.json"
    target.write_text(json.dumps({"size": 5}))
    monkeypatch.setattr(otk.pick, "replay", pick_stub)
    assert pick_params.main([str(capture), "--min-seconds", "60", "--status", str(status), "--apply", str(target)]) == 0
    assert json.loads(target.read_text()) == {"size": 5, "skew_ticks": 2}


def test_json_gives_the_watcher_the_pick_and_the_next_step(capture, monkeypatch, capsys):
    monkeypatch.setattr(otk.pick, "replay", pick_stub)
    assert pick_params.main([str(capture), "--min-seconds", "60", "--json"]) == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["picked"] == {"skew_ticks": 2}
    assert report["ran"] is True
    assert report["line"].startswith("DECISION:")
    assert "live_test.py start" in report["next"]


def test_json_says_when_nothing_was_replayed(tmp_path, capsys):
    path = tmp_path / "short.jsonl"
    synthetic_capture(path, seconds=10)
    assert pick_params.main([str(path), "--min-seconds", "1800", "--json"]) == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["picked"] is None
    assert report["ran"] is False
    assert "too short" in report["line"]


def synthetic_capture(path: Path, seconds=200):
    rng = random.Random(9)
    mid = 100.0
    lines = []
    for second in range(seconds):
        mid = round(mid + rng.choice((-0.1, 0.0, 0.1)), 1)
        for iid, width in (("A", 0.1), ("A_DUAL", 0.3)):
            lines.append({"t": 1000.0 + second, "k": "book", "iid": iid, "bids": [[round(mid - width, 1), 50]], "asks": [[round(mid + width, 1), 50]]})
        lines.append({"t": 1000.5 + second, "k": "trade", "iid": "A_DUAL", "px": round(mid + rng.choice((-0.3, 0.3)), 1), "vol": 10, "aggr": "bid"})
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


@pytest.fixture
def capture(tmp_path):
    path = tmp_path / "market.jsonl"
    synthetic_capture(path)
    return path


def run_cli(capture, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pick_params.py"), str(capture), "--min-seconds", "60", *extra],
        capture_output=True, text=True, timeout=120, check=False,
    )


def test_a_missing_capture_is_an_answer_rather_than_a_crash(tmp_path):
    report = otk.pick.pick_from_capture(tmp_path / "none.jsonl.gz", {}, 3600.0, 60.0)
    assert (report["picked"], report["ran"]) == (None, False)
    assert "cannot read" in report["line"]


def test_a_format_it_cannot_stand_for_says_trying_again_will_not_help(tmp_path):
    path = tmp_path / "market.jsonl"
    path.write_text("\n".join(json.dumps({"t": float(t), "k": "book", "iid": "ASML_202612_100C", "bids": [], "asks": []}) for t in (0, 100)))
    report = otk.pick.pick_from_capture(path, {}, None, 60.0)
    assert report["never"] is True
    short = tmp_path / "short.jsonl"
    synthetic_capture(short, seconds=10)
    assert "never" not in otk.pick.pick_from_capture(short, {}, None, 60.0)


def test_only_the_window_is_ever_held_in_memory(capture, monkeypatch):
    held = []
    real = otk.pick.deque

    class Watched(real):
        def append(self, item):
            super().append(item)
            held.append(len(self))

    monkeypatch.setattr(otk.pick, "deque", Watched)
    everything = otk.pick.load_events(capture)
    recent = otk.pick.recent_events(capture, 10.0)
    assert recent == [event for event in everything if event["t"] >= everything[-1]["t"] - 10.0]
    assert max(held) <= len(recent) + 3
    assert max(held) < len(everything) / 5


def test_it_runs_end_to_end_and_always_ends_with_a_decision(capture):
    done = run_cli(capture)
    assert done.returncode == 0, done.stderr
    assert "base" in done.stdout
    assert "DECISION" in done.stdout


def test_apply_keeps_what_is_already_in_the_file(capture, tmp_path, monkeypatch):
    target = tmp_path / "params.json"
    target.write_text(json.dumps({"size": 5}))
    monkeypatch.setattr(otk.pick, "replay", lambda _e, _i, _t, overrides, _b: BASE if not overrides else result(walk(0.70 if overrides == {"skew_ticks": 2} else 0.30, noise=0.5, seed=7)))
    assert pick_params.main([str(capture), "--min-seconds", "60", "--apply", str(target)]) == 0
    assert json.loads(target.read_text()) == {"size": 5, "skew_ticks": 2}


def test_apply_writes_nothing_when_nothing_passes(capture, tmp_path, monkeypatch):
    target = tmp_path / "params.json"
    monkeypatch.setattr(otk.pick, "replay", lambda *_a: BASE)
    assert pick_params.main([str(capture), "--min-seconds", "60", "--apply", str(target)]) == 0
    assert not target.exists()
