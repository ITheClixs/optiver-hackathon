"""With no agent able to reach it, the bot still picks, checks for harm and adopts, with no tokens and no network."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_single_file

import otk.main
from otk.selftune import TUNE_OFF_NAME, TUNE_RETRY_S, SelfTuner, TuneConfig, build_self_tuner
from otk.updater import parse_freeze

FREEZE = 100_000.0
HOUR = TuneConfig(FREEZE).every_s
PICK = {
    "ran": True, "picked": {"skew_ticks": 2}, "line": "DECISION: x", "next": "",
    "variants": {"A": {"skew_ticks": 4, "size": 5}, "B": {"skew_ticks": 2, "size": 5}},
}


class Wall:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def make(tmp_path, report=PICK, wall=None, **kwargs):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    if not (run / "params.json").exists():
        (run / "params.json").write_text('{"size": 10}')
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "a log line\n" + json.dumps(report), "")

    tuner = SelfTuner(tmp_path / "bot.py", run, TuneConfig(FREEZE), runner=runner, wall=wall or Wall(), **kwargs)
    return tuner, run, calls


def stub_tests(tuner):
    """For what comes before and after a test. The tests at the end run a real one."""
    started = []
    tuner.tests.begin = lambda request, _test_id: started.append(request)
    return started


def finish(run, tuner, verdict):
    (run / f"test-{tuner.pending['id']}.json").write_text(json.dumps({"done": True, "decision": {"verdict": verdict}}))


def params(run):
    return json.loads((run / "params.json").read_text())


def notes(run):
    return (run / "selftune.jsonl").read_text()


def test_it_runs_the_replay_as_its_own_process_of_the_same_file(tmp_path):
    tuner, run, calls = make(tmp_path)
    stub_tests(tuner)
    tuner.step(0.0)
    assert calls[0][1] == str(tmp_path / "bot.py")
    assert calls[0][calls[0].index("--self-tune-pick") + 1] == str(TuneConfig(FREEZE).min_capture_s)
    assert calls[0][calls[0].index("--run-dir") + 1] == str(run)


def test_a_pick_starts_a_half_size_test_that_only_checks_for_harm(tmp_path):
    tuner, _run, _calls = make(tmp_path)
    started = stub_tests(tuner)
    tuner.step(0.0)
    assert started[0]["purpose"] == "no_harm"
    assert started[0]["adopt"] == {"skew_ticks": 2}
    assert started[0]["variants"]["B"]["size"] == 5
    assert started[0]["id"] == tuner.pending["id"]


def test_no_harm_goes_in_at_full_size_on_top_of_what_is_in_force(tmp_path):
    tuner, run, _calls = make(tmp_path)
    stub_tests(tuner)
    tuner.step(0.0)
    finish(run, tuner, "NO HARM")
    tuner.step(1.0)
    assert params(run) == {"size": 10, "skew_ticks": 2}
    assert "adopted" in notes(run)
    assert tuner.pending is None


@pytest.mark.parametrize("verdict", ["HARM", "NOT ENOUGH", "NO CHANGE"])
def test_anything_else_changes_nothing(tmp_path, verdict):
    tuner, run, _calls = make(tmp_path)
    stub_tests(tuner)
    tuner.step(0.0)
    finish(run, tuner, verdict)
    tuner.step(1.0)
    assert params(run) == {"size": 10}
    assert tuner.pending is None


def test_nothing_is_adopted_past_the_freeze(tmp_path):
    wall = Wall()
    tuner, run, _calls = make(tmp_path, wall=wall)
    stub_tests(tuner)
    tuner.step(0.0)
    finish(run, tuner, "NO HARM")
    wall.now = FREEZE + 1
    tuner.step(1.0)
    assert params(run) == {"size": 10}


def test_nothing_new_starts_while_a_test_runs(tmp_path):
    tuner, _run, calls = make(tmp_path)
    stub_tests(tuner)
    tuner.step(0.0)
    tuner.tests.running = lambda: True
    tuner.step(HOUR + 1)
    assert len(calls) == 1


def test_it_runs_once_an_hour_and_the_same_pick_is_never_tried_twice(tmp_path):
    tuner, run, calls = make(tmp_path)
    started = stub_tests(tuner)
    tuner.step(0.0)
    finish(run, tuner, "HARM")
    tuner.step(1.0)
    tuner.step(HOUR - 1)
    assert len(calls) == 1
    tuner.step(HOUR + 1)
    assert len(calls) == 2
    assert len(started) == 1


@pytest.mark.parametrize("verdict", ["NOT ENOUGH", "NO CHANGE"])
def test_a_test_that_could_not_judge_does_not_burn_the_pick(tmp_path, verdict):
    tuner, run, _calls = make(tmp_path)
    started = stub_tests(tuner)
    tuner.step(0.0)
    finish(run, tuner, verdict)
    tuner.step(1.0)
    tuner.step(HOUR + 1)
    assert len(started) == 2


def test_a_replay_that_blows_up_waits_the_hour_rather_than_running_every_few_seconds(tmp_path):
    tuner, _run, calls = make(tmp_path)

    def boom(command, **_kwargs):
        calls.append(command)
        raise RuntimeError("something nobody expected")

    tuner.runner = boom
    with pytest.raises(RuntimeError):
        tuner.step(0.0)
    tuner.step(5.0)
    assert len(calls) == 1


def test_a_capture_still_too_short_is_tried_again_soon_and_logged_once(tmp_path):
    tuner, run, calls = make(tmp_path, {"ran": False, "picked": None, "line": "too short", "next": ""})
    tuner.step(0.0)
    tuner.step(TUNE_RETRY_S + 1)
    assert len(calls) == 2
    assert notes(run).count("too short") == 1


def test_a_format_the_replay_cannot_stand_for_ends_it_for_good(tmp_path):
    tuner, _run, calls = make(tmp_path, {"ran": False, "never": True, "picked": None, "line": "this format", "next": ""})
    tuner.step(0.0)
    tuner.step(100 * HOUR)
    assert len(calls) == 1


@pytest.mark.parametrize("seconds_left", [0.0, TuneConfig(FREEZE).test_s - 1])
def test_no_replay_starts_when_no_test_could_finish_before_the_freeze(tmp_path, seconds_left):
    tuner, _run, calls = make(tmp_path, wall=Wall(FREEZE - seconds_left))
    tuner.step(0.0)
    assert not calls


def test_a_replay_that_took_too_long_starts_no_test(tmp_path):
    wall = Wall(FREEZE - TuneConfig(FREEZE).test_s - 1)
    tuner, _run, _calls = make(tmp_path, wall=wall)
    started = stub_tests(tuner)
    real = tuner.runner

    def slow(command, **kwargs):
        wall.now += 2
        return real(command, **kwargs)

    tuner.runner = slow
    tuner.step(0.0)
    assert not started
    assert tuner.pending is None


def test_a_restart_remembers_what_was_tried_and_judges_what_was_pending(tmp_path):
    tuner, run, _calls = make(tmp_path)
    stub_tests(tuner)
    tuner.step(0.0)
    finish(run, tuner, "NO HARM")
    again, run, _calls = make(tmp_path)
    again.step(1.0)
    assert params(run) == {"size": 10, "skew_ticks": 2}
    assert again.tried


def test_a_test_that_left_no_result_does_not_hold_up_what_comes_next(tmp_path):
    tuner, run, _calls = make(tmp_path)
    stub_tests(tuner)
    tuner.step(0.0)
    again, run, calls = make(tmp_path)
    stub_tests(again)
    again.next_run = HOUR
    again.step(1.0)
    assert again.pending is None
    assert params(run) == {"size": 10}
    assert "no result" in notes(run)
    again.step(HOUR + 1)
    assert len(calls) == 1


@pytest.mark.parametrize("stdout", ["Traceback", "", "[1, 2]"])
def test_a_replay_that_fails_is_nothing_rather_than_a_crash_and_is_not_rushed(tmp_path, stdout):
    tuner, run, _calls = make(tmp_path)
    started = stub_tests(tuner)
    tuner.runner = lambda command, **_kwargs: subprocess.CompletedProcess(command, 1 if stdout == "" else 0, stdout, "it broke")
    tuner.step(0.0)
    assert not started
    assert tuner.next_run == HOUR
    assert "replay failed" in notes(run)


def test_a_replay_that_hangs_is_nothing_rather_than_a_crash(tmp_path):
    tuner, run, _calls = make(tmp_path)
    started = stub_tests(tuner)

    def hang(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 1)

    tuner.runner = hang
    tuner.step(0.0)
    assert not started
    assert "replay failed" in notes(run)


def test_a_step_that_blows_up_does_not_end_the_thread(tmp_path, monkeypatch):
    monkeypatch.setattr("otk.selftune.TUNE_POLL_S", 0.01)
    tuner, _run, _calls = make(tmp_path)
    steps = []

    def boom(_now):
        steps.append(1)
        raise RuntimeError("no")

    tuner.step = boom
    thread = threading.Thread(target=tuner._run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while len(steps) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    tuner._stop.set()
    thread.join(5)
    assert len(steps) >= 3


def test_a_person_can_switch_it_off_with_a_file_and_on_again(tmp_path):
    tuner, run, calls = make(tmp_path)
    stub_tests(tuner)
    (run / TUNE_OFF_NAME).touch()
    tuner.step(0.0)
    assert not calls
    (run / TUNE_OFF_NAME).unlink()
    tuner.step(1.0)
    assert len(calls) == 1


def test_the_switch_drops_a_test_and_what_it_found(tmp_path):
    tuner, run, _calls = make(tmp_path)
    stub_tests(tuner)
    tuner.step(0.0)
    finish(run, tuner, "NO HARM")
    (run / TUNE_OFF_NAME).touch()
    tuner.step(1.0)
    assert tuner.pending is None
    assert params(run) == {"size": 10}


STOP = "2026-11-06T21:00:00+01:00"


@pytest.mark.parametrize(
    "event",
    [{}, {"self_tune": {}}, {"self_tune": False}, {"self_tune": True}, {"self_tune": {"every_s": 60}},
     {"self_tune": {"freeze_at": "2026-11-06T21:00:00"}}, {"self_tune": {"freeze_at": STOP, "every_s": 0}},
     {"self_tune": True, "update": {"repo": "o/r"}}, {"self_tune": True, "update": "nonsense"},
     {"update": {"freeze_at": STOP}}, {"self_tune": {"freeze_at": STOP}, "pairs": [["FOO", "BAR"]]}],
)
def test_it_is_off_unless_it_is_asked_for_and_has_a_real_time_to_stop(tmp_path, event):
    assert build_self_tuner(event, tmp_path, tmp_path / "bot.py") is None


def test_it_takes_the_time_to_stop_from_the_update_settings_so_there_is_one_place_to_set_it(tmp_path):
    tuner = build_self_tuner({"self_tune": True, "update": {"freeze_at": STOP}}, tmp_path, tmp_path / "bot.py")
    assert tuner.config.freeze_at == parse_freeze(STOP)
    assert (tuner.config.every_s, tuner.config.test_s) == (3600.0, 1200.0)


def test_a_time_given_here_wins(tmp_path):
    settings = {"self_tune": {"freeze_at": "2026-11-06T20:00:00+01:00"}, "update": {"freeze_at": STOP}}
    assert build_self_tuner(settings, tmp_path, tmp_path / "bot.py").config.freeze_at == parse_freeze("2026-11-06T20:00:00+01:00")


def test_a_rehearsal_can_shorten_every_wait(tmp_path):
    settings = {"freeze_at": STOP, "every_s": 120, "min_capture_s": 60, "test_s": 240, "slice_s": 20}
    config = build_self_tuner({"self_tune": settings}, tmp_path, tmp_path / "bot.py").config
    assert (config.every_s, config.min_capture_s, config.test_s, config.slice_s) == (120.0, 60.0, 240.0, 20.0)


# A real live test from here on, with only the clock made fast.


class Fast:
    """Each pause moves time on, and the bot's status moves with it, as it would once a second."""

    def __init__(self, run):
        self.run, self.now, self.seen, self.hook = run, 0.0, [], None
        self._write(0)

    def _write(self, pnl):
        (self.run / "status.json").write_text(json.dumps({"pnl": pnl, "connected": True, "elapsed_s": self.now, "positions": {}}))

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self._write(self.now)
        self.seen.append(json.dumps(params(self.run), sort_keys=True))
        if self.hook:
            self.hook(self.now)


def real_test(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    fast = Fast(run)
    tuner, run, calls = make(tmp_path, wall=Wall(), clock=fast.clock, sleep=fast.sleep)
    return tuner, run, fast, calls


def wait_for_the_test(tuner):
    deadline = time.monotonic() + 30
    while tuner.tests.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not tuner.tests.running()


def test_the_whole_way_from_a_pick_to_full_size(tmp_path):
    tuner, run, fast, _calls = real_test(tmp_path)
    tuner.step(0.0)
    wait_for_the_test(tuner)
    assert fast.seen, "the test never ran"
    assert {json.loads(seen)["size"] for seen in fast.seen} == {5}, "both sides at half size"
    assert {json.loads(seen)["skew_ticks"] for seen in fast.seen} == {2, 4}, "and the two sides really alternate"
    assert params(run) == {"size": 10}, "the bot goes back to what is in force before the verdict"
    tuner.step(1.0)
    assert params(run) == {"size": 10, "skew_ticks": 2}
    assert "NO HARM" in notes(run)


def test_a_persons_hand_edit_during_a_test_stays_and_ends_the_test(tmp_path):
    tuner, run, fast, _calls = real_test(tmp_path)
    edited = []

    def person(now):
        if now >= 100 and not edited:
            edited.append(now)
            (run / "params.json").write_text('{"enabled": 0}')

    fast.hook = person
    tuner.step(0.0)
    wait_for_the_test(tuner)
    assert fast.now < 200, "the test stopped at the next slice"
    assert params(run) == {"enabled": 0}
    tuner.step(1.0)
    assert params(run) == {"enabled": 0}, "and nothing was adopted over it"
    assert "NO CHANGE" in notes(run)


def test_the_switch_ends_a_running_test_and_puts_back_what_was_in_force(tmp_path):
    tuner, run, fast, _calls = real_test(tmp_path)
    switched = threading.Event()

    def wait_for_the_switch(now):
        if now >= 30:
            switched.set()
            fast.hook = None
            time.sleep(0.2)

    fast.hook = wait_for_the_switch
    tuner.step(0.0)
    assert switched.wait(10)
    (run / TUNE_OFF_NAME).touch()
    tuner.step(1.0)
    wait_for_the_test(tuner)
    assert params(run) == {"size": 10}


def test_a_parent_that_stops_during_a_test_leaves_no_half_size_behind(tmp_path):
    tuner, run, fast, _calls = real_test(tmp_path)
    fast.hook = lambda _now: time.sleep(0.01)
    tuner.step(0.0)
    deadline = time.monotonic() + 10
    while not fast.seen and time.monotonic() < deadline:
        time.sleep(0.01)
    tuner.stop()
    assert params(run) == {"size": 10}


def test_a_parent_killed_during_a_test_puts_it_back_when_it_starts_again(tmp_path):
    tuner, run, fast, _calls = real_test(tmp_path)
    fast.hook = lambda _now: time.sleep(10)  # stuck in a slice, as if killed
    tuner.step(0.0)
    deadline = time.monotonic() + 10
    while not fast.seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert params(run)["size"] == 5
    again, run, _calls = make(tmp_path)
    assert params(run) == {"size": 10}
    assert again.pending


# The real file, with no client and no data.


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("built") / "bot.py"
    make_single_file.build(out, {})
    return out


def ask(built, run_dir, cwd):
    return subprocess.run(
        [sys.executable, str(built), "--self-tune-pick", "1800", "--run-dir", str(run_dir)],
        capture_output=True, text=True, timeout=60, cwd=cwd, check=False,
    )


def test_the_replay_and_the_chooser_ship_inside_the_single_file(built):
    text = built.read_text()
    for needed in ("def pick_from_capture", "class ReplayExchange", "class SelfTuner"):
        assert needed in text


def said_in_force(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"baseline": {"size": 5}}))
    return run


def answer(done):
    assert done.returncode == 0, done.stderr[-600:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_built_file_answers_with_no_login_and_no_capture(built, tmp_path):
    report = answer(ask(built, said_in_force(tmp_path), tmp_path))
    assert (report["picked"], report["ran"]) == (None, False)
    assert "cannot read the capture" in report["line"]


def test_the_built_file_answers_on_a_capture_with_nothing_in_it_yet(built, tmp_path):
    run = said_in_force(tmp_path)
    (run / "capture.jsonl.gz").write_bytes(b"")
    assert "too short" in answer(ask(built, run, tmp_path))["line"]


def test_it_picks_nothing_until_the_bot_has_said_what_is_in_force(built, tmp_path):
    report = answer(ask(built, tmp_path / "run", tmp_path))
    assert (report["picked"], report["ran"]) == (None, False)
    assert "in force" in report["line"]


# The parent's wiring.


class Spy:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def command(self, base):
        return base


def run_parent(monkeypatch, tmp_path, updater):
    spy = Spy()
    seen = []
    monkeypatch.setattr(otk.main, "build_updater", lambda *_a: updater)
    monkeypatch.setattr(otk.main, "build_self_tuner", lambda *_a: seen.append(1) or spy)
    monkeypatch.setattr(otk.main, "supervise", lambda *_a, **_k: spy.calls.append("supervise") or 7)
    code = otk.main.main(["--run-dir", str(tmp_path / "run")], event={"self_tune": {}}, entry=str(tmp_path / "bot.py"))
    return code, spy, seen


def test_with_no_updater_the_parent_runs_it_around_the_child(monkeypatch, tmp_path):
    code, spy, _seen = run_parent(monkeypatch, tmp_path, None)
    assert code == 7
    assert spy.calls == ["start", "supervise", "stop"]


def test_with_an_updater_it_never_starts_so_one_thing_writes_the_file(monkeypatch, tmp_path):
    _code, spy, seen = run_parent(monkeypatch, tmp_path, Spy())
    assert not seen
    assert "start" not in spy.calls
