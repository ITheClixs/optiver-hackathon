"""A live test starts from a laptop, runs beside the bot, and decides by rule. Default: no change."""

import json
import math
import random
import threading
import time
from pathlib import Path

import pytest

from otk.evaluation import (
    MIN_PAIRS,
    adoption_rule,
    decision_line,
    deflated_sharpe_normal,
    no_harm_rule,
    sign_flip_p,
)
from otk.livetest import LiveTest, LiveTests, parse_spec, result_report
from otk.updater import LiveFiles, UpdateConfig, Uplink

# --- the rule --------------------------------------------------------------------------------------


def pair_slices(differences, base=10.0, group=None, peak=(10, 10), drawdown=(1.0, 1.0), rejects=(0, 0), noise=0.0, seed=0):
    rng = random.Random(seed)
    slices = []
    for block, difference in enumerate(differences):
        shared = base + rng.gauss(0.0, noise) if noise else base
        slices.append({"block": block, "variant": "A", "pnl": shared, "valid": True, "peak_inventory": peak[0], "drawdown": drawdown[0], "rejects": rejects[0], "by_group": group or {}})
        slices.append({"block": block, "variant": "B", "pnl": shared + difference, "valid": True, "peak_inventory": peak[1], "drawdown": drawdown[1], "rejects": rejects[1], "by_group": group or {}})
    return slices


WINNER = [9.0, 11.0, 10.0, 12.0, 8.0, 10.0, 11.0, 9.0, 10.0, 12.0]
NOISE = [5.0, -6.0, 4.0, -5.0, 6.0, -4.0, 5.0, -6.0, 4.0, -3.0]


def test_a_sign_flip_is_exact_on_a_small_sample():
    assert sign_flip_p([1.0] * 6) == pytest.approx(2 / 2**6)
    assert sign_flip_p([0.0, 0.0, 0.0]) == 1.0
    assert sign_flip_p([1.0, -1.0, 1.0, -1.0]) == 1.0


def test_a_sign_flip_on_a_large_sample_is_repeatable_and_sensible():
    values = [1.0] * 25
    assert sign_flip_p(values) == sign_flip_p(values)
    assert sign_flip_p(values) < 0.01


@pytest.mark.parametrize("pairs,trials,sharpe_needed", [(30, 10, 0.63), (60, 20, 0.47), (120, 50, 0.36), (15, 2, 0.62)])
def test_the_deflated_sharpe_matches_the_table_in_the_research(pairs, trials, sharpe_needed):
    assert deflated_sharpe_normal(sharpe_needed, pairs, trials) == pytest.approx(0.95, abs=0.015)


def test_a_clear_winner_is_adopted():
    decision = adoption_rule(pair_slices(WINNER), "A", "B", trials=2)
    assert decision.verdict == "ADOPT", decision.reasons
    assert decision.stats["pairs"] == 10
    assert decision.stats["mean"] == pytest.approx(10.2)


def test_noise_changes_nothing():
    decision = adoption_rule(pair_slices(NOISE), "A", "B", trials=2)
    assert decision.verdict == "NO CHANGE"
    assert any("p " in reason for reason in decision.reasons)


def test_a_clear_loser_changes_nothing():
    decision = adoption_rule(pair_slices([-value for value in WINNER]), "A", "B", trials=2)
    assert decision.verdict == "NO CHANGE"
    assert any("did not earn more" in reason for reason in decision.reasons)


def test_too_few_pairs_change_nothing():
    decision = adoption_rule(pair_slices(WINNER[: MIN_PAIRS - 1]), "A", "B", trials=2)
    assert decision.verdict == "NO CHANGE"
    assert "clean pairs" in decision.reasons[0]


def test_a_gain_from_only_one_half_changes_nothing():
    lopsided = [0.0, -1.0, 0.5, -0.5, 0.0, 20.0, 22.0, 19.0, 21.0, 20.0]
    decision = adoption_rule(pair_slices(lopsided), "A", "B", trials=2)
    assert decision.verdict == "NO CHANGE"
    assert any("halves" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    "worse,word",
    [({"peak": (10, 90)}, "peak_inventory"), ({"drawdown": (1.0, 40.0)}, "drawdown"), ({"rejects": (0, 9)}, "rejects")],
)
def test_more_risk_changes_nothing_even_with_more_pnl(worse, word):
    decision = adoption_rule(pair_slices(WINNER, **worse), "A", "B", trials=2)
    assert decision.verdict == "NO CHANGE"
    assert any(word in reason for reason in decision.reasons)


def test_a_slice_that_cannot_be_trusted_takes_its_pair_with_it():
    slices = pair_slices(WINNER)
    slices[0]["valid"] = False
    assert adoption_rule(slices, "A", "B", trials=2).stats["pairs"] == 9


def test_three_or_more_underlyings_must_mostly_agree():
    slices = pair_slices(WINNER)
    for item in slices:
        sign = 1.0 if item["variant"] == "B" else -1.0
        item["by_group"] = {"ASML": 100.0 * sign, "SAP": -sign, "NVDA": -sign, "TSLA": -sign}
    decision = adoption_rule(slices, "A", "B", trials=2)
    assert decision.verdict == "NO CHANGE"
    assert any("underlyings" in reason for reason in decision.reasons)


def test_one_or_two_underlyings_are_not_held_to_it():
    slices = pair_slices(WINNER)
    for item in slices:
        sign = 1.0 if item["variant"] == "B" else -1.0
        item["by_group"] = {"ASML": 100.0 * sign, "SAP": -sign}
    assert adoption_rule(slices, "A", "B", trials=2).verdict == "ADOPT"


def test_many_trials_downgrade_a_win_to_practice_only():
    moderate = [-1.0, 11.0, 0.0, 10.0, 1.0, 9.0, 2.0, 12.0, -2.0, 11.0, 3.0, 8.0]
    assert adoption_rule(pair_slices(moderate), "A", "B", trials=2).verdict == "ADOPT"
    assert adoption_rule(pair_slices(moderate), "A", "B", trials=2000).verdict == "NO CHANGE"
    weak = adoption_rule(pair_slices(moderate), "A", "B", trials=40)
    assert weak.verdict == "ADOPT IN PRACTICE", (weak.reasons, weak.stats)
    assert "not for the final" in decision_line(weak, "A", "B")


def test_an_effect_too_small_to_matter_changes_nothing():
    small = [0.2, 0.25, 0.3, 0.2, 0.25, 0.3, 0.2, 0.25, 0.3, 0.2]
    slices = pair_slices(small, base=10.0, noise=8.0, seed=4)
    decision = adoption_rule(slices, "A", "B", trials=2)
    assert decision.verdict == "NO CHANGE"
    assert any("effect" in reason for reason in decision.reasons)


def test_the_decision_line_is_one_line_a_person_can_read():
    line = decision_line(adoption_rule(pair_slices(NOISE), "A", "B", trials=2), "A", "B")
    assert line.startswith("DECISION: NO CHANGE")
    assert "\n" not in line


# --- the check on a replay pick --------------------------------------------------------------------


def test_a_pick_that_earns_about_the_same_does_no_harm():
    decision = no_harm_rule(pair_slices(NOISE), "A", "B")
    assert decision.verdict == "NO HARM"
    assert decision_line(decision, "A", "B").startswith("DECISION: NO HARM")


def test_a_pick_that_clearly_earns_less_is_harm():
    decision = no_harm_rule(pair_slices([-value for value in WINNER]), "A", "B")
    assert decision.verdict == "HARM"
    assert "earned less" in decision.reasons[0]
    assert decision_line(decision, "A", "B").startswith("DECISION: HARM")


def test_a_pick_that_earns_more_but_takes_more_risk_is_harm():
    decision = no_harm_rule(pair_slices(WINNER, peak=(10, 90)), "A", "B")
    assert decision.verdict == "HARM"
    assert "peak_inventory" in decision.reasons[0]


def test_too_few_pairs_say_nothing_either_way():
    decision = no_harm_rule(pair_slices(NOISE[:3]), "A", "B")
    assert decision.verdict == "NOT ENOUGH"
    assert decision_line(decision, "A", "B").startswith("DECISION: NOT ENOUGH")


def test_a_no_harm_test_is_judged_by_the_no_harm_rule(world):
    world.earn = lambda _params: 1.0
    result, _ = run_test(world, purpose="no_harm", adopt={"skew_ticks": 2}, duration_s=1200.0)
    assert result["decision"]["verdict"] == "NO HARM"


def test_publishing_during_a_no_harm_test_still_spoils_it(world):
    files = make_files(world)

    def publish(current):
        if current.now == 1200.0:
            files._apply_params(json.dumps({"size": 7}).encode())

    world.earn = lambda _params: 1.0
    world.hook = publish
    result, _ = run_test(world, files=files, purpose="no_harm", duration_s=1200.0)
    assert result["decision"]["verdict"] == "NO CHANGE"


# --- the request -----------------------------------------------------------------------------------


GOOD = {"id": "exp-017", "variants": {"A": {}, "B": {"skew_ticks": 2}}}


def test_a_request_gets_the_usual_shape_by_default():
    spec = parse_spec(GOOD)
    assert (spec["slice_s"], spec["washout_s"], spec["duration_s"]) == (60.0, 10.0, 1200.0)
    assert spec["control"] == "A"
    assert spec["challenger"] == "B"
    assert spec["trials"] == 2


def test_the_control_can_be_named():
    spec = parse_spec({**GOOD, "control": "B"})
    assert (spec["control"], spec["challenger"]) == ("B", "A")


def test_a_test_asks_for_adoption_unless_it_says_otherwise():
    assert parse_spec(GOOD)["purpose"] == "adopt"
    spec = parse_spec({**GOOD, "purpose": "no_harm", "adopt": {"skew_ticks": 2}})
    assert (spec["purpose"], spec["adopt"]) == ("no_harm", {"skew_ticks": 2})


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {},
        {"id": "../../etc", "variants": GOOD["variants"]},
        {"id": "x", "variants": {"A": {}}},
        {"id": "x", "variants": {"A": {}, "B": {}, "C": {}}},
        {"id": "x", "variants": {"A": {}, "B": {"nonsense": 1}}},
        {"id": "x", "variants": {"A": {}, "B": "size"}},
        {**GOOD, "slice_s": 1},
        {**GOOD, "washout_s": 60},
        {**GOOD, "duration_s": 100000},
        {**GOOD, "control": "Z"},
        {**GOOD, "trials": 0},
        {**GOOD, "slice_s": "soon"},
        {**GOOD, "purpose": "whatever"},
        {**GOOD, "adopt": {"nonsense": 1}},
        {**GOOD, "adopt": "size"},
    ],
)
def test_a_bad_request_is_refused(bad):
    with pytest.raises(ValueError):
        parse_spec(bad)


# --- running one -----------------------------------------------------------------------------------


class World:
    """A pretend bot. It earns by whichever parameters are in force, and writes status as the child does."""

    def __init__(self, run_dir, earn=None, freeze_at=None):
        self.run = Path(run_dir)
        self.now = 1000.0
        self.pnl = 0.0
        self.elapsed = 0.0
        self.earn = earn or (lambda params: 3.0 if params.get("skew_ticks") == 2 else 1.0)
        self.status = {}
        self.params_seen = []
        self.freeze_at = freeze_at
        self.hook = None
        self.write_status()

    def clock(self):
        return self.now

    def params(self):
        try:
            return json.loads((self.run / "params.json").read_text())
        except OSError:
            return {}

    def write_status(self):
        payload = {
            "connected": True, "halted": False, "pnl": self.pnl, "elapsed_s": self.elapsed, "rejects": 0,
            "positions": {"ASML": 5}, "by_instrument": {"ASML": {"pnl": self.pnl, "group": "ASML"}}, **self.status,
        }
        (self.run / "status.json").write_text(json.dumps(payload))

    def sleep(self, seconds):
        self.now += seconds
        self.elapsed += seconds
        params = self.params()
        if not self.params_seen or self.params_seen[-1] != params:
            self.params_seen.append(params)
        self.pnl += self.earn(params) * seconds
        if self.hook:
            self.hook(self)
        self.write_status()


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def make_files(world, published=None, freeze_at=None):
    files = LiveFiles(UpdateConfig(repo="o/r", token="t", freeze_at=freeze_at), None, world.run, lambda _text: None, wall=lambda: world.now)
    (world.run / "params.json").write_text(json.dumps(published or {"size": 5}))
    return files


def run_test(world, files=None, stop=None, freeze_at=None, **over):
    files = files or make_files(world)
    spec = parse_spec({**GOOD, "duration_s": 600.0, "seed": 3, **over})
    test = LiveTest(spec, files, world.run, clock=world.clock, sleep=world.sleep, wall=lambda: world.now, freeze_at=freeze_at)
    return test.run(stop or threading.Event()), files


def test_each_variant_runs_on_top_of_what_is_published_and_it_ends_where_it_began(world):
    run_test(world)
    assert {"size": 5} in world.params_seen
    assert {"size": 5, "skew_ticks": 2} in world.params_seen
    assert world.params() == {"size": 5}


def test_each_slice_measures_only_after_its_washout_and_pairs_come_from_one_block(world):
    result, _ = run_test(world)
    slices = result["slices"]
    assert len(slices) == 10
    for item in slices:
        assert item["valid"]
        rate = 3.0 if item["variant"] == "B" else 1.0
        assert item["pnl"] == pytest.approx(rate * 50.0, abs=rate * 3.0)
    for block in range(5):
        assert {item["variant"] for item in slices if item["block"] == block} == {"A", "B"}


def test_a_clear_gain_is_adopted_by_the_rule(world):
    result, _ = run_test(world, duration_s=1200.0)
    assert result["decision"]["verdict"] in {"ADOPT", "ADOPT IN PRACTICE"}, result["decision"]
    assert result["done"]


def test_no_difference_changes_nothing(world):
    world.earn = lambda _params: 1.0
    result, _ = run_test(world, duration_s=1200.0)
    assert result["decision"]["verdict"] == "NO CHANGE"


def test_the_result_is_on_disk_after_every_slice(world):
    seen = []

    def look(current):
        path = current.run / "test-exp-017.json"
        if path.exists():
            seen.append(json.loads(path.read_text()))

    world.hook = look
    result, _ = run_test(world)
    assert seen
    assert not seen[-1]["done"]
    assert len(seen[-1]["slices"]) >= 8
    assert json.loads((world.run / "test-exp-017.json").read_text())["done"]
    assert result["done"]


def test_a_halted_bot_spoils_only_the_slices_it_touched(world):
    def halt(current):
        current.status = {"halted": 1100.0 <= current.now <= 1110.0}

    world.hook = halt
    result, _ = run_test(world)
    assert [item["valid"] for item in result["slices"]].count(False) == 1


def test_a_disconnected_bot_spoils_the_slice(world):
    def drop(current):
        current.status = {"connected": not 1100.0 <= current.now <= 1110.0}

    world.hook = drop
    result, _ = run_test(world)
    assert [item["valid"] for item in result["slices"]].count(False) == 1


def test_a_restarted_child_spoils_the_slice_because_its_numbers_started_over(world):
    def restart(current):
        if current.now == 1150.0:
            current.elapsed = 0.0

    world.hook = restart
    result, _ = run_test(world)
    assert [item["valid"] for item in result["slices"]].count(False) == 1


def test_a_restart_during_a_washout_spoils_nothing_because_nothing_was_measured_yet(world):
    def restart(current):
        if current.now == 1125.0:
            current.elapsed = 0.0

    world.hook = restart
    result, _ = run_test(world)
    assert all(item["valid"] for item in result["slices"])


def test_a_missing_status_spoils_the_slice_and_does_not_crash(world):
    original = world.write_status

    def write():
        if 1100.0 <= world.now <= 1110.0:
            (world.run / "status.json").unlink(missing_ok=True)
        else:
            original()

    world.write_status = write
    result, _ = run_test(world)
    assert [item["valid"] for item in result["slices"]].count(False) == 1


def test_risk_is_measured_inside_each_slice(world):
    def wobble(current):
        current.status = {"positions": {"ASML": 40 if current.now % 120 < 60 else -70}, "rejects": int(current.now - 1000.0) // 60}

    world.hook = wobble
    result, _ = run_test(world)
    assert {item["peak_inventory"] for item in result["slices"]} <= {40, 70}
    assert all(item["rejects"] in (0, 1) for item in result["slices"])


def test_a_drawdown_inside_a_slice_is_measured(world):
    world.earn = lambda _params: -2.0
    result, _ = run_test(world)
    assert all(item["drawdown"] == pytest.approx(100.0, abs=6.0) for item in result["slices"])


def test_pnl_by_underlying_is_split_per_slice(world):
    result, _ = run_test(world)
    item = result["slices"][0]
    assert item["by_group"]["ASML"] == pytest.approx(item["pnl"])


def test_stopping_early_restores_the_published_set_and_says_why(world):
    stop = threading.Event()

    def halt_at(current):
        if current.now > 1200.0:
            stop.set()

    world.hook = halt_at
    files = make_files(world)
    result, _ = run_test(world, files=files, stop=stop, why="not needed")
    assert result["stopped"]
    assert world.params() == {"size": 5}
    assert result["done"]


def test_the_freeze_ends_a_test_at_once(world):
    result, _ = run_test(world, freeze_at=1300.0)
    assert "freeze" in result["stopped"]
    assert world.params() == {"size": 5}
    assert world.now < 1305.0


def test_publishing_during_a_test_is_kept_and_spoils_the_test(world):
    files = make_files(world)

    def publish(current):
        if current.now == 1200.0:
            files._apply_params(json.dumps({"size": 7}).encode())

    world.hook = publish
    result, _ = run_test(world, files=files, duration_s=1200.0)
    assert result["published_during_test"]
    assert result["decision"]["verdict"] == "NO CHANGE"
    assert any("published" in reason for reason in result["decision"]["reasons"])
    assert world.params() == {"size": 7}, "what was published stands, and the variant does not"


def test_a_file_that_cannot_be_written_ends_the_test_and_never_raises(world):
    files = make_files(world)
    calls = []

    def fail(overlay):
        calls.append(overlay)
        if len(calls) == 2:
            raise OSError("disk full")

    files.set_overlay = fail
    result, _ = run_test(world, files=files)
    assert "disk full" in result["stopped"]
    assert result["done"]


# --- from the branch to the thread -----------------------------------------------------------------


class Hub:
    def __init__(self):
        self.body = None
        self.forgotten = []
        self.uploads = {}
        self.fail_upload = False

    def fetch(self, _path, _accept=None):
        body, self.body = self.body, None
        return body

    def forget(self, path):
        self.forgotten.append(path)

    def read(self, _path):
        return {"id": 77}

    def post(self, _path, _payload):
        return {"id": 77}

    def upload(self, _release, name, data):
        if self.fail_upload:
            raise OSError("network")
        self.uploads[name] = data


def make_tests(tmp_path, freeze_at=None, wall=lambda: 1000.0):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "params.json").write_text('{"size": 5}')
    reports = []
    config = UpdateConfig(repo="o/r", token="t", freeze_at=freeze_at)
    hub = Hub()
    files = LiveFiles(config, hub, run, reports.append, wall=wall)
    tests = LiveTests(config, hub, run, files, reports.append, wall=wall, sleep=lambda _seconds: time.sleep(0.001))
    return tests, hub, files, run, reports


def request(**over):
    return json.dumps({**GOOD, "duration_s": 0.01, **over}).encode()


def wait_until_done(tests, seconds=5.0):
    deadline = time.monotonic() + seconds
    while tests.running() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not tests.running()


def test_a_request_on_the_branch_starts_a_test_and_it_finishes(tmp_path):
    tests, hub, _files, run, _reports = make_tests(tmp_path)
    hub.body = request()
    tests.sync()
    wait_until_done(tests)
    assert json.loads((run / "test-exp-017.json").read_text())["done"]


def test_the_same_request_never_runs_twice_not_even_after_a_restart(tmp_path):
    tests, hub, _files, run, _reports = make_tests(tmp_path)
    hub.body = request()
    tests.sync()
    wait_until_done(tests)
    (run / "test-exp-017.json").unlink()
    again, hub, _files, run, _reports = make_tests(tmp_path)
    hub.body = request()
    again.sync()
    assert not again.running()
    assert not (run / "test-exp-017.json").exists()


def test_a_second_test_is_refused_while_one_runs(tmp_path):
    tests, hub, _files, _run, reports = make_tests(tmp_path)
    tests._stop.clear()
    hub.body = request(duration_s=60.0)
    tests.sync()
    assert tests.running()
    hub.body = request(id="exp-018")
    tests.sync()
    assert any("exp-018" in text and "already running" in text for text in reports)
    tests.stop("done with this")
    wait_until_done(tests)


def test_no_test_starts_past_the_freeze(tmp_path):
    tests, hub, _files, _run, reports = make_tests(tmp_path, freeze_at=500.0)
    hub.body = request()
    tests.sync()
    assert not tests.running()
    assert any("freeze" in text for text in reports)


@pytest.mark.parametrize("body", [b"not json", b'{"id": "x", "variants": {"A": {}}}'])
def test_a_bad_request_is_reported_and_nothing_runs(tmp_path, body):
    tests, hub, _files, _run, reports = make_tests(tmp_path)
    hub.body = body
    tests.sync()
    assert not tests.running()
    assert reports and reports[0].startswith("@A task=")


def test_an_empty_file_means_nothing_to_do(tmp_path):
    tests, hub, _files, _run, reports = make_tests(tmp_path)
    hub.body = b"{}"
    tests.sync()
    assert not tests.running()
    assert not reports


def test_a_stop_request_stops_a_running_test(tmp_path):
    tests, hub, _files, run, _reports = make_tests(tmp_path)
    hub.body = request(duration_s=600.0)
    tests.sync()
    assert tests.running()
    hub.body = json.dumps({"id": "exp-017", "stop": True, "why": "looks wrong"}).encode()
    tests.sync()
    wait_until_done(tests)
    result = json.loads((run / "test-exp-017.json").read_text())
    assert "looks wrong" in result["stopped"]
    assert json.loads((run / "params.json").read_text()) == {"size": 5}


def test_a_restart_during_a_test_restores_the_published_set_and_closes_the_result(tmp_path):
    tests, hub, _files, run, _reports = make_tests(tmp_path)
    hub.body = request(duration_s=600.0)
    tests.sync()
    assert tests.running()
    state = json.loads((run / "test_state.json").read_text())
    assert state["running"]["id"] == "exp-017"
    tests.stop("the parent is stopping")
    wait_until_done(tests)

    # Now as if the parent had died mid-test: the variant is still in the file.
    (run / "params.json").write_text('{"size": 5, "skew_ticks": 2}')
    (run / "test-exp-017.json").write_text(json.dumps({"id": "exp-017", "slices": [], "done": False, "control": "A", "challenger": "B", "trials": 2}))
    state["started"] = ["exp-017"]
    (run / "test_state.json").write_text(json.dumps(state))

    config = UpdateConfig(repo="o/r", token="t")
    reports = []
    files = LiveFiles(config, hub, run, reports.append, wall=lambda: 1000.0)
    again = LiveTests(config, hub, run, files, reports.append, wall=lambda: 1000.0)
    assert json.loads((run / "params.json").read_text()) == {"size": 5}
    result = json.loads((run / "test-exp-017.json").read_text())
    assert result["done"]
    assert "restarted" in result["stopped"]
    assert not again.running()


def test_a_network_failure_is_seen_again_next_poll(tmp_path):
    tests, hub, _files, _run, _reports = make_tests(tmp_path)
    hub.body = b"not json"

    def fail(_text):
        raise RuntimeError("no")

    tests.report = fail
    with pytest.raises(RuntimeError):
        tests.sync()
    assert hub.forgotten


def test_a_test_that_could_not_begin_can_still_run_later(tmp_path):
    tests, hub, files, _run, _reports = make_tests(tmp_path)
    real = files.set_overlay

    def full_disk(_overlay):
        raise OSError("disk full")

    files.set_overlay = full_disk
    hub.body = request()
    with pytest.raises(OSError):
        tests.sync()
    assert "exp-017" not in tests.started
    files.set_overlay = real
    hub.body = request()
    tests.sync()
    wait_until_done(tests)
    assert "exp-017" in tests.started


# --- the result reaches everyone -------------------------------------------------------------------


def finished(run, test_id="exp-017", verdict="NO CHANGE", **over):
    result = {
        "id": test_id, "done": True, "control": "A", "challenger": "B", "variants": {"A": {}, "B": {"skew_ticks": 2}},
        "slices": [], "stopped": None, "decision": {"verdict": verdict, "reasons": ["paired p 0.31 is not below 0.10"], "stats": {"pairs": 10, "mean": 1.5, "p": 0.31}},
        **over,
    }
    (run / f"test-{test_id}.json").write_text(json.dumps(result))
    return result


def make_uplink(tmp_path):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    hub = Hub()
    reports = []
    return Uplink(UpdateConfig(repo="o/r", token="t", data_tag="data-1"), hub, run, reports.append), hub, run, reports


def test_a_finished_result_goes_up_once_and_the_manager_is_told(tmp_path):
    uplink, hub, run, reports = make_uplink(tmp_path)
    finished(run)
    uplink.send_tests()
    uplink.send_tests()
    assert list(hub.uploads) == ["test-exp-017.json"]
    assert len(reports) == 1
    assert reports[0].startswith("@A task=exp-017 from=updater")
    assert "DECISION: NO CHANGE" in reports[0]


def test_a_running_test_does_not_go_up(tmp_path):
    uplink, hub, run, reports = make_uplink(tmp_path)
    (run / "test-exp-017.json").write_text(json.dumps({"id": "exp-017", "done": False}))
    uplink.send_tests()
    assert not hub.uploads
    assert not reports


def test_a_failed_upload_is_tried_again_and_only_then_told(tmp_path):
    uplink, hub, run, reports = make_uplink(tmp_path)
    finished(run)
    hub.fail_upload = True
    with pytest.raises(OSError):
        uplink.send_tests()
    assert not reports
    hub.fail_upload = False
    uplink.send_tests()
    assert len(reports) == 1


def test_a_restart_of_the_parent_does_not_send_it_again(tmp_path):
    uplink, hub, run, reports = make_uplink(tmp_path)
    finished(run)
    uplink.send_tests()
    again = Uplink(UpdateConfig(repo="o/r", token="t", data_tag="data-1"), hub, run, reports.append)
    again.send_tests()
    assert len(reports) == 1


def test_the_report_says_how_to_adopt_only_if_the_rule_says_so(tmp_path):
    run = tmp_path
    adopted = result_report(finished(run, verdict="ADOPT"))
    assert "publish_params.py" in adopted
    assert '"skew_ticks": 2' in adopted
    assert "publish_params.py" not in result_report(finished(run, verdict="NO CHANGE"))


def test_a_pick_that_did_no_harm_is_adopted_at_full_size_without_the_tests_half_size(tmp_path):
    result = finished(tmp_path, verdict="NO HARM", purpose="no_harm", adopt={"skew_ticks": 2}, variants={"A": {"skew_ticks": 4, "size": 5}, "B": {"skew_ticks": 2, "size": 5}})
    report = result_report(result)
    assert """publish_params.py '{"skew_ticks": 2}'""" in report
    assert '"size"' not in report


@pytest.mark.parametrize("verdict", ["HARM", "NOT ENOUGH"])
def test_a_pick_that_failed_its_check_is_not_offered(tmp_path, verdict):
    assert "publish_params.py" not in result_report(finished(tmp_path, verdict=verdict, purpose="no_harm", adopt={"skew_ticks": 2}))


def test_the_report_says_when_it_was_cut_short(tmp_path):
    assert "the freeze" in result_report(finished(tmp_path, stopped="the freeze"))


def test_the_arithmetic_of_a_default_test_gives_ten_pairs():
    spec = parse_spec(GOOD)
    assert math.floor(spec["duration_s"] / spec["slice_s"]) // 2 == 10
