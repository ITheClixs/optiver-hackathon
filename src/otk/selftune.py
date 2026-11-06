"""The bot tunes itself when no agent can reach it. No tokens, no model, no network.

Once an hour the supervising parent, which holds no exchange session, runs a replay of the last hour in a separate
low-priority process, from the five changes fixed before the event. A clear winner goes to a live test at half size on
both sides, judged only for harm, and then in at full size. It runs only if the event settings name a time to stop, so
that no test can land in the final, and only with no updater, so that one thing writes the parameter file. A person can
always overrule it: `touch run/NO_TUNE` stops it, and a hand edit of the parameter file ends any test and stays.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from otk.livetest import LiveTests
from otk.pick import no_pick, pick_from_capture
from otk.updater import LiveFiles, UpdateConfig, parse_freeze

logger = logging.getLogger("otk.selftune")

TUNE_WINDOW_S = 3600.0  # replay the last hour, as the watcher does
TUNE_RETRY_S = 300.0  # how often new capture arrives
TUNE_POLL_S = 5.0
TUNE_PICK_TIMEOUT_S = 900.0
TUNE_EXIT_WAIT_S = 5.0
TUNE_NICENESS = 10
TUNE_STATE_NAME = "selftune_state.json"
TUNE_LOG_NAME = "selftune.jsonl"
TUNE_OFF_NAME = "NO_TUNE"
TUNE_NUMBERS = ("every_s", "min_capture_s", "test_s", "slice_s")
TUNE_JUDGED = ("NO HARM", "HARM")  # anything else, such as a test an outage cut short, says nothing about the change


@dataclass(frozen=True)
class TuneConfig:
    freeze_at: float
    every_s: float = 3600.0
    min_capture_s: float = 1800.0
    test_s: float = 1200.0
    slice_s: float = 60.0


def _lower_priority() -> None:
    """So a replay cannot slow the bot's ticks. Each may be refused, and neither is worth failing over."""
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))  # Linux: runs only on what the bot leaves
    except (AttributeError, OSError) as exc:
        logger.debug("no idle scheduling: %s", exc)
    try:
        os.nice(TUNE_NICENESS)
    except (AttributeError, OSError) as exc:
        logger.debug("no lower priority: %s", exc)


def run_self_tune_pick(run_dir: Path, min_capture_s: float) -> int:
    """Run by the parent as its own process. Prints one line of JSON last, and the parent reads that."""
    _lower_priority()
    try:
        status = json.loads((Path(run_dir) / "status.json").read_text())
    except (OSError, ValueError):
        status = {}
    baseline = status.get("baseline") if isinstance(status, dict) else None
    if isinstance(baseline, dict):
        print(json.dumps(pick_from_capture(Path(run_dir) / "capture.jsonl.gz", baseline, TUNE_WINDOW_S, min_capture_s)))
    else:  # half of a size we do not know would be half of the wrong thing
        print(json.dumps(no_pick("the bot has not said yet what is in force")))
    return 0


class SelfTuner:
    def __init__(
        self,
        entry: Path,
        run_dir: Path,
        config: TuneConfig,
        runner: Callable = subprocess.run,
        wall: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.entry, self.run_dir, self.config = Path(entry), Path(run_dir), config
        self.runner, self.wall = runner, wall
        files_config = UpdateConfig(repo="", token="", freeze_at=config.freeze_at)  # nothing here calls GitHub
        self.files = LiveFiles(files_config, None, self.run_dir, self._note, wall)
        self.tests = LiveTests(files_config, None, self.run_dir, self.files, self._note, wall, clock, sleep)
        saved = self._load_state()
        self.tried: set[str] = set(saved.get("tried", ()))
        pending = saved.get("pending")
        self.pending: dict | None = pending if isinstance(pending, dict) else None
        self.next_run = 0.0
        self._last_line = ""
        self._stop = threading.Event()

    def _load_state(self) -> dict:
        try:
            loaded = json.loads((self.run_dir / TUNE_STATE_NAME).read_text())
        except (OSError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _set_pending(self, pending: dict | None) -> None:
        self.pending = pending
        (self.run_dir / TUNE_STATE_NAME).write_text(json.dumps({"tried": sorted(self.tried), "pending": pending}))

    def _note(self, text: str) -> None:
        """A person reads this with `tail run/selftune.jsonl`. It must never stop anything."""
        logger.info("self-tune: %s", text.replace("\n", " "))
        try:
            with open(self.run_dir / TUNE_LOG_NAME, "a") as handle:
                handle.write(json.dumps({"t": self.wall(), "note": text}) + "\n")
        except OSError as exc:
            logger.warning("could not log: %s", exc)

    def _pick(self) -> dict | None:
        command = [sys.executable, str(self.entry), "--self-tune-pick", str(self.config.min_capture_s), "--run-dir", str(self.run_dir)]
        try:
            done = self.runner(command, capture_output=True, text=True, timeout=TUNE_PICK_TIMEOUT_S, check=False)
            lines = done.stdout.strip().splitlines()
            report = json.loads(lines[-1]) if done.returncode == 0 and lines else None
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self._note(f"replay failed: {type(exc).__name__}: {exc}")
            return None
        if not isinstance(report, dict):
            self._note(f"replay failed: {done.stderr.strip()[-300:]}")
            return None
        return report

    def _switched_off(self) -> bool:
        if not (self.run_dir / TUNE_OFF_NAME).exists():
            return False
        if self.tests.running():
            self.tests.stop(f"{TUNE_OFF_NAME} is there")
        if self.pending:
            self._note(f"{self.pending['id']}: dropped, {TUNE_OFF_NAME} is there")
            self._set_pending(None)
        return True

    def _judge_finished(self) -> None:
        """A test that left no finished result, because a restart cut it short, changes nothing."""
        if not self.pending or self.tests.running():
            return
        try:
            result = json.loads((self.run_dir / f"test-{self.pending['id']}.json").read_text())
        except (OSError, ValueError):
            result = {}
        verdict = (result.get("decision") or {}).get("verdict", "no result") if result.get("done") else "no result"
        change = self.pending["change"]
        self._note(f"{self.pending['id']} {json.dumps(change)}: {verdict}")
        if verdict in TUNE_JUDGED:
            self.tried.add(json.dumps(change, sort_keys=True))
        if verdict == "NO HARM" and self.wall() < self.config.freeze_at:
            self.files.adopt(change)
            self._note(f"adopted at full size: {json.dumps(change)}")
        self._set_pending(None)

    def _time_for_a_test(self) -> bool:
        return self.wall() + self.config.test_s < self.config.freeze_at

    def _start_test(self, picked: Mapping, variants: Mapping) -> None:
        if json.dumps(picked, sort_keys=True) in self.tried or not self._time_for_a_test():
            return
        test_id = f"tune-{int(self.wall())}"
        self._set_pending({"id": test_id, "change": dict(picked)})
        request = {
            "id": test_id, "variants": variants, "purpose": "no_harm", "adopt": picked, "duration_s": self.config.test_s,
            "slice_s": self.config.slice_s, "seed": int(self.wall()) % 100_000, "why": "self-tune",
        }
        self.tests.begin(request, test_id)

    def _after_pick(self, report: dict | None, now: float) -> None:
        if report is None:
            return
        if report["line"] != self._last_line:
            self._last_line = report["line"]
            self._note(report["line"])
        if report.get("never"):
            self.next_run = float("inf")
        elif not report.get("ran"):
            self.next_run = now + TUNE_RETRY_S
        elif report.get("picked"):
            self._start_test(report["picked"], report["variants"])

    def step(self, now: float) -> None:
        """`now` is monotonic, for the schedule. The freeze is on the wall clock, like everything else that has one."""
        if self._switched_off():
            return
        self._judge_finished()
        if self.pending or self.tests.running() or now < self.next_run or not self._time_for_a_test():
            return
        self.next_run = now + self.config.every_s  # first, so a replay that keeps failing can never run in a tight loop
        self._after_pick(self._pick(), now)

    def start(self) -> None:
        self._note("on: it starts no test that could not finish before the freeze")
        threading.Thread(target=self._run, name="self-tune", daemon=True).start()

    def _run(self) -> None:
        while not self._stop.wait(TUNE_POLL_S):
            try:
                self.step(time.monotonic())
            except Exception:
                logger.exception("self-tune step failed, carrying on")  # nothing here may touch trading

    def stop(self) -> None:
        self._stop.set()
        self.tests.stop("the bot is stopping")
        self.tests.wait(TUNE_EXIT_WAIT_S)


def build_self_tuner(event: Mapping, run_dir: Path, entry: Path) -> SelfTuner | None:
    """Only if asked for, with a time to stop, taken from `update` unless given here. Without one, a half-size test could land in the final."""
    asked = event.get("self_tune")
    if not asked:
        return None
    if event.get("pairs"):
        logger.error("self-tuning off: the replay does not know pairs named by hand, and would judge another strategy")
        return None
    settings = asked if isinstance(asked, Mapping) else {}
    update = event.get("update")
    try:
        freeze_at = parse_freeze(settings.get("freeze_at") or (update.get("freeze_at") if isinstance(update, Mapping) else None))
        numbers = {key: float(settings[key]) for key in TUNE_NUMBERS if key in settings}
        if freeze_at is None or any(value <= 0 for value in numbers.values()):
            raise ValueError("it needs a time to stop, and numbers above zero")
        return SelfTuner(Path(entry).resolve(), Path(run_dir), TuneConfig(freeze_at, **numbers))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        logger.error("self-tuning off: bad settings: %s: %s", type(exc).__name__, exc)
        return None
