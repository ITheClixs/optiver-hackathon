"""A paired live test, started from a laptop and run beside the bot.

A request on the `live` branch reaches the updater in the supervising parent, which holds no exchange session. It
alternates two parameter sets in time slices by rewriting only the parameter file, and it measures each slice from
the status file the bot already writes, so the 60 s upload to laptops does not limit it. The adoption rule then decides,
with no model involved, and the default is no change. Whatever goes wrong here, the bot keeps trading on what was
published.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from otk.allocator import SwitchbackSchedule
from otk.evaluation import adoption_rule, decision_line, no_harm_rule
from otk.params import validate_params

logger = logging.getLogger("otk.livetest")

TEST_PATH = "contents/test.json?ref={branch}"
TEST_ACCEPT = "application/vnd.github.raw+json"
TEST_STATE_NAME = "test_state.json"
TEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,60}$")
MIN_SLICE_S, MAX_SLICE_S = 20.0, 600.0
MAX_DURATION_S = 7200.0
POLL_S = 1.0  # the bot writes its status once a second
MIN_SAMPLES = 2
PURPOSES = ("adopt", "no_harm")


class HandEdited(Exception):
    """A person changed the parameter file during a test. What they wrote stays, and the test is over."""


def parse_spec(raw: object) -> dict:
    """A complete, checked request, or ValueError saying what is wrong with it."""
    if not isinstance(raw, Mapping):
        raise ValueError("a test request must be a JSON object")  # noqa: TRY004
    test_id = raw.get("id")
    if not isinstance(test_id, str) or not TEST_ID.match(test_id):
        raise ValueError(f"bad test id: {test_id!r}")
    variants = raw.get("variants")
    if not isinstance(variants, Mapping) or len(variants) != 2:
        raise ValueError("a test compares exactly two variants")
    for params in variants.values():
        if not isinstance(params, Mapping):
            raise ValueError("each variant must be an object of parameters")  # noqa: TRY004
        validate_params(params)
    names = sorted(variants)
    control = raw.get("control", names[0])
    if control not in names:
        raise ValueError(f"control {control!r} is not one of the variants")
    try:
        slice_s, washout_s = float(raw.get("slice_s", 60.0)), float(raw.get("washout_s", 10.0))
        duration_s, seed, trials = float(raw.get("duration_s", 1200.0)), int(raw.get("seed", 0)), int(raw.get("trials", 2))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bad number in the test request: {exc}") from exc
    if not MIN_SLICE_S <= slice_s <= MAX_SLICE_S or not 0.0 <= washout_s < slice_s:
        raise ValueError("slices run 20 to 600 s, with a washout shorter than the slice")
    if not 0.0 < duration_s <= MAX_DURATION_S or trials < 1:
        raise ValueError("bad duration or trial count")
    purpose, adopt = raw.get("purpose", "adopt"), raw.get("adopt")
    if purpose not in PURPOSES:
        raise ValueError(f"a test's purpose is one of {PURPOSES}")
    if adopt is not None:
        if not isinstance(adopt, Mapping):
            raise ValueError("adopt must be an object of parameters")
        validate_params(adopt)
    challenger = next(name for name in names if name != control)
    return {
        "id": test_id, "variants": {name: dict(params) for name, params in variants.items()}, "control": control,
        "challenger": challenger, "slice_s": slice_s, "washout_s": washout_s, "duration_s": duration_s, "seed": seed,
        "trials": trials, "why": str(raw.get("why", "")), "purpose": purpose, "adopt": dict(adopt) if adopt else None,
    }


def _load_json(path: Path) -> dict:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _save_json(path: Path, payload: Mapping) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)


def _by_group(status: Mapping) -> dict[str, float]:
    totals: dict[str, float] = {}
    rows = status.get("by_instrument", {})
    for iid, row in rows.items() if isinstance(rows, Mapping) else ():
        totals[row.get("group", iid)] = totals.get(row.get("group", iid), 0.0) + row.get("pnl", 0.0)
    return totals


class _Slice:
    """What one slice saw, after its washout. Anything odd makes it untrustworthy rather than wrong."""

    def __init__(self, index: int, block: int, variant: str) -> None:
        self.index, self.block, self.variant = index, block, variant
        self.first: Mapping | None = None
        self.last: Mapping | None = None
        self.samples = 0
        self.valid = True
        self.peak_inventory = 0
        self.high = float("-inf")
        self.drawdown = 0.0

    def sample(self, status: Mapping) -> None:
        pnl = status.get("pnl")
        if pnl is None or not status.get("connected", True) or status.get("halted"):
            self.valid = False
            return
        if self.last is not None and status.get("elapsed_s", 0.0) < self.last.get("elapsed_s", 0.0):
            self.valid = False  # a new child: its own tallies started over
        self.first = self.first or status
        self.last = status
        self.samples += 1
        positions = status.get("positions") or {}
        self.peak_inventory = max(self.peak_inventory, *(abs(lots) for lots in positions.values()), 0)
        self.high = max(self.high, pnl)
        self.drawdown = max(self.drawdown, self.high - pnl)

    def closed(self) -> dict:
        usable = self.valid and self.samples >= MIN_SAMPLES
        first, last = self.first or {}, self.last or {}
        before, after = _by_group(first), _by_group(last)
        return {
            "index": self.index, "block": self.block, "variant": self.variant, "valid": usable,
            "pnl": round(last.get("pnl", 0.0) - first.get("pnl", 0.0), 4), "samples": self.samples,
            "peak_inventory": self.peak_inventory, "drawdown": round(self.drawdown, 4),
            "rejects": last.get("rejects", 0) - first.get("rejects", 0),
            "by_group": {name: round(after.get(name, 0.0) - before.get(name, 0.0), 4) for name in sorted({*before, *after})},
        }


class LiveTest:
    def __init__(
        self,
        spec: Mapping,
        files,
        run_dir: Path,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall: Callable[[], float] = time.time,
        freeze_at: float | None = None,
    ) -> None:
        self.spec, self.files, self.run_dir = spec, files, Path(run_dir)
        self.clock, self.sleep, self.wall, self.freeze_at = clock, sleep, wall, freeze_at
        self.path = self.run_dir / f"test-{spec['id']}.json"
        self.stop_reason = ""

    def run(self, stop: threading.Event) -> dict:
        """Never raises, and always leaves the published set in force and a finished result on disk."""
        spec = self.spec
        result = {**spec, "slices": [], "stopped": None, "done": False, "started_at": self.wall()}
        publishes = getattr(self.files, "publishes", 0)
        try:
            self._slices(stop, result)
        except HandEdited as exc:
            result["stopped"] = str(exc)
        except Exception as exc:  # this must never reach the parent
            logger.exception("live test %s failed", spec["id"])
            result["stopped"] = f"it failed: {type(exc).__name__}: {exc}"
        try:
            self.files.set_overlay({})
        except Exception as exc:  # the next published change repairs the file
            logger.exception("live test %s could not restore the parameters", spec["id"])
            result["stopped"] = result["stopped"] or f"it could not restore the parameters: {exc}"
        result["published_during_test"] = getattr(self.files, "publishes", 0) != publishes
        return self._finish(result)

    def _finish(self, result: dict) -> dict:
        if result.get("purpose") == "no_harm":
            decision = no_harm_rule(result["slices"], result["control"], result["challenger"])._asdict()
        else:
            decision = adoption_rule(result["slices"], result["control"], result["challenger"], result["trials"])._asdict()
        if result.get("published_during_test"):
            decision = {**decision, "verdict": "NO CHANGE", "reasons": ["parameters were published during the test", *decision["reasons"]]}
        result.update(decision=decision, done=True, ended_at=self.wall())
        try:
            _save_json(self.path, result)
        except OSError:
            logger.exception("could not save the result of live test %s", result["id"])
        return result

    def _slices(self, stop: threading.Event, result: dict) -> None:
        spec = self.spec
        names = sorted(spec["variants"])
        schedule = SwitchbackSchedule(names, spec["slice_s"], spec["washout_s"], self.clock(), spec["seed"])
        end = schedule.start + spec["duration_s"]
        current: _Slice | None = None
        while True:
            now = self.clock()
            why = self._why_stop(stop, now, end)
            if why is not None:
                result["stopped"] = why or None
                break
            index = int((now - schedule.start) // spec["slice_s"])
            if current is None or index != current.index:
                self._close(current, result)
                name = schedule.variant_at(now)
                self.files.set_overlay(spec["variants"][name])
                current = _Slice(index, index // len(names), name)
            if not schedule.in_washout(now):
                current.sample(_load_json(self.run_dir / "status.json"))
            self.sleep(POLL_S)
        self._close(current, result, final=True)

    def _why_stop(self, stop: threading.Event, now: float, end: float) -> str | None:
        """None to carry on, an empty string for a normal end, or the reason it was cut short."""
        if stop.is_set():
            return self.stop_reason or "it was stopped"
        if self.freeze_at is not None and self.wall() >= self.freeze_at:
            return "the freeze"
        return "" if now >= end else None

    def _close(self, current: _Slice | None, result: dict, final: bool = False) -> None:
        if current is None or (final and current.samples < MIN_SAMPLES):
            return  # a last slice cut off before it measured anything is no slice at all
        result["slices"].append(current.closed())
        _save_json(self.path, result)


class LiveTests:
    """Reads the request from the branch, and runs one test at a time on its own thread, so no slice waits on the network."""

    def __init__(
        self,
        config,
        github,
        run_dir: Path,
        files,
        report: Callable[[str], None],
        wall: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config, self.gh, self.run_dir, self.files = config, github, Path(run_dir), files
        self.report, self.wall, self.clock, self.sleep = report, wall, clock, sleep
        self.state_path = self.run_dir / TEST_STATE_NAME
        saved = _load_json(self.state_path)
        started = saved.get("started", ())
        self.started: set[str] = {item for item in started if isinstance(item, str)} if isinstance(started, list) else set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._test: LiveTest | None = None
        self._recover(saved.get("running"))

    def _recover(self, running: object) -> None:
        """The parent died during a test, so a variant may still be in force. Put back what was published."""
        if not isinstance(running, Mapping):
            return
        logger.warning("live test %s was cut off by a restart", running.get("id"))
        self.files.restore(running.get("published", {}))
        path = self.run_dir / f"test-{running.get('id')}.json"
        partial = _load_json(path)
        if partial and not partial.get("done"):
            partial["stopped"] = "the parent restarted"
            spec = {key: partial[key] for key in ("id", "control", "challenger") if key in partial}
            LiveTest({**spec, "id": running.get("id")}, self.files, self.run_dir, wall=self.wall)._finish(partial)
        self._save(None)

    def _save(self, running: Mapping | None) -> None:
        _save_json(self.state_path, {"started": sorted(self.started), "running": running})

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, why: str) -> None:
        if self._test is not None:
            self._test.stop_reason = why
        self._stop.set()

    def wait(self, timeout_s: float) -> None:
        """For a parent about to exit, so what was published is back in the file before it goes."""
        if self._thread is not None:
            self._thread.join(timeout_s)

    def sync(self) -> None:
        path = TEST_PATH.format(branch=self.config.branch)
        body = self.gh.fetch(path, TEST_ACCEPT)
        if body is None:
            return
        try:
            self._apply(body)
        except Exception:
            self.gh.forget(path)  # or the next poll's 304 would hide it for good
            raise

    def _refuse(self, task: str, why: str) -> None:
        self.report(f"@A task={task} from=updater\nLive test refused: {why}")

    def _apply(self, body: bytes) -> None:
        try:
            raw = json.loads(body)
        except ValueError:
            return self._refuse("live-test", "test.json is not JSON")
        if raw == {}:
            return None
        if isinstance(raw, Mapping) and raw.get("stop"):
            return self.stop(f"stopped: {raw.get('why', 'by request')}")
        test_id = raw.get("id") if isinstance(raw, Mapping) else None
        if isinstance(test_id, str) and test_id in self.started:
            return None
        return self.begin(raw, test_id)

    def begin(self, raw: object, test_id: object) -> None:
        task = test_id if isinstance(test_id, str) and TEST_ID.match(test_id) else "live-test"
        if self.config.freeze_at is not None and self.wall() >= self.config.freeze_at:
            return self._refuse(task, f"{task} is past the freeze, where nothing is explored")
        if self.running():
            return self._refuse(task, f"{task}: another test is already running")
        try:
            spec = parse_spec(raw)
        except ValueError as exc:
            return self._refuse(task, str(exc))
        self.files.set_overlay({})  # what is in force now is what the test comes back to
        self.started.add(spec["id"])
        try:
            self._save({"id": spec["id"], "published": self.files.published})
        except OSError:
            self.started.discard(spec["id"])  # or it could never run, once the disk is back
            raise
        self._stop = threading.Event()
        self._test = LiveTest(spec, self.files, self.run_dir, self.clock, self.sleep, self.wall, self.config.freeze_at)
        self._thread = threading.Thread(target=self._run, args=(self._test, self._stop), name="live-test", daemon=True)
        self._thread.start()
        return None

    def _run(self, test: LiveTest, stop: threading.Event) -> None:
        test.run(stop)
        try:
            self._save(None)
        except OSError:
            logger.exception("could not save the live test state")


def result_report(result: Mapping) -> str:
    """What the manager is told, and everyone can read, once a result has gone up."""
    control, challenger = result["control"], result["challenger"]
    lines = [f"@A task={result['id']} from=updater", f"Live test {result['id']} finished: {challenger} against {control}."]
    if result.get("stopped"):
        lines.append(f"Cut short: {result['stopped']}.")
    decision = result["decision"]
    lines.append(decision_line(decision, control, challenger))
    if decision["verdict"] in ("ADOPT", "ADOPT IN PRACTICE", "NO HARM"):
        change = json.dumps(result.get("adopt") or result["variants"][challenger])
        lines.append(f"To adopt it: python scripts/publish_params.py '{change}' --why \"{result['id']} adopted\"")
    lines.append(f"Slices: test-{result['id']}.json in the data release.")
    return "\n".join(lines)
