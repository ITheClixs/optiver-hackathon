"""Keeps the bot current with nobody uploading: merged code and parameters come to it, and status goes out.

It runs in the supervising parent, which holds no exchange session, so swapping the child needs no second
login, and nothing has to reach into the cloud. A release is downloaded, checked against its sha256 and run
with `--selfcheck`, and only then does the running bot stop for it. The new file is watched, and any trouble
restores the last good one. If GitHub is unreachable nothing changes and the bot trades on. Network calls run
on their own threads, because a slow request must not delay the parent's watch over the child, and a slow
download must not delay a kill.

The parent itself is not replaced: it stays the first upload, so a bad release cannot break its own rollback.
Nothing here may stop the parent: every call it makes is guarded, and fails towards keeping a bot running.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from otk.livetest import HandEdited, LiveTests, result_report
from otk.params import INSTRUMENTS_KEY, merge_params, validate_params

logger = logging.getLogger("otk.updater")

TOKEN_ENV = "OPTIBOOK_GH_TOKEN"
DEFAULT_TOKEN_FILE = "~/.optibook_github"
BOT_ASSET = "optibook_bot.py"
SHA_ASSET = "optibook_bot.py.sha256"
STATE_NAME = "update_state.json"
UPLINK_STATE_NAME = "uplink_state.json"
KILL_STATE_NAME = "kill_state.json"
SAFE_TAG = re.compile(r"^[A-Za-z0-9._-]+$")
SELFCHECK_TIMEOUT_S = 120.0
SELFCHECK_DETAIL_CHARS = 600
REQUEST_TIMEOUT_S = 15.0
WATCH_ERRORS = 5
WATCH_REJECTS = 30
WATCH_CRASHES = 2  # one crash may be the exchange, not the code; a second condemns the new file
MAX_PART_BYTES = 50_000_000
MAX_REPORT_CHARS = 60_000
MAX_REPORTS = 200
REPORT_TRIES = 5
HTTP_ALREADY_EXISTS = 422
REDIRECTS = (301, 302, 303, 307, 308)
JSON_ACCEPT = "application/vnd.github+json"
RAW_ACCEPT = "application/vnd.github.raw+json"
BINARY_ACCEPT = "application/octet-stream"
# After the freeze a parameter may only move these, and only down.
DECREASE_ONLY = frozenset(
    {"size", "enabled", "position_limit", "mr_enabled", "gap_enabled", "lag_enabled", "lag_clip", "ap_enabled", "taker_mode"}
)
# Where 0 means off, so setting one tightens it, and setting it to 0 does not.
STOPS = frozenset({"kill_drawdown", "kill_max_loss", "kill_loss_per_min"})
EVENT_IN_FORCE = "event_in_force.json"  # written by otk.main
TAILED_FILES = ("capture.jsonl.gz", "autopilot.jsonl")


@dataclass(frozen=True)
class UpdateConfig:
    repo: str
    token: str
    api: str = "https://api.github.com"
    uploads: str = "https://uploads.github.com"
    branch: str = "live"
    issue: int | None = None
    freeze_at: float | None = None
    code_poll_s: float = 30.0
    params_poll_s: float = 10.0
    status_s: float = 60.0
    data_s: float = 300.0
    watch_s: float = 300.0
    connect_grace_s: float = 60.0
    data_tag: str = ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class GitHub:
    """The few REST calls the updater needs. An ETag'd 304 costs no rate limit, so polling is cheap.

    No call follows a redirect with the token on it. Only `download` follows one at all, without the token.
    """

    def __init__(self, config: UpdateConfig) -> None:
        self.config = config
        self._etags: dict[str, str] = {}
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _request(self, url: str, accept: str = JSON_ACCEPT, data: bytes | None = None, kind: str = "") -> urllib.request.Request:
        headers = {"Authorization": f"Bearer {self.config.token}", "Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
        if kind:
            headers["Content-Type"] = kind
        return urllib.request.Request(url, data=data, headers=headers)

    def _url(self, path: str) -> str:
        return f"{self.config.api}/repos/{self.config.repo}/{path}"

    def fetch(self, path: str, accept: str = JSON_ACCEPT) -> bytes | None:
        """The body if it changed. None if unchanged or absent, so callers keep what they have."""
        request = self._request(self._url(path), accept)
        if path in self._etags:
            request.add_header("If-None-Match", self._etags[path])
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
                etag = response.headers.get("ETag")
                if etag:
                    self._etags[path] = etag
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (304, 404):
                return None
            raise

    def forget(self, path: str) -> None:
        """A caller that could not act on a body says so, or the next poll's 304 would hide it for good."""
        self._etags.pop(path, None)

    def read(self, path: str) -> dict | None:
        """A plain read that ignores ETags, for looking something up once."""
        try:
            with self._opener.open(self._request(self._url(path)), timeout=REQUEST_TIMEOUT_S) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def download(self, asset_id: int | str) -> bytes:
        """Storage is on another host, and refuses a request that still carries our token."""
        request = self._request(self._url(f"releases/assets/{asset_id}"), BINARY_ACCEPT)
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in REDIRECTS:
                raise
            location = exc.headers["Location"]
        if not location.startswith(("https://", self.config.api)):
            raise ValueError(f"refusing to download from {location}")
        with urllib.request.urlopen(location, timeout=REQUEST_TIMEOUT_S * 4) as response:
            return response.read()

    def post(self, path: str, payload: Mapping) -> dict:
        request = self._request(self._url(path), data=json.dumps(payload).encode(), kind="application/json")
        with self._opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read() or b"{}")

    def upload(self, release_id: int, name: str, data: bytes) -> None:
        """An asset already there means an earlier try worked and its reply was lost, so this one has too."""
        url = f"{self.config.uploads}/repos/{self.config.repo}/releases/{release_id}/assets?name={name}"
        request = self._request(url, data=data, kind="application/octet-stream")
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_S * 4) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != HTTP_ALREADY_EXISTS:
                raise
            logger.warning("%s was already uploaded", name)


def _read_json(path: Path) -> dict:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: Mapping) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _usable_release(saved) -> dict | None:
    """A saved release to run, if the record is sound and the file is still there. Otherwise forget it."""
    if not isinstance(saved, dict) or not isinstance(saved.get("tag"), str) or not isinstance(saved.get("path"), str):
        return None
    return saved if Path(saved["path"]).is_file() else None


def _started_watch(now: float, trips: int = 0, latched: bool = False) -> dict:
    return {"since": now, "trips": trips, "latched": latched, "crashes": 0, "seen_ok": False}


class ReleaseWatch:
    """Which file should be running. The parent asks it what to do; it never touches the child itself."""

    def __init__(
        self,
        config: UpdateConfig,
        github: GitHub,
        run_dir: Path,
        entry: Path,
        report: Callable[[str], None],
        wall: Callable[[], float] = time.time,
    ) -> None:
        # Absolute, since the self-check runs from a scratch folder.
        self.config, self.gh, self.run_dir, self.report, self.wall = config, github, Path(run_dir).resolve(), report, wall
        self.releases = self.run_dir / "releases"
        self.releases.mkdir(parents=True, exist_ok=True)
        self.entry_sha = _sha256_of(Path(entry)) if Path(entry).exists() else ""
        self._lock = threading.Lock()
        self._ready: dict | None = None
        saved = _read_json(self.run_dir / STATE_NAME)
        if saved.get("entry_sha") != self.entry_sha:  # a person uploaded a new first file: it wins
            saved = {}
        self.good: dict | None = _usable_release(saved.get("good"))
        self.current: dict | None = _usable_release(saved.get("current")) or self.good
        bad = saved.get("bad", ())
        self.bad: set[str] = {tag for tag in bad if isinstance(tag, str)} if isinstance(bad, list) else set()
        # Clocks do not survive a reboot, so a watch cut short starts again.
        self.trying: dict | None = _started_watch(time.monotonic()) if saved.get("trying") and self.current else None
        # What a person just uploaded is newer than any release before it, so only later ones count.
        started = saved.get("started_at")
        self.started_at: float = started if isinstance(started, (int, float)) else wall()
        self._save()

    @property
    def good_tag(self) -> str | None:
        return self.good["tag"] if self.good else None

    def _save(self) -> None:
        """A lost record costs at most a repeated download, so it must never stop a swap or a rollback."""
        state = {
            "entry_sha": self.entry_sha, "current": self.current, "good": self.good,
            "bad": sorted(self.bad), "trying": self.trying, "started_at": self.started_at,
        }
        with self._lock:  # the worker refuses releases while the parent swaps them
            try:
                _write_json(self.run_dir / STATE_NAME, state)
            except OSError as exc:
                logger.error("could not save update state: %s", exc)

    def _frozen(self) -> bool:
        return self.config.freeze_at is not None and self.wall() >= self.config.freeze_at

    # --- the worker thread: may block on the network ---------------------------------------------

    def check(self) -> None:
        if self._frozen():
            return
        body = self.gh.fetch("releases/latest")
        if body is None:
            return
        try:
            self._consider(json.loads(body))
        except Exception:
            self.gh.forget("releases/latest")  # so the next poll sees this release again
            raise

    def _consider(self, release: Mapping) -> None:
        tag = str(release["tag_name"])
        with self._lock:
            known = {self.current["tag"] if self.current else None, self._ready["tag"] if self._ready else None}
        if tag in self.bad or tag in known or self._older_than_this_run(release):
            return None
        if not SAFE_TAG.match(tag):
            return self._refuse(tag, "its tag cannot be a file name")
        assets = {asset["name"]: asset["id"] for asset in release.get("assets", ())}
        if BOT_ASSET not in assets or SHA_ASSET not in assets:
            return None
        code = self.gh.download(assets[BOT_ASSET])
        try:
            wanted = self.gh.download(assets[SHA_ASSET]).decode().split()[0]
        except (IndexError, UnicodeDecodeError):
            return self._refuse(tag, "its checksum file is unreadable")
        if hashlib.sha256(code).hexdigest() != wanted:
            return self._refuse(tag, "its checksum does not match")
        path = self.releases / f"{tag}.py"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(code)
        temporary.replace(path)
        failure = self._selfcheck_failure(path)
        if failure is not None:
            return self._refuse(tag, f"it failed its self-check: {failure}")
        with self._lock:
            self._ready = {"tag": tag, "path": str(path)}
        return None

    def _older_than_this_run(self, release: Mapping) -> bool:
        """Without this, a bot freshly uploaded from main would swap itself back to an older release."""
        published = release.get("published_at")
        return bool(published) and datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp() <= self.started_at

    def _refuse(self, tag: str, why: str) -> None:
        self.bad.add(tag)
        self._save()
        self.report(f"@A task=deploy from=updater\n{tag} refused: {why}. Still running {self._running_tag()}.")

    def _running_tag(self) -> str:
        return self.current["tag"] if self.current else "the first upload"

    @staticmethod
    def _selfcheck_failure(path: Path) -> str | None:
        """What the file printed, if it fails. A timeout or a busy machine raises instead, so the check is retried."""
        with tempfile.TemporaryDirectory() as scratch:
            result = subprocess.run(
                [sys.executable, str(path), "--selfcheck"], cwd=scratch, capture_output=True, text=True,
                timeout=SELFCHECK_TIMEOUT_S, check=False,
            )
        return None if result.returncode == 0 else result.stderr[-SELFCHECK_DETAIL_CHARS:]

    # --- the parent's thread: never blocks on the network ----------------------------------------

    def command(self, base: Sequence[str]) -> list[str]:
        for release in (self.current, self.good):
            if release and Path(release["path"]).is_file():
                return [base[0], release["path"], *base[2:]]
        return list(base)

    def action(self, now: float) -> str | None:
        if self.trying:
            return self._watch(now)
        with self._lock:
            if self._ready and self._frozen():  # checked before the freeze, too late to run
                self._ready = None
            return "swap" if self._ready else None

    def _watch(self, now: float) -> str | None:
        status = _read_json(self.run_dir / "status.json")
        if status.get("connected"):
            self.trying["seen_ok"] = True
        trouble = self._trouble(status, now)
        if trouble:
            self.trying = {**self.trying, "why": trouble}
            return "rollback"
        if now - self.trying["since"] < self.config.watch_s:
            return None
        if not self.trying["seen_ok"]:
            self.trying = {**self.trying, "why": "it never reported being connected"}
            return "rollback"
        self.good = self.current
        self.trying = None
        self._save()
        self.report(f"updater: {self.good_tag} is now the good version.")
        return None

    def _trouble(self, status: Mapping, now: float) -> str | None:
        if not status:
            return None
        kill = status.get("kill", {})
        if kill.get("latched") and not self.trying["latched"]:
            return "the kill switch latched"
        if kill.get("trips", 0) > self.trying["trips"]:
            return f"the kill switch tripped: {kill.get('reason')}"
        if status.get("errors", 0) >= WATCH_ERRORS:
            return f"{status['errors']} errors in the loop"
        if status.get("rejects", 0) >= WATCH_REJECTS:
            return f"{status['rejects']} rejected orders"
        if not status.get("connected", True) and now - self.trying["since"] > self.config.connect_grace_s:
            return "it is not connected"
        return None

    def apply(self, action: str, now: float) -> None:
        if action == "swap":
            self._swap(now)
        else:
            self._roll_back(self.trying.get("why", "trouble") if self.trying else "trouble")

    def _kill_counts(self) -> tuple[int, bool]:
        """From the switch's own record if the bot has not reported yet: it survives restarts, so must the count."""
        kill = _read_json(self.run_dir / "status.json").get("kill") or _read_json(self.run_dir / KILL_STATE_NAME)
        return int(kill.get("trips", 0)), bool(kill.get("latched"))

    def _swap(self, now: float) -> None:
        trips, latched = self._kill_counts()
        with self._lock:
            ready, self._ready = self._ready, None
        if ready is None:
            return
        self.current = ready
        self.trying = _started_watch(now, trips, latched)
        self._clear_status()
        self._save()
        self.report(f"updater: now running {ready['tag']}, watching it.")

    def _clear_status(self) -> None:
        """The old bot's numbers must not judge the new one."""
        try:
            (self.run_dir / "status.json").unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("could not clear the old status: %s", exc)

    def _roll_back(self, why: str) -> None:
        failed = self._running_tag()
        self.bad.add(failed)
        self.current, self.trying = self.good, None
        self._clear_status()
        self._save()
        self.report(f"@A task=deploy from=updater\n{failed} rolled back: {why}. Now running {self._running_tag()}.")

    def on_crash(self, _now: float) -> bool:
        """True if this crash condemned a new file, so the parent restarts with no back-off."""
        if not self.trying:
            return False
        self.trying["crashes"] = self.trying.get("crashes", 0) + 1
        if self.trying["crashes"] < WATCH_CRASHES:
            return False
        self._roll_back(f"it crashed {self.trying['crashes']} times")
        return True


def risk_cuts(old: Mapping, new: Mapping, base: Mapping | None = None) -> dict:
    """What the freeze lets through: `old`, with only the known risk knobs in `new` that move down applied.

    `base` is what the event settings run at, since a value set there is in no published file. Without it, a cut
    from size 20 to 15 reads as a raise from the default of 10, and a raise from 5 to 8 reads as a cut.
    Anything else in `new` is left out, so one refused increase cannot block a later cut published on top of it.
    """
    base = base or {}
    defaults = {**validate_params({}), **{key: value for key, value in base.items() if key != INSTRUMENTS_KEY}}
    cut = {key: value for key, value in old.items() if key != INSTRUMENTS_KEY}
    for key in DECREASE_ONLY:
        if _lower(new.get(key), old.get(key, defaults.get(key))):
            cut[key] = new[key]
    for key in STOPS:
        if _tighter(new.get(key), old.get(key, defaults.get(key))):
            cut[key] = new[key]
    instruments = {iid: dict(values) for iid, values in old.get(INSTRUMENTS_KEY, {}).items()}
    for iid, values in new.get(INSTRUMENTS_KEY, {}).items():
        for key in DECREASE_ONLY & set(values):
            fallback = base.get(INSTRUMENTS_KEY, {}).get(iid, {}).get(key, cut.get(key, defaults.get(key)))
            before = instruments.get(iid, {}).get(key, fallback)
            if _lower(values[key], before):
                instruments.setdefault(iid, {})[key] = values[key]
    if instruments:
        cut[INSTRUMENTS_KEY] = instruments
    return cut


def _lower(value, before) -> bool:
    return isinstance(value, (int, float)) and isinstance(before, (int, float)) and value < before


def _tighter(value, before) -> bool:
    if not isinstance(value, (int, float)) or value <= 0:
        return False
    return not isinstance(before, (int, float)) or before <= 0 or value < before


def _event_settings(run_dir: Path) -> dict:
    """The settings the bot was started with, as otk.main lays them under anything published."""
    event = _read_json(Path(run_dir) / EVENT_IN_FORCE)
    params = event.get("params")
    limit = {"position_limit": event["position_limit"]} if "position_limit" in event else {}
    return {**limit, **(params if isinstance(params, Mapping) else {})}


class LiveFiles:
    """`params.json` and `control.json`, from a branch, so every change has a commit saying why."""

    PARAMS = "contents/params.json?ref={branch}"
    CONTROL = "contents/control.json?ref={branch}"

    def __init__(
        self,
        config: UpdateConfig,
        github: GitHub,
        run_dir: Path,
        report: Callable[[str], None],
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.config, self.gh, self.run_dir, self.report, self.wall = config, github, Path(run_dir), report, wall
        self.made_path = self.run_dir / "control_state.json"
        self.made: set[str] = set(_read_json(self.made_path).get("made", ()))
        # A live test lays one variant over what is published. Both threads write the one file, so it takes a lock.
        self.published: dict | None = None
        self.overlay: dict = {}
        self.publishes = 0
        self._written: dict = {}
        self._write_lock = threading.Lock()

    def _write_in_force(self) -> None:
        path = self.run_dir / "params.json"
        _write_json(path, merge_params(self.published or {}, self.overlay))
        self._written = _read_json(path)

    def set_overlay(self, overlay: Mapping) -> None:
        with self._write_lock:
            on_disk = _read_json(self.run_dir / "params.json")
            edited = bool(self.overlay) and on_disk != self._written  # someone else wrote it during a test
            if edited:
                self.publishes += 1  # and so the test proves nothing
            if edited or not self.overlay:  # what is in force with nothing laid over it, which may include a person's edit
                self.published = on_disk
            self.overlay = {} if edited else dict(overlay)
            self._write_in_force()
        if edited and overlay:
            raise HandEdited("a person changed the parameter file, and what they wrote stays")

    def adopt(self, change: Mapping) -> None:
        """What a check for harm passed, on top of what is in force. Only for when no agent can publish it."""
        with self._write_lock:
            current = self.published if self.overlay else _read_json(self.run_dir / "params.json")
            self.published = merge_params(current or {}, change)
            self._write_in_force()

    def restore(self, published: Mapping) -> None:
        """After a restart cut a test short: its variant may still be in the file."""
        with self._write_lock:
            self.published, self.overlay = dict(published), {}
            self._write_in_force()

    def sync(self) -> None:
        self._sync(self.CONTROL, self._apply_control)
        self._sync(self.PARAMS, self._apply_params)

    def _sync(self, template: str, apply: Callable[[bytes], None]) -> None:
        """If this cannot be applied now, the next poll must try again, or a kill could be lost to one full disk."""
        path = template.format(branch=self.config.branch)
        body = self.gh.fetch(path, RAW_ACCEPT)
        if body is None:
            return
        try:
            apply(body)
        except Exception:
            self.gh.forget(path)
            raise

    def _apply_params(self, body: bytes) -> None:
        try:
            wanted = json.loads(body)
            validate_params(wanted)
        except ValueError as exc:
            self.report(f"@A task=params from=updater\nPublished parameters refused, keeping the last good set: {exc}")
            return
        path = self.run_dir / "params.json"
        with self._write_lock:
            if self.config.freeze_at is not None and self.wall() >= self.config.freeze_at:
                current = self.published if self.overlay else _read_json(path)
                cut = risk_cuts(current or {}, wanted, _event_settings(self.run_dir))
                if cut != wanted:
                    self.report("@A task=params from=updater\nPast the freeze: only risk reductions were applied.")
                wanted = cut
            self.publishes += 1
            self.published = wanted
            self._write_in_force()

    def _apply_control(self, body: bytes) -> None:
        try:
            wanted = json.loads(body)
        except ValueError:
            wanted = None
        if not isinstance(wanted, dict):
            self.report("@A task=control from=updater\nPublished control refused: it must be a JSON object.")
            return
        failures = []
        for key, name in (("kill", "KILL"), ("flatten", "FLATTEN")):  # each on its own: one failing must not skip the other
            try:
                self._set_flag(name, bool(wanted.get(key)))
            except OSError as exc:
                failures.append(exc)
        _write_json(self.made_path, {"made": sorted(self.made)})
        if failures:
            raise failures[0]

    def _set_flag(self, name: str, on: bool) -> None:
        path = self.run_dir / name
        if on:
            if not path.exists():
                path.touch()
                self.made.add(name)
        elif name in self.made:  # a kill a person set stays until a person clears it
            path.unlink(missing_ok=True)
            self.made.discard(name)


class Uplink:
    """Status and captures go to assets on a private prerelease, so agents can read them and git stays clean.

    Only new bytes of a growing file go up. Agents join the parts in order to get the file back.
    """

    def __init__(self, config: UpdateConfig, github: GitHub, run_dir: Path, report: Callable[[str], None] = lambda _text: None) -> None:
        self.config, self.gh, self.run_dir, self.report = config, github, Path(run_dir), report
        self.tag = config.data_tag or "data-" + time.strftime("%Y%m%d")
        self.state_path = self.run_dir / UPLINK_STATE_NAME
        saved = _read_json(self.state_path)
        self.status_seq: int = saved.get("status_seq", 0)
        self.offsets: dict[str, int] = saved.get("offsets", {})
        self.parts: dict[str, int] = saved.get("parts", {})
        self.tests_sent: list[str] = saved.get("tests_sent", [])
        self._release_id: int | None = None

    def _release(self) -> int:
        if self._release_id is None:
            found = self.gh.read(f"releases/tags/{self.tag}")
            if found is None:
                found = self.gh.post("releases", {"tag_name": self.tag, "name": self.tag, "prerelease": True, "body": "Live data"})
            self._release_id = int(found["id"])
        return self._release_id

    def _save(self) -> None:
        _write_json(self.state_path, {"status_seq": self.status_seq, "offsets": self.offsets, "parts": self.parts, "tests_sent": self.tests_sent})

    def send_tests(self) -> None:
        """A finished live test goes up once, and only then is the manager told, so the file is there to read."""
        for path in sorted(self.run_dir.glob("test-*.json")):
            if path.name in self.tests_sent:
                continue
            result = _read_json(path)
            if not result.get("done"):
                continue
            self.gh.upload(self._release(), path.name, path.read_bytes())
            self.tests_sent.append(path.name)
            self._save()
            self.report(result_report(result))

    def send_status(self) -> None:
        path = self.run_dir / "status.json"
        if not path.exists():
            return
        self.gh.upload(self._release(), f"status-{self.status_seq:05d}.json", path.read_bytes())
        self.status_seq += 1
        self._save()

    def send_data(self) -> None:
        for name in TAILED_FILES:
            self._send_tail(name)

    def _send_tail(self, name: str) -> None:
        path = self.run_dir / name
        if not path.exists():
            return
        offset = self.offsets.get(name, 0)
        if path.stat().st_size < offset:
            offset = 0  # replaced by a new file
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(MAX_PART_BYTES)
        if not data:
            return
        part = self.parts.get(name, 0)
        self.gh.upload(self._release(), f"{name}.{part:05d}.part", data)
        self.offsets[name], self.parts[name] = offset + len(data), part + 1
        self._save()


class Updater:
    """Ties the pieces together: the parent calls `command`, `action`, `apply` and `on_crash`; threads do the rest."""

    def __init__(self, config: UpdateConfig, run_dir: Path, entry: Path) -> None:
        self.config, self.run_dir = config, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.gh = GitHub(config)
        self.reports: list[str] = []
        self._report_tries = 0
        self.watch = ReleaseWatch(config, self.gh, self.run_dir, Path(entry), self._report)
        self.files = LiveFiles(config, self.gh, self.run_dir, self._report)
        self.uplink = Uplink(config, self.gh, self.run_dir, self._report)
        self.tests = LiveTests(config, self.gh, self.run_dir, self.files, self._report)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _report(self, text: str) -> None:
        self.reports.append(text[:MAX_REPORT_CHARS])
        del self.reports[:-MAX_REPORTS]

    # --- called by the parent. Whatever goes wrong here, the parent must keep a bot running ---------

    def command(self, base: Sequence[str]) -> list[str]:
        try:
            return self.watch.command(base)
        except Exception:
            logger.exception("updater: no command, running the first upload")
            return list(base)

    def action(self, now: float) -> str | None:
        try:
            wanted = self.watch.action(now)
            # A swap restarts the child and would spoil a live test. It waits. A rollback does not.
            return None if wanted == "swap" and self.tests.running() else wanted
        except Exception:
            logger.exception("updater: action failed, carrying on")
            return None

    def apply(self, action: str, now: float) -> None:
        """The child is already stopped by now. If this fails, the next start runs the first upload."""
        try:
            self.watch.apply(action, now)
        except Exception:
            logger.exception("updater: %s failed, falling back to the first upload", action)
            self.watch.current, self.watch.trying = None, None

    def on_crash(self, now: float) -> bool:
        try:
            return self.watch.on_crash(now)
        except Exception:
            logger.exception("updater: could not judge a crash")
            return False

    # --- the threads --------------------------------------------------------------------------------

    def start(self) -> None:
        # Parameters and kills get a thread of their own: a release's self-check can take two minutes.
        groups = (
            [(self.config.params_poll_s, self.files.sync, "parameters"), (self.config.params_poll_s, self.tests.sync, "live test")],
            [
                (self.config.code_poll_s, self.watch.check, "release"),
                (self.config.status_s, self.uplink.send_status, "status"),
                (self.config.status_s, self.uplink.send_tests, "test results"),
                (self.config.data_s, self.uplink.send_data, "data"),
                (5.0, self._send_reports, "reports"),
            ],
        )
        self._threads = [threading.Thread(target=self._run, args=(tasks,), name="updater", daemon=True) for tasks in groups]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.tests.stop("the parent is stopping")

    def worker_alive(self) -> bool:
        return bool(self._threads) and all(thread.is_alive() for thread in self._threads)

    def _send_reports(self) -> None:
        while self.reports:
            text = self.reports[0]
            logger.info(text.replace("\n", " "))
            try:
                if self.config.issue:
                    self.gh.post(f"issues/{self.config.issue}/comments", {"body": text})
            except (OSError, ValueError):
                self._report_tries += 1
                if self._report_tries < REPORT_TRIES:
                    raise
                logger.error("updater: giving up on a report: %s", text[:200])  # or it would block every later one
            self._report_tries = 0
            self.reports.pop(0)

    def _run(self, tasks: list) -> None:
        due = [0.0] * len(tasks)
        while not self._stop.is_set():
            for index, (interval, work, name) in enumerate(tasks):
                if time.monotonic() >= due[index]:
                    due[index] = time.monotonic() + interval
                    self._safely(work, name)
            self._stop.wait(min(0.2, min(task[0] for task in tasks)))

    @staticmethod
    def _safely(work: Callable[[], None], name: str) -> None:
        """Nothing out here may affect trading, so a failure is logged and tried again at the next interval."""
        try:
            work()
        except Exception as exc:  # noqa: BLE001 - this thread must outlive anything GitHub does
            logger.warning("updater: %s failed: %s: %s", name, type(exc).__name__, exc)


def _read_token(settings: Mapping) -> str:
    token = os.environ.pop(TOKEN_ENV, "")  # the bot's child process has no use for it
    if token:
        return token.strip()
    try:
        return Path(settings.get("token_file", DEFAULT_TOKEN_FILE)).expanduser().read_text().strip()
    except OSError:
        return ""


def parse_freeze(text: str | None) -> float | None:
    """With a UTC offset, or refuse: the devenv's clock may be in another zone, and a wrong freeze is a silent one."""
    if not text:
        return None
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError(f"freeze_at needs a UTC offset: {text}")
    return moment.timestamp()


NUMERIC_SETTINGS = ("code_poll_s", "params_poll_s", "status_s", "data_s", "watch_s", "connect_grace_s")
TEXT_SETTINGS = ("api", "uploads", "branch", "data_tag")


def build_updater(event: Mapping, run_dir: Path, entry: Path) -> Updater | None:
    """None if it is not set up or the setup is broken. The bot then runs as uploaded, which is always safe."""
    settings = event.get("update")
    if not settings:
        return None
    token = _read_token(settings)
    if not token:
        logger.error("updater off: no token in %s or %s", TOKEN_ENV, settings.get("token_file", DEFAULT_TOKEN_FILE))
        return None
    try:
        config = UpdateConfig(
            repo=str(settings["repo"]),
            token=token,
            issue=settings.get("issue"),
            freeze_at=parse_freeze(settings.get("freeze_at")),
            **{key: float(settings[key]) for key in NUMERIC_SETTINGS if key in settings},
            **{key: str(settings[key]) for key in TEXT_SETTINGS if key in settings},
        )
        return Updater(config, run_dir, entry)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        logger.error("updater off: bad settings: %s: %s", type(exc).__name__, exc)
        return None
