"""The bot updates itself: merged code and parameters arrive with nobody touching it, and a bad one rolls back."""

import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_single_file

from otk.livetest import HandEdited
from otk.main import main as bot_main
from otk.main import supervise
from otk.updater import (
    GitHub,
    LiveFiles,
    ReleaseWatch,
    UpdateConfig,
    Updater,
    Uplink,
    build_updater,
    parse_freeze,
    risk_cuts,
)

TOKEN = "ghp_secret_token"
GOOD = "import sys\nsys.exit(0)\n"
BROKEN = "import sys\nsys.exit(1)\n"


class FakeGitHub:
    """Just enough of GitHub, over real HTTP, so redirects, ETags and headers are really exercised."""

    def __init__(self):
        self.latest = None
        self.assets = {}
        self.files = {}
        self.releases = {}
        self.uploaded = {}
        self.comments = []
        self.requests = []
        self.fail = False
        self.saw_auth_on_storage = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _send(self, code, body=b"", headers=None):
                self.send_response(code)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _conditional(self, body):
                etag = '"' + hashlib.md5(body).hexdigest() + '"'
                if self.headers.get("If-None-Match") == etag:
                    return self._send(304)
                return self._send(200, body, {"ETag": etag})

            def do_GET(self):
                owner.requests.append(("GET", self.path, self.headers.get("Authorization")))
                if owner.fail:
                    return self._send(500)
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/repos/o/r/releases/latest":
                    if owner.latest is None:
                        return self._send(404)
                    return self._conditional(json.dumps(owner.latest).encode())
                if path.startswith("/repos/o/r/releases/assets/"):
                    asset = path.rsplit("/", 1)[1]
                    return self._send(302, b"", {"Location": f"http://127.0.0.1:{owner.port}/storage/{asset}"})
                if path.startswith("/storage/"):
                    owner.saw_auth_on_storage |= self.headers.get("Authorization") is not None
                    return self._send(200, owner.assets[path.rsplit("/", 1)[1]])
                if path.startswith("/repos/o/r/contents/"):
                    name = path.rsplit("/", 1)[1]
                    assert parse_qs(parsed.query)["ref"] == ["live"]
                    if name not in owner.files:
                        return self._send(404)
                    return self._conditional(owner.files[name].encode())
                if path.startswith("/repos/o/r/releases/tags/"):
                    tag = path.rsplit("/", 1)[1]
                    if tag not in owner.releases:
                        return self._send(404)
                    return self._send(200, json.dumps(owner.releases[tag]).encode())
                return self._send(404)

            def do_POST(self):
                owner.requests.append(("POST", self.path, self.headers.get("Authorization")))
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if owner.fail:
                    return self._send(500)
                parsed = urlparse(self.path)
                if parsed.path == "/repos/o/r/releases":
                    data = json.loads(body)
                    assert data["prerelease"] is True, "a data release must never become the latest bot"
                    owner.releases[data["tag_name"]] = {"id": 77, "tag_name": data["tag_name"]}
                    return self._send(201, json.dumps(owner.releases[data["tag_name"]]).encode())
                if parsed.path == "/repos/o/r/releases/77/assets":
                    owner.uploaded[parse_qs(parsed.query)["name"][0]] = body
                    return self._send(201, b"{}")
                if parsed.path == "/repos/o/r/issues/5/comments":
                    owner.comments.append(json.loads(body)["body"])
                    return self._send(201, b"{}")
                return self._send(404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, tag, text, sha=None, published=None):
        number = len(self.assets) + 1
        data = text.encode()
        self.assets[str(number)] = data
        self.assets[str(number + 1)] = (sha or hashlib.sha256(data).hexdigest()).encode()
        self.latest = {
            "tag_name": tag,
            "published_at": published or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "assets": [
                {"id": number, "name": "optibook_bot.py"},
                {"id": number + 1, "name": "optibook_bot.py.sha256"},
            ],
        }

    def count(self, fragment):
        return sum(fragment in request[1] for request in self.requests)


@pytest.fixture
def hub():
    server = FakeGitHub()
    yield server
    server.server.shutdown()


def config(hub, **over):
    values = {"repo": "o/r", "token": TOKEN, "api": hub.url, "uploads": hub.url, "issue": 5}
    return UpdateConfig(**{**values, **over})


def make_watch(hub, tmp_path, now=lambda: 1000.0, wall=lambda: 1000.0, **over):
    entry = tmp_path / "first.py"
    entry.write_text("# first upload\n")
    cfg = config(hub, **over)
    reports = []
    watch = ReleaseWatch(cfg, GitHub(cfg), tmp_path / "run", entry, reports.append, wall=wall)
    return watch, reports


def write_status(run, **values):
    (run / "status.json").write_text(json.dumps({
        "connected": True, "errors": 0, "rejects": 0, "kill": {"trips": 0, "latched": False}, **values,
    }))


# --- getting a new version -------------------------------------------------------------------------


def test_a_new_release_is_checked_and_made_ready(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    assert watch.action(1000.0) == "swap"


def test_nothing_published_means_nothing_happens(hub, tmp_path):
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    assert watch.action(1000.0) is None


def test_the_token_is_not_sent_to_the_storage_a_download_redirects_to(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    assert watch.action(1000.0) == "swap"
    assert not hub.saw_auth_on_storage
    assert any(auth == f"Bearer {TOKEN}" for _, path, auth in hub.requests if "assets/" in path)


def test_a_file_that_does_not_match_its_checksum_is_refused_for_good(hub, tmp_path):
    hub.publish("bot-1", GOOD, sha="0" * 64)
    watch, reports = make_watch(hub, tmp_path)
    watch.check()
    watch.check()
    assert watch.action(1000.0) is None
    assert "checksum" in reports[0]
    assert len(reports) == 1


def test_a_file_that_fails_its_self_check_is_refused_for_good(hub, tmp_path):
    hub.publish("bot-1", BROKEN)
    watch, reports = make_watch(hub, tmp_path)
    watch.check()
    assert watch.action(1000.0) is None
    assert "self-check" in reports[0]
    hub.requests.clear()
    watch.check()
    assert hub.count("assets/") == 0, "not downloaded again"


def test_a_relative_run_folder_does_not_break_the_self_check(hub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "first.py").write_text("# first\n")
    hub.publish("bot-1", GOOD)
    cfg = config(hub)
    watch = ReleaseWatch(cfg, GitHub(cfg), Path("run"), Path("first.py"), lambda _text: None, wall=lambda: 1000.0)
    watch.check()
    assert watch.action(1000.0) == "swap"


def test_a_refusal_says_what_the_self_check_printed(hub, tmp_path):
    hub.publish("bot-1", "import sys\nprint('the reason', file=sys.stderr)\nsys.exit(1)\n")
    watch, reports = make_watch(hub, tmp_path)
    watch.check()
    assert "the reason" in reports[0]


def test_a_failed_download_is_tried_again(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    real = watch.gh.download
    watch.gh.download = lambda _asset: (_ for _ in ()).throw(OSError("network"))
    with pytest.raises(OSError):
        watch.check()
    watch.gh.download = real
    watch.check()
    assert watch.action(1000.0) == "swap"


def test_polling_an_unchanged_release_downloads_nothing(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    watch.apply("swap", 1000.0)
    hub.requests.clear()
    watch.check()
    assert hub.count("assets/") == 0


def test_no_new_code_after_the_freeze(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, reports = make_watch(hub, tmp_path, freeze_at=500.0)
    watch.check()
    assert watch.action(1000.0) is None
    assert hub.count("assets/") == 0
    assert not reports


def test_a_slow_self_check_is_retried_rather_than_condemned(hub, tmp_path, monkeypatch):
    hub.publish("bot-1", GOOD)
    watch, reports = make_watch(hub, tmp_path)
    monkeypatch.setattr("otk.updater.SELFCHECK_TIMEOUT_S", 0.0)
    with pytest.raises(subprocess.TimeoutExpired):
        watch.check()
    assert not reports
    monkeypatch.setattr("otk.updater.SELFCHECK_TIMEOUT_S", 60.0)
    watch.check()
    assert watch.action(1000.0) == "swap"


def test_a_bot_just_uploaded_never_swaps_itself_back_to_an_older_release(hub, tmp_path):
    hub.publish("bot-1", GOOD, published="2026-11-06T15:00:00Z")
    watch, reports = make_watch(hub, tmp_path, wall=lambda: datetime(2026, 11, 6, 15, 30, tzinfo=timezone.utc).timestamp())
    watch.check()
    assert watch.action(1000.0) is None
    assert hub.count("assets/") == 0
    assert not reports


def test_a_release_after_the_start_is_taken(hub, tmp_path):
    hub.publish("bot-2", GOOD, published="2026-11-06T16:00:00Z")
    watch, _ = make_watch(hub, tmp_path, wall=lambda: datetime(2026, 11, 6, 15, 30, tzinfo=timezone.utc).timestamp())
    watch.check()
    assert watch.action(1000.0) == "swap"


def test_a_parent_restart_keeps_the_original_start_time(hub, tmp_path):
    first, _ = make_watch(hub, tmp_path, wall=lambda: 1000.0)
    again, _ = make_watch(hub, tmp_path, wall=lambda: 5000.0)
    assert again.started_at == first.started_at == 1000.0


def test_a_tag_that_cannot_be_a_file_name_is_refused_once(hub, tmp_path):
    hub.publish("release/1", GOOD)
    watch, reports = make_watch(hub, tmp_path)
    watch.check()
    hub.requests.clear()
    watch.check()
    assert "file name" in reports[0]
    assert hub.count("assets/") == 0


def test_an_unreadable_checksum_is_refused_once(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    number = hub.latest["assets"][1]["id"]
    hub.assets[str(number)] = b""
    watch, reports = make_watch(hub, tmp_path)
    watch.check()
    assert "unreadable" in reports[0]


def test_a_redirect_from_the_api_is_not_followed_with_the_token(hub, tmp_path):
    other = FakeGitHub()
    original = hub.server.RequestHandlerClass.do_GET

    def redirecting(handler):
        if "releases/latest" in handler.path:
            handler._send(301, b"", {"Location": f"{other.url}/repos/o/r/releases/latest"})
            return
        original(handler)

    hub.server.RequestHandlerClass.do_GET = redirecting
    cfg = config(hub)
    with pytest.raises(OSError):
        GitHub(cfg).fetch("releases/latest")
    assert not other.requests
    other.server.shutdown()


def test_a_download_is_only_followed_to_https_or_our_own_host(hub, tmp_path):
    cfg = config(hub, api="https://api.example")
    gh = GitHub(cfg)

    class Redirect:
        def open(self, request, timeout):
            import email.message
            import urllib.error

            headers = email.message.Message()
            headers["Location"] = "file:///etc/passwd"
            raise urllib.error.HTTPError(request.full_url, 302, "found", headers, None)

    gh._opener = Redirect()
    with pytest.raises(ValueError):
        gh.download(1)


# --- swapping and rolling back ---------------------------------------------------------------------


def swapped(hub, tmp_path, tag="bot-1", text=GOOD):
    hub.publish(tag, text)
    watch, reports = make_watch(hub, tmp_path)
    watch.check()
    assert watch.action(1000.0) == "swap"
    watch.apply("swap", 1000.0)
    return watch, reports


def test_a_swap_runs_the_new_file_in_place_of_the_first(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    command = watch.command(["python", "first.py", "--parent-pid", "9"])
    assert command[0] == "python"
    assert command[1].endswith("bot-1.py")
    assert command[2:] == ["--parent-pid", "9"]


def test_a_quiet_watch_makes_the_new_file_the_good_one(hub, tmp_path):
    watch, reports = swapped(hub, tmp_path)
    write_status(watch.run_dir)
    assert watch.action(1100.0) is None
    assert watch.action(1000.0 + watch.config.watch_s + 1) is None
    assert watch.good_tag == "bot-1"
    assert any("good" in report for report in reports)


@pytest.mark.parametrize(
    "bad",
    [
        {"kill": {"trips": 1, "latched": False}},
        {"kill": {"trips": 0, "latched": True}},
        {"errors": 5},
        {"rejects": 30},
    ],
)
def test_trouble_during_the_watch_rolls_back(hub, tmp_path, bad):
    watch, reports = swapped(hub, tmp_path)
    write_status(watch.run_dir, **bad)
    assert watch.action(1010.0) == "rollback"
    watch.apply("rollback", 1010.0)
    assert watch.command(["python", "first.py"]) == ["python", "first.py"]
    assert any(report.startswith("@A") and "bot-1" in report for report in reports)


def test_being_disconnected_rolls_back_only_after_a_grace_period(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    write_status(watch.run_dir, connected=False)
    assert watch.action(1010.0) is None
    assert watch.action(1000.0 + watch.config.connect_grace_s + 1) == "rollback"


def test_status_left_by_the_old_bot_does_not_count_against_the_new_one(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    write_status(watch.run_dir, errors=50)
    watch.apply("swap", 1000.0)
    assert watch.action(1010.0) is None


def test_trips_from_before_the_swap_do_not_count(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    write_status(watch.run_dir, kill={"trips": 1, "latched": False})
    watch.apply("swap", 1000.0)
    write_status(watch.run_dir, kill={"trips": 1, "latched": False})
    assert watch.action(1010.0) is None


def test_a_latch_from_before_the_swap_does_not_roll_back(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    write_status(watch.run_dir, kill={"trips": 3, "latched": True})
    watch.apply("swap", 1000.0)
    write_status(watch.run_dir, kill={"trips": 3, "latched": True})
    assert watch.action(1010.0) is None


def test_one_crash_during_the_watch_gets_a_second_chance_and_two_roll_back(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    assert watch.on_crash(1010.0) is False, "it may be the exchange, not the code"
    assert watch.on_crash(1020.0) is True
    assert watch.command(["python", "first.py"]) == ["python", "first.py"]


def test_a_crash_after_the_watch_just_restarts_the_good_file(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    write_status(watch.run_dir)
    watch.action(1000.0 + watch.config.watch_s + 1)
    assert watch.on_crash(5000.0) is False
    assert watch.command(["python", "first.py"])[1].endswith("bot-1.py")


def test_a_rolled_back_release_is_never_taken_again(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    write_status(watch.run_dir, errors=9)
    watch.apply(watch.action(1010.0), 1010.0)
    hub.requests.clear()
    watch.check()
    assert watch.action(1020.0) is None
    assert hub.count("assets/") == 0


def test_rollback_goes_to_the_last_good_release_not_the_first(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    write_status(watch.run_dir)
    watch.action(1000.0 + watch.config.watch_s + 1)
    hub.publish("bot-2", GOOD + "# two\n")
    watch.check()
    watch.apply(watch.action(2000.0), 2000.0)
    write_status(watch.run_dir, errors=9)
    watch.apply(watch.action(2010.0), 2010.0)
    assert watch.command(["python", "first.py"])[1].endswith("bot-1.py")


def test_no_new_swap_while_one_is_being_watched(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    write_status(watch.run_dir)
    hub.publish("bot-2", GOOD + "# two\n")
    watch.check()
    assert watch.action(1010.0) is None
    assert watch.action(1000.0 + watch.config.watch_s + 1) is None  # this one settles the watch
    assert watch.action(1000.0 + watch.config.watch_s + 2) == "swap"


def test_a_restart_of_the_parent_carries_on_from_where_it_was(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    write_status(watch.run_dir)
    watch.action(1000.0 + watch.config.watch_s + 1)
    again, _ = make_watch(hub, tmp_path)
    assert again.command(["python", "first.py"])[1].endswith("bot-1.py")
    assert again.good_tag == "bot-1"


def test_a_new_first_upload_starts_clean(hub, tmp_path):
    swapped(hub, tmp_path)
    entry = tmp_path / "first.py"
    entry.write_text("# a different first upload\n")
    cfg = config(hub)
    fresh = ReleaseWatch(cfg, GitHub(cfg), tmp_path / "run", entry, lambda _text: None, wall=lambda: 1000.0)
    assert fresh.command(["python", "first.py"]) == ["python", "first.py"]


def test_no_status_for_the_whole_watch_is_not_evidence_of_health(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    assert watch.action(1000.0 + watch.config.watch_s + 1) == "rollback"


def test_a_release_checked_before_the_freeze_does_not_run_after_it(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    clock = [1000.0]
    watch, _ = make_watch(hub, tmp_path, wall=lambda: clock[0], freeze_at=2000.0)
    watch.check()
    clock[0] = 2001.0
    assert watch.action(1.0) is None


def test_trips_are_read_from_the_switch_when_the_new_bot_has_not_reported_yet(hub, tmp_path):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    (watch.run_dir / "kill_state.json").write_text(json.dumps({"owner": 1, "trips": 2, "latched": False, "reason": ""}))
    watch.apply("swap", 1000.0)
    write_status(watch.run_dir, kill={"trips": 2, "latched": False})
    assert watch.action(1010.0) is None, "an old trip must not condemn an innocent release"


@pytest.mark.parametrize(
    "state",
    [
        {"current": {"tag": "t"}},
        {"current": "nonsense"},
        {"current": {"tag": "t", "path": "/nowhere/absent.py"}},
        {"trying": {"odd": 1}},
        {"trying": True, "bad": "nonsense"},
    ],
)
def test_a_broken_state_file_never_stops_the_parent(hub, tmp_path, state):
    watch, _ = make_watch(hub, tmp_path)
    (watch.run_dir / "update_state.json").write_text(json.dumps({"entry_sha": watch.entry_sha, **state}))
    again, _ = make_watch(hub, tmp_path)
    assert again.command(["python", "first.py"]) == ["python", "first.py"]
    assert again.action(1000.0) is None


def test_a_release_file_that_vanished_falls_back_rather_than_failing_forever(hub, tmp_path):
    watch, _ = swapped(hub, tmp_path)
    write_status(watch.run_dir)
    watch.action(1000.0 + watch.config.watch_s + 1)
    Path(watch.current["path"]).unlink()
    assert watch.command(["python", "first.py"]) == ["python", "first.py"]


def test_a_watch_cut_short_by_a_restart_starts_again(hub, tmp_path):
    swapped(hub, tmp_path)
    again, _ = make_watch(hub, tmp_path)
    assert again.trying["since"] == pytest.approx(time.monotonic(), abs=5.0)


def test_failing_to_save_state_does_not_stop_a_swap(hub, tmp_path, monkeypatch):
    hub.publish("bot-1", GOOD)
    watch, _ = make_watch(hub, tmp_path)
    watch.check()
    monkeypatch.setattr("otk.updater._write_json", lambda *_a: (_ for _ in ()).throw(OSError("disk full")))
    watch.apply("swap", 1000.0)
    assert watch.command(["python", "first.py"])[1].endswith("bot-1.py")


@pytest.mark.parametrize("name", ["command", "action", "apply", "on_crash"])
def test_nothing_the_parent_calls_can_raise(tmp_path, monkeypatch, name):
    monkeypatch.setenv("OPTIBOOK_GH_TOKEN", TOKEN)
    entry = tmp_path / "first.py"
    entry.write_text("# first\n")
    updater = build_updater({"update": {"repo": "o/r"}}, tmp_path / "run", entry)
    boom = lambda *_a: 1 / 0
    monkeypatch.setattr(updater.watch, name, boom)
    args = {"command": (["python", "first.py"],), "action": (1.0,), "apply": ("swap", 1.0), "on_crash": (1.0,)}[name]
    getattr(updater, name)(*args)
    assert updater.command(["python", "first.py"])


def test_a_kill_does_not_wait_for_a_slow_release_check(hub, tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIBOOK_GH_TOKEN", TOKEN)
    entry = tmp_path / "first.py"
    entry.write_text("# first\n")
    event = {"update": {"repo": "o/r", "api": hub.url, "uploads": hub.url, "params_poll_s": 0.05, "code_poll_s": 0.05}}
    updater = build_updater(event, tmp_path / "run", entry)
    release = threading.Event()
    monkeypatch.setattr(updater.watch, "check", lambda: release.wait(10.0))
    updater.start()
    hub.files["control.json"] = json.dumps({"kill": True})
    deadline = time.monotonic() + 5.0
    while not (tmp_path / "run" / "KILL").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    release.set()
    updater.stop()
    assert (tmp_path / "run" / "KILL").exists()


def test_one_report_that_cannot_be_sent_does_not_block_the_rest_for_ever(hub, tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIBOOK_GH_TOKEN", TOKEN)
    entry = tmp_path / "first.py"
    entry.write_text("# first\n")
    updater = build_updater({"update": {"repo": "o/r", "api": hub.url, "uploads": hub.url, "issue": 404}}, tmp_path / "run", entry)
    updater.reports.extend(["one", "two"])
    for _ in range(12):
        try:
            updater._send_reports()
        except OSError:
            pass
    assert not updater.reports


@pytest.mark.parametrize(
    "text,good",
    [
        ("2026-11-06T21:00:00+01:00", True), ("2026-11-06T20:00:00Z", True), ("2026-11-06T21:00:00", False), ("soon", False),
    ],
)
def test_a_freeze_time_needs_a_zone(text, good):
    if good:
        assert parse_freeze(text) == pytest.approx(1793304000.0, abs=86400 * 400)
    else:
        with pytest.raises(ValueError):
            parse_freeze(text)


# --- a live test, from the branch to the manager ------------------------------------------------------


def make_updater(hub, tmp_path, **over):
    entry = tmp_path / "first.py"
    entry.write_text(GOOD)
    return Updater(config(hub, **over), tmp_path / "run", entry)


def test_a_swap_waits_for_a_running_live_test_but_a_rollback_does_not(hub, tmp_path):
    updater = make_updater(hub, tmp_path)
    updater.tests.running = lambda: True
    updater.watch.action = lambda _now: "swap"
    assert updater.action(1000.0) is None
    updater.watch.action = lambda _now: "rollback"
    assert updater.action(1000.0) == "rollback"
    updater.tests.running = lambda: False
    updater.watch.action = lambda _now: "swap"
    assert updater.action(1000.0) == "swap"


def test_a_request_on_the_branch_ends_with_the_result_uploaded_and_the_manager_told(hub, tmp_path):
    updater = make_updater(hub, tmp_path, data_tag="data-1")
    run = tmp_path / "run"
    (run / "params.json").write_text('{"size": 5}')
    hub.files["test.json"] = json.dumps({"id": "exp-1", "variants": {"A": {}, "B": {"size": 3}}, "duration_s": 0.01})
    updater.tests.sync()
    deadline = time.monotonic() + 10
    while updater.tests.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not updater.tests.running()
    updater.uplink.send_tests()
    updater._send_reports()
    assert "test-exp-1.json" in hub.uploaded
    assert json.loads(hub.uploaded["test-exp-1.json"])["decision"]["verdict"] == "NO CHANGE"
    assert hub.comments[0].startswith("@A task=exp-1 from=updater")
    assert "DECISION: NO CHANGE" in hub.comments[0]
    assert json.loads((run / "params.json").read_text()) == {"size": 5}


def test_the_parent_stopping_stops_a_running_test(hub, tmp_path):
    updater = make_updater(hub, tmp_path)
    (tmp_path / "run" / "params.json").write_text('{"size": 5}')
    hub.files["test.json"] = json.dumps({"id": "exp-1", "variants": {"A": {}, "B": {"size": 3}}})
    updater.tests.sync()
    assert updater.tests.running()
    updater.stop()
    deadline = time.monotonic() + 10
    while updater.tests.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not updater.tests.running()
    assert json.loads((tmp_path / "run" / "params.json").read_text()) == {"size": 5}


def test_a_person_editing_the_file_by_hand_is_what_a_test_comes_back_to(hub, tmp_path):
    files, run, _ = live(hub, tmp_path)
    hub.files["params.json"] = json.dumps({"size": 7})
    files.sync()
    (run / "params.json").write_text('{"size": 2}')
    files.set_overlay({"skew_ticks": 2})
    assert json.loads((run / "params.json").read_text()) == {"size": 2, "skew_ticks": 2}
    files.set_overlay({})
    assert json.loads((run / "params.json").read_text()) == {"size": 2}


def test_past_the_freeze_a_cut_applies_to_what_was_published_not_to_a_test_variant(hub, tmp_path):
    files, run, _ = live(hub, tmp_path, freeze_at=500.0)
    (run / "params.json").write_text('{"size": 8}')
    files.set_overlay({"size": 20})
    hub.files["params.json"] = json.dumps({"size": 5})
    files.sync()
    assert json.loads((run / "params.json").read_text()) == {"size": 20}, "the test still runs until it stops itself"
    files.set_overlay({})
    assert json.loads((run / "params.json").read_text()) == {"size": 5}


# --- parameters and control ------------------------------------------------------------------------


def live(hub, tmp_path, **over):
    cfg = config(hub, **over)
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    reports = []
    return LiveFiles(cfg, GitHub(cfg), run, reports.append, wall=lambda: 1000.0), run, reports


def test_published_parameters_reach_the_file_the_bot_reads(hub, tmp_path):
    hub.files["params.json"] = json.dumps({"size": 7})
    files, run, _ = live(hub, tmp_path)
    files.sync()
    assert json.loads((run / "params.json").read_text()) == {"size": 7}


def test_invalid_parameters_are_refused_and_the_last_good_set_stays(hub, tmp_path):
    hub.files["params.json"] = json.dumps({"size": 7})
    files, run, _ = live(hub, tmp_path)
    files.sync()
    hub.files["params.json"] = json.dumps({"nonsense": 1})
    files.sync()
    assert json.loads((run / "params.json").read_text()) == {"size": 7}
    hub.files["params.json"] = "not json"
    files.sync()
    assert json.loads((run / "params.json").read_text()) == {"size": 7}


def test_after_the_freeze_only_risk_reductions_apply(hub, tmp_path):
    hub.files["params.json"] = json.dumps({"size": 7})
    files, run, reports = live(hub, tmp_path, freeze_at=500.0)
    (run / "params.json").write_text(json.dumps({"size": 8}))
    files.sync()
    assert json.loads((run / "params.json").read_text()) == {"size": 7}
    hub.files["params.json"] = json.dumps({"size": 9})
    files.sync()
    assert json.loads((run / "params.json").read_text()) == {"size": 7}
    assert any("freeze" in report for report in reports)


def test_a_refused_increase_cannot_block_a_later_cut_published_on_top_of_it(hub, tmp_path):
    files, run, _ = live(hub, tmp_path, freeze_at=500.0)
    (run / "params.json").write_text(json.dumps({"size": 8, "enabled": 1}))
    hub.files["params.json"] = json.dumps({"size": 7, "enabled": 1, "base_spread_ticks": 9})
    files.sync()
    assert json.loads((run / "params.json").read_text()) == {"size": 7, "enabled": 1}


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ({"size": 8}, {"size": 7}, {"size": 7}),
        ({"size": 8}, {"size": 9}, {"size": 8}),
        ({"enabled": 1}, {"enabled": 0}, {"enabled": 0}),
        ({"enabled": 0}, {"enabled": 1}, {"enabled": 0}),
        ({}, {"size": 1}, {"size": 1}),
        ({"size": 8}, {"base_spread_ticks": 9}, {"size": 8}),
        ({"size": 8}, {"instruments": {"A": {"size": 1}}}, {"size": 8, "instruments": {"A": {"size": 1}}}),
        ({"size": 8}, {"instruments": {"A": {"size": 90}}}, {"size": 8}),
        ({"size": 8}, {"instruments": {"A": {"base_spread_ticks": 9}}}, {"size": 8}),
        ({"instruments": {"A": {"size": 2}}}, {"instruments": {"A": {"size": 5}}}, {"instruments": {"A": {"size": 2}}}),
    ],
)
def test_what_the_freeze_lets_through(old, new, expected):
    assert risk_cuts(old, new) == expected


def test_control_kills_and_clears(hub, tmp_path):
    hub.files["control.json"] = json.dumps({"kill": True})
    files, run, _ = live(hub, tmp_path)
    files.sync()
    assert (run / "KILL").exists()
    hub.files["control.json"] = json.dumps({"flatten": True})
    files.sync()
    assert not (run / "KILL").exists()
    assert (run / "FLATTEN").exists()


def test_control_never_clears_a_kill_a_person_set(hub, tmp_path):
    files, run, _ = live(hub, tmp_path)
    (run / "KILL").touch()
    hub.files["control.json"] = json.dumps({"kill": False})
    files.sync()
    assert (run / "KILL").exists()


def test_what_control_set_is_remembered_across_a_parent_restart(hub, tmp_path):
    hub.files["control.json"] = json.dumps({"kill": True})
    files, run, _ = live(hub, tmp_path)
    files.sync()
    again, _, _ = live(hub, tmp_path)
    hub.files["control.json"] = json.dumps({"kill": False})
    again.sync()
    assert not (run / "KILL").exists()


def test_a_kill_that_could_not_be_written_is_tried_again(hub, tmp_path, monkeypatch):
    hub.files["control.json"] = json.dumps({"kill": True, "flatten": True})
    files, run, _ = live(hub, tmp_path)
    real = Path.touch
    tries = []

    def fail_kill_once(path, *args, **kwargs):
        if path.name == "KILL" and not tries:
            tries.append(1)
            raise OSError("disk full")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fail_kill_once)
    with pytest.raises(OSError):
        files.sync()
    assert (run / "FLATTEN").exists(), "one failing must not skip the other"
    assert not (run / "KILL").exists()
    files.sync()
    assert (run / "KILL").exists()


def test_control_that_is_not_an_object_changes_nothing(hub, tmp_path):
    hub.files["control.json"] = "[1]"
    files, run, reports = live(hub, tmp_path)
    files.sync()
    assert not (run / "KILL").exists()
    assert reports


# --- data going out --------------------------------------------------------------------------------


def make_uplink(hub, tmp_path):
    cfg = config(hub)
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    return Uplink(cfg, GitHub(cfg), run), run


def test_status_is_uploaded_to_a_data_release_that_cannot_become_the_latest_bot(hub, tmp_path):
    uplink, run = make_uplink(hub, tmp_path)
    write_status(run, pnl=12)
    uplink.send_status()
    assert list(hub.uploaded) == ["status-00000.json"]
    assert json.loads(hub.uploaded["status-00000.json"])["pnl"] == 12


def test_only_new_capture_bytes_go_up_and_the_parts_join_into_the_file(hub, tmp_path):
    uplink, run = make_uplink(hub, tmp_path)
    (run / "capture.jsonl.gz").write_bytes(b"first")
    uplink.send_data()
    uplink.send_data()
    with open(run / "capture.jsonl.gz", "ab") as handle:
        handle.write(b"second")
    uplink.send_data()
    parts = sorted(name for name in hub.uploaded if name.startswith("capture.jsonl.gz"))
    assert len(parts) == 2
    assert b"".join(hub.uploaded[name] for name in parts) == b"firstsecond"


def test_a_failed_upload_loses_nothing(hub, tmp_path):
    uplink, run = make_uplink(hub, tmp_path)
    (run / "capture.jsonl.gz").write_bytes(b"first")
    hub.fail = True
    with pytest.raises(OSError):
        uplink.send_data()
    hub.fail = False
    uplink.send_data()
    assert b"".join(hub.uploaded.values()) == b"first"


def test_an_upload_that_worked_but_lost_its_reply_does_not_stick_for_ever(hub, tmp_path):
    uplink, run = make_uplink(hub, tmp_path)
    (run / "capture.jsonl.gz").write_bytes(b"first")
    original = hub.server.RequestHandlerClass.do_POST

    def already_there(handler):
        if "/assets" in handler.path:
            handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            handler._send(422, b'{"errors":[{"code":"already_exists"}]}')
            return
        original(handler)

    hub.server.RequestHandlerClass.do_POST = already_there
    uplink.send_data()
    assert uplink.offsets["capture.jsonl.gz"] == 5


def test_an_upload_position_survives_a_parent_restart(hub, tmp_path):
    uplink, run = make_uplink(hub, tmp_path)
    (run / "capture.jsonl.gz").write_bytes(b"first")
    uplink.send_data()
    again, _ = make_uplink(hub, tmp_path)
    again.send_data()
    assert len(hub.uploaded) == 1


# --- building it, and failing safe -----------------------------------------------------------------


def test_no_settings_means_no_updater_and_the_bot_just_runs(tmp_path):
    assert build_updater({}, tmp_path, tmp_path / "first.py") is None


def test_no_token_means_no_updater(tmp_path, monkeypatch):
    monkeypatch.delenv("OPTIBOOK_GH_TOKEN", raising=False)
    event = {"update": {"repo": "o/r", "token_file": str(tmp_path / "absent")}}
    assert build_updater(event, tmp_path, tmp_path / "first.py") is None


def test_a_broken_freeze_time_means_no_updater_rather_than_no_freeze(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIBOOK_GH_TOKEN", TOKEN)
    event = {"update": {"repo": "o/r", "freeze_at": "soon"}}
    assert build_updater(event, tmp_path, tmp_path / "first.py") is None


def test_the_token_is_taken_out_of_the_environment_the_bot_inherits(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIBOOK_GH_TOKEN", TOKEN)
    event = {"update": {"repo": "o/r", "freeze_at": "2026-11-06T21:00:00+01:00"}}
    updater = build_updater(event, tmp_path, tmp_path / "first.py")
    assert updater is not None
    assert updater.config.freeze_at == pytest.approx(1793304000.0, abs=86400 * 400)
    assert "OPTIBOOK_GH_TOKEN" not in __import__("os").environ


def test_an_unreachable_github_changes_nothing_and_the_worker_survives(hub, tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIBOOK_GH_TOKEN", TOKEN)
    entry = tmp_path / "first.py"
    entry.write_text("# first\n")
    event = {"update": {"repo": "o/r", "api": "http://127.0.0.1:9", "uploads": "http://127.0.0.1:9", "code_poll_s": 0.05, "params_poll_s": 0.05}}
    updater = build_updater(event, tmp_path / "run", entry)
    updater.start()
    time.sleep(0.5)
    assert updater.worker_alive()
    assert updater.action(1000.0) is None
    updater.stop()


# --- with real processes ---------------------------------------------------------------------------


def child_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


class ScriptedUpdater:
    """Asks for one swap once the first child has started."""

    def __init__(self, new_file, marker):
        self.new_file, self.marker = new_file, marker
        self.swapped = False

    def command(self, base):
        return [base[0], str(self.new_file), *base[2:]] if self.swapped else list(base)

    def action(self, _now):
        return "swap" if not self.swapped and self.marker.exists() else None

    def apply(self, action, _now):
        assert action == "swap"
        self.swapped = True

    def on_crash(self, _now):
        return False


def test_supervise_swaps_a_running_child_for_the_new_file(tmp_path):
    marker = tmp_path / "started"
    done = tmp_path / "done"
    old = child_script(tmp_path, "old.py", f"import pathlib, time\npathlib.Path({str(marker)!r}).touch()\ntime.sleep(60)\n")
    new = child_script(tmp_path, "new.py", f"import pathlib\npathlib.Path({str(done)!r}).touch()\n")
    updater = ScriptedUpdater(new, marker)
    code = supervise([sys.executable, str(old)], tmp_path / "heartbeat", poll_s=0.02, updater=updater)
    assert code == 0
    assert done.exists()


def test_supervise_restarts_at_once_when_a_crash_rolls_back(tmp_path):
    good = tmp_path / "good"
    bad = child_script(tmp_path, "bad.py", "import sys\nsys.exit(7)\n")
    fine = child_script(tmp_path, "fine.py", f"import pathlib\npathlib.Path({str(good)!r}).touch()\n")

    class Rollback:
        rolled = False

        def command(self, base):
            return [base[0], str(fine if self.rolled else bad)]

        def action(self, _now):
            return None

        def apply(self, _action, _now):
            raise AssertionError("no swap expected")

        def on_crash(self, _now):
            self.rolled = True
            return True

    pauses = []
    code = supervise([sys.executable, "x"], tmp_path / "heartbeat", poll_s=0.02, updater=Rollback(), sleep=lambda s: (pauses.append(s), time.sleep(0.01)))
    assert code == 0
    assert good.exists()
    assert not [pause for pause in pauses if pause >= 1.0], "no back-off: the fix is already known"


def test_a_new_child_finding_the_lock_just_after_a_swap_is_retried_not_final(tmp_path):
    marker = tmp_path / "started"
    tries = tmp_path / "tries"
    old = child_script(tmp_path, "old.py", f"import pathlib, time\npathlib.Path({str(marker)!r}).touch()\ntime.sleep(60)\n")
    new = child_script(
        tmp_path, "new.py",
        f"import pathlib, sys\np = pathlib.Path({str(tries)!r})\nn = int(p.read_text()) if p.exists() else 0\np.write_text(str(n + 1))\nsys.exit(3 if n < 2 else 0)\n",
    )
    code = supervise(
        [sys.executable, str(old)], tmp_path / "heartbeat", poll_s=0.02, restart_pause_s=0.01,
        updater=ScriptedUpdater(new, marker),
    )
    assert code == 0
    assert tries.read_text() == "3"


STUB = """
import time

from otk.sim import scenarios
from otk.sim.exchange import SimExchange

_market = scenarios.dual_listing(1, bases=("PHILIPS",))
_last = time.monotonic()


class Exchange(SimExchange):
    def __init__(self):
        super().__init__(_market, "us")

    def _pump(self):
        global _last
        now = time.monotonic()
        while now - _last >= 0.05:
            _market.advance(0.05)
            _last += 0.05

    def get_positions(self):
        self._pump()
        return super().get_positions()
"""


@pytest.fixture(scope="module")
def bot(tmp_path_factory):
    return make_single_file.build(tmp_path_factory.mktemp("dist") / "optibook_bot.py")


def test_the_single_file_passes_its_self_check_with_no_login(bot, tmp_path):
    env = {"PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, str(bot), "--selfcheck"], cwd=tmp_path, env=env, capture_output=True, text=True,
        timeout=60.0, check=False,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert not (tmp_path / "run").exists(), "a self-check leaves nothing behind"


def test_the_self_check_fails_on_a_stack_that_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("otk.engine.Engine._act", lambda *_a: 1 / 0)
    assert bot_main(["--selfcheck"]) != 0


def test_a_merged_release_reaches_the_running_bot_with_nobody_touching_it(hub, bot, tmp_path):
    """The whole path: first upload runs, a release appears, the child is swapped, and it trades on."""
    (tmp_path / "optibook").mkdir()
    (tmp_path / "optibook" / "__init__.py").write_text("")
    (tmp_path / "optibook" / "synchronous_client.py").write_text(STUB)
    hub.files["params.json"] = json.dumps({"size": 3})
    hub.releases["data-test"] = {"id": 77, "tag_name": "data-test"}
    first = tmp_path / "first.py"
    event = {"update": {
        "repo": "o/r", "api": hub.url, "uploads": hub.url, "issue": 5, "code_poll_s": 0.5, "params_poll_s": 0.5,
        "status_s": 1, "data_s": 1, "watch_s": 3, "data_tag": "data-test",
    }}
    first.write_text(bot.read_text().replace("EVENT = {}", f"EVENT = {event!r}"))
    env = {"PYTHONPATH": f"{tmp_path}:{ROOT / 'src'}", "PATH": "/usr/bin:/bin", "OPTIBOOK_GH_TOKEN": TOKEN}
    process = subprocess.Popen(
        [sys.executable, str(first)], cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    state = tmp_path / "run" / "update_state.json"
    status = tmp_path / "run" / "status.json"

    def wait_for(check, seconds=45.0):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                if check():
                    return True
            except (OSError, ValueError, KeyError, TypeError):
                pass
            time.sleep(0.2)
        return False

    try:
        assert wait_for(status.exists), "the first upload runs"
        assert wait_for(lambda: json.loads((tmp_path / "run" / "params.json").read_text()) == {"size": 3})
        hub.publish("bot-2", bot.read_text() + "\n# the next version\n")
        assert wait_for(lambda: json.loads(state.read_text())["good"]["tag"] == "bot-2"), (tmp_path / "run" / "bot.log").read_text()[-1500:]
        before = json.loads(status.read_text())["elapsed_s"]
        assert wait_for(lambda: json.loads(status.read_text())["elapsed_s"] > before), "it trades on"
        assert json.loads(status.read_text())["connected"] is True
        assert wait_for(lambda: any(name.startswith("status-") for name in hub.uploaded))
        assert wait_for(lambda: any("bot-2" in comment for comment in hub.comments))
    finally:
        process.terminate()
        process.wait(timeout=20.0)
    assert TOKEN not in (tmp_path / "run" / "bot.log").read_text()


# --- a new version keeps the event's settings ------------------------------------------------------


def test_every_child_gets_the_settings_of_the_first_upload(tmp_path, monkeypatch):
    """Event settings are not committed, so a release has none of its own: it must not lose them on a swap."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPTIBOOK_GH_TOKEN", raising=False)
    entry = tmp_path / "first.py"
    entry.write_text("# first\n")
    seen = {}

    def fake_supervise(command, *_args, **_kwargs):
        seen["command"] = command
        return 0

    monkeypatch.setattr("otk.main.supervise", fake_supervise)
    event = {"position_limit": 50, "pairs": [["A", "A_DUAL"]]}
    assert bot_main([], event=event, entry=str(entry)) == 0
    command = seen["command"]
    path = Path(command[command.index("--event") + 1])
    assert json.loads(path.read_text())["position_limit"] == 50
    assert path.is_absolute()


def test_a_child_reads_those_settings_over_its_own_empty_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr("otk.main.run_live", lambda _args, event: seen.update(event) or 0)
    settings = tmp_path / "event_in_force.json"
    settings.write_text(json.dumps({"position_limit": 50}))
    assert bot_main(["--parent-pid", "1", "--event", str(settings)], event={}, entry="x") == 0
    assert seen["position_limit"] == 50


def test_settings_that_cannot_be_written_do_not_stop_the_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entry = tmp_path / "first.py"
    entry.write_text("# first\n")
    seen = {}
    monkeypatch.setattr("otk.main.supervise", lambda command, *_a, **_k: seen.update(command=command) or 0)
    monkeypatch.setattr("otk.main.save_event", lambda *_a: None)
    assert bot_main([], event={"position_limit": 50}, entry=str(entry)) == 0
    assert "--event" not in seen["command"]


def test_an_event_file_given_by_hand_is_still_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entry = tmp_path / "first.py"
    entry.write_text("# first\n")
    given = tmp_path / "given.json"
    given.write_text(json.dumps({"position_limit": 7}))
    seen = {}
    monkeypatch.setattr("otk.main.supervise", lambda command, *_a, **_k: seen.update(command=command) or 0)
    assert bot_main(["--event", str(given)], event={"position_limit": 50, "option_vol": 2.0}, entry=str(entry)) == 0
    command = seen["command"]
    merged = json.loads(Path(command[len(command) - 1 - command[::-1].index("--event") + 1]).read_text())
    assert merged == {"position_limit": 7, "option_vol": 2.0}


def _files_in_a_test(tmp_path):
    (tmp_path / "params.json").write_text('{"size": 10}')
    files = LiveFiles(UpdateConfig(repo="", token=""), None, tmp_path, lambda _text: None)
    files.set_overlay({"size": 5})
    return files


def test_a_hand_edit_during_a_test_stays_and_ends_the_test(tmp_path):
    files = _files_in_a_test(tmp_path)
    (tmp_path / "params.json").write_text('{"enabled": 0}')
    with pytest.raises(HandEdited):
        files.set_overlay({"size": 6})
    assert json.loads((tmp_path / "params.json").read_text()) == {"enabled": 0}
    assert files.publishes == 1
    files.set_overlay({})
    assert json.loads((tmp_path / "params.json").read_text()) == {"enabled": 0}


def test_a_hand_edit_as_a_test_ends_stays_with_no_fuss(tmp_path):
    files = _files_in_a_test(tmp_path)
    (tmp_path / "params.json").write_text('{"enabled": 0}')
    files.set_overlay({})
    assert json.loads((tmp_path / "params.json").read_text()) == {"enabled": 0}
    assert files.publishes == 1


def test_no_edit_means_no_alarm_however_many_slices_run(tmp_path):
    files = _files_in_a_test(tmp_path)
    for size in (6, 5, 6):
        files.set_overlay({"size": size, "instruments": {"A": {"skew_ticks": 2.5}}})
    files.set_overlay({})
    assert json.loads((tmp_path / "params.json").read_text()) == {"size": 10}
    assert files.publishes == 0


def test_a_publish_during_a_test_is_not_taken_for_a_hand_edit(tmp_path):
    files = _files_in_a_test(tmp_path)
    files._apply_params(b'{"size": 8}')
    files.set_overlay({"size": 6})
    assert json.loads((tmp_path / "params.json").read_text()) == {"size": 6}
    files.set_overlay({})
    assert json.loads((tmp_path / "params.json").read_text()) == {"size": 8}


def test_the_freeze_lets_the_lead_lag_signal_be_switched_off_but_not_on():
    assert risk_cuts({"lag_enabled": 1}, {"lag_enabled": 0})["lag_enabled"] == 0
    assert risk_cuts({"lag_enabled": 0}, {"lag_enabled": 1})["lag_enabled"] == 0
