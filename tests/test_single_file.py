"""The one file submitted to Optibook must work with nothing else beside it."""

import ast
import json
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_single_file

from otk.main import HEARTBEAT_NAME, supervise

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


def run_bot(bot: Path, cwd: Path, *arguments: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    (cwd / "optibook").mkdir(exist_ok=True)
    (cwd / "optibook" / "__init__.py").write_text("")
    (cwd / "optibook" / "synchronous_client.py").write_text(STUB)
    env = {"PYTHONPATH": f"{cwd}{':'}{ROOT / 'src'}", "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [sys.executable, str(bot), *arguments],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False,
    )


def test_the_build_has_no_repo_imports_and_only_the_standard_library(bot):
    tree = ast.parse(bot.read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    assert "otk" not in top
    assert top <= set(sys.stdlib_module_names)
    assert "from otk" not in bot.read_text()


def test_the_event_block_comes_first_and_is_embedded(tmp_path):
    path = make_single_file.build(tmp_path / "bot.py", {"position_limit": 50})
    text = path.read_text()
    assert text.index("EVENT = ") < text.index("# params")
    assert '"position_limit": 50' in text


def test_every_strategy_reaches_the_file(bot):
    text = bot.read_text()
    for name in ("class MarketMaker", "class DualListingArb", "class NewsTrader", "class MeanReversionTaker", "class Autopilot", "class KillSwitch", "def plan_event"):
        assert name in text


def test_a_name_defined_differently_in_two_modules_stops_the_build():
    first = make_single_file.Module("a", ast.parse("X = 1\n"), ("X = 1",))
    second = make_single_file.Module("b", ast.parse("X = 2\n"), ("X = 2",))
    with pytest.raises(SystemExit, match="X: a and b"):
        make_single_file.check_collisions([first, second])


def test_the_same_definition_twice_is_fine():
    first = make_single_file.Module("a", ast.parse("X = 1\n"), ("X = 1",))
    second = make_single_file.Module("b", ast.parse("X = 1\n"), ("X = 1",))
    make_single_file.check_collisions([first, second])


def test_it_runs_from_the_one_file_detects_the_format_and_trades(bot, tmp_path):
    result = run_bot(bot, tmp_path, "--seconds", "4", "--no-supervise")
    assert result.returncode == 0, result.stderr[-1500:]
    card = json.loads((tmp_path / "run" / "format.json").read_text())
    assert card["format"] == "dual_listing"
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["errors"] == 0
    assert status["connected"] is True
    assert status["rejects"] == 0
    assert status["writes_last_second"] > 0 or status["positions"]
    assert (tmp_path / "run" / "capture.jsonl.gz").exists()
    assert "FORMAT: dual_listing" in (tmp_path / "run" / "bot.log").read_text()


def test_it_stops_cleanly_on_a_signal(bot, tmp_path):
    (tmp_path / "optibook").mkdir()
    (tmp_path / "optibook" / "__init__.py").write_text("")
    (tmp_path / "optibook" / "synchronous_client.py").write_text(STUB)
    env = {"PYTHONPATH": f"{tmp_path}:{ROOT / 'src'}", "PATH": "/usr/bin:/bin"}
    child = subprocess.Popen(
        [sys.executable, str(bot), "--no-supervise"], cwd=tmp_path, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20.0
    while not (tmp_path / "run" / "status.json").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    child.send_signal(signal.SIGTERM)
    assert child.wait(timeout=15.0) == 0
    assert not (tmp_path / "run" / "LIVE.lock").exists(), "a restart must not find a stale lock"


def test_a_second_copy_does_not_log_in_at_all(bot, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "LIVE.lock").write_text(str(subprocess.os.getpid()))
    (run / "heartbeat").touch()
    (tmp_path / "optibook").mkdir()
    (tmp_path / "optibook" / "__init__.py").write_text("")
    (tmp_path / "optibook" / "synchronous_client.py").write_text("raise SystemExit('logged in')")
    env = {"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, str(bot), "--no-supervise"], cwd=tmp_path, env=env, capture_output=True, text=True,
        timeout=30.0, check=False,
    )
    assert result.returncode == 3, result.stderr[-800:]


def test_a_broken_event_block_still_trades(bot, tmp_path):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "instruments": ["NOPE"], "news_training_csv": "absent.csv", "params": {"nonsense": 1}, "position_limit": 100,
    }))
    result = run_bot(bot, tmp_path, "--seconds", "3", "--no-supervise", "--event", str(event))
    assert result.returncode == 0, result.stderr[-1500:]
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["errors"] == 0
    assert status["writes_last_second"] > 0 or status["positions"]


def test_a_read_only_folder_still_trades(bot, tmp_path):
    (tmp_path / "optibook").mkdir()
    (tmp_path / "optibook" / "__init__.py").write_text("")
    (tmp_path / "optibook" / "synchronous_client.py").write_text(STUB)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    env = {"PYTHONPATH": f"{tmp_path}:{ROOT / 'src'}", "PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)}
    try:
        result = subprocess.run(
            [sys.executable, str(bot), "--seconds", "3", "--no-supervise"], cwd=locked, env=env,
            capture_output=True, text=True, timeout=60.0, check=False,
        )
    finally:
        locked.chmod(0o755)
    assert result.returncode == 0, result.stderr[-1500:]
    assert json.loads((scratch / "optibook_run" / "status.json").read_text())["errors"] == 0


# --- the supervisor --------------------------------------------------------------------------------


def child_script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, str(script)]


def test_a_crash_is_restarted(tmp_path):
    count = tmp_path / "count"
    command = child_script(
        tmp_path,
        f"""
        import pathlib, sys
        path = pathlib.Path({str(count)!r})
        seen = int(path.read_text()) if path.exists() else 0
        path.write_text(str(seen + 1))
        sys.exit(0 if seen >= 2 else 4)
        """,
    )
    assert supervise(command, tmp_path / HEARTBEAT_NAME, poll_s=0.02, restart_pause_s=0.01) == 0
    assert count.read_text() == "3"


def test_a_clean_exit_is_not_restarted(tmp_path):
    count = tmp_path / "count"
    command = child_script(
        tmp_path,
        f"""
        import pathlib
        pathlib.Path({str(count)!r}).write_text("ran")
        """,
    )
    assert supervise(command, tmp_path / HEARTBEAT_NAME, poll_s=0.02) == 0
    assert count.read_text() == "ran"


def test_a_hang_is_killed_and_restarted(tmp_path):
    count = tmp_path / "count"
    command = child_script(
        tmp_path,
        f"""
        import pathlib, time
        path = pathlib.Path({str(count)!r})
        seen = int(path.read_text()) if path.exists() else 0
        path.write_text(str(seen + 1))
        if seen == 0:
            time.sleep(60)
        """,
    )
    started = time.monotonic()
    code = supervise(
        command, tmp_path / HEARTBEAT_NAME, stale_s=0.3, grace_s=0.1, poll_s=0.02, restart_pause_s=0.01
    )
    assert code == 0
    assert count.read_text() == "2"
    assert time.monotonic() - started < 20.0


def test_restarts_can_be_capped(tmp_path):
    command = child_script(tmp_path, "import sys; sys.exit(7)")
    assert (
        supervise(command, tmp_path / HEARTBEAT_NAME, poll_s=0.02, restart_pause_s=0.01, max_restarts=2) == 7
    )


def test_stopping_the_supervisor_stops_the_child(tmp_path):
    marker = tmp_path / "alive"
    command = child_script(
        tmp_path,
        f"""
        import pathlib, time
        pathlib.Path({str(marker)!r}).write_text("yes")
        time.sleep(60)
        """,
    )
    started = time.monotonic()

    def stop():
        return marker.exists() or time.monotonic() - started > 15.0

    assert supervise(command, tmp_path / HEARTBEAT_NAME, stop, poll_s=0.02) == 0
    assert time.monotonic() - started < 20.0


def test_it_runs_when_pasted_into_a_notebook_cell(bot, tmp_path):
    """A notebook has no __file__, and its own arguments on sys.argv. The devenv has offered Jupyter at past events."""
    (tmp_path / "optibook").mkdir()
    (tmp_path / "optibook" / "__init__.py").write_text("")
    (tmp_path / "optibook" / "synchronous_client.py").write_text(STUB)
    (tmp_path / "cell.py").write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.argv = ["ipykernel_launcher.py", "-f", "/tmp/kernel-demo.json"]
            source = open({str(bot)!r}).read()
            exec(compile(source, "<ipython-input-1>", "exec"), {{"__name__": "__main__"}})
            """
        )
    )
    env = {"PYTHONPATH": f"{tmp_path}:{ROOT / 'src'}", "PATH": "/usr/bin:/bin"}
    process = subprocess.Popen([sys.executable, "cell.py"], cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    status = tmp_path / "run" / "status.json"
    deadline = time.monotonic() + 20
    while not status.exists() and time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.1)
    alive = process.poll() is None
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=20)
    assert alive, output[-800:]
    assert json.loads(status.read_text())["connected"] is True


def test_health_works_from_the_one_file_with_no_login(bot, tmp_path):
    (tmp_path / "optibook").mkdir()
    (tmp_path / "optibook" / "__init__.py").write_text("")
    (tmp_path / "optibook" / "synchronous_client.py").write_text("raise SystemExit('logged in')")
    done = subprocess.run([sys.executable, str(bot), "--health"], cwd=tmp_path, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path)}, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 2
    assert "RED" in done.stdout
    assert "logged in" not in done.stdout + done.stderr


def test_settings_with_a_true_or_a_null_do_not_stop_the_file_at_start(tmp_path):
    """Our own example has `book_includes_own: true`, and JSON's `true` is not Python's."""
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"book_includes_own": True, "news_training_csv": None, "self_tune": False, "params": {"size": 5}}))
    out = tmp_path / "bot.py"
    assert make_single_file.main(["--event", str(event), "--out", str(out)]) == 0
    done = subprocess.run([sys.executable, str(out), "--selfcheck"], capture_output=True, text=True, timeout=120, cwd=tmp_path, check=False)
    assert done.returncode == 0, done.stderr[-600:]


def test_our_own_example_settings_build_a_file_that_starts(tmp_path):
    out = tmp_path / "bot.py"
    assert make_single_file.main(["--event", str(ROOT / "config" / "event.example.json"), "--out", str(out)]) == 0
    done = subprocess.run([sys.executable, str(out), "--selfcheck"], capture_output=True, text=True, timeout=120, cwd=tmp_path, check=False)
    assert done.returncode == 0, done.stderr[-600:]
