"""The main algo: connect, work out the format, run the matching strategies, stay up.

This is what `dist/optibook_bot.py` runs, and the one file submitted to Optibook is this module and
its dependencies flattened by `scripts/make_single_file.py`. It needs no arguments and no other files:
what it can read from the exchange it reads, the rest comes from `EVENT`, and everything it writes goes
under `run/`. A supervising parent, which holds no exchange session, restarts it if it crashes or hangs,
because a final run lost to one error is the most expensive failure there is.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from otk.adapter import LiveExchange, optibook_client_factory
from otk.autopilot import Autopilot, ParticipantIntel
from otk.engine import EXIT_LISTING_CHANGED, Engine, EngineConfig
from otk.formats import apply_pairs, choose_liquid_leg, detect_pairs, plan_event, render_plan
from otk.health import print_health
from otk.instruments import Kind, Universe
from otk.monitor import LiveMonitor, TeeRecorder
from otk.news import (
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    AsyncClassifier,
    CombinedClassifier,
    KeywordClassifier,
    LlmClassifier,
    NaiveBayesNews,
    default_classifier,
    http_completer,
    parse_labelled_rows,
)
from otk.params import (
    INSTRUMENTS_KEY,
    ParamStore,
    fine_tick_overrides,
    merge_params,
    validate_params,
)
from otk.recorder import JsonlRecorder
from otk.risk import Limits
from otk.selfcheck import CheckExchange
from otk.selftune import build_self_tuner, run_self_tune_pick
from otk.strategies import default_stack
from otk.updater import build_updater

logger = logging.getLogger("otk.main")

# Live safety nets. Off in the library so tests and replays see only what they ask for.
LIVE_KILL_DEFAULTS = {"kill_drawdown_frac": 0.25, "kill_stale_s": 30.0, "kill_reject_limit": 30.0}
BOOK_WAIT_S = 3.0
SELFCHECK_TICKS = 200
EXIT_ALREADY_RUNNING = 3
FAST_FAILURE_S = 30.0
MAX_RESTART_PAUSE_S = 30.0
SWITCH_INTERVAL_S = 0.001
HEARTBEAT_NAME = "heartbeat"
EVENT_IN_FORCE_NAME = "event_in_force.json"
LOCK_NAME = "LIVE.lock"
HEARTBEAT_STALE_S = 20.0
SUPERVISOR_GRACE_S = 60.0
SUPERVISOR_POLL_S = 0.5
RESTART_PAUSE_S = 1.0
SWAP_SETTLE_TRIES = 5


def load_event_config(path: str | None) -> dict:
    """Facts confirmed at the briefing: limits, index weights, volatility. See config/."""
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle)


def usable_run_dir(wanted: Path) -> Path:
    """Where to write. The submitted file may run where it cannot write, and that must not stop trading."""
    for candidate in (wanted, Path(tempfile.gettempdir()) / "optibook_run"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix="optibook_run_"))


def beat(heartbeat: Path) -> None:
    try:
        heartbeat.touch()
    except OSError as exc:
        logger.warning("could not touch %s: %s", heartbeat, exc)


def _owner_is_running(lock: Path, heartbeat: Path) -> bool:
    """A lock counts only if its owner exists and is still beating, since a pid can be reused."""
    try:
        owner = int(lock.read_text().strip() or 0)
        if not owner:
            return False
        os.kill(owner, 0)
    except (OSError, ValueError) as exc:
        return isinstance(exc, PermissionError)
    return time.time() - _heartbeat_time(heartbeat) < HEARTBEAT_STALE_S


def _heartbeat_time(heartbeat: Path) -> float:
    try:
        return heartbeat.stat().st_mtime
    except OSError:
        return 0.0


def claim_session(lock: Path, heartbeat: Path) -> int | None:
    """Before logging in, or a second copy would kick the first. None if another copy really holds it."""
    if lock.exists() and _owner_is_running(lock, heartbeat):
        return None
    try:
        lock.unlink(missing_ok=True)
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    except OSError as exc:
        logger.warning("no session lock at %s: %s", lock, exc)
        return -1
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    return os.getpid()


def release_session(lock: Path) -> None:
    try:
        lock.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not remove %s: %s", lock, exc)


def engine_config(event: Mapping, run_dir: Path, max_drawdown: float | None, owner: int | None = None) -> EngineConfig:
    limits = Limits(
        position=int(event.get("position_limit", 100)),
        outstanding=int(event.get("outstanding_limit", 200)),
    )
    return EngineConfig(
        limits=limits,
        book_includes_own=bool(event.get("book_includes_own", True)),
        max_drawdown=max_drawdown,
        kill_file=str(run_dir / "KILL"),
        flatten_file=str(run_dir / "FLATTEN"),
        live_lock=str(run_dir / LOCK_NAME),
        kill_state_file=str(run_dir / "kill_state.json"),
        kill_state_owner=owner,
        follow_limit_param=True,
    )


def option_vol_from(event: Mapping) -> float | None:
    """What the team was told. None keeps stock options out: no guess is ever used to price them."""
    raw = event.get("option_vol")
    if raw is None:
        return None
    try:
        value = None if isinstance(raw, bool) else float(raw)
    except (TypeError, ValueError):
        value = None
    if value is None or not 0 < value < float("inf"):
        logger.error("option_vol must be a positive number, not %r: stock options stay out", raw)
        return None
    return value


# --- news -----------------------------------------------------------------------------------------


def build_classifier(event: Mapping):
    """News model from the event config: labelled CSV if supplied, else keywords, else none."""
    training = event.get("news_training_csv")
    if training:
        with open(training, newline="") as handle:
            rows = list(csv.reader(handle))
        return NaiveBayesNews.fit(parse_labelled_rows(rows[0], rows[1:]))
    aliases = event.get("news_aliases")
    if aliases:
        return KeywordClassifier(
            aliases, event.get("news_positive", POSITIVE_WORDS), event.get("news_negative", NEGATIVE_WORDS)
        )
    return None


def news_parts(event: Mapping, universe: Universe) -> tuple:
    """The fast classifier, always present, and an LLM off-thread if the event provides one."""
    stocks = universe.ids_of(Kind.STOCK)
    rules = default_classifier(stocks)
    try:
        configured = build_classifier(event)
    except (OSError, IndexError, ValueError) as exc:
        logger.error("news model not loaded, using rules: %s: %s", type(exc).__name__, exc)
        configured = None
    fast = CombinedClassifier(configured, rules) if configured else rules
    slow = None
    if event.get("llm_url"):
        complete = http_completer(
            event["llm_url"], event.get("llm_model", ""), event.get("llm_key_env"),
            event.get("llm_style", "openai"),
        )
        slow = AsyncClassifier(LlmClassifier(complete, stocks).classify_all)
    return fast, slow


# --- status ---------------------------------------------------------------------------------------


def write_status(
    path: Path, engine: Engine, started: float, monitor: LiveMonitor | None = None, autopilot=None
) -> None:
    """Agents and the dashboard read this; it is replaced atomically once per second."""
    connected = engine.exchange.is_connected()
    payload = {
        "elapsed_s": round(time.monotonic() - started, 1),
        "connected": connected,
        "pnl": engine._read_pnl() if connected else None,
        "positions": {iid: pos for iid, pos in engine.exchange.get_positions().items() if pos}
        if connected
        else {},
        "errors": engine.errors,
        "orders_sent": engine.manager.sent,
        "rejects": engine.manager.rejects,
        "last_reject": engine.manager.last_reject,
        "telemetry_errors": engine.telemetry_errors,
        "reconnects": engine.reconnects,
        "halted": engine.halted,
        "kill": {"trips": engine.kill.trips, "latched": engine.kill.latched, "reason": engine.kill.last_reason},
        "writes_last_second": engine.budget.used_last_second(),
        "write_ms": engine.manager.write_ms(),
        "taker_mode": int(engine.params_in_force().get("taker_mode", 0)),
        "params_version": engine.params.version,
        "params_error": engine.params.last_error,
        "field": monitor.snapshot() if monitor else {},
        "autopilot": autopilot.snapshot() if autopilot else {},
        "by_strategy": monitor.by_strategy() if monitor else {},
        "by_instrument": monitor.by_instrument() if monitor else {},
        "baseline": engine.params.baseline(),  # what is in force apart from Autopilot, so a replay can start from it
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def safe_status(path: Path, engine: Engine, started: float, monitor: LiveMonitor, autopilot=None) -> None:
    """The status file is for humans and agents. It must never be able to stop the loop."""
    try:
        write_status(path, engine, started, monitor, autopilot)
    except Exception:
        logger.exception("status write failed")


# --- start-up -------------------------------------------------------------------------------------


def tune_runtime() -> None:
    """Cheap latency wins that need no assumptions about the client.

    The client's network thread needs the GIL to deliver book updates, and Python's default 5 ms
    switch interval delays it behind a busy trading loop. The client also logs every order unless told
    not to, and a garbage collection pause can arrive mid-tick.
    """
    sys.setswitchinterval(SWITCH_INTERVAL_S)
    logging.getLogger("client").setLevel(logging.ERROR)
    gc.collect()
    gc.freeze()
    gc.set_threshold(50_000, 50, 100)


def _wait_for_books(exchange, ids: Sequence[str], clock=time.monotonic, sleep=time.sleep) -> dict:
    """Books arrive by push just after connecting, and choosing the liquid leg needs them."""
    deadline = clock() + BOOK_WAIT_S
    books: dict = {}
    while True:
        books = {iid: exchange.get_last_price_book(iid) for iid in ids}
        if all(books.values()) or clock() >= deadline:
            return books
        sleep(0.05)


def prepare_universe(exchange, event: Mapping) -> Universe:
    instruments = exchange.get_instruments()
    universe = Universe.from_instruments(instruments)
    wanted = [iid for iid in event.get("instruments", ()) if iid in universe.by_id]
    unknown = set(event.get("instruments", ())) - set(wanted)
    if unknown:
        logger.error("not listed, ignoring: %s", sorted(unknown))
    if wanted:
        universe = Universe(universe.by_id[iid] for iid in wanted)
    ids = sorted(universe.by_id)
    pairs = [tuple(pair) for pair in event.get("pairs", ())]
    if not pairs:
        groups = {iid: getattr(instruments[iid], "instrument_group", None) for iid in ids}
        books = _wait_for_books(exchange, ids)
        pairs = [choose_liquid_leg(pair, books) for pair in detect_pairs(ids, groups)]
    return apply_pairs(universe, pairs)


def event_base(event: Mapping, universe: Universe, plan_params: Mapping | None = None) -> dict:
    """Event limits, live safety nets, the format's preset, then anything the team set by hand."""
    base: dict = {"position_limit": int(event.get("position_limit", 100)), **LIVE_KILL_DEFAULTS}
    fine = fine_tick_overrides(universe)
    return merge_params(
        base, {INSTRUMENTS_KEY: fine} if fine else {}, plan_params or {}, _usable_params(event.get("params", {}))
    )


def _usable_params(raw: Mapping) -> dict:
    """A typo in the event block costs that one setting, not the session."""
    usable = {}
    for key, value in raw.items():
        try:
            validate_params({key: value})
            usable[key] = value
        except ValueError as exc:
            logger.error("ignoring event parameter: %s", exc)
    return usable


def build_autopilot(
    store: ParamStore, universe: Universe, monitor: LiveMonitor, run_dir: Path, args: argparse.Namespace
) -> tuple[Autopilot, JsonlRecorder]:
    """Decisions go to their own log, so anyone can see why the bot changed itself."""
    log = JsonlRecorder(run_dir / "autopilot.jsonl", clock=time.time)
    intel = ParticipantIntel(args.intel or run_dir / "participants.json")
    return Autopilot(store, universe, monitor=monitor, intel=intel, sink=log.record), log


def _write_card(run_dir: Path, plan) -> None:
    card = render_plan(plan)
    for line in card:
        logger.info(line)
    try:
        (run_dir / "format.json").write_text(json.dumps({"format": plan.name, "card": card}, indent=2))
    except OSError as exc:
        logger.warning("could not write the format card: %s", exc)


# --- the live loop --------------------------------------------------------------------------------


def _build(args: argparse.Namespace, event: Mapping, run_dir: Path, exchange=None) -> tuple:
    read_budget = event.get("read_budget")
    tick_sleep = args.sleep
    exchange = exchange or LiveExchange(optibook_client_factory())
    exchange.connect()
    universe = prepare_universe(exchange, event)
    ids = sorted(universe.by_id)
    if read_budget:  # the probe showed reads are rate limited: slow the tick rather than go stale
        tick_sleep = max(tick_sleep, (4 * len(ids) + 2) / float(read_budget))
        exchange.set_read_ttl(tick_sleep)
    plan = plan_event(universe, event, has_news_feed=exchange.has_news_feed())
    _write_card(run_dir, plan)
    fast, slow = news_parts(event, universe)
    stack = default_stack(
        ids,
        index_weights=event.get("index_weights"),
        classifier=fast,
        vol=option_vol_from(event),
        flatten_at=time.monotonic() + args.flatten_after if args.flatten_after else None,
        universe=universe,
        slow_classifier=slow,
    )
    monitor = LiveMonitor(universe.tick_of, group_of=lambda iid: universe.by_id[iid].base if iid in universe.by_id else iid)
    capture = JsonlRecorder(run_dir / "capture.jsonl.gz" if not args.capture else args.capture)
    store = ParamStore(args.params or run_dir / "params.json", base=event_base(event, universe, plan.params))
    autopilot, log = (None, None) if args.no_autopilot else build_autopilot(store, universe, monitor, run_dir, args)
    recorder = TeeRecorder([r for r in (monitor, capture, autopilot) if r])
    config = engine_config(event, run_dir, args.max_drawdown, args.parent_pid)
    engine = Engine(exchange, universe, stack, replace(config, live_lock=None), store, recorder)
    monitor.strategy_of = capture.strategy_of = engine.manager.strategy_of  # read only, to report who earned what
    return engine, recorder, monitor, autopilot, log, tick_sleep


def run_selfcheck(args: argparse.Namespace, event: Mapping) -> int:
    """Everything up to the exchange, on a made-up market with no login. Run on a new file before it replaces the bot."""
    with tempfile.TemporaryDirectory() as scratch:
        run_dir = Path(scratch)
        engine, recorder, monitor, autopilot, log, _sleep = _build(args, event, run_dir, CheckExchange())
        for _ in range(SELFCHECK_TICKS):
            engine.tick()
        write_status(run_dir / "status.json", engine, time.monotonic(), monitor, autopilot)
        engine.shutdown()
        recorder.close()
        if log:
            log.close()
    if engine.errors or engine.manager.rejects:
        logger.error("self-check failed: %d errors, %d rejects", engine.errors, engine.manager.rejects)
        return 1
    return 0


def run_live(args: argparse.Namespace, event: Mapping) -> int:
    run_dir = usable_run_dir(Path(args.run_dir))
    heartbeat = run_dir / HEARTBEAT_NAME
    lock = run_dir / LOCK_NAME
    if claim_session(lock, heartbeat) is None:
        logger.error("%s: another copy holds the exchange session. A second login would disconnect it.", lock)
        return EXIT_ALREADY_RUNNING
    beat(heartbeat)
    try:
        return _run_claimed(args, event, run_dir, heartbeat)
    finally:
        release_session(lock)


def _run_claimed(args: argparse.Namespace, event: Mapping, run_dir: Path, heartbeat: Path) -> int:
    engine, recorder, monitor, autopilot, log, tick_sleep = _build(args, event, run_dir)
    tune_runtime()
    stopping: list[bool] = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    signal.signal(signal.SIGINT, lambda *_: stopping.append(True))
    started = time.monotonic()
    deadline = started + args.seconds if args.seconds else float("inf")
    next_status = started
    try:
        # If the supervisor dies, keep trading: unmanaged positions cost more than a missing parent.
        while not stopping and time.monotonic() < deadline:
            engine.tick()
            if time.monotonic() >= next_status:
                safe_status(run_dir / "status.json", engine, started, monitor, autopilot)
                beat(heartbeat)
                next_status = time.monotonic() + 1.0
            if tick_sleep:
                time.sleep(tick_sleep)
    finally:
        engine.shutdown()
        recorder.close()
        if log:
            log.close()
    return 0


# --- staying up -----------------------------------------------------------------------------------


def supervise(
    command: Sequence[str],
    heartbeat: Path,
    should_stop: Callable[[], bool] = lambda: False,
    stale_s: float = HEARTBEAT_STALE_S,
    grace_s: float = SUPERVISOR_GRACE_S,
    poll_s: float = SUPERVISOR_POLL_S,
    restart_pause_s: float = RESTART_PAUSE_S,
    max_restarts: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    updater=None,
) -> int:
    """Restart the bot if it dies or stops beating. A hung exchange call has no timeout of its own.

    This parent holds no exchange session, so it cannot disconnect the child by existing. A child that
    exits with 0 was stopped on purpose, so the parent exits too. So does one that found another copy
    running. A child that fails fast is probably failing the same way each time, and each try logs in,
    so the pause doubles until it has run for a while. An `updater` can ask for the child to be swapped
    for a new file, or a new file to be rolled back.
    """
    restarts = 0
    pause = restart_pause_s
    settling = 0
    while not should_stop():
        child = subprocess.Popen(updater.command(command) if updater else command)
        started = time.monotonic()
        action = _attend(child, started, heartbeat, should_stop, stale_s, grace_s, poll_s, sleep, updater)
        if should_stop():
            _stop_child(child)
            return 0
        if action:
            _stop_child(child)
            updater.apply(action, time.monotonic())
            restarts, pause, settling = 0, restart_pause_s, SWAP_SETTLE_TRIES
            continue
        code = child.wait()
        if code == EXIT_ALREADY_RUNNING and settling > 0:  # the bot just stopped may still hold the lock for a moment
            settling -= 1
            sleep(restart_pause_s)
            continue
        if code == EXIT_LISTING_CHANGED:
            logger.warning("the listing changed: restarting the bot to plan for it")
            sleep(restart_pause_s)
            continue
        if code in (0, EXIT_ALREADY_RUNNING):
            return code
        if updater and updater.on_crash(time.monotonic()):
            continue
        restarts += 1
        logger.error("bot exited with code %s: restart %d", code, restarts)
        if max_restarts is not None and restarts >= max_restarts:
            return code
        pause = restart_pause_s if time.monotonic() - started > FAST_FAILURE_S else pause
        sleep(pause)
        pause = min(pause * 2.0, MAX_RESTART_PAUSE_S)
    return 0


def _attend(child, started, heartbeat, should_stop, stale_s, grace_s, poll_s, sleep, updater) -> str | None:
    """Wait on the child until it ends, a stop is asked for, or the updater wants something done to it."""
    while child.poll() is None and not should_stop():
        sleep(poll_s)
        age = time.monotonic() - max(_mtime(heartbeat), started)
        if time.monotonic() - started > grace_s and age > stale_s:
            logger.error("no heartbeat for %.0fs: killing the bot to restart it", age)
            child.kill()
        action = updater.action(time.monotonic()) if updater else None
        if action:
            return action
    return None


def _stop_child(child: subprocess.Popen) -> None:
    child.terminate()
    try:
        child.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()  # gone for certain, before anything logs in again


def _mtime(path: Path) -> float:
    """As monotonic time, so it compares with the supervisor's own clock."""
    try:
        return time.monotonic() - (time.time() - path.stat().st_mtime)
    except OSError:
        return float("-inf")


def add_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--event", help="event config JSON, on top of the EVENT block")
    parser.add_argument("--params", help="live parameter file; default run/params.json")
    parser.add_argument("--capture", help="capture path; default run/capture.jsonl.gz")
    parser.add_argument("--run-dir", default="run")
    # A busy Python loop can starve the client's network thread, so its data would go stale.
    # Replays on real books lost little up to 50 ms, so 20 ms is enough and leaves the client room.
    parser.add_argument("--sleep", type=float, default=0.02, help="seconds between ticks")
    parser.add_argument("--max-drawdown", type=float, help="latching stop on top of the live kill switch")
    parser.add_argument("--flatten-after", type=float, help="flatten N seconds after start")
    parser.add_argument("--no-autopilot", action="store_true", help="run parameters as given")
    parser.add_argument("--intel", help="participants.json from visualiser_feed; default run/")
    parser.add_argument("--seconds", type=float, help="stop after this long; default runs until killed")
    parser.add_argument("--no-supervise", action="store_true", help="no parent to restart a crash")
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--selfcheck", action="store_true", help="run the stack on a made-up market with no login, then exit")
    parser.add_argument("--self-tune-pick", type=float, metavar="MIN_SECONDS", help=argparse.SUPPRESS)  # what the parent runs, hourly
    parser.add_argument("--health", action="store_true", help="say whether the running bot is really trading; no login")


def main(argv: list[str] | None = None, event: Mapping | None = None, entry: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(parser)
    args = parser.parse_args(argv)
    if args.health:
        return print_health(usable_run_dir(Path(args.run_dir)))
    if args.self_tune_pick is not None:
        logging.basicConfig(level=logging.ERROR)
        return run_self_tune_pick(usable_run_dir(Path(args.run_dir)), args.self_tune_pick)
    if args.selfcheck:
        logging.basicConfig(level=logging.ERROR)
        return run_selfcheck(args, {**(event or {}), **_event_file(args.event)})
    run_dir = usable_run_dir(Path(args.run_dir))
    _set_up_logging(run_dir)
    if entry and not args.no_supervise and args.parent_pid is None:
        passed = list(sys.argv[1:] if argv is None else argv)
        merged = {**(event or {}), **_event_file(args.event)}
        # Event settings are not committed, so a release has none. Every child gets the first upload's, last wins.
        saved = save_event(run_dir, merged)
        handed = ["--event", saved] if saved else []
        command = [sys.executable, entry, *passed, *handed, "--parent-pid", str(os.getpid())]
        stopping: list[bool] = []
        signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
        signal.signal(signal.SIGINT, lambda *_: stopping.append(True))
        updater = build_updater(merged, run_dir, Path(entry))
        if updater:
            updater.start()
        tuner = None if updater else build_self_tuner(merged, run_dir, Path(entry))  # agents reach the bot, or this does
        if tuner:
            tuner.start()
        try:
            code = supervise(command, run_dir / HEARTBEAT_NAME, lambda: bool(stopping), updater=updater)
        except OSError as exc:
            logger.error("cannot start a child process, so no restarts: %s", exc)
        else:
            if tuner:
                tuner.stop()  # so a test's half size is not left in the file
            return code
    return run_live(args, {**(event or {}), **_event_file(args.event)})


def save_event(run_dir: Path, event: Mapping) -> str | None:
    """The settings in force, as an absolute path for the child. None if there are none, or they cannot be written."""
    if not event:
        return None
    path = Path(run_dir).resolve() / EVENT_IN_FORCE_NAME
    try:
        path.write_text(json.dumps(event, indent=2))
    except OSError as exc:
        logger.error("could not save the event settings, so a new release would run without them: %s", exc)
        return None
    return str(path)


def _event_file(path: str | None) -> dict:
    """A broken file must not cost the session: the EVENT block in the file still applies."""
    try:
        return load_event_config(path)
    except (OSError, ValueError) as exc:
        logger.error("event config %s not loaded: %s", path, exc)
        return {}


def _set_up_logging(run_dir: Path) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(run_dir / "bot.log"))
    except OSError as exc:
        print(f"no log file in {run_dir}: {exc}", file=sys.stderr)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)
