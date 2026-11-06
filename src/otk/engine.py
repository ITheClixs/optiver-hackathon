"""Deterministic trading loop. No LLM call ever runs here; agents act through params.json.

Each tick: reconnect if needed, read state, ask strategies for intents, execute takes before
quotes, all through the risk clip and write budget. Any strategy error pulls that tick's quotes
and is counted, because a crash mid-window costs more than a missed tick.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from otk.budget import WriteBudget
from otk.fair import Book, Top, external_book, mid, top_of_book
from otk.instruments import Kind, Universe
from otk.intents import Pull, Take
from otk.kill import KillAction, KillSwitch
from otk.params import INSTRUMENTS_KEY, ParamStore, params_for, scaled_sizes
from otk.quote_manager import QuoteManager, group_intents
from otk.recorder import NullRecorder
from otk.risk import DrawdownStop, Limits

logger = logging.getLogger("otk.engine")

# The taker's position is its edge, and hedging it cost 19 to 32% on real books. So the hedge waits for a real breach.
TAKER_MODE_KEEPS = frozenset({"flattener", "news_trader", "hedger", "thin_leg_taker"})
TAKER_HEDGE_THRESHOLD = 98
TAKER_HEDGE_TO = 90
TAKER_MIN_SAMPLES = 10

# A teammate's second login kicks us; reconnecting at once kicks them back in a duel that lost one
# public team 36k log lines. Back off, and alert a human if it keeps happening.
RECONNECT_MIN_S = 1.0
RECONNECT_MAX_S = 30.0
RECONNECT_RESET_S = 60.0
# After a reconnect, quote and trade nothing this long: a reset restarts prices, and strategies need fresh books.
RESET_QUIET_S = 10.0  # the first trade comes 6 to 7 s after we reconnect, with the house walls just behind it
MAX_TICK_ERRORS = 5
EXIT_LISTING_CHANGED = 4  # the parent restarts it at once, and it is not a crash
KILL_CHECK_S = 0.5
FLATTEN_INTERVAL_S = 0.5  # positions arrive late: wait, or a second IOC overshoots to the other side


@dataclass(frozen=True)
class EngineConfig:
    limits: Limits = field(default_factory=Limits)
    max_writes_per_second: int = 22
    risk_reserve: int = 6
    max_drawdown: float | None = None
    kill_file: str | None = None
    flatten_file: str | None = None
    kill_state_file: str | None = None
    kill_state_owner: int | None = None
    live_lock: str | None = None
    book_includes_own: bool = True  # verified by scripts/probe_exchange.py on the day
    flatten_on_stop: bool = True
    follow_limit_param: bool = False  # live: a published position_limit moves the gate. Replays and sims keep theirs


class MarketState:
    """Read-only view handed to strategies for one tick."""

    def __init__(
        self,
        now: float,
        universe: Universe,
        positions: Mapping[str, int],
        books: Mapping[str, object],
        own_orders: Mapping[str, list],
        params: Mapping[str, float],
        new_ticks: Mapping[str, list],
        new_fills: Mapping[str, list],
        new_news: list | None = None,
        book_includes_own: bool = True,
        paused: frozenset[str] = frozenset(),
        source_positions: Mapping[tuple[str, str], int] | None = None,
    ) -> None:
        self.paused = paused
        # Kept by reference: fills noted later this tick must reach the strategies that run after them.
        self.source_positions = {} if source_positions is None else source_positions
        self.new_news = new_news or []
        self.book_includes_own = book_includes_own
        self.now = now
        self.universe = universe
        self.positions = positions
        self.books = books
        self.own_orders = own_orders
        self.params = params
        self.new_ticks = new_ticks
        self.new_fills = new_fills
        self._external: dict[str, Book] = {}

    def external(self, instrument_id: str) -> Book:
        """The book with our resting volume removed, computed lazily and cached for the tick."""
        if instrument_id not in self._external:
            own = self.own_orders.get(instrument_id, []) if self.book_includes_own else []
            self._external[instrument_id] = external_book(
                self.books.get(instrument_id), own, self.tick_of(instrument_id)
            )
        return self._external[instrument_id]

    def tick_of(self, instrument_id: str) -> float:
        return self.universe.tick_of(instrument_id)

    def top(self, instrument_id: str) -> Top | None:
        return top_of_book(self.external(instrument_id))

    def raw_top(self, instrument_id: str) -> Top | None:
        return top_of_book(self.books.get(instrument_id))

    def position(self, instrument_id: str) -> int:
        return self.positions.get(instrument_id, 0)

    def position_of(self, source: str, instrument_id: str) -> int:
        """What one strategy's own takes have built, as far as fills have reported."""
        return self.source_positions.get((source, instrument_id), 0)


class Engine:
    def __init__(
        self,
        exchange,
        universe: Universe,
        strategies: list,
        config: EngineConfig | None = None,
        params: ParamStore | None = None,
        recorder=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.exchange = exchange
        self.universe = universe
        self.strategies = strategies
        self._can_take = any(getattr(strategy, "name", "") == "thin_leg_taker" for strategy in strategies)
        self._slow_connection = False
        self.config = config = config or EngineConfig()
        self.params = params or ParamStore()
        self.recorder = recorder or NullRecorder()
        self.clock = clock
        self.budget = WriteBudget(config.max_writes_per_second, config.risk_reserve, clock=clock)
        self.manager = QuoteManager(exchange, self.budget, config.limits, tick=universe.tick_of)
        self.stop = DrawdownStop(config.max_drawdown) if config.max_drawdown else None
        self.errors = 0
        self.reconnects = 0
        self.halted = False
        self._quoted: set[str] = set()
        self._rotation = 0
        self._consecutive_errors = 0
        self._listed: set | None = None
        self.telemetry_errors = 0
        self._backoff = 0.0
        self._next_connect = float("-inf")
        self._last_connect = float("-inf")
        self._lock_fd: int | None = None
        self._meta_recorded = False
        self._last_flatten = float("-inf")
        self.kill = KillSwitch(config.kill_state_file, config.kill_state_owner)
        self._next_kill_check = float("-inf")
        self._kill_verdict = None
        self._last_tops: dict[str, tuple] = {}
        self._quiet_until = float("-inf")
        self._last_data_change: float | None = None

    # --- single-writer lock: a second login takes over and breaks the live session -----------

    def acquire_lock(self) -> None:
        path = self.config.live_lock
        if not path:
            return
        self._clear_stale_lock(path)
        try:
            self._lock_fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"{path} exists: another process owns the exchange session. "
                "A second login would disconnect it."
            ) from exc
        os.write(self._lock_fd, str(os.getpid()).encode())

    @staticmethod
    def _clear_stale_lock(path: str) -> None:
        """A crashed bot must not block its own restart during the scored window."""
        try:
            with open(path) as handle:
                owner = int(handle.read().strip() or 0)
            if owner:
                os.kill(owner, 0)
        except FileNotFoundError:
            return
        except (ProcessLookupError, ValueError):
            os.unlink(path)
        except PermissionError:
            return

    def release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
            try:
                os.unlink(self.config.live_lock)
            except OSError:
                logger.warning("could not remove %s", self.config.live_lock)

    # --- tick --------------------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        if self.exchange.is_connected():
            if self.clock() - self._last_connect >= RECONNECT_RESET_S:
                self._backoff = 0.0
            return True
        now = self.clock()
        if now < self._next_connect:
            return False
        self._backoff = min(max(self._backoff * 2.0, RECONNECT_MIN_S), RECONNECT_MAX_S)
        self._next_connect = now + self._backoff
        self._last_connect = now
        try:
            self.exchange.connect()
        except Exception as exc:  # noqa: BLE001 - the client raises bare Exception on connect failure
            logger.error("reconnect failed: %s", exc)
            self._safe_record("reconnect_failed", {"error": str(exc)})
            return False
        self.reconnects += 1
        self._safe_record("reconnect", {"count": self.reconnects})
        self._quoted.clear()
        self.manager.forget_orders()
        self._last_data_change = None  # time spent disconnected says nothing about the feed
        self._on_reset()
        return True

    def _on_reset(self) -> None:
        """Every reconnect may be an exchange reset: prices, positions and trade ids restart. Nothing from the old round may
        trade in the new one, so each strategy drops its price memory and the engine stays quiet for `RESET_QUIET_S`."""
        self._quiet_until = self.clock() + RESET_QUIET_S
        self._last_tops.clear()
        self.kill.on_reset()
        for strategy in self.strategies:
            hook = getattr(strategy, "on_reset", None)
            if hook is None:
                continue
            try:
                hook()
            except Exception:  # a strategy bug must not stop the reconnect
                self.errors += 1
                logger.exception("strategy %s failed to reset", getattr(strategy, "name", "?"))
        self._safe_record("reset_quiet", {"seconds": RESET_QUIET_S})

    def _paused(self) -> frozenset[str]:
        """Markets can close and reopen at any moment, and the client keeps these flags current."""
        listed = self.exchange.get_instruments()
        self._check_listing(listed)
        return frozenset(
            iid
            for iid in self.universe.by_id
            if getattr(listed.get(iid), "paused", False) or getattr(listed.get(iid), "expired", False)
        )

    def _check_listing(self, listed: Mapping) -> None:
        """The plan is built from one listing. If a new one appears, as when a competition opens, only a rebuild can
        trade it, and otherwise the bot beats and earns nothing with no error anywhere. It compares with the listing
        and not the universe, so an instrument the event left out on purpose cannot restart it for ever."""
        if self._listed is None:
            self._listed = set(listed)
            return
        new = sorted(iid for iid, item in listed.items() if iid not in self._listed and not getattr(item, "expired", False))
        if new:
            logger.error("new instruments listed, restarting to plan for them: %s", new)
            self._safe_record("listing_changed", {"new": new})
            raise SystemExit(EXIT_LISTING_CHANGED)

    def _snapshot(self) -> MarketState:
        ids = list(self.universe.by_id)
        paused = self._paused()
        # A closed book reads as empty, so every strategy already knows not to trade or price off it.
        books = {iid: None if iid in paused else self.exchange.get_last_price_book(iid) for iid in ids}
        # The manager's view, not the raw cache: otherwise a quote the cache has not listed yet looks
        # like someone else's in the public book, and we would penny ourselves.
        own = {iid: self.manager.resting(iid) for iid in ids}
        ticks = {iid: self.exchange.poll_new_trade_ticks(iid) for iid in ids}
        fills = {iid: self.exchange.poll_new_trades(iid) for iid in ids}
        poll_news = getattr(self.exchange, "poll_new_social_media_feeds", None)
        news = poll_news() if poll_news else []
        for post in news:
            self._safe_record("news", {"post": getattr(post, "post", str(post))})
        return MarketState(
            self.clock(), self.universe, self.exchange.get_positions(), books, own,
            self._current_params(), ticks, fills, news,
            self.config.book_includes_own, paused, self.manager.source_positions,
        )

    def _current_params(self) -> Mapping:
        params = self.params.refresh()
        self._follow_limit(params)
        return self._with_taker_mode(scaled_sizes(params, self.kill.size_factor(params)))

    def _follow_limit(self, params: Mapping) -> None:
        """The challenge text may arrive after the start. Without this, a published limit changes quoting and not what is sent."""
        wanted = int(params["position_limit"])
        current = self.manager.limits.position
        if self.config.follow_limit_param and wanted != current:
            logger.warning("position limit now %d, was %d", wanted, current)
            self.manager.limits = self.manager.limits._replace(position=wanted)

    def params_in_force(self) -> Mapping:
        return self._current_params()

    def _with_taker_mode(self, params: Mapping) -> Mapping:
        """Switched on by hand, or by a slow connection if `taker_auto_ms` is set, which replays do not support by default."""
        if not self._can_take:
            return {**params, "taker_mode": 0}
        if not params["taker_mode"] and not self._connection_is_slow(params):
            return params
        return {
            **params, "taker_mode": 1,
            "hedge_threshold": max(params["hedge_threshold"], TAKER_HEDGE_THRESHOLD),
            "hedge_to": max(params["hedge_to"], TAKER_HEDGE_TO),
        }

    def _connection_is_slow(self, params: Mapping) -> bool:
        """On past the line, off only well below it, so it does not flap. Needs enough writes to mean anything."""
        line = params["taker_auto_ms"]
        if not line or self.manager.write_samples() < TAKER_MIN_SAMPLES:
            return False
        median = self.manager.write_ms()
        slow = median >= line if not self._slow_connection else median >= line / 2.0
        if slow != self._slow_connection:
            logger.warning("write time %.0f ms: taker mode %s", median, "on" if slow else "off")
            self._safe_record("taker_mode", {"on": slow, "write_ms": median})
        self._slow_connection = slow
        return slow

    def _pull_all(self) -> bool:
        """Pull everywhere, not only where we think we quoted. True once nothing is left.

        The write budget can run out mid-pull, so halted ticks call this again until it is true.
        """
        done = [self.manager.pull(iid) for iid in self.universe.by_id]
        self._quoted.clear()
        return all(done)

    def shutdown(self) -> None:
        """Best effort on exit: the exchange deletes our orders when the session ends anyway."""
        try:
            self._pull_all()
        except Exception:
            logger.exception("could not pull quotes on shutdown")
        self.release_lock()

    def _safe_record(self, kind: str, payload: Mapping) -> None:
        try:
            self.recorder.record(kind, payload)
        except Exception:
            logger.exception("recorder failed on %s", kind)

    def _safe_observe(self, state: MarketState) -> None:
        try:
            self.recorder.tick(state)
        except Exception:
            self.telemetry_errors += 1
            if self.telemetry_errors <= 5:
                logger.exception("recorder tick failed")

    def _notional(self, state: MarketState) -> float:
        """What the book could lose at the limit, so a loss kill scales with the event."""
        total = 0.0
        for iid, info in self.universe.by_id.items():
            if info.kind is not Kind.OPTION:
                current = mid(state.top(iid))
                if current is not None:
                    total += current * params_for(state.params, iid)["position_limit"]
        return total

    def _note_data_change(self, state: MarketState) -> None:
        """Every tick, even while halted, or a long halt would read as stale data on resuming."""
        # The external book, or pulling our own quotes would look like the market moving.
        tops = {iid: state.top(iid) for iid in self.universe.by_id if iid not in state.paused}
        if self._last_data_change is None or tops != self._last_tops:
            self._last_tops = tops
            self._last_data_change = state.now

    def _read_pnl(self) -> float | None:
        try:
            return self.exchange.get_pnl()
        except Exception:  # noqa: BLE001 - a broken read must not be what stops trading
            self.telemetry_errors += 1
            return None

    def _check_kill(self, state: MarketState):
        """Twice a second, since reading PnL costs more than anything else in a tick."""
        data_age = state.now - self._last_data_change
        if state.now >= self._next_kill_check:
            self._next_kill_check = state.now + KILL_CHECK_S
            self._kill_verdict = self.kill.check(
                state.now, self._read_pnl(), self._notional(state), self.manager.rejects, data_age,
                state.params,
            )
        return self._kill_verdict

    def _halt_reason(self, state: MarketState) -> tuple[str, bool] | None:
        """Why to stop and whether to flatten, or None."""
        if self.config.flatten_file and os.path.exists(self.config.flatten_file):
            return "flatten file present", True
        if self.config.kill_file and os.path.exists(self.config.kill_file):
            return "kill file present", False
        if not state.params.get("enabled", 1):
            return "disabled by params", False
        if self.stop is not None and self.stop.update(self.exchange.get_pnl()):
            return "drawdown stop", self.config.flatten_on_stop
        verdict = self._check_kill(state)
        if verdict:
            return verdict.reason, verdict.action is KillAction.FLATTEN
        return None

    def _collect(self, state: MarketState) -> list:
        intents: list = []
        taker_only = state.params["taker_mode"]
        for strategy in self.strategies:
            if taker_only and getattr(strategy, "name", "") not in TAKER_MODE_KEEPS:
                continue
            try:
                produced = strategy.on_tick(state)
                if getattr(strategy, "exclusive", False) and produced:
                    return produced  # the Flattener stops every other strategy opening risk
                intents.extend(produced)
            except Exception as exc:  # a strategy bug must not stop hedging or the session
                self.errors += 1
                logger.exception("strategy %s failed", getattr(strategy, "name", "?"))
                self._safe_record("strategy_error", {"name": getattr(strategy, "name", "?"), "error": str(exc)})
        return intents

    def _execute(self, intents: list, positions: Mapping[str, int]) -> None:
        pulled = {i.instrument_id for i in intents if isinstance(i, Pull)}
        takes = sorted(
            (i for i in intents if isinstance(i, Take)), key=lambda take: not take.hedge
        )
        for instrument_id in pulled:
            self.manager.pull(instrument_id)
            self._quoted.discard(instrument_id)
        for take in takes:
            self.manager.take(take, positions)
        desired = group_intents(intents)
        for instrument_id in self._quoted - set(desired) - pulled:
            self.manager.pull(instrument_id)
        self._quoted = set(desired) - pulled
        # Rotate the start so a spent write budget cannot starve the same instruments every tick,
        # then fix the stalest quotes first: those are the ones that can be picked off.
        ordered = sorted(desired)
        self._rotation = (self._rotation + 1) % max(len(ordered), 1)
        rotated = ordered[self._rotation :] + ordered[: self._rotation]
        urgency = {iid: self.manager.urgency(desired[iid]) for iid in rotated if iid not in pulled}
        for instrument_id in sorted(urgency, key=lambda iid: -urgency[iid]):
            self.manager.sync(desired[instrument_id], positions)

    def tick(self) -> None:
        if not self._ensure_connected():
            return
        try:
            state = self._snapshot()
            self._record_meta()
            self.manager.note_fills(state.new_fills)
            self._act(state)
            self._safe_observe(state)
            self._consecutive_errors = 0
        except ConnectionError as exc:
            logger.warning("connection lost mid-tick: %s", exc)
            self._safe_record("disconnect", {"error": str(exc)})
            self._on_tick_error(exc)  # a client still claiming to be connected would otherwise idle for ever
        except Exception as exc:  # noqa: BLE001 - the loop must survive any client or data error
            self._on_tick_error(exc)

    def _record_meta(self) -> None:
        """Tick sizes differ by instrument, and analysis needs them to read a capture correctly."""
        if not self._meta_recorded:
            self._meta_recorded = True
            ticks = {iid: info.tick for iid, info in self.universe.by_id.items()}
            self._safe_record("meta", {"ticks": ticks})

    def _act(self, state: MarketState) -> None:
        self._note_data_change(state)
        halt = self._halt_reason(state)
        if halt:
            reason, flatten = halt
            if not self.halted:
                logger.warning("halting: %s", reason)
                self._safe_record("halt", {"reason": reason, "flatten": flatten})
            self.halted = True
            self._pull_all()
            if flatten:
                self._flatten_out(state)
            return
        if self.halted:
            logger.warning("resuming")
            self._safe_record("resume", {})
        self.halted = False
        self.manager.reprice_ticks = state.params.get("reprice_ticks", 0.0)
        self.manager.reprice_by = {
            iid: values["reprice_ticks"]
            for iid, values in state.params.get(INSTRUMENTS_KEY, {}).items()
            if "reprice_ticks" in values
        }
        intents = self._collect(state)  # strategies still see the fresh books while quiet
        if self.clock() < self._quiet_until:
            self._pull_all()
            return
        self._execute(intents, state.positions)

    def _flatten_out(self, state: MarketState) -> None:
        """Kill switch: leave nothing open to lose more, as far as the visible depth allows."""
        now = self.clock()
        if now - self._last_flatten < FLATTEN_INTERVAL_S:
            return
        self._last_flatten = now
        for iid, position in state.positions.items():
            top = state.top(iid)
            if top is None:
                continue
            if position > 0 and top.bid is not None:
                exit_take = Take(iid, "ask", top.bid, min(position, top.bid_vol), True, "kill", True)
            elif position < 0 and top.ask is not None:
                exit_take = Take(iid, "bid", top.ask, min(-position, top.ask_vol), True, "kill", True)
            else:
                continue
            self.manager.take(exit_take, state.positions)

    def _on_tick_error(self, exc: Exception) -> None:
        """A dead session may raise anything while still claiming to be connected."""
        self.errors += 1
        self._consecutive_errors += 1
        if self._consecutive_errors <= 3 or self._consecutive_errors % 100 == 0:
            logger.exception("tick failed (%d in a row)", self._consecutive_errors)
        self._safe_record("tick_error", {"error": str(exc)})
        if self._consecutive_errors >= MAX_TICK_ERRORS:
            self._consecutive_errors = 0
            try:
                self.exchange.disconnect()
            except Exception:
                logger.exception("disconnect failed")

    def run(self, sleep_s: float = 0.0, should_stop: Callable[[], bool] = lambda: False) -> None:
        self.acquire_lock()
        try:
            while not should_stop():
                self.tick()
                if sleep_s:
                    time.sleep(sleep_s)
        finally:
            self.shutdown()
