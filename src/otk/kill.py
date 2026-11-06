"""Kill switches. Each answers one question: is carrying on worse than stopping?

Two kinds of stop, because they fix different problems:

  * a loss (drawdown, max loss, fast bleed) means our model is wrong, so close everything, wait out
    `kill_cooloff_s`, and come back at `kill_resume_size`. `kill_max_trips` losses in a row latch it
    until a new launch, and a crash restart keeps the latch: a bot that keeps tripping should not keep
    trying. Latching on the first stop would give up too much PnL on one bad minute, so the first are temporary.
  * bad plumbing (a reject storm, stale market data) says nothing about our inventory, so only pull
    quotes, and resume by itself once the cause clears.

Every limit is a live parameter, off at 0, and `scripts/kill_drill.py` rehearses them all.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger("otk.kill")

VELOCITY_WINDOW_S = 60.0
REJECT_WINDOW_S = 10.0
_NEVER = float("-inf")


class KillAction(Enum):
    PULL = "pull"
    FLATTEN = "flatten"


class KillVerdict(NamedTuple):
    action: KillAction
    reason: str


class KillSwitch:
    """`state_file` and `owner` let a latch survive a crash. `owner` is the supervising parent, so a
    restart keeps the state, while a new launch starts clean and practice cannot latch the final.
    """

    def __init__(self, state_file: str | Path | None = None, owner: int | None = None) -> None:
        self.state_file = Path(state_file) if state_file else None
        self.owner = owner
        self.trips = 0
        self.latched = False
        self.last_reason = ""
        self._load_state()
        self._start: float | None = None
        self._peak: float | None = None
        self._cooling_until = _NEVER
        self._paused_until = _NEVER
        self._pnl_points: deque[tuple[float, float]] = deque()
        self._reject_points: deque[tuple[float, int]] = deque()

    def on_reset(self) -> None:
        """An exchange reset takes PnL back to zero, and that is not a loss. The limit scales with what we hold, so a peak
        kept from the old round could trip on the first small position of the new one. Trips and latches stay."""
        self._start = None
        self._peak = None
        self._pnl_points.clear()

    def _load_state(self) -> None:
        if self.state_file is None or self.owner is None:
            return
        try:
            saved = json.loads(self.state_file.read_text())
            if saved["owner"] != self.owner:
                return
            trips, latched, reason = int(saved["trips"]), bool(saved["latched"]), str(saved["reason"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        self.trips, self.latched, self.last_reason = trips, latched, reason

    def _save_state(self) -> None:
        if self.state_file is None or self.owner is None:
            return
        state = {"owner": self.owner, "trips": self.trips, "latched": self.latched, "reason": self.last_reason}
        try:
            self.state_file.write_text(json.dumps(state))
        except OSError:
            logger.warning("could not save kill state to %s", self.state_file)

    def size_factor(self, params: Mapping) -> float:
        return params["kill_resume_size"] if self.trips else 1.0

    # --- losses --------------------------------------------------------------------------------

    def _loss_reason(self, now: float, pnl: float, notional: float, params: Mapping) -> str | None:
        limit = params["kill_drawdown"] or params["kill_drawdown_frac"] * notional
        if limit and self._peak - pnl > limit:
            return f"drawdown {self._peak - pnl:.0f} above {limit:.0f}"
        if params["kill_max_loss"] and self._start - pnl > params["kill_max_loss"]:
            return f"session loss {self._start - pnl:.0f} above {params['kill_max_loss']:.0f}"
        rate = params["kill_loss_per_min"]
        if rate and self._pnl_points:
            then, past = self._pnl_points[0]
            if now - then >= VELOCITY_WINDOW_S / 2.0 and past - pnl > rate:
                return f"lost {past - pnl:.0f} within a minute, above {rate:.0f}"
        return None

    def _track_pnl(self, now: float, pnl: float) -> None:
        self._start = pnl if self._start is None else self._start
        self._peak = pnl if self._peak is None else max(self._peak, pnl)
        self._pnl_points.append((now, pnl))
        while now - self._pnl_points[0][0] > VELOCITY_WINDOW_S:
            self._pnl_points.popleft()

    def _trip(self, now: float, reason: str, params: Mapping) -> KillVerdict:
        self.trips += 1
        self.last_reason = reason
        self._cooling_until = now + params["kill_cooloff_s"]
        self.latched = self.trips >= params["kill_max_trips"]
        self._save_state()
        return KillVerdict(KillAction.FLATTEN, reason)

    def _resume(self, pnl: float) -> None:
        """Measure afresh, or the same old loss would trip it again on the first check."""
        self._cooling_until = _NEVER
        self._start = pnl
        self._peak = pnl
        self._pnl_points.clear()

    def _check_losses(self, now: float, pnl: float | None, notional: float, params: Mapping) -> KillVerdict | None:
        if self.latched:
            return KillVerdict(KillAction.FLATTEN, f"latched after {self.trips} loss stops: {self.last_reason}")
        if self._cooling_until != _NEVER:
            if now < self._cooling_until:
                return KillVerdict(KillAction.FLATTEN, f"cooling off: {self.last_reason}")
            if pnl is not None:
                self._resume(pnl)
            return None
        if pnl is None:
            return None
        self._track_pnl(now, pnl)
        reason = self._loss_reason(now, pnl, notional, params)
        return self._trip(now, reason, params) if reason else None

    # --- plumbing ------------------------------------------------------------------------------

    def _check_rejects(self, now: float, rejects: int, params: Mapping) -> KillVerdict | None:
        limit = params["kill_reject_limit"]
        self._reject_points.append((now, rejects))
        while now - self._reject_points[0][0] > REJECT_WINDOW_S:
            self._reject_points.popleft()
        if limit and rejects - self._reject_points[0][1] > limit:
            self._paused_until = now + params["kill_pause_s"]
            self._reject_points.clear()
            self._reject_points.append((now, rejects))
            self.last_reason = f"more than {limit:.0f} rejects in {REJECT_WINDOW_S:.0f}s"
        if now < self._paused_until:
            return KillVerdict(KillAction.PULL, self.last_reason)
        return None

    def check(
        self, now: float, pnl: float | None, notional: float, rejects: int, data_age: float, params: Mapping
    ) -> KillVerdict | None:
        """`pnl` is None when it could not be read this time: losses stay as last known."""
        if not params["kill_enabled"]:
            return None
        verdict = self._check_losses(now, pnl, notional, params)
        if verdict:
            return verdict
        verdict = self._check_rejects(now, rejects, params)
        if verdict:
            return verdict
        if params["kill_stale_s"] and data_age > params["kill_stale_s"]:
            self.last_reason = f"market data stale for {data_age:.1f}s"
            return KillVerdict(KillAction.PULL, self.last_reason)
        return None
