"""Trade the house bots' known script: after each reset the market replays the same price path.

From 16:43 UTC on 6 Nov every round replayed the one before it tick for tick: the house bot sweeps a leader at the same
second after the round's first trade, and teams then race to take the follower's stale quotes. X timed 16 sweeps in two
rounds to 0.1 s (they differ by a constant 0.26 s, sd 0.07). This strategy takes the position a fraction of a second
before each sweep and closes it a few seconds after, and it pulls our market-making quotes on those stocks meanwhile.

Round start (t0) is the first public trade of a round: trade ids restart at 1 to 3 after a reset. A person can pin it
with `sched_t0` (epoch seconds). If the first moves of a round go the wrong way, the path is not the known one and the
strategy stands down until the next round. Off by default (`sched_enabled` 0).
"""

from __future__ import annotations

import time
from typing import NamedTuple

from otk.intents import Pull, Take
from otk.params import params_for

# A trade id this low marks the start of a round.
FIRST_TRADE_IDS = 20
# A new low id this long after t0 is a new round, not the same one.
MIN_ROUND_S = 60.0
# Stand down for the round after this many consecutive legs that moved at least WRONG_TICKS against us (C, M3):
# a flat leg is not wrong, or round 3's flat MU/CRWV leg at 651.9 would have given up about $2,000 of later edge.
WRONG_TO_STOP, WRONG_TICKS = 2, 2.0


class Leg(NamedTuple):
    at: float  # seconds after t0 when the house bot sweeps the leader
    instrument_id: str
    side: int  # +1 the price rises, -1 it falls
    lots: int


# X's table (round 2 offsets after the first trade), directions from A's and E's move lists. Followers ride the leader's
# sweep; moves 2 and 16 skip the follower, whose stale volume was too thin to pay.
SCHEDULE: tuple[Leg, ...] = (
    Leg(77.9, "NVDA", -1, 90), Leg(77.9, "SNDK", -1, 90),
    Leg(405.9, "AVGO", 1, 80), Leg(405.9, "NBIS", 1, 60),
    Leg(512.0, "VRT", -1, 50),
    Leg(651.9, "MU", 1, 70), Leg(651.9, "CRWV", 1, 70),
    Leg(739.9, "NVDA", -1, 50), Leg(739.9, "SNDK", -1, 50),
    Leg(823.9, "MU", 1, 70), Leg(823.9, "CRWV", 1, 70),
    Leg(834.9, "NVDA", 1, 90), Leg(834.9, "SNDK", 1, 90),
    Leg(881.9, "MU", -1, 90), Leg(881.9, "CRWV", -1, 70),
    Leg(1054.9, "VRT", 1, 90),
    Leg(1061.0, "AVGO", -1, 90), Leg(1061.0, "NBIS", -1, 60),
    Leg(1160.0, "NVDA", -1, 70), Leg(1160.0, "SNDK", -1, 70),
    Leg(1221.9, "AVGO", -1, 90), Leg(1221.9, "NBIS", -1, 60),
    Leg(1296.9, "MU", -1, 60), Leg(1296.9, "CRWV", -1, 60),
    Leg(1356.9, "AMD", 1, 80),
    Leg(1378.9, "AVGO", -1, 80),
    # Round 3 ran 4,710 s, and round 4 replays it (15 of 17 moves). E timed these from round 3's dashboard feed and mapped
    # them through 9 legs above seen in both rounds: round 4 = round 3 + 4705.24 s, spread +-0.5 s. Each `at` is 0.3 s
    # early, since a late entry trades after the sweep. Followers only where they moved 10 ticks or more in round 3.
    Leg(1715.5, "MU", -1, 70), Leg(1715.5, "CRWV", -1, 70),
    Leg(1750.5, "AMD", 1, 80),
    Leg(1763.6, "AMD", 1, 80),  # B: round 3's second AMD jump, +75 ticks, 13 s after the first
    Leg(1896.0, "AVGO", 1, 80), Leg(1896.0, "NBIS", 1, 50),
    Leg(1995.5, "MU", -1, 70), Leg(1995.5, "CRWV", -1, 70),
    Leg(2084.0, "AVGO", -1, 90), Leg(2084.0, "NBIS", -1, 60),
    Leg(2130.1, "AVGO", -1, 90), Leg(2130.1, "NBIS", -1, 90),
    Leg(2138.2, "AVGO", -1, 90), Leg(2138.2, "NBIS", -1, 60),  # B: AVGO -75 ticks, NBIS -16, 8 s after the crash
    Leg(2627.0, "AVGO", -1, 50),
    Leg(3036.5, "VRT", -1, 80),
    Leg(3233.0, "AVGO", 1, 90), Leg(3233.0, "NBIS", 1, 60),
    Leg(3412.5, "AMD", -1, 80),
    Leg(3489.0, "MU", 1, 70), Leg(3489.0, "CRWV", 1, 70),
    Leg(3648.5, "AMD", 1, 80),
    Leg(3961.5, "MU", -1, 80), Leg(3961.5, "CRWV", -1, 70),
    Leg(4045.5, "AVGO", 1, 60),
    Leg(4283.5, "VRT", 1, 60),
    Leg(4366.5, "AVGO", -1, 90), Leg(4366.5, "NBIS", -1, 50),
    Leg(4491.6, "AVGO", -1, 90), Leg(4491.6, "NBIS", -1, 60),
    Leg(4649.0, "AVGO", -1, 70),
)


def _mid(top) -> float | None:
    if top is None or top.bid is None or top.ask is None:
        return None
    return (top.bid + top.ask) / 2.0


class ScheduleTrader:
    """Enters each leg `sched_enter_s` before its sweep, exits `sched_exit_s` after, and pulls quotes in between."""

    name = "schedule"

    def __init__(self, instruments, schedule: tuple[Leg, ...] = SCHEDULE) -> None:
        listed = set(instruments)
        self.schedule = tuple(leg for leg in schedule if leg.instrument_id in listed)
        self.t0: float | None = None
        self._entry_mid: dict[int, float] = {}
        self._judged: set[int] = set()
        self._sent: set[int] = set()
        self._wrong = 0
        self._stood_down = False
        self._pinned = 0.0
        self._old_pin = 0.0

    def _new_round(self, t0: float) -> None:
        self.t0 = t0
        self._entry_mid.clear()
        self._judged.clear()
        self._sent.clear()
        self._wrong = 0
        self._stood_down = False

    def on_reset(self) -> None:
        """The exchange was reset: wait for the new round's first trade rather than carry the old clock over.

        A pin belongs to the round it was set in. Left in force it would keep the new round's first trade from starting
        the clock, and the schedule would stay silent. A new value is a new pin.
        """
        self.t0 = None
        self._old_pin = self._pinned

    def _watch_t0(self, state, pinned: float) -> None:
        if pinned > 0 and pinned != self._old_pin:
            if pinned != self._pinned:  # the pin is epoch seconds; state.now is the engine's monotonic clock (C, M2)
                self._pinned = pinned
                self._new_round(pinned - (time.time() - state.now))
            return
        for ticks in state.new_ticks.values():
            for tick in ticks:
                trade_id = getattr(tick, "trade_id", None)
                if trade_id is None or trade_id > FIRST_TRADE_IDS:
                    continue
                if self.t0 is None or state.now - self.t0 > MIN_ROUND_S:
                    self._new_round(state.now)
                return

    def _judge(self, index: int, leg: Leg, state) -> None:
        """At a leg's exit, count it wrong if the price moved against it; enough wrong legs early means a new path."""
        if index in self._judged or self._entry_mid.get(index) is None:
            return
        self._judged.add(index)
        now_mid = _mid(state.raw_top(leg.instrument_id))
        if now_mid is None:
            return
        against = leg.side * (self._entry_mid[index] - now_mid) >= WRONG_TICKS * state.tick_of(leg.instrument_id) - 1e-9
        self._wrong = self._wrong + 1 if against else 0
        if self._wrong >= WRONG_TO_STOP:
            self._stood_down = True

    def _toward(self, state, leg: Leg, target: int, params, reduce_only: bool) -> Take | None:
        held = state.position_of(self.name, leg.instrument_id)
        need = target - held
        top = state.raw_top(leg.instrument_id)
        if need == 0 or top is None or top.bid is None or top.ask is None:
            return None
        reach = params["sched_reach_ticks"] * state.tick_of(leg.instrument_id)
        buy = need > 0
        price = top.ask + reach if buy else top.bid - reach
        return Take(leg.instrument_id, "bid" if buy else "ask", round(price, 10), abs(need), False, self.name, reduce_only)

    def _enter(self, index: int, leg: Leg, state, params) -> Take | None:
        """One IOC per leg (C, M1): fills report late, so resending until the position shows would overshoot."""
        if index in self._sent:
            return None
        lots = round(leg.lots * params["sched_scale"])
        take = self._toward(state, leg, leg.side * lots, params, reduce_only=False)
        if take is not None:
            self._sent.add(index)
            self._entry_mid[index] = _mid(state.raw_top(leg.instrument_id))
        return take

    def on_tick(self, state) -> list:
        base = state.params
        self._watch_t0(state, base["sched_t0"])
        if self.t0 is None:
            return []
        elapsed = state.now - self.t0 - base["sched_offset_s"]
        intents: list = []
        entering: set[str] = set()
        for index, leg in enumerate(self.schedule):
            params = params_for(state.params, leg.instrument_id)
            start, end = leg.at - params["sched_enter_s"], leg.at + params["sched_exit_s"]
            if elapsed >= end:
                self._judge(index, leg, state)
            if not base["sched_enabled"] or not start - params["sched_pull_s"] <= elapsed <= end + params["sched_pull_s"]:
                continue
            intents.append(Pull(leg.instrument_id, self.name))
            if start <= elapsed < end and (index in self._sent or not self._stood_down):
                entering.add(leg.instrument_id)  # held until the exit time
                if elapsed < leg.at and not self._stood_down:  # never enter after the sweep (C, M1)
                    take = self._enter(index, leg, state, params)
                    if take is not None:
                        intents.append(take)
        # Whatever this strategy holds outside an entry window goes back to zero, even after it was switched off.
        for leg in {leg.instrument_id: leg for leg in self.schedule}.values():
            if leg.instrument_id not in entering and state.position_of(self.name, leg.instrument_id):
                take = self._toward(state, leg, 0, params_for(state.params, leg.instrument_id), reduce_only=True)
                if take is not None:
                    intents.append(take)
        return intents
