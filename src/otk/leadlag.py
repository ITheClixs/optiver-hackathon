"""Some stocks copy another a moment later: learn the pairs from the book, then lean on the lag. Off until `lag_enabled` = 1.

On the ETH 2026 practice exchange three pairs moved with a 60 s return correlation of 1.00 and the follower 1 to 3 s behind,
at a fixed ratio that is neither 1 nor the price ratio. Nothing in the instrument list says so, and Friday's names may differ, so
the pairs are learned while running. It always watches, so switching it on live needs no warm-up. It does two things:

- Quotes on the follower centre on where the leader says it is going, so a resting order is not left stale.
- When the follower is off by more than its half spread, it takes the follower by IOC. Market making then sheds the position.

The gap has no slow reversion worth fading: it closes in about 2 s, too fast and too small to cross both spreads for.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations, pairwise
from typing import NamedTuple

from otk.fair import mid
from otk.intents import Pull, Take
from otk.params import params_for

logger = logging.getLogger("otk.leadlag")

SAMPLE_S = 1.0
HISTORY_S = 1800
LEARN_EVERY_S = 30.0
RETURN_S = 30
MIN_RETURNS = 10  # with 28 pairs of 8 stocks, random walks clear this bar about once in thousands of tries
MIN_CORRELATION = 0.9
MAX_LAG_S = 3
MIN_LEAD = 0.2  # how much better one direction must explain the other, or there is no leader to lean on
BETA_TOLERANCE = 0.05
MEAN_HALF_LIFE_S = 30.0
HEDGE_SOURCE = "lead_lag_hedge"
# After a reset the book is a few thin quotes for about 6 s, and the house walls land just after the first trade. A baseline
# set on that is wrong, and at 20:10:30 it sold 60 CRWV in the round's first second. Start after the open instead.
OPEN_WAIT_S = 15.0


class LagLink(NamedTuple):
    leader: str
    beta: float  # follower's price move per unit of the leader's
    correlation: float


# The pairs in the organisers' order log for 6 Nov, follower to leader. The ratios come from 14:00 to 16:00 UTC and held in
# later rounds. Learning needs 300 s after every start, and CRWV and NBIS sit near MIN_CORRELATION, so they drop in and out.
KNOWN_LINKS = {
    "SNDK": LagLink("NVDA", 0.43, 1.0),
    "NBIS": LagLink("AVGO", 0.40, 1.0),
    "CRWV": LagLink("MU", 1.10, 1.0),
}


def _changes(series: Sequence[float], step: int) -> list[float]:
    """Changes over `step` samples, counted back from the newest."""
    start = (len(series) - 1) % step
    points = series[start::step]
    return [after - before for before, after in pairwise(points)]


def _correlation(x: Sequence[float], y: Sequence[float]) -> float:
    count = min(len(x), len(y))
    if count < 2:
        return 0.0
    mean_x, mean_y = sum(x[:count]) / count, sum(y[:count]) / count
    cross = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x[:count])
    var_y = sum((b - mean_y) ** 2 for b in y[:count])
    return cross / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0.0


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    var_x = sum((a - mean_x) ** 2 for a in x)
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / var_x if var_x > 0 else 0.0


def _lead(first: Sequence[float], second: Sequence[float]) -> float:
    """How well `first` now explains `second` a moment later. Positive means `first` leads."""
    x, y = _changes(first, 1), _changes(second, 1)
    return sum(_correlation(x[:-lag], y[lag:]) - _correlation(y[:-lag], x[lag:]) for lag in range(1, MAX_LAG_S + 1))


def _aligned_moves(first: Sequence[float], second: Sequence[float]) -> tuple[float, list[float], list[float]]:
    """The best correlation of the two's moves, and those moves, with either allowed to run up to MAX_LAG_S behind.

    Without this, a jump just before a window's end lands in different windows for the two, and a true pair looks broken.
    """
    best = (-2.0, [], [])
    for lag in range(-MAX_LAG_S, MAX_LAG_S + 1):
        x = first[: len(first) - max(lag, 0)] if lag >= 0 else first[-lag:]
        y = second[lag:] if lag >= 0 else second[: len(second) + lag]
        moves_x, moves_y = _changes(x, RETURN_S), _changes(y, RETURN_S)
        correlation = _correlation(moves_x, moves_y)
        if correlation > best[0]:
            best = (correlation, moves_x, moves_y)
    return best


def learn_links(history: Mapping[str, Sequence[float]]) -> dict[str, LagLink]:
    """{follower: link} from mids sampled together once a second. A stock joins at most one pair, best correlation first."""
    if not history or min(len(series) for series in history.values()) < RETURN_S * MIN_RETURNS:
        return {}
    scored = sorted(
        ((*_aligned_moves(history[a], history[b]), a, b) for a, b in combinations(sorted(history), 2)),
        key=lambda item: item[0],
        reverse=True,
    )
    links: dict[str, LagLink] = {}
    used: set[str] = set()
    for correlation, moves_a, moves_b, a, b in scored:
        if correlation < MIN_CORRELATION:
            break
        if a in used or b in used:
            continue
        lead = _lead(history[a], history[b])
        if abs(lead) < MIN_LEAD:
            continue
        link = LagLink(a, _slope(moves_a, moves_b), correlation) if lead > 0 else LagLink(b, _slope(moves_b, moves_a), correlation)
        links[b if lead > 0 else a] = link
        used |= {a, b}
    return links


class LeadLag:
    """Watches every stock it is given. Trades and shifts quotes only with `lag_enabled` on for the follower."""

    name = "lead_lag"

    def __init__(self, instruments: Sequence[str]) -> None:
        self.instruments = list(instruments)
        self.links: dict[str, LagLink] = {}
        self._history: dict[str, deque] = {iid: deque(maxlen=HISTORY_S) for iid in self.instruments}
        self._next_sample = float("-inf")
        self._next_learn = 0.0
        self._residual: dict[str, float] = {}
        self._mean: dict[str, float] = {}
        self._mean_at: dict[str, float] = {}
        self._last_take: dict[str, float] = {}
        self._guard_until: dict[str, float] = {}
        self._warmup_until: dict[str, float] = {}
        self._guard_armed: dict[str, bool] = {}
        self._hedged: dict[str, int] = {}
        self._carry: dict[str, float] = {}
        self._kept_links = False
        self._hold_until: float | None = None
        self._holding = False

    def on_reset(self) -> None:
        """The exchange was reset: prices restart at the round's levels, so every price memory is from another market.

        The slow follower-leader gap is the dangerous one: at 18:11 it read the restart as a huge gap and traded on it,
        carrying a trade of one round into the next. The pairs themselves hold across rounds, so they are kept and the
        take works again at once, while the history refills before they are relearned.
        """
        for series in self._history.values():
            series.clear()
        for memory in (self._residual, self._mean, self._mean_at, self._last_take, self._guard_until,
                       self._warmup_until, self._guard_armed, self._hedged, self._carry):
            memory.clear()
        self._next_sample = float("-inf")
        self._kept_links = bool(self.links)
        self._holding, self._hold_until = True, None

    def _waiting_for_the_open(self, now: float) -> bool:
        if not self._holding:
            return False
        if self._hold_until is None:
            self._hold_until = now + OPEN_WAIT_S
        self._holding = now < self._hold_until
        return self._holding

    def _sample(self, state, mids: Mapping[str, float | None]) -> None:
        if state.now < self._next_sample or any(mids[iid] is None for iid in self.instruments):
            return
        self._next_sample = state.now + SAMPLE_S
        for iid in self.instruments:
            self._history[iid].append(mids[iid])

    def _seeded(self, state) -> dict[str, LagLink]:
        """The known pairs on this exchange, with `lag_seed` on. They hold from the first tick and nothing learned displaces them."""
        if not state.params["lag_seed"]:
            return {}
        return {
            follower: link
            for follower, link in KNOWN_LINKS.items()
            if follower in self.instruments and link.leader in self.instruments
        }

    def _learn(self, now: float, seeded: Mapping[str, LagLink] | None = None) -> None:
        if now < self._next_learn:
            return
        self._next_learn = now + LEARN_EVERY_S
        history = {iid: list(values) for iid, values in self._history.items()}
        if self._kept_links and min(len(values) for values in history.values()) < HISTORY_S // 3:
            return  # after a reset, keep the known pairs until enough of the new round has been seen
        self._kept_links = False
        learned = {**learn_links(history), **(seeded or {})}
        for follower, link in learned.items():
            old = self.links.get(follower)
            if old and old.leader == link.leader and abs(link.beta - old.beta) <= BETA_TOLERANCE * abs(old.beta):
                learned[follower] = old  # the slow mean is in this ratio's units, so a small change must not reset it
            else:
                self._mean.pop(follower, None)
                logger.info("lead-lag: %s follows %s, ratio %.3f, correlation %.3f", follower, link.leader, link.beta, link.correlation)
        for follower in self.links.keys() - learned.keys():
            logger.info("lead-lag: %s no longer follows %s", follower, self.links[follower].leader)
        self.links = learned

    def _update_residual(self, state, follower: str, link: LagLink, mids: Mapping[str, float | None]) -> None:
        leader_mid, follower_mid = mids.get(link.leader), mids.get(follower)
        if leader_mid is None or follower_mid is None:
            self._residual.pop(follower, None)
            return
        gap = follower_mid - link.beta * leader_mid
        if follower not in self._mean:
            self._mean[follower], self._mean_at[follower] = gap, state.now
        residual = gap - self._mean[follower]
        elapsed = state.now - self._mean_at[follower]
        self._mean[follower] += (1.0 - 0.5 ** (elapsed / MEAN_HALF_LIFE_S)) * residual
        self._mean_at[follower] = state.now
        self._residual[follower] = residual

    @staticmethod
    def _reach(params, residual: float, needed: float, tick: float) -> float:
        """How far past the touch a take may go, in whole ticks: `lag_sweep_ticks`, never past half the expected follower
        move, and never so far that what is left of the move no longer beats the half spread and the edge.

        After a leader jump the follower's touch is thin and a stale wall sits a few levels behind it. An IOC at the touch
        stops at the first level; one priced a few ticks through takes the wall too.
        """
        extra = min(params["lag_sweep_ticks"] * tick, abs(residual) / 2.0, abs(residual) - needed)
        return max(0, math.floor(extra / tick + 1e-9)) * tick

    def _hedge(self, state, follower: str, link: LagLink) -> Take | None:
        """Hedge each lot lead-lag's own takes filled with -ratio lots of the leader, once, from fills as attributed.

        One IOC per change, so market making can unwind both legs afterwards without this re-adding the hedge. Rounding is
        carried over, and after a disconnect forgets the attributions it starts afresh rather than hedging back.
        """
        attributed = getattr(state, "source_positions", {})
        key = (self.name, follower)
        if key not in attributed:
            self._hedged.pop(follower, None)
            self._carry.pop(follower, None)
            return None
        if not params_for(state.params, follower)["lag_hedge"]:
            # Keep up while off, or switching it on would hedge every lot taken meanwhile, at once.
            self._hedged[follower] = attributed[key]
            self._carry.pop(follower, None)
            return None
        top = state.top(link.leader)
        if top is None or top.bid is None or top.ask is None:
            return None  # wait for a leader book: the change stays unhedged until then
        held = attributed[key]
        wanted = -link.beta * (held - self._hedged.get(follower, 0)) + self._carry.get(follower, 0.0)
        self._hedged[follower] = held
        lots = round(wanted)
        self._carry[follower] = wanted - lots
        if lots == 0:
            return None
        buy = lots > 0
        return Take(link.leader, "bid" if buy else "ask", top.ask if buy else top.bid, abs(lots), True, HEDGE_SOURCE)

    def _take(self, state, follower: str) -> Take | None:
        params = params_for(state.params, follower)
        residual, top = self._residual.get(follower), state.top(follower)
        enabled = params["lag_enabled"] or params["lag_take_enabled"]  # the take alone, without the quote shift
        if not enabled or residual is None or top is None or top.bid is None or top.ask is None:
            return None
        if state.now - self._last_take.get(follower, float("-inf")) < params["lag_pause_s"]:
            return None  # fills report late: without a pause the same view would be sent again
        needed = (top.ask - top.bid) / 2.0 + params["lag_edge_ticks"] * state.tick_of(follower)
        if abs(residual) <= needed:
            return None
        cheap = residual < 0
        room = int(params["lag_max_position"]) - (state.position(follower) if cheap else -state.position(follower))
        volume = min(int(params["lag_clip"]), room)
        if volume <= 0:
            return None
        self._last_take[follower] = state.now
        reach = self._reach(params, residual, needed, state.tick_of(follower))
        price = top.ask + reach if cheap else top.bid - reach
        return Take(follower, "bid" if cheap else "ask", round(price, 10), volume, False, self.name)

    def _warmup_pulls(self, state, mids: Mapping[str, float | None]) -> list:
        """Until any pair is learned, a jump of `lag_warmup_ticks` in one stock pulls every other stock for `lag_guard_s`.

        For about 300 s after each start nothing is learned, so the follower guard knows no follower. At 18:13 NVDA gapped
        78 s after a restart and teams swept about 220 lots into our SNDK bids. Not knowing which stock follows, it stands
        every other one aside briefly. The jump is measured against the last 1 s sample, so a gap over a few updates counts.
        """
        if self.links:
            self._warmup_until.clear()
            return []
        for iid in self.instruments:
            params = params_for(state.params, iid)
            history, value = self._history[iid], mids.get(iid)
            if params["lag_warmup_ticks"] <= 0 or value is None or not history:
                continue
            if abs(value - history[-1]) >= params["lag_warmup_ticks"] * state.tick_of(iid) - 1e-9:
                for other in self.instruments:
                    if other != iid:
                        until = state.now + params_for(state.params, other)["lag_guard_s"]
                        self._warmup_until[other] = max(self._warmup_until.get(other, float("-inf")), until)
        return [Pull(iid, self.name) for iid in self.instruments if state.now < self._warmup_until.get(iid, float("-inf"))]

    def on_tick(self, state) -> list:
        mids = {iid: mid(state.top(iid)) for iid in self.instruments}
        warmup = self._warmup_pulls(state, mids)
        self._sample(state, mids)
        if self._waiting_for_the_open(state.now):
            return list(warmup)
        self._learn(state.now, self._seeded(state))
        intents: list = list(warmup)
        for follower, link in self.links.items():
            self._update_residual(state, follower, link, mids)
            take = self._take(state, follower)
            if take is not None:
                intents.append(take)
            guard = self._guard(state, follower)
            if guard is not None:
                intents.append(guard)
            hedge = self._hedge(state, follower, link)
            if hedge is not None:
                intents.append(hedge)
        return intents

    def _guard(self, state, follower: str) -> Pull | None:
        """After a leader jump, take our follower quotes out for `lag_guard_s`: teams sweep the stale follower, us included.

        The trigger is the take's: the follower is off by more than its half spread plus `lag_edge_ticks`. Unlike the quote
        shift, this adds no inventory. A Pull suppresses every quote on the follower for the tick, whoever would send it.
        Each guard lasts `lag_guard_s` from its first trigger, and the next one waits until the follower is back in line,
        so a pair that stays apart, as a false one would, cannot keep the follower unquoted.
        """
        params = params_for(state.params, follower)
        if params["lag_guard_s"] <= 0:
            return None
        residual, top = self._residual.get(follower), state.top(follower)
        if residual is not None and top is not None and top.bid is not None and top.ask is not None:
            needed = (top.ask - top.bid) / 2.0 + params["lag_edge_ticks"] * state.tick_of(follower)
            if abs(residual) <= needed:
                self._guard_armed[follower] = True
            elif self._guard_armed.get(follower, True):
                self._guard_until[follower] = state.now + params["lag_guard_s"]
                self._guard_armed[follower] = False
        if state.now < self._guard_until.get(follower, float("-inf")):
            return Pull(follower, self.name)
        return None

    def shift(self, state, instrument_id: str) -> float:
        """How far this stock's fair value should move to where its leader says it is going."""
        if instrument_id not in self.links or not params_for(state.params, instrument_id)["lag_enabled"]:
            return 0.0
        return -self._residual.get(instrument_id, 0.0)


class LagAdjustedFair:
    """A fair value source, moved by the lead-lag shift. With `lag_enabled` off it gives what `base` gives."""

    def __init__(self, base: Callable, lag: LeadLag) -> None:
        self.base, self.lag = base, lag

    def __call__(self, state, instrument_id: str) -> float | None:
        fair = self.base(state, instrument_id)
        return None if fair is None else fair + self.lag.shift(state, instrument_id)
