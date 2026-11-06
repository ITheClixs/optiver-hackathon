"""Reusable strategy components, composed into a priority-ordered stack per event format.

Each component is a mechanism named in a disclosed top Optibook result. All read
`state` and return intents, and all parameters come from `state.params`, so agents retune them
without a restart. Order in the stack is priority: Pull beats quotes, and the first quote wins.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

from otk.fair import (
    Top,
    bs_price,
    etf_fair,
    future_fair,
    imbalance,
    index_value,
    microprice,
    mid,
    near_touch_imbalance,
    round_ask,
    round_bid,
    wall_mid,
)
from otk.features import ChangeVol, Ewma, TimedZ
from otk.instruments import Kind, delta_per_lot, parse_instrument_id, theo_ready, years_to_expiry
from otk.intents import LadderIntent, Pull, QuoteIntent, Take
from otk.leadlag import LagAdjustedFair, LeadLag
from otk.news import AsyncClassifier, NewsSignal
from otk.optioncheck import OptionMarketCheck
from otk.params import params_for
from otk.quoter import QuoteParams, compute_quotes
from otk.risk import DeltaWatch, Limits, clip_order
from otk.schedule import ScheduleTrader

FairSource = Callable[[object, str], float | None]
DEFAULT_OPTION_VOL = 0.25
ARB_RESEND_S = 0.5
THIN_TAKER_MIN_GAP_S = 0.03
THIN_TAKER_HEARTBEAT_S = 0.5
TICK_ROUNDING = 1e-9


def quote_params(params: Mapping[str, float]) -> QuoteParams:
    """Everything `compute_quotes` needs, in one place, from the live parameters."""
    return QuoteParams(
        base_spread_ticks=params["base_spread_ticks"],
        vol_factor=params["vol_factor"],
        max_vol_ticks=params["max_vol_ticks"],
        imbalance_ticks=params["imbalance_ticks"],
        skew_ticks=params["skew_ticks"],
        skew_curve=params["skew_curve"],
        max_deviation_ticks=params["max_deviation_ticks"],
        size=int(params["size"]),
        signal_size=params["signal_size"],
    )


def micro_fair(state, instrument_id: str) -> float | None:
    """Microprice of the external book, falling back on the wall mid."""
    fair = microprice(state.top(instrument_id))
    return fair if fair is not None else wall_mid(state.external(instrument_id))


class MarketMaker:
    """Two-sided quoting around a fair value, with a jump guard that steps out of the way."""

    name = "market_maker"

    def __init__(
        self,
        instruments: Sequence[str],
        fair: FairSource = micro_fair,
        position_of: Callable[[object, str], int] | None = None,
        use_imbalance: bool = True,
        guard_own_spread: bool = True,
    ) -> None:
        self.guard_own_spread = guard_own_spread
        self.instruments = list(instruments)
        self.fair = fair
        self.position_of = position_of or (lambda state, iid: state.position(iid))
        self.use_imbalance = use_imbalance
        self._vol: dict[str, ChangeVol] = {}
        self._last_fair: dict[str, float] = {}
        self._paused_until: dict[str, float] = {}

    def _jumped(self, state, iid: str, fair: float) -> bool:
        last = self._last_fair.get(iid)
        self._last_fair[iid] = fair
        params = params_for(state.params, iid)
        if last is not None and abs(fair - last) >= params["jump_ticks"] * state.tick_of(iid):
            self._paused_until[iid] = state.now + params["jump_cooldown_s"]
            return True
        return state.now < self._paused_until.get(iid, float("-inf"))

    def _imbalance(self, state, iid: str, top: Top) -> float:
        if not self.use_imbalance:
            return 0.0
        band = params_for(state.params, iid)["near_touch_ticks"] * state.tick_of(iid)
        return near_touch_imbalance(state.external(iid), band) if band > 0 else imbalance(top)

    def _intent(self, state, iid: str):
        fair = self.fair(state, iid)
        top = state.top(iid)
        if fair is None or top is None or top.bid is None or top.ask is None:
            return Pull(iid, self.name)
        tick = state.tick_of(iid)
        params = params_for(state.params, iid)
        too_wide = top.ask - top.bid > params["max_spread_ticks"] * tick
        if self.guard_own_spread and too_wide:
            return Pull(iid, self.name)  # junk levels: public books show 10 and 210 around 153
        if abs(fair - (top.bid + top.ask) / 2.0) > params["circuit_ticks"] * tick:
            return Pull(iid, self.name)  # circuit breaker from the Imperial 2026 winner
        vol = self._vol.setdefault(iid, ChangeVol()).update(fair)
        if self._jumped(state, iid, fair):
            return Pull(iid, self.name)
        quote = compute_quotes(
            fair,
            self.position_of(state, iid),
            int(params["position_limit"]),
            vol,
            self._imbalance(state, iid, top),
            quote_params(params),
            tick,
            top,
        )
        if params["ladder_levels"] > 1:
            return self._ladder(iid, quote, params, tick)
        return QuoteIntent(iid, quote.bid, quote.bid_size, quote.ask, quote.ask_size, self.name)

    def _ladder(self, iid: str, quote, params: Mapping, tick: float) -> LadderIntent:
        """Deeper rungs catch sweeps at better prices. Takers of 200 to 500 lots walk the book."""
        step = params["ladder_step_ticks"] * tick
        decay = params["ladder_size_decay"]

        def rungs(price: float, size: int, direction: int) -> tuple:
            levels = []
            for level in range(int(params["ladder_levels"])):
                rung_size = round(size * decay**level)
                if rung_size > 0:
                    levels.append((price + direction * level * step, rung_size))
            return tuple(levels)

        return LadderIntent(
            iid, rungs(quote.bid, quote.bid_size, -1), rungs(quote.ask, quote.ask_size, 1), self.name
        )

    def on_tick(self, state) -> list:
        return [self._intent(state, iid) for iid in self.instruments]


def pair_room(state, thin: str, thin_side: str, liquid: str, liquid_side: str, wanted: int) -> int:
    """Clip both legs together, so a limit on either leg cannot strand the other."""
    limits = Limits(position=int(state.params["position_limit"]))

    def room(iid: str, side: str) -> int:
        orders = getattr(state, "own_orders", {}).get(iid, [])
        open_buy = sum(o.volume for o in orders if o.side == "bid")
        open_sell = sum(o.volume for o in orders if o.side == "ask")
        return clip_order(side, wanted, state.position(iid), open_buy, open_sell, limits)

    return min(room(thin, thin_side), room(liquid, liquid_side))


class DualOffset:
    """Slow estimate of a persistent gap between the listings, capped in ticks.

    One Imperial 2026 team measured offsets near +/-0.5 and found a 1:1 assumption pinned it at the
    position limit 60 to 70% of the time. Its fix was primary plus an EWMA of the gap, capped.
    """

    def __init__(self) -> None:
        self._gap: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def update(self, state, primary: str, dual: str) -> float:
        top_p, top_d = state.top(primary), state.top(dual)
        mid_p, mid_d = mid(top_p), mid(top_d)
        if mid_p is not None and mid_d is not None:
            gap = mid_d - mid_p
            elapsed = state.now - self._last.get(dual, state.now)
            half_life = params_for(state.params, dual)["dual_offset_half_life_s"]
            weight = 1.0 - 0.5 ** (elapsed / half_life)
            previous = self._gap.get(dual, 0.0)  # the organisers' prior: long-run equal prices
            self._gap[dual] = previous + weight * (gap - previous)
            self._last[dual] = state.now
        cap = params_for(state.params, dual)["dual_offset_cap_ticks"] * state.tick_of(dual)
        return max(-cap, min(cap, self._gap.get(dual, 0.0)))


class DualListingArb:
    """Optiver's first exercise: IOC arb across the pair, then passive quotes on the thin leg."""

    name = "dual_listing"

    def __init__(self, pairs: Sequence[tuple[str, str]], quote: bool = True) -> None:
        self.pairs = list(pairs)
        self._duals = {dual: primary for primary, dual in self.pairs}
        self._offset = DualOffset()
        self._last_arb: dict[str, tuple] = {}
        self._quoter = (
            MarketMaker(
                sorted(self._duals), fair=self._dual_fair, use_imbalance=False,
                guard_own_spread=False,
            )
            if quote
            else None
        )

    def _dual_fair(self, state, dual: str) -> float | None:
        primary = self._duals[dual]
        top = state.top(primary)
        fair = microprice(top)
        params = params_for(state.params, dual)
        if fair is None and primary in state.paused and params["dual_quote_when_closed"]:
            return micro_fair(state, dual)  # the primary is shut: this book is all there is
        limit = params_for(state.params, primary)["max_spread_ticks"] * state.tick_of(primary)
        if fair is None or top.ask - top.bid > limit:
            return None
        return fair + self._offset.update(state, primary, dual)

    @staticmethod
    def _crossed(buy: Top, sell: Top, edge: float) -> tuple[float, float, int] | None:
        if buy.ask is None or sell.bid is None or sell.bid - buy.ask < edge - 1e-9:
            return None
        return buy.ask, sell.bid, min(buy.ask_vol, sell.bid_vol)

    def _book_is_news(self, state, dual: str, seen: tuple) -> bool:
        """A book we already traded against has not reported that trade yet: sending again would double up."""
        last = self._last_arb.get(dual)
        return last is None or last[0] != seen or state.now - last[1] >= ARB_RESEND_S

    def _arb(self, state, primary: str, dual: str) -> list:
        top_p, top_d = state.top(primary), state.top(dual)
        if top_p is None or top_d is None or not self._book_is_news(state, dual, (top_p, top_d)):
            return []
        params = params_for(state.params, dual)
        tick = max(state.tick_of(primary), state.tick_of(dual))
        mid_p, mid_d = mid(top_p), mid(top_d)
        if mid_p is not None and mid_d is not None and abs(mid_p - mid_d) > params["circuit_ticks"] * tick:
            return []  # `instrument_group` can pair two different stocks, and that gap is no edge and never closes
        edge = params["arb_edge_ticks"] * tick
        size = int(params["size"])
        offset = self._offset.update(state, primary, dual)
        pos_d = state.position(dual)
        # Listings cannot be transferred, so a reducing arb is what frees capacity: it needs no edge.
        sell_edge = 0.0 if pos_d > 0 else edge + max(offset, 0.0)
        buy_edge = 0.0 if pos_d < 0 else edge + max(-offset, 0.0)
        sell_dual = self._crossed(top_p, top_d, sell_edge)
        if sell_dual:
            clip = self._clip(size, sell_dual[1] - sell_dual[0], pos_d <= 0, pos_d, params, tick)
            volume = self._pair_volume(state, dual, "ask", primary, "bid", min(sell_dual[2], clip))
            if pos_d > 0 and sell_edge == 0.0 and sell_dual[1] - sell_dual[0] < edge:
                volume = min(volume, pos_d)
            return self._pair(dual, "ask", sell_dual[1], primary, "bid", sell_dual[0], volume)
        buy_dual = self._crossed(top_d, top_p, buy_edge)
        if buy_dual:
            clip = self._clip(size, buy_dual[1] - buy_dual[0], pos_d >= 0, pos_d, params, tick)
            volume = self._pair_volume(state, dual, "bid", primary, "ask", min(buy_dual[2], clip))
            if pos_d < 0 and buy_edge == 0.0 and buy_dual[1] - buy_dual[0] < edge:
                volume = min(volume, -pos_d)
            return self._pair(dual, "bid", buy_dual[0], primary, "ask", buy_dual[1], volume)
        return []

    @staticmethod
    def _clip(size: int, edge: float, adding: bool, position: int, params: Mapping, tick: float) -> int:
        """The position limit is what runs out: small edges must not spend it on their own.

        The IC Hack 2024 winner scaled by how good the edge was and how much room was left, so a
        small opportunity could not use up the capacity a large one would need.
        """
        scale = params["arb_scale_ticks"]
        if not adding or scale <= 0:
            return size
        strength = min(1.0, edge / (scale * tick))
        room = max(0.0, 1.0 - abs(position) / params["position_limit"])
        return max(1, round(size * strength * room))

    def _pair(self, thin, thin_side, thin_px, liquid, liquid_side, liquid_px, volume) -> list:
        """Thin leg first: it is the one that can miss. The Hedger closes any difference."""
        if volume <= 0:
            return []
        return [
            Take(thin, thin_side, thin_px, volume, False, self.name),
            Take(liquid, liquid_side, liquid_px, volume, False, self.name),
        ]

    _pair_volume = staticmethod(lambda *args: pair_room(*args))

    @staticmethod
    def _capped(intent, bid_cap: float | None, ask_floor: float | None):
        def low(price: float | None) -> float | None:
            return min(price, bid_cap) if price is not None and bid_cap is not None else price

        def high(price: float | None) -> float | None:
            return max(price, ask_floor) if price is not None and ask_floor is not None else price

        if isinstance(intent, LadderIntent):
            return intent._replace(
                bids=tuple((low(price), size) for price, size in intent.bids),
                asks=tuple((high(price), size) for price, size in intent.asks),
            )
        return intent._replace(bid=low(intent.bid), ask=high(intent.ask))

    def _bounded(self, state, intent):
        """Never quote where a fill cannot be hedged at a profit on the liquid leg (MAD-LAD, Parity).

        A fill that would cancel delta we already hold needs no hedge, so that side may stay tighter,
        exactly when we want the trade.
        """
        if isinstance(intent, Pull):
            return intent
        dual = intent.instrument_id
        primary = self._duals[dual]
        top = state.top(primary)
        params = params_for(state.params, dual)
        pillow = params["dual_pillow_ticks"]
        if not pillow or top is None or top.bid is None or top.ask is None:
            return intent
        tick = state.tick_of(dual)
        offset = self._offset.update(state, primary, dual)
        delta = state.position(primary) + state.position(dual)
        size = int(params["size"])
        bid_cap = None if delta <= -size else round_bid(top.bid + offset - pillow * tick, tick)
        ask_floor = None if delta >= size else round_ask(top.ask + offset + pillow * tick, tick)
        return self._capped(intent, bid_cap, ask_floor)

    def on_tick(self, state) -> list:
        intents: list = []
        for primary, dual in self.pairs:
            legs = self._arb(state, primary, dual)
            if legs:
                self._last_arb[dual] = ((state.top(primary), state.top(dual)), state.now)
            intents.extend(legs)
        if self._quoter:
            intents.extend(self._bounded(state, intent) for intent in self._quoter.on_tick(state))
        return intents


class GapReversion:
    """Trade the dual gap back to its own slow mean, both legs by IOC. Off until `gap_enabled` = 1.

    The gap is the largest structural edge in public Optibook data, and its shape differs by event:
    a 180 s cycle (Sep 2026), swings of +/-8 over minutes (FutureFocus 2026), or fast noise (UNSW
    2026). Turn it on only once `analyze_capture.py` shows a half-life or period that fits, and
    expected convergence beats crossing both spreads.
    """

    name = "gap_reversion"
    WARMUP_S = 30.0

    def __init__(self, pairs: Sequence[tuple[str, str]]) -> None:
        self.pairs = list(pairs)
        self._stats: dict[str, TimedZ] = {}
        self._last_take: dict[str, float] = {}

    @staticmethod
    def _target(z: float, position: int, params: Mapping) -> int:
        cap = int(params["gap_max_position"])
        if z >= params["gap_entry_z"]:
            return -cap
        if z <= -params["gap_entry_z"]:
            return cap
        return 0 if abs(z) <= params["gap_exit_z"] else position

    def _trade(self, state, primary: str, dual: str) -> list:
        top_p, top_d = state.top(primary), state.top(dual)
        mid_p, mid_d = mid(top_p), mid(top_d)
        if mid_p is None or mid_d is None:
            return []
        params = params_for(state.params, dual)
        stats = self._stats.setdefault(dual, TimedZ())
        # Always tracked, so switching this on live needs no warm-up.
        z = stats.update_z(state.now, mid_d - mid_p, params["gap_half_life_s"], self.WARMUP_S)
        if z is None or not (params["gap_enabled"] or state.position_of(self.name, dual)):
            return []
        position = state.position_of(self.name, dual)
        target = self._target(z, position, params) if params["gap_enabled"] else 0
        need = target - position
        if need == 0:
            return []
        # Fills report late: without a pause the same view would be sent again before the first shows.
        if state.now - self._last_take.get(dual, float("-inf")) < params["mr_interval_s"]:
            return []
        mean = stats.mean
        entering = abs(target) > abs(position)
        if need < 0:
            expected = (top_d.bid - top_p.ask) - mean
            depth, legs = min(top_d.bid_vol, top_p.ask_vol), (("ask", top_d.bid), ("bid", top_p.ask))
        else:
            expected = mean - (top_d.ask - top_p.bid)
            depth, legs = min(top_d.ask_vol, top_p.bid_vol), (("bid", top_d.ask), ("ask", top_p.bid))
        if entering and expected < params["gap_min_edge_ticks"] * state.tick_of(dual):
            return []
        wanted = min(abs(need), int(params["gap_clip"]), depth)
        volume = pair_room(state, dual, legs[0][0], primary, legs[1][0], wanted)
        if volume <= 0:
            return []
        self._last_take[dual] = state.now
        # A trade that also clears inventory we hold goes ahead of everything else.
        urgent = state.position(dual) * need < 0
        return [
            Take(dual, legs[0][0], legs[0][1], volume, urgent, self.name),
            Take(primary, legs[1][0], legs[1][1], volume, urgent, self.name),
        ]

    def on_tick(self, state) -> list:
        intents: list = []
        for primary, dual in self.pairs:
            intents.extend(self._trade(state, primary, dual))
        return intents


class MeanReversionTaker:
    """Aggressive mean reversion on one book, by IOC. Off until `mr_enabled` = 1 for that instrument.

    The organisers confirmed a team may run this while the suggested strategy is passive: if a passive
    crowd keeps a book anchored, every push away from the anchor can be faded, even after paying the
    spread. Autopilot enables it only where paper trades on the live book paid after costs. Exits at
    the mean, and a move that keeps running past `mr_stop_z` is treated as news rather than noise:
    close the position and stay out for `mr_cooldown_s`. It never adds to a stopped loser.
    """

    name = "mean_reversion"
    WARMUP_S = 30.0

    def __init__(self, instruments: Sequence[str]) -> None:
        self.instruments = list(instruments)
        self._stats: dict[str, TimedZ] = {}
        self._last_take: dict[str, float] = {}
        self._paused_until: dict[str, float] = {}

    def stopped_out(self, instrument_id: str, now: float) -> bool:
        return now < self._paused_until.get(instrument_id, float("-inf"))

    @staticmethod
    def _target(z: float, position: int, params: Mapping) -> int:
        cap = int(params["mr_max_position"])
        if z >= params["mr_entry_z"]:
            return -cap
        if z <= -params["mr_entry_z"]:
            return cap
        return 0 if abs(z) <= params["mr_exit_z"] else position

    def _take(self, state, iid: str, top: Top, need: int, stopped: bool, params: Mapping, entering: bool):
        mean = self._stats[iid].mean
        if need > 0:
            side, price, depth, edge = "bid", top.ask, top.ask_vol, mean - top.ask
        else:
            side, price, depth, edge = "ask", top.bid, top.bid_vol, top.bid - mean
        if entering and edge < params["mr_min_edge_ticks"] * state.tick_of(iid):
            return None
        volume = min(abs(need), int(params["mr_clip"]), depth)
        if volume <= 0:
            return None
        self._last_take[iid] = state.now
        return Take(iid, side, price, volume, stopped, self.name)

    def _trade(self, state, iid: str):
        top = state.top(iid)
        value = mid(top)
        if value is None:
            return None
        params = params_for(state.params, iid)
        stats = self._stats.setdefault(iid, TimedZ())
        # Always tracked, so switching this on live needs no warm-up.
        z = stats.update_z(state.now, value, params["mr_half_life_s"], self.WARMUP_S)
        # Only what this strategy built is its to trade: inventory from quoting or arb is not a view.
        position = state.position_of(self.name, iid)
        if z is None or not (params["mr_enabled"] or position):
            return None
        if top.ask - top.bid > params["max_spread_ticks"] * state.tick_of(iid):
            return None
        if position * z < 0 and abs(z) >= params["mr_stop_z"]:
            self._paused_until[iid] = state.now + params["mr_cooldown_s"]
        stopped = self.stopped_out(iid, state.now) or not params["mr_enabled"]
        target = 0 if stopped else self._target(z, position, params)
        if target == position:
            return None
        if state.now - self._last_take.get(iid, float("-inf")) < params["mr_interval_s"]:
            return None
        return self._take(state, iid, top, target - position, stopped, params, abs(target) > abs(position))

    def on_tick(self, state) -> list:
        takes = (self._trade(state, iid) for iid in self.instruments)
        return [take for take in takes if take]


class CrossListingFair:
    """Lean a listing's fair toward what the other listing's microprice says, within a few ticks.

    From the hackaTUM 2025 write-up: when only the microprices diverge, there is no arb to take, but it
    still says where the price is going, so nudge the theoretical price the market maker quotes around.
    """

    def __init__(self, base: FairSource, other: Mapping[str, str]) -> None:
        self.base = base
        self.other = dict(other)
        self._offset = DualOffset()

    def __call__(self, state, instrument_id: str) -> float | None:
        fair = self.base(state, instrument_id)
        other = self.other.get(instrument_id)
        if fair is None or other is None:
            return fair
        params = params_for(state.params, instrument_id)
        here, there = microprice(state.top(instrument_id)), microprice(state.top(other))
        if not params["dual_nudge"] or here is None or there is None:
            return fair
        offset = self._offset.update(state, instrument_id, other)
        limit = params["max_alpha_ticks"] * state.tick_of(instrument_id)
        lean = params["dual_nudge"] * (there - here - offset)
        return fair + max(-limit, min(limit, lean))


class FlowFair:
    """Fair value shifted toward recent large aggressive prints. Off until `follow_min_size` is set.

    One Sep 2026 study: aggressors of 1,000+ lots marked +0.485 at 5 s, so large prints inform. Set
    the threshold from `markouts` in the behaviour report, on a size that passes on two price
    definitions. The shift pulls quotes off the side about to be run over and leans with the flow.
    """

    def __init__(self, base: FairSource = micro_fair) -> None:
        self.base = base
        self._sign: dict[str, int] = {}
        self._until: dict[str, float] = {}

    def __call__(self, state, instrument_id: str) -> float | None:
        fair = self.base(state, instrument_id)
        if fair is None:
            return None
        params = params_for(state.params, instrument_id)
        minimum = params["follow_min_size"]
        if minimum > 0:
            for trade in state.new_ticks.get(instrument_id, []):
                if trade.volume >= minimum and trade.aggressor_side in ("bid", "ask"):
                    self._sign[instrument_id] = 1 if trade.aggressor_side == "bid" else -1
                    self._until[instrument_id] = state.now + params["follow_hold_s"]
        if state.now < self._until.get(instrument_id, float("-inf")):
            return fair + self._sign[instrument_id] * params["follow_ticks"] * state.tick_of(instrument_id)
        return fair


class Hedger:
    """Flattens net delta per underlying from actual positions, which covers one-sided fills."""

    name = "hedger"

    def __init__(
        self,
        hedge_on: Mapping[str, str],
        vol: float = 0.25,
        rate: float = 0.03,
        min_interval_s: float = 0.25,
        calendar_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        holding: Callable[[object, str], bool] | None = None,
    ) -> None:
        self.holding = holding
        self.hedge_on = dict(hedge_on)
        self.vol = vol
        self.rate = rate
        self.min_interval_s = min_interval_s
        self.calendar_now = calendar_now
        self._last_hedge: dict[str, float] = {}
        self._watch = DeltaWatch()

    def group_delta(self, state, base: str) -> float:
        hedge_id = self.hedge_on[base]
        spot = micro_fair(state, hedge_id) or 0.0
        total = 0.0
        for iid in state.universe.groups.get(base, []):
            position = state.position(iid)
            if position:
                info = state.universe.by_id[iid]
                total += position * delta_per_lot(info, spot, self.vol, self.rate, self.calendar_now())
        return total

    def _stands_down(self, state, base: str) -> bool:
        return not params_for(state.params, self.hedge_on[base])["hedge_enabled"]

    def _hedge_book(self, state, base: str) -> tuple:
        """The chosen book, or the other listing when that market is closed."""
        chosen = self.hedge_on[base]
        members = [m for m in state.universe.groups.get(base, ()) if m != chosen]
        listings = [m for m in members if state.universe.by_id[m].kind in (Kind.STOCK, Kind.DUAL)]
        for iid in [chosen, *listings]:
            top = state.top(iid)
            if top is not None and top.bid is not None and top.ask is not None:
                return iid, top
        return chosen, state.top(chosen)

    def _hedge(self, state, base: str):
        now = state.now
        if self.holding is not None and self.holding(state, base):
            return None  # a news position is the trade: the soft delta charge is cheaper than the move
        true_delta = self.group_delta(state, base)
        hedge_id, top = self._hedge_book(state, base)
        params = params_for(state.params, hedge_id)
        # Nearing the exchange's delta hack-out, which counts everything. Teams have earned more by paying it.
        urgent = self._watch.update(base, true_delta, now) and params["hedge_urgent"]
        # A fading position is a view, so hedging it away would mean it can never earn. The rest is not.
        members = state.universe.groups.get(base, ())
        delta = true_delta if urgent else true_delta - sum(state.position_of("mean_reversion", iid) for iid in members)
        if not urgent and self._stands_down(state, base):
            return None
        if not urgent and now - self._last_hedge.get(base, float("-inf")) < self.min_interval_s:
            return None
        threshold = params["hedge_threshold"]
        if abs(delta) <= threshold:
            return None
        # Part of the way only: fills on other products often cancel it, and each hedge pays the spread.
        lots = int(abs(delta) - min(params["hedge_to"], threshold))
        if top is None or lots <= 0:
            return None
        self._last_hedge[base] = now
        if delta > 0 and top.bid is not None:
            return Take(hedge_id, "ask", top.bid, min(lots, top.bid_vol), True, self.name)
        if delta < 0 and top.ask is not None:
            return Take(hedge_id, "bid", top.ask, min(lots, top.ask_vol), True, self.name)
        return None

    def on_tick(self, state) -> list:
        hedges = (self._hedge(state, base) for base in self.hedge_on)
        return [hedge for hedge in hedges if hedge]


class Flattener:
    """From `flatten_at` on the engine clock: pull everything and IOC out of every position."""

    name = "flattener"
    exclusive = True

    def __init__(self, flatten_at: float) -> None:
        self.flatten_at = flatten_at

    def on_tick(self, state) -> list:
        if state.now < self.flatten_at:
            return []
        intents: list = []
        for iid in state.universe.by_id:
            intents.append(Pull(iid, self.name))
            position = state.position(iid)
            top = state.top(iid)
            if position > 0 and top and top.bid is not None:
                intents.append(Take(iid, "ask", top.bid, min(position, top.bid_vol), True, self.name, True))
            elif position < 0 and top and top.ask is not None:
                intents.append(Take(iid, "bid", top.ask, min(-position, top.ask_vol), True, self.name, True))
        return intents


class OptionTheo:
    """Theoretical values off a stock's microprice. Futures use carry; options use Black-Scholes."""

    def __init__(
        self,
        vol: float | None = DEFAULT_OPTION_VOL,
        rate: float = 0.03,
        calendar_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.vol = vol
        self.rate = rate
        self.calendar_now = calendar_now

    def spot(self, state, info) -> float | None:
        """The underlying's level: the client's index if it gave one, or the base's book."""
        if info.index is None:
            return micro_fair(state, info.base)
        prices = {name: micro_fair(state, name) for name in info.index.weights}
        return index_value(prices, info.index.weights, info.index.divisor)

    def etf(self, info, level: float) -> float:
        if info.etf_multiplier is None:
            return level  # named after a listed stock, with no terms from the client
        return etf_fair(level, info.etf_multiplier, info.etf_cash or 0.0)

    def __call__(self, state, instrument_id: str) -> float | None:
        info = state.universe.by_id.get(instrument_id) or parse_instrument_id(instrument_id)
        spot = self.spot(state, info)
        if spot is None:
            return None
        rate = self.rate if info.rate is None else info.rate
        if info.kind is Kind.FUTURE:
            return None if info.expiry is None else future_fair(spot, rate, years_to_expiry(info.expiry, self.calendar_now()))
        if info.kind is Kind.OPTION:
            vol = self.vol if info.vol is None else info.vol
            if None in (info.expiry, info.strike, info.option_kind, vol):
                return None
            tau = years_to_expiry(info.expiry, self.calendar_now())
            return bs_price(info.option_kind, spot, info.strike, tau, rate, vol)
        return self.etf(info, spot) if info.kind is Kind.ETF else spot


class IndexTheo(OptionTheo):
    """Weights from the event, which outrank the client's. Divisor and ETF terms come from the client if it gave them."""

    def __init__(
        self,
        weights: Mapping[str, float],
        divisor: float = 1000.0,
        multiplier: float = 0.25,
        cash: float = 2.5,
        index_id: str = "OB5X",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.weights = dict(weights)
        self.divisor = divisor
        self.multiplier = multiplier
        self.cash = cash
        self.index_id = index_id

    def spot(self, state, info) -> float | None:
        on_index = info.index is not None or info.base == self.index_id or info.base not in state.universe.by_id
        if not on_index:
            return micro_fair(state, info.base)
        prices = {name: micro_fair(state, name) for name in self.weights}
        return index_value(prices, self.weights, info.index.divisor if info.index else self.divisor)

    def etf(self, info, level: float) -> float:
        multiplier = self.multiplier if info.etf_multiplier is None else info.etf_multiplier
        cash = self.cash if info.etf_multiplier is None else info.etf_cash or 0.0
        return etf_fair(level, multiplier, cash)


class TheoQuoter(MarketMaker):
    """Market making around a theoretical value rather than the instrument's own book."""

    name = "theo_quoter"

    def __init__(self, instruments: Sequence[str], theo: FairSource) -> None:
        super().__init__(instruments, fair=theo, use_imbalance=False, guard_own_spread=False)


class ThinLegTaker:
    """Takes a stale quote on the thin listing, priced off the liquid leg's mid. Off unless `taker_mode` is on.

    It uses about 1/70 of the stack's writes, so it is the backup for when writes run out on a full board, which
    is untested. Replays on one pair: 14,113 against the stack's 17,863 over 13 real hours, and it does not beat
    the stack on a slow connection either. Never alongside the stack, which lost money.
    """

    name = "thin_leg_taker"

    def __init__(self, pairs: Sequence[tuple[str, str]]) -> None:
        self.pairs = list(pairs)
        self._fair: dict[str, float] = {}
        self._sent: dict[str, float] = {}

    def on_tick(self, state) -> list:
        if not state.params["taker_mode"]:
            return []
        intents: list = []
        for primary, dual in self.pairs:
            intents.extend(self._take(state, primary, dual))
        return intents

    def _due(self, state, dual: str, fair: float) -> bool:
        """When the fair moves, or as a slow heartbeat. Anything faster repeats itself before the reply comes."""
        moved = self._fair.get(dual) != fair
        self._fair[dual] = fair
        waited = state.now - self._sent.get(dual, float("-inf"))
        return waited >= THIN_TAKER_MIN_GAP_S and (moved or waited >= THIN_TAKER_HEARTBEAT_S)

    def _take(self, state, primary: str, dual: str) -> list:
        liquid, thin = state.top(primary), state.top(dual)
        if liquid is None or thin is None or None in (liquid.bid, liquid.ask, thin.bid, thin.ask):
            return []
        fair = (liquid.bid + liquid.ask) / 2.0
        if not self._due(state, dual, fair):
            return []
        params = params_for(state.params, dual)
        tick = state.tick_of(dual)
        edge = params["taker_edge_ticks"] * tick
        buy = math.floor((fair - edge) / tick + TICK_ROUNDING) * tick
        sell = math.ceil((fair + edge) / tick - TICK_ROUNDING) * tick
        clip = max(1, int(params["taker_clip"]))
        takes = []
        if thin.ask <= buy:
            takes.append(Take(dual, "bid", buy, clip, False, self.name))
        if thin.bid >= sell:
            takes.append(Take(dual, "ask", sell, clip, False, self.name))
        if takes:
            self._sent[dual] = state.now
        return takes


class TheoTaker:
    """IOC when the book crosses theoretical value by more than the required edge."""

    name = "theo_taker"

    def __init__(self, instruments: Sequence[str], theo: FairSource) -> None:
        self.instruments = list(instruments)
        self.theo = theo

    def _take(self, state, iid: str):
        theo = self.theo(state, iid)
        top = state.top(iid)
        if theo is None or top is None:
            return None
        params = params_for(state.params, iid)
        book_mid = mid(top)
        if book_mid is not None and abs(theo - book_mid) > params["circuit_ticks"] * state.tick_of(iid):
            return None  # a gap this large means our formula is wrong: quoting has the same breaker
        edge = params["theo_edge_ticks"] * state.tick_of(iid)
        size = int(params["size"])
        if top.ask is not None and top.ask <= theo - edge:
            return Take(iid, "bid", top.ask, min(size, top.ask_vol), False, self.name)
        if top.bid is not None and top.bid >= theo + edge:
            return Take(iid, "ask", top.bid, min(size, top.bid_vol), False, self.name)
        return None

    def on_tick(self, state) -> list:
        takes = (self._take(state, iid) for iid in self.instruments)
        return [take for take in takes if take]


class NewsTrader:
    """Duck, then optionally go directional, then hand back to market making.

    Mechanism from the 1st-place TUM 2025 write-up: pull quotes on the affected stock, take up to
    the position limit via IOC in the predicted direction, and exit when the microprice z-score
    flips sign. Low confidence only ducks.
    """

    name = "news_trader"

    def __init__(
        self,
        classifier,
        aggression_ticks: float = 1.0,
        z_half_life: float = 200.0,
        slow: AsyncClassifier | None = None,
    ) -> None:
        self.classifier = classifier
        self.slow = slow
        self._scale: dict[str, float] = {}
        self.aggression_ticks = aggression_ticks
        self.z_half_life = z_half_life
        self._ducked_until: dict[str, float] = {}
        self._direction: dict[str, int] = {}
        self._hold_until: dict[str, float] = {}
        self._ewma: dict[str, Ewma] = {}
        self._min_hold_until: dict[str, float] = {}
        self._entry_mid: dict[str, float] = {}
        self._closing: set[str] = set()

    def holding(self, state, base: str) -> bool:
        """True while a news position is open on this underlying, so nothing hedges it away."""
        active = set(self._direction) | self._closing
        return any(state.universe.by_id[iid].base == base for iid in active)

    def holding_instrument(self, instrument_id: str) -> bool:
        return instrument_id in self._direction or instrument_id in self._closing

    def _signals(self, text: str) -> list[NewsSignal]:
        """One post can move several stocks, each its own way."""
        classify_all = getattr(self.classifier, "classify_all", None)
        return classify_all(text) if classify_all else [self.classifier.classify(text)]

    @staticmethod
    def _confidence_scale(state, confidence: float) -> float:
        return min(1.0, confidence / state.params["news_full_confidence"])

    @staticmethod
    def _listings(state, iid: str) -> list[str]:
        """Conviction goes on both listings: twice the position, as a top hackaTUM 2025 team did."""
        info = state.universe.by_id[iid]
        if not state.params["news_both_listings"] or info.kind not in (Kind.STOCK, Kind.DUAL):
            return [iid]
        members = state.universe.groups.get(info.base, ())
        return [m for m in members if state.universe.by_id[m].kind in (Kind.STOCK, Kind.DUAL)]

    def _enter(self, state, iid: str, direction: int, scale: float) -> None:
        self._ducked_until[iid] = state.now + state.params["news_cooldown_s"]
        self._direction[iid] = direction
        self._scale[iid] = scale
        self._closing.discard(iid)
        current = mid(state.top(iid))
        if current is not None:
            self._entry_mid[iid] = current
        self._hold_until[iid] = state.now + state.params["news_hold_max_s"]
        self._min_hold_until[iid] = state.now + state.params["news_check_s"]

    def _absorb_slow(self, state) -> None:
        """An LLM verdict arrives later: agreement scales to full size, disagreement flattens."""
        for _, result in self.slow.poll() if self.slow else ():
            for signal in result if isinstance(result, list) else [result]:
                self._absorb_verdict(state, signal)

    def _absorb_verdict(self, state, signal: NewsSignal | None) -> None:
        if signal is None or signal.instrument_id not in state.universe.by_id:
            return
        confident = signal.confidence >= state.params["news_min_confidence"]
        scale = self._confidence_scale(state, signal.confidence)
        for iid in self._listings(state, signal.instrument_id):
            active = self._direction.get(iid)
            if active is None:
                if signal.direction and confident:
                    self._enter(state, iid, signal.direction, scale)
            elif signal.direction == active:
                self._scale[iid] = scale
            elif signal.direction == -active:
                self._direction.pop(iid)

    def _absorb(self, state) -> None:
        self._absorb_slow(state)
        for post in getattr(state, "new_news", []):
            if self.slow:
                self.slow.submit(post.post)
            for signal in self._signals(post.post):
                self._absorb_fast(state, signal)

    def _absorb_fast(self, state, signal: NewsSignal) -> None:
        if signal.instrument_id not in state.universe.by_id:
            return
        listings = self._listings(state, signal.instrument_id)
        for listing in listings:
            self._ducked_until[listing] = state.now + state.params["news_cooldown_s"]
        if signal.direction and signal.confidence >= state.params["news_min_confidence"]:
            scale = self._confidence_scale(state, signal.confidence)
            if self.slow:
                scale *= state.params["news_stage1_fraction"]
            for listing in listings:
                self._enter(state, listing, signal.direction, scale)

    def _z(self, state, iid: str) -> float:
        fair = micro_fair(state, iid)
        if fair is None:
            return 0.0
        ewma = self._ewma.setdefault(iid, Ewma(self.z_half_life))
        score = ewma.z(fair)
        ewma.update(fair)
        return score

    def _proved_wrong(self, state, iid: str, direction: int) -> bool:
        """Past the short delay, a price running against the call means the call was wrong."""
        stop = params_for(state.params, iid)["news_stop_ticks"] * state.tick_of(iid)
        entry, current = self._entry_mid.get(iid), mid(state.top(iid))
        if not stop or entry is None or current is None or state.now < self._min_hold_until[iid]:
            return False
        return (entry - current) * direction >= stop

    def _exit_due(self, state, iid: str, direction: int, z: float) -> bool:
        if state.now >= self._hold_until[iid]:
            return True
        return state.now >= self._min_hold_until[iid] and z * direction < 0

    def _close_out(self, state, iid: str) -> list:
        """Leave a wrong call now, rather than hand the position to the market maker."""
        position = state.position(iid)
        top = state.top(iid)
        if position == 0:
            self._closing.discard(iid)
            return []
        if top is None:
            return []
        if position > 0 and top.bid is not None:
            return [Take(iid, "ask", top.bid, min(position, top.bid_vol), True, self.name, True)]
        if position < 0 and top.ask is not None:
            return [Take(iid, "bid", top.ask, min(-position, top.ask_vol), True, self.name, True)]
        return []

    def _chase(self, state, iid: str, direction: int) -> list:
        fraction = state.params["news_size_fraction"] * self._scale.get(iid, 1.0)
        target = int(direction * state.params["position_limit"] * fraction)
        missing = target - state.position(iid)
        top = state.top(iid)
        if top is None or missing == 0 or (missing > 0) != (direction > 0):
            return []
        tick = state.tick_of(iid)
        slip = self.aggression_ticks * tick
        if direction > 0 and top.ask is not None:
            price = round_ask(top.ask + slip, tick)
            return [Take(iid, "bid", price, missing, False, self.name)]
        if direction < 0 and top.bid is not None:
            price = round_bid(top.bid - slip, tick)
            return [Take(iid, "ask", price, -missing, False, self.name)]
        return []

    def on_tick(self, state) -> list:
        self._absorb(state)
        intents: list = []
        for iid in state.universe.by_id:
            z = self._z(state, iid)
            direction = self._direction.get(iid)
            if direction and self._proved_wrong(state, iid, direction):
                self._direction.pop(iid)
                self._closing.add(iid)
                direction = None
            if iid in self._closing:
                intents.append(Pull(iid, self.name))
                intents.extend(self._close_out(state, iid))
                continue
            if direction:
                if self._exit_due(state, iid, direction, z):
                    self._direction.pop(iid)
                    self._ducked_until.pop(iid, None)
                else:
                    intents.append(Pull(iid, self.name))
                    intents.extend(self._chase(state, iid, direction))
                    continue
            if state.now < self._ducked_until.get(iid, float("-inf")):
                intents.append(Pull(iid, self.name))
        return intents


def default_stack(
    instrument_ids: Sequence[str],
    index_weights: Mapping[str, float] | None = None,
    classifier=None,
    vol: float | None = DEFAULT_OPTION_VOL,
    flatten_at: float | None = None,
    universe=None,
    slow_classifier: AsyncClassifier | None = None,
) -> list:
    """Build a stack from whatever the exchange lists: no assumption about Friday's instruments.

    Pass `universe` when pairs were detected under names the id conventions do not cover. With `vol` None, nobody has said
    what the volatility is, so only options whose volatility the client gives can trade.
    """
    infos = [universe.by_id[iid] if universe else parse_instrument_id(iid) for iid in instrument_ids]
    ids: set[str] = set(instrument_ids)
    pairs = sorted((i.base, i.instrument_id) for i in infos if i.kind is Kind.DUAL and i.base in ids)
    derivatives = sorted(
        i.instrument_id for i in infos if i.kind in (Kind.ETF, Kind.FUTURE, Kind.OPTION)
    )
    dualled = {primary for primary, _ in pairs}
    plain = sorted(i.instrument_id for i in infos if i.kind is Kind.STOCK and i.instrument_id not in dualled)
    theo: FairSource = IndexTheo(index_weights, vol=vol) if index_weights else OptionTheo(vol=vol)

    hedge_on: dict[str, str] = {primary: primary for primary, _ in pairs}
    for info in infos:
        if info.base == info.instrument_id:
            continue  # nothing to hedge against: a stock, or a derivative the client gave no underlying for
        if info.kind is Kind.FUTURE:
            hedge_on.setdefault(info.base, info.instrument_id)
        if info.kind is Kind.OPTION and info.base in ids:
            hedge_on.setdefault(info.base, info.base)

    stack: list = []
    if flatten_at is not None:
        stack.append(Flattener(flatten_at))
    news = NewsTrader(classifier, slow=slow_classifier) if classifier is not None else None
    if news is not None:
        stack.append(news)
    if hedge_on:
        hedge_vol = DEFAULT_OPTION_VOL if vol is None else vol  # no option position can arise without a volatility
        stack.append(Hedger(hedge_on, vol=hedge_vol, holding=news.holding if news else None))
    directional = sorted(i.instrument_id for i in infos if i.kind is not Kind.OPTION)
    if directional:
        stack.append(MeanReversionTaker(directional))
    if pairs:
        stack.append(ThinLegTaker(pairs))
        stack.append(GapReversion(pairs))
        stack.append(DualListingArb(pairs))
    by_id = {info.instrument_id: info for info in infos}
    no_vol = {iid for iid in derivatives if by_id[iid].kind is Kind.OPTION and vol is None and by_id[iid].vol is None}
    tradable = [
        iid for iid in derivatives
        if iid not in no_vol and ((index_weights and by_id[iid].base != iid) or theo_ready(by_id[iid], ids))
    ]
    options = {iid: by_id[iid].base for iid in tradable if by_id[iid].kind is Kind.OPTION}
    if options:
        theo = OptionMarketCheck(theo, options)
    if tradable:
        stack.append(TheoTaker(tradable, theo))
        stack.append(TheoQuoter(tradable, theo))
    # Without index weights an ETF or future has no theo, but its own book still pays to quote.
    unpriced = [iid for iid in derivatives if iid not in tradable and by_id[iid].kind is not Kind.OPTION]
    if plain or dualled or unpriced:
        fair = CrossListingFair(FlowFair(), {primary: dual for primary, dual in pairs})
        if len(plain) >= 2:
            lag = LeadLag(plain)
            stack.append(lag)
            fair = LagAdjustedFair(fair, lag)
            stack.append(ScheduleTrader(plain))
        stack.append(MarketMaker(plain + sorted(dualled) + unpriced, fair=fair))
    return stack


__all__ = [
    "CrossListingFair",
    "DualListingArb",
    "Flattener",
    "FlowFair",
    "GapReversion",
    "Hedger",
    "IndexTheo",
    "MarketMaker",
    "MeanReversionTaker",
    "NewsTrader",
    "OptionTheo",
    "TheoQuoter",
    "TheoTaker",
    "ThinLegTaker",
    "default_stack",
    "micro_fair",
    "quote_params",
]
