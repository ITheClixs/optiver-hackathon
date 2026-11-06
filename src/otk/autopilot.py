"""Autonomous adaptation of the live bot: deterministic, bounded, and logged.

The organisers confirmed that AI may trade autonomously, so no person has to approve a change. What
keeps this from chasing noise is that each change needs evidence from the live market:

  * reversion goes on only where paper trades on this book paid after costs, and goes off when they
    stop paying or the instrument's own PnL breaks down;
  * quoting follows the live flow labels and our own fill quality, as a factor on the baseline that
    lapses when the evidence does, so no adjustment can ratchet;
  * names from the dashboard feed, when available, add size signatures to lean on.

It only writes the overlay layer of `ParamStore`. Hard bounds, the risk clip and the write budget all
still apply, a key set in `params.json` outranks it, and no LLM runs in this path. It observes through
the recorder interface, so an error here is counted and cannot stop trading.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from otk.allocator import Cusum
from otk.fair import mid
from otk.instruments import Kind, Universe
from otk.opponents import Welford, recommend
from otk.params import INSTRUMENTS_KEY, params_for
from otk.recorder import NullRecorder
from otk.reversion import ArmConfig, ReversionArm, best_arm, make_arms

logger = logging.getLogger("otk.autopilot")

_NEVER = float("-inf")
FILL_HORIZON_S = 5.0
SAMPLES_PER_SIZE_STEP = 16


@dataclass(frozen=True)
class AutopilotConfig:
    interval_s: float = 30.0
    sample_s: float = 1.0
    hold_s: float = 60.0  # a change stays at least this long, unless it reduces risk
    arms: ArmConfig = field(default_factory=ArmConfig)
    min_round_trips: int = 8
    enable_t: float = 2.5  # above 2 because six arms compete per instrument
    disable_t: float = 0.5
    stale_s: float = 300.0  # an arm that stopped trading has stopped being evidence
    min_edge_ticks: float = 0.5
    start_position: int = 15
    max_position: int = 30
    min_factor: float = 0.25
    max_factor: float = 3.0
    min_fills: int = 20
    fill_t: float = 2.0
    fill_ticks: float = 0.5
    penalty_s: float = 300.0
    min_pnl_slices: int = 8
    intel_max_age_s: float = 60.0
    min_signature_share: float = 0.5


def arm_failure(arm: ReversionArm, now: float, config: AutopilotConfig) -> str | None:
    """Why an active arm should stop, or None. The same rule runs live and in `walk_forward`."""
    if now - arm.last_close > config.stale_s:
        return f"no paper round trip for {config.stale_s:.0f}s"
    if arm.n >= config.min_round_trips and (arm.t < config.disable_t or arm.recent_mean() <= 0.0):
        return f"paper round trips stopped paying, t={arm.t:.2f}"
    return None


def promotable_arm(
    arms: Sequence[ReversionArm], now: float, tick: float, config: AutopilotConfig
) -> ReversionArm | None:
    """The arm evidence supports right now, or None."""
    current = [arm for arm in arms if now - arm.last_close <= config.stale_s]
    if not current:
        return None
    arm = best_arm(current)
    if (
        arm.n < config.min_round_trips
        or arm.t < config.enable_t
        or arm.mean < config.min_edge_ticks * tick
        or arm.recent_mean() <= 0.0
    ):
        return None
    return arm


def walk_forward(points: Iterable[tuple], tick: float, config: AutopilotConfig | None = None) -> dict:
    """What the promotion rule would have earned per unit: only round trips made while it was on count.

    Evidence is gathered before each decision and judged after it, which is the honest test of the
    rule. The best arm picked in hindsight is not.
    """
    chosen = config or AutopilotConfig()
    arms = make_arms(tick, chosen.arms)
    active: ReversionArm | None = None
    earned: list[float] = []
    promotions = 0
    next_decision: float | None = None
    opened_before_rule = False
    for now, value, buy_px, sell_px in points:
        before = active.total_trades if active else 0
        for arm in arms:
            arm.update(now, value, buy_px, sell_px)
        if active is not None and active.total_trades > before:
            if not opened_before_rule:  # a trip already open when the rule switched on was not ours
                earned.append(active.results[-1])
            opened_before_rule = False
        if next_decision is None:
            next_decision = now + chosen.interval_s
        if now < next_decision:
            continue
        next_decision = now + chosen.interval_s
        if active is not None and arm_failure(active, now, chosen):
            active = None
        if active is None:
            active = promotable_arm(arms, now, tick, chosen)
            promotions += active is not None
            opened_before_rule = active is not None and active.position != 0
    count = len(earned)
    mean = sum(earned) / count if count else 0.0
    var = sum((value - mean) ** 2 for value in earned) / (count - 1) if count > 1 else 0.0
    return {
        "promotions": promotions,
        "round_trips": count,
        "total_ticks": round(sum(earned) / tick, 1),
        "mean_ticks": round(mean / tick, 3),
        "t": round(mean / math.sqrt(var / count), 2) if var > 0 else 0.0,
    }


class ParticipantIntel:
    """Reads `participants.json` from the dashboard feed process. Stale or broken means no intel."""

    def __init__(
        self, path: str | Path, max_age_s: float = 60.0, clock: Callable[[], float] = time.time
    ) -> None:
        self.path = Path(path)
        self.max_age_s = max_age_s
        self.clock = clock
        self.errors = 0

    def load(self) -> dict:
        try:
            if self.clock() - self.path.stat().st_mtime > self.max_age_s:
                return {}
            report = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            self.errors += 1
            logger.warning("cannot read %s: %s", self.path, exc)
            return {}
        return report if isinstance(report, dict) else {}


class _Ledger:
    """Mark-to-mid PnL of one instrument from our fills, for its slices and the session total."""

    def __init__(self) -> None:
        self.cash = 0.0
        self.position = 0
        self.mid: float | None = None
        self.last_pnl = 0.0
        self.slices: deque[float] = deque(maxlen=30)
        self.cusum = Cusum()

    def fill(self, sign: int, price: float, volume: int) -> None:
        self.cash -= sign * price * volume
        self.position += sign * volume

    @property
    def pnl(self) -> float:
        return self.cash + self.position * (self.mid or 0.0)

    def close_slice(self, min_slices: int) -> bool:
        """True when this instrument's PnL per slice has fallen by more than chance would explain."""
        current = self.pnl
        change = current - self.last_pnl
        self.last_pnl = current
        alarm = False
        if len(self.slices) >= min_slices:
            mean = sum(self.slices) / len(self.slices)
            var = sum((value - mean) ** 2 for value in self.slices) / (len(self.slices) - 1)
            alarm = var > 0 and self.cusum.update((change - mean) / math.sqrt(var))
        self.slices.append(change)
        return alarm


class _FillQuality:
    """Signed markout of our recent fills in ticks. Negative means we are being picked off."""

    def __init__(self) -> None:
        self.pending: deque[tuple] = deque()
        self.scores = Welford()

    def add(self, now: float, sign: int, price: float) -> None:
        self.pending.append((now + FILL_HORIZON_S, sign, price))

    def settle(self, now: float, current: float, tick: float) -> None:
        while self.pending and self.pending[0][0] <= now:
            _, sign, price = self.pending.popleft()
            self.scores.add(sign * (current - price) / tick)

    def verdict(self, min_fills: int, min_t: float, min_ticks: float) -> int:
        stats = self.scores.stats()
        if stats["n"] < min_fills or abs(stats["mean"]) < min_ticks or abs(stats["t"]) < min_t:
            return 0
        return 1 if stats["mean"] > 0 else -1

    def reset(self) -> None:
        self.scores = Welford()


class _Target:
    """Something reversion can trade: one book, or the gap between a dual and its primary."""

    def __init__(self, key: str, prefix: str, base: str, tick: float, arms: list[ReversionArm]) -> None:
        self.key = key
        self.prefix = prefix
        self.base = base
        self.tick = tick
        self.arms = arms
        self.active: ReversionArm | None = None

    def values(self, config: AutopilotConfig) -> dict[str, float]:
        arm = self.active
        lots = int(min(config.max_position, config.start_position * self._steps(config)))
        common = {
            f"{self.prefix}_enabled": 1,
            f"{self.prefix}_entry_z": arm.entry_z,
            f"{self.prefix}_exit_z": config.arms.exit_z,
            f"{self.prefix}_half_life_s": arm.half_life_s,
            f"{self.prefix}_min_edge_ticks": config.arms.min_edge_ticks,
            f"{self.prefix}_max_position": lots,
            f"{self.prefix}_clip": max(5, lots // 3),
        }
        if self.prefix == "mr":
            common["mr_stop_z"] = config.arms.stop_z
            common["mr_cooldown_s"] = config.arms.cooldown_s
        return common

    def _steps(self, config: AutopilotConfig) -> int:
        """Size grows with both confidence and how much has been seen: a clean short record stays small."""
        by_confidence = int(self.active.t / config.enable_t)
        by_sample = 1 + self.active.n // SAMPLES_PER_SIZE_STEP
        return max(1, min(by_confidence, by_sample))


class Autopilot(NullRecorder):
    def __init__(
        self,
        store,
        universe: Universe,
        config: AutopilotConfig | None = None,
        monitor=None,
        intel: ParticipantIntel | None = None,
        sink: Callable[[str, Mapping], None] | None = None,
    ) -> None:
        self.store = store
        self.universe = universe
        self.config = config or AutopilotConfig()
        self.monitor = monitor
        self.intel = intel
        self.sink = sink
        self.decisions: deque[dict] = deque(maxlen=50)
        self.overlay: dict[str, dict[str, float]] = {}
        self._targets = self._build_targets()
        self._pairs = {dual: primary for primary, dual in universe.dual_pairs()}
        self._ledgers: dict[str, _Ledger] = defaultdict(_Ledger)
        self._fills: dict[str, _FillQuality] = defaultdict(_FillQuality)
        self._next_sample = _NEVER
        self._next_decision: float | None = None
        self._changed: dict[str, float] = {}
        self._penalty_until: dict[str, float] = {}
        self._peak = 0.0
        self._risk_off_until = _NEVER

    def _build_targets(self) -> dict[str, _Target]:
        targets: dict[str, _Target] = {}
        arms = self.config.arms
        for iid, info in self.universe.by_id.items():
            if info.kind is not Kind.OPTION:
                targets[iid] = _Target(iid, "mr", info.base, info.tick, make_arms(info.tick, arms))
        for primary, dual in self.universe.dual_pairs():
            tick = self.universe.tick_of(dual)
            targets[f"{dual}-gap"] = _Target(dual, "gap", primary, tick, make_arms(tick, arms))
        return targets

    # --- observing ----------------------------------------------------------------------------

    def tick(self, state) -> None:
        self._observe_fills(state)
        if state.now >= self._next_sample:
            self._next_sample = state.now + self.config.sample_s
            self._sample(state)
        if self._next_decision is None:
            self._next_decision = state.now + self.config.interval_s
        elif state.now >= self._next_decision:
            self._next_decision = state.now + self.config.interval_s
            self._decide(state)

    def _observe_fills(self, state) -> None:
        for iid, fills in state.new_fills.items():
            for fill in fills:
                sign = 1 if fill.side == "bid" else -1
                self._ledgers[iid].fill(sign, fill.price, fill.volume)
                self._fills[iid].add(state.now, sign, fill.price)

    def _sample(self, state) -> None:
        tops = {}
        for iid in self.universe.by_id:
            top = state.top(iid)
            value = mid(top)
            if value is None:
                continue
            tops[iid] = top
            self._ledgers[iid].mid = value
            self._fills[iid].settle(state.now, value, state.tick_of(iid))
        for target in self._targets.values():
            point = self._reversion_point(target, tops)
            if point is not None:
                for arm in target.arms:
                    arm.update(state.now, *point)

    def _reversion_point(self, target: _Target, tops: Mapping) -> tuple | None:
        top = tops.get(target.key)
        if top is None:
            return None
        if target.prefix == "mr":
            return (top.bid + top.ask) / 2.0, top.ask, top.bid
        other = tops.get(target.base)
        if other is None:
            return None
        gap = (top.bid + top.ask) / 2.0 - (other.bid + other.ask) / 2.0
        return gap, top.ask - other.bid, top.bid - other.ask

    # --- deciding -----------------------------------------------------------------------------

    def _decide(self, state) -> None:
        now = state.now
        params = state.params
        if not params["ap_enabled"]:
            if self.overlay:
                self._log(now, "all", {}, "ap_enabled is 0: back to baseline")
            self._publish({}, now)
            return
        self._check_risk(now, params)
        baseline = self.store.baseline()
        intel = self.intel.load() if self.intel else {}
        plan: dict[str, dict[str, float]] = defaultdict(dict)
        reasons: dict[str, list[str]] = defaultdict(list)
        if params["ap_quoting"]:
            for iid in self.universe.by_id:
                self._plan_quoting(iid, params_for(baseline, iid), now, intel, plan[iid], reasons[iid])
        self._plan_directional(now, params, plan, reasons)
        self._commit({iid: values for iid, values in plan.items() if values}, reasons, now)

    def _check_risk(self, now: float, params: Mapping) -> None:
        for iid, ledger in self._ledgers.items():
            if ledger.close_slice(self.config.min_pnl_slices):
                self._penalty_until[iid] = now + self.config.penalty_s
                self._log(now, iid, {}, "PnL per slice broke down: size down, directional off")
        total = sum(ledger.pnl for ledger in self._ledgers.values())
        self._peak = max(self._peak, total)
        limit = params["ap_drawdown"]
        if limit and self._peak - total > limit and now >= self._risk_off_until:
            self._risk_off_until = now + self.config.penalty_s
            self._peak = total
            self._log(now, "all", {}, f"session drawdown above {limit}: risk off")

    def _plan_quoting(
        self, iid: str, base: Mapping, now: float, intel: Mapping, values: dict, why: list
    ) -> None:
        factors: dict[str, float] = {}

        def apply(parameter: str, factor: float, reason: str) -> None:
            factors[parameter] = factors.get(parameter, 1.0) * factor
            why.append(reason)

        labels = self.monitor.labels(iid) if self.monitor else []
        for change in recommend(labels):
            apply(change.parameter, change.factor, change.reason)
        config = self.config
        verdict = self._fills[iid].verdict(config.min_fills, config.fill_t, config.fill_ticks)
        if verdict < 0:
            apply("base_spread_ticks", 1.5, "our fills are picked off: widen")
            apply("size", 0.7, "our fills are picked off: show less")
        elif verdict > 0:
            apply("size", 1.25, "our fills mark out well: show more")
        if now < max(self._penalty_until.get(iid, _NEVER), self._risk_off_until):
            apply("size", 0.5, "in penalty: half size")
        self._apply_intel(iid, base, intel, values, apply, why)
        for parameter, factor in factors.items():
            bounded = max(config.min_factor, min(config.max_factor, factor))
            if abs(bounded - 1.0) > 0.01:
                values[parameter] = round(base[parameter] * bounded, 6)

    def _apply_intel(self, iid: str, base: Mapping, intel: Mapping, values: dict, apply, why: list) -> None:
        targets = intel.get("targets") if isinstance(intel, Mapping) else None
        if not isinstance(targets, Mapping):
            return
        for row in targets.get("overpayers", ()):
            if row.get("instrument") == iid:
                apply("size", 1.25, f"{row.get('kind', 'someone')} on this book overpays: quote more")
                break
        signatures = [
            row["modal_size"]
            for row in targets.get("informed", ())
            if row.get("instrument") == iid
            and row.get("modal_share", 0.0) >= self.config.min_signature_share
            and row.get("modal_size", 0) > base["size"]
        ]
        if signatures:
            values["follow_min_size"] = float(min(signatures))
            why.append(f"prints of {min(signatures)} lots are informed: lean with them")

    def _plan_directional(self, now: float, params: Mapping, plan: dict, reasons: dict) -> None:
        taken: set[str] = set()
        allowed = params["ap_directional"] and now >= self._risk_off_until
        def permitted(target: _Target) -> bool:
            return bool(allowed and params[f"ap_{target.prefix}"])

        for target in self._targets.values():
            if target.active is not None and (
                not permitted(target) or self._failing(target, now, reasons[target.key])
            ):
                target.active = None
        for target in self._targets.values():
            if target.active is not None:
                taken.add(target.base)
        if allowed:
            for target in sorted(self._targets.values(), key=lambda t: -best_arm(t.arms).t):
                if permitted(target) and target.base not in taken and self._promote(target, now, reasons[target.key]):
                    taken.add(target.base)
        for target in self._targets.values():
            if target.active is not None:
                plan[target.key].update(target.values(self.config))

    def _failing(self, target: _Target, now: float, why: list) -> bool:
        if now < self._penalty_until.get(target.key, _NEVER):
            why.append(f"{target.prefix} off: instrument in penalty")
            return True
        reason = arm_failure(target.active, now, self.config)
        if reason:
            why.append(f"{target.prefix} off: {reason}")
        return bool(reason)

    def _promote(self, target: _Target, now: float, why: list) -> bool:
        if target.active is not None or now < self._penalty_until.get(target.key, _NEVER):
            return False
        arm = promotable_arm(target.arms, now, target.tick, self.config)
        if arm is None:
            return False
        target.active = arm
        why.append(
            f"{target.prefix} on: {arm.n} paper round trips at {arm.mean / target.tick:.2f} ticks "
            f"after costs, t={arm.t:.2f}"
        )
        return True

    # --- committing ---------------------------------------------------------------------------

    @staticmethod
    def _reduces_risk(before: Mapping, after: Mapping) -> bool:
        for prefix in ("mr", "gap"):
            if before.get(f"{prefix}_enabled") and not after.get(f"{prefix}_enabled"):
                return True
        return after.get("size", math.inf) < before.get("size", math.inf)

    def _commit(self, plan: Mapping, reasons: Mapping, now: float) -> None:
        merged: dict[str, dict[str, float]] = {}
        for iid in self.universe.by_id:
            have, want = self.overlay.get(iid, {}), plan.get(iid, {})
            if want == have:
                if have:
                    merged[iid] = have
                continue
            held = now - self._changed.get(iid, _NEVER) < self.config.hold_s
            if held and not self._reduces_risk(have, want):
                if have:
                    merged[iid] = have
                continue
            self._changed[iid] = now
            self._fills[iid].reset()  # evidence must be about the new settings
            self._log(now, iid, _difference(have, want), "; ".join(reasons.get(iid, ())) or "evidence lapsed")
            if want:
                merged[iid] = want
        self._publish(merged, now)

    def _publish(self, overlay: dict, now: float) -> None:
        if overlay == self.overlay:
            return
        self.store.set_overlay({INSTRUMENTS_KEY: overlay} if overlay else {})
        self.overlay = overlay

    def _log(self, now: float, iid: str, change: Mapping, reason: str) -> None:
        entry = {"at": round(now, 3), "instrument": iid, "change": dict(change), "why": reason}
        self.decisions.append(entry)
        logger.info("autopilot %s %s: %s", iid, entry["change"], reason)
        if self.sink:
            self.sink("decision", entry)

    def snapshot(self) -> dict:
        return {
            "overlay": self.overlay,
            "directional": {
                key: target.active.summary()
                for key, target in self._targets.items()
                if target.active is not None
            },
            "best_paper": {
                key: best_arm(target.arms).summary()
                for key, target in self._targets.items()
                if best_arm(target.arms).n
            },
            "decisions": list(self.decisions)[-10:],
        }


def _difference(before: Mapping, after: Mapping) -> dict:
    changed = {key: value for key, value in after.items() if before.get(key) != value}
    changed.update({key: None for key in before if key not in after})
    return changed
