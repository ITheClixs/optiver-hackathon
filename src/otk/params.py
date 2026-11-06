"""Runtime parameters: the only channel by which anything adapts the live loop.

Layers, later wins: defaults, event base, Autopilot's overlay, `params.json`. A person or an agent
therefore always outranks Autopilot, and a key set in the file pins it. The loop re-reads the file on
change, clamps every value to hard bounds, and keeps the last good set if the file is broken. Bounds
are deliberately not configurable from the file, so a bad edit cannot widen them.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from otk.instruments import DEFAULT_TICK


class Bound(NamedTuple):
    default: float
    low: float
    high: float


BOUNDS: dict[str, Bound] = {
    "position_limit": Bound(100, 1, 1000),
    "size": Bound(10, 0, 100),
    "reprice_ticks": Bound(1.0, 0, 10),
    "near_touch_ticks": Bound(4.0, 0, 50),
    "max_alpha_ticks": Bound(2.0, 0, 10),
    "news_stage1_fraction": Bound(0.5, 0, 1),
    "base_spread_ticks": Bound(2.0, 0, 40),
    "vol_factor": Bound(1.0, 0, 10),
    "max_vol_ticks": Bound(6.0, 0, 400),
    "imbalance_ticks": Bound(1.0, 0, 5),
    "skew_ticks": Bound(4.0, 0, 400),
    "skew_curve": Bound(2.0, 0.1, 10),
    "max_deviation_ticks": Bound(15.0, 1, 1000),
    "arb_edge_ticks": Bound(1.0, 0, 20),
    "theo_edge_ticks": Bound(2.0, 0, 50),
    "dual_offset_half_life_s": Bound(30.0, 1, 600),
    "dual_offset_cap_ticks": Bound(15.0, 0, 1000),
    "max_spread_ticks": Bound(50.0, 2, 500),
    "circuit_ticks": Bound(100.0, 1, 1000),
    "hedge_threshold": Bound(60, 0, 100),
    "hedge_to": Bound(30, 0, 100),
    "hedge_enabled": Bound(1, 0, 1),
    "hedge_urgent": Bound(1, 0, 1),
    "taker_mode": Bound(0, 0, 1),
    "taker_edge_ticks": Bound(1.0, 0.5, 20),
    "taker_clip": Bound(20, 1, 100),
    "taker_auto_ms": Bound(0, 0, 60000),
    "dual_pillow_ticks": Bound(1.0, 0, 500),
    "dual_quote_when_closed": Bound(1, 0, 1),
    "arb_scale_ticks": Bound(3.0, 0, 500),
    "news_both_listings": Bound(1, 0, 1),
    "jump_ticks": Bound(5.0, 1, 100),
    "jump_cooldown_s": Bound(1.0, 0, 30),
    "news_hold_max_s": Bound(30.0, 0, 300),
    "news_cooldown_s": Bound(15.0, 0, 120),
    "news_min_confidence": Bound(0.75, 0.5, 1.0),
    "news_size_fraction": Bound(1.0, 0, 1),
    "enabled": Bound(1, 0, 1),
    "lag_enabled": Bound(0, 0, 1),
    "lag_edge_ticks": Bound(1.0, 0, 500),
    "lag_clip": Bound(40, 0, 200),
    "lag_max_position": Bound(80, 0, 1000),
    "lag_pause_s": Bound(0.5, 0.05, 60),
    "lag_take_enabled": Bound(0, 0, 1),
    "lag_sweep_ticks": Bound(0, 0, 50),
    "lag_guard_s": Bound(0, 0, 10),
    "lag_hedge": Bound(0, 0, 1),
    "lag_warmup_ticks": Bound(0, 0, 500),
    "lag_seed": Bound(0, 0, 1),
    # The replayed-path schedule (otk/schedule.py), off by default. sched_t0 pins the round start in epoch seconds (0: auto).
    "sched_enabled": Bound(0, 0, 1),
    "sched_scale": Bound(1.0, 0, 1),
    "sched_enter_s": Bound(0.3, 0, 15),
    "sched_exit_s": Bound(6.0, 0, 30),
    "sched_pull_s": Bound(2.0, 0, 15),
    "sched_reach_ticks": Bound(3.0, 0, 50),
    "sched_offset_s": Bound(0.0, -30, 30),
    "sched_t0": Bound(0.0, 0, 1e10),
    "gap_enabled": Bound(0, 0, 1),
    "gap_entry_z": Bound(2.0, 0.25, 10),
    "gap_exit_z": Bound(0.5, 0, 5),
    "gap_half_life_s": Bound(120.0, 5, 3600),
    "gap_min_edge_ticks": Bound(2.0, 0, 500),
    "gap_clip": Bound(10, 0, 100),
    "gap_max_position": Bound(60, 0, 1000),
    "follow_min_size": Bound(0, 0, 100000),
    "follow_ticks": Bound(2.0, 0, 20),
    "follow_hold_s": Bound(5.0, 0, 120),
    "ladder_levels": Bound(1, 1, 6),
    "ladder_step_ticks": Bound(3.0, 1, 50),
    "ladder_size_decay": Bound(0.7, 0.1, 1),
    "mr_enabled": Bound(0, 0, 1),
    "mr_entry_z": Bound(2.0, 0.25, 10),
    "mr_exit_z": Bound(0.5, 0, 5),
    "mr_stop_z": Bound(6.0, 1, 50),
    "mr_half_life_s": Bound(60.0, 5, 3600),
    "mr_min_edge_ticks": Bound(1.0, 0, 500),
    "mr_max_position": Bound(30, 0, 1000),
    "mr_clip": Bound(10, 0, 100),
    "mr_cooldown_s": Bound(60.0, 0, 3600),
    "mr_interval_s": Bound(0.5, 0, 30),
    "kill_enabled": Bound(1, 0, 1),
    "kill_drawdown": Bound(0, 0, 100_000_000),
    "kill_drawdown_frac": Bound(0, 0, 10),
    "kill_max_loss": Bound(0, 0, 100_000_000),
    "kill_loss_per_min": Bound(0, 0, 100_000_000),
    "kill_cooloff_s": Bound(120.0, 0, 3600),
    "kill_max_trips": Bound(3, 1, 100),
    "kill_resume_size": Bound(0.5, 0.05, 1),
    "kill_reject_limit": Bound(0, 0, 10_000),
    "kill_pause_s": Bound(5.0, 0, 600),
    "kill_stale_s": Bound(0, 0, 3600),
    "dual_nudge": Bound(0.25, 0, 1),
    "signal_size": Bound(0.5, 0, 1),
    "news_full_confidence": Bound(0.9, 0.5, 1),
    "news_stop_ticks": Bound(5.0, 0, 500),
    "news_check_s": Bound(1.0, 0, 60),
    "ap_enabled": Bound(1, 0, 1),
    "ap_directional": Bound(1, 0, 1),
    "ap_mr": Bound(1, 0, 1),
    "ap_gap": Bound(1, 0, 1),
    "ap_quoting": Bound(1, 0, 1),
    "ap_drawdown": Bound(0, 0, 10_000_000),
}
INSTRUMENTS_KEY = "instruments"
REFERENCE_TICK = DEFAULT_TICK
# Distances in price, written in ticks: on a finer grid they need more ticks to mean the same thing.
# base_spread_ticks stays out: on a fine grid, quoting closer is the point.
FINE_TICK_SCALED = (
    "jump_ticks",
    "max_spread_ticks",
    "circuit_ticks",
    "max_deviation_ticks",
    "max_vol_ticks",
    "skew_ticks",
    "near_touch_ticks",
    "reprice_ticks",
    "follow_ticks",
    "ladder_step_ticks",
    "dual_offset_cap_ticks",
    "gap_min_edge_ticks",
    "mr_min_edge_ticks",
    "arb_scale_ticks",
    "news_stop_ticks",
)


def defaults() -> dict[str, float]:
    return {name: bound.default for name, bound in BOUNDS.items()}


def _clamped(key: str, value: object) -> float:
    if key not in BOUNDS:
        raise ValueError(f"unknown parameter {key!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameter {key!r} must be a number, got {value!r}")  # noqa: TRY004
    if not math.isfinite(value):
        raise ValueError(f"parameter {key!r} must be finite, got {value!r}")
    bound = BOUNDS[key]
    return max(bound.low, min(bound.high, value))


def _instrument_overrides(raw: object) -> dict[str, dict[str, float]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{INSTRUMENTS_KEY!r} must map instrument ids to parameters")  # noqa: TRY004
    overrides = {}
    for instrument_id, values in raw.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"overrides for {instrument_id!r} must be an object")  # noqa: TRY004
        overrides[instrument_id] = {key: _clamped(key, value) for key, value in values.items()}
    return overrides


def validate_params(raw: Mapping[str, object]) -> dict:
    """Defaults overlaid with `raw`, clamped; raises ValueError on unknown keys or non-numbers.

    `{"instruments": {"ASML": {"size": 30}}}` overrides values for one instrument, for example to
    quote more into a bot on that instrument. The same hard bounds apply.
    """
    merged: dict = defaults()
    for key, value in raw.items():
        if key == INSTRUMENTS_KEY:
            merged[INSTRUMENTS_KEY] = _instrument_overrides(value)
        else:
            merged[key] = _clamped(key, value)
    return merged


def merge_params(*layers: Mapping) -> dict:
    """Later layers win, and per-instrument overrides merge rather than replace each other."""
    merged: dict = {}
    for layer in layers:
        for key, value in layer.items():
            if key != INSTRUMENTS_KEY:
                merged[key] = value
                continue
            instruments = merged.setdefault(INSTRUMENTS_KEY, {})
            for instrument_id, values in value.items():
                instruments[instrument_id] = {**instruments.get(instrument_id, {}), **values}
    return merged


def fine_tick_overrides(universe) -> dict[str, dict[str, float]]:
    """Keep price distances meaningful on instruments finer than the reference tick."""
    overrides: dict[str, dict[str, float]] = {}
    for instrument_id, info in universe.by_id.items():
        scale = REFERENCE_TICK / info.tick
        if scale > 1.0 + 1e-9:
            overrides[instrument_id] = {
                key: round(BOUNDS[key].default * scale, 6) for key in FINE_TICK_SCALED
            }
    return overrides


def scaled_sizes(params: Mapping, factor: float) -> Mapping:
    """A copy with quoted size scaled, globally and per instrument, for trading back in after a stop."""
    if factor >= 1.0:
        return params
    scaled = {**params, "size": params["size"] * factor, "taker_clip": params["taker_clip"] * factor}
    overrides = params.get(INSTRUMENTS_KEY)
    if overrides:
        scaled[INSTRUMENTS_KEY] = {
            iid: {**values, "size": values["size"] * factor} if "size" in values else values
            for iid, values in overrides.items()
        }
    return scaled


def params_for(params: Mapping, instrument_id: str) -> Mapping:
    """Effective parameters for one instrument."""
    overrides = params.get(INSTRUMENTS_KEY, {}).get(instrument_id)
    return {**params, **overrides} if overrides else params


class ParamStore:
    def __init__(self, path: str | Path | None = None, base: Mapping | None = None) -> None:
        self.path = Path(path) if path else None
        self._base = dict(base or {})
        self._overlay: dict = {}
        self._file: dict = {}
        self.current = validate_params(self._base)
        self.last_error = ""
        self._mtime: int | None = None
        self.version = 0

    def _pinned_overlay(self) -> dict:
        """A key set in the file also outranks Autopilot's per-instrument values for that key."""
        instruments = {
            instrument_id: {k: v for k, v in values.items() if k not in self._file}
            for instrument_id, values in self._overlay.get(INSTRUMENTS_KEY, {}).items()
        }
        globals_ = {k: v for k, v in self._overlay.items() if k != INSTRUMENTS_KEY}
        return {**globals_, INSTRUMENTS_KEY: instruments} if instruments else globals_

    def baseline(self) -> dict:
        """Everything except Autopilot: what its adjustments are measured against."""
        return validate_params(merge_params(self._base, self._file))

    def set_overlay(self, overlay: Mapping) -> None:
        """Autopilot's layer. Raises ValueError on unknown keys; values clamp to the same bounds."""
        validate_params(overlay)
        self._overlay = dict(overlay)
        self.current = validate_params(merge_params(self._base, self._pinned_overlay(), self._file))
        self.version += 1

    def refresh(self) -> dict[str, float]:
        """Cheap when unchanged: one stat call. Never raises; a bad file keeps the last good set."""
        if self.path is None:
            return self.current
        try:
            mtime = os.stat(self.path).st_mtime_ns
            if mtime == self._mtime:
                return self.current
            self._mtime = mtime
            with open(self.path) as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, Mapping):
                raise ValueError("params file must hold a JSON object")  # noqa: TRY004
            validate_params(loaded)
            self._file = loaded
            self.current = validate_params(merge_params(self._base, self._pinned_overlay(), self._file))
            self.last_error = ""
            self.version += 1
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return self.current


def write_params_atomic(path: str | Path, params: Mapping[str, object]) -> None:
    """Validated write through a rename, so the loop never reads a half-written file."""
    validate_params(params)
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(params, indent=2, sort_keys=True))
    os.replace(temporary, target)
