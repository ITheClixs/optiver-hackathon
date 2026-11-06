"""Pure quoting function: fair value and state in, two-sided quote out.

Follows the disclosed structure of the 1st-place TUM 2025 market maker: microprice theoretical,
base spread, volatility widening, imbalance shift bounded in ticks, tanh inventory skew, and hard
bounds around fair. Each knob is scaled in ticks so it stays meaningful with no backtest data.
https://github.com/fabianbaiertum/Optiver-Hackathon-TUM
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from otk.fair import Top, round_ask, round_bid


@dataclass(frozen=True)
class QuoteParams:
    base_spread_ticks: float = 2.0
    vol_factor: float = 1.0
    max_vol_ticks: float = 6.0
    imbalance_ticks: float = 1.0
    skew_ticks: float = 4.0
    skew_curve: float = 2.0
    max_deviation_ticks: float = 15.0
    size: int = 10
    min_size: int = 1
    penny_only: bool = True
    signal_size: float = 0.0


class Quote(NamedTuple):
    bid: float
    ask: float
    bid_size: int
    ask_size: int


def _sizes(position: int, limit: int, book_imbalance: float, params: QuoteParams) -> tuple:
    """Linear taper on the side that adds to inventory; the reducing side keeps full size.

    Then lean with the signal: show more on the side the imbalance favours, less on the other.
    """
    load = min(abs(position) / float(limit), 1.0) if limit > 0 else 1.0
    tapered = max(round(params.size * (1.0 - load)), 0)
    if position > 0:
        bid, ask = tapered, params.size
    elif position < 0:
        bid, ask = params.size, tapered
    else:
        bid, ask = params.size, params.size
    tilt = params.signal_size * max(min(book_imbalance, 1.0), -1.0)
    return max(round(bid * (1.0 + tilt)), 0), max(round(ask * (1.0 - tilt)), 0)


def _improve_at_most_one_tick(bid: float, ask: float, top: Top, tick: float) -> tuple:
    """Price-time priority: one tick better wins the queue, more gives edge away for nothing."""
    if top.bid is not None:
        bid = min(bid, top.bid + tick)
    if top.ask is not None:
        ask = max(ask, top.ask - tick)
    return bid, ask


def _no_cross(bid: float, ask: float, top: Top, tick: float) -> tuple:
    if top.ask is not None:
        bid = min(bid, top.ask - tick)
    if top.bid is not None:
        ask = max(ask, top.bid + tick)
    return bid, ask


def compute_quotes(
    fair: float,
    position: int,
    limit: int,
    vol: float,
    book_imbalance: float,
    params: QuoteParams,
    tick: float,
    top: Top | None = None,
) -> Quote:
    """`vol` is per-update fair-value stdev in price units; `top` must exclude our own orders."""
    half = params.base_spread_ticks * tick / 2.0
    half += min(params.vol_factor * vol, params.max_vol_ticks * tick)
    shift = params.imbalance_ticks * tick * max(min(book_imbalance, 1.0), -1.0)
    load = position / float(limit) if limit > 0 else 0.0
    skew = params.skew_ticks * tick * math.tanh(params.skew_curve * load)
    center = fair + shift - skew

    floor_bid, floor_ask = center - half, center + half
    bid, ask = floor_bid, floor_ask
    if top is not None and params.penny_only:
        bid, ask = _improve_at_most_one_tick(bid, ask, top, tick)

    deviation = params.max_deviation_ticks * tick
    bid = max(bid, fair - deviation)
    ask = min(ask, fair + deviation)
    if top is not None:  # last, or the clamp pulls a quote through the book when fair is far from it
        bid, ask = _no_cross(bid, ask, top, tick)

    bid_px, ask_px = round_bid(bid, tick), round_ask(ask, tick)
    if ask_px - bid_px < tick - 1e-9:
        ask_px = round_ask(bid_px + tick, tick)
    bid_size, ask_size = _sizes(position, limit, book_imbalance, params)
    return Quote(bid_px, ask_px, bid_size, ask_size)
