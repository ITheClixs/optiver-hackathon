"""Fair-value primitives. Runtime code: stdlib only, Python 3.8+, no I/O."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import NormalDist
from typing import NamedTuple

_NORMAL = NormalDist()
_PRICE_DIGITS = 10
_EPS = 1e-6  # in ticks: a price this close to the grid is on the grid


class Level(NamedTuple):
    price: float
    volume: int


class Book(NamedTuple):
    bids: Sequence[Level]
    asks: Sequence[Level]


class Top(NamedTuple):
    bid: float | None
    bid_vol: int | None
    ask: float | None
    ask_vol: int | None


def snap(price: float, tick: float) -> float:
    """Nearest tick, cleaned of float noise such as 86.80000000000001."""
    return round(round(price / tick) * tick, _PRICE_DIGITS)


def round_bid(price: float, tick: float) -> float:
    return round(math.floor(price / tick + _EPS) * tick, _PRICE_DIGITS)


def round_ask(price: float, tick: float) -> float:
    return round(math.ceil(price / tick - _EPS) * tick, _PRICE_DIGITS)


def top_of_book(book) -> Top | None:
    if book is None or (not book.bids and not book.asks):
        return None
    bid = book.bids[0] if book.bids else None
    ask = book.asks[0] if book.asks else None
    return Top(
        bid.price if bid else None,
        bid.volume if bid else None,
        ask.price if ask else None,
        ask.volume if ask else None,
    )


def mid(top: Top | None) -> float | None:
    if top is None or top.bid is None or top.ask is None:
        return None
    return (top.bid + top.ask) / 2.0


def microprice(top: Top | None) -> float | None:
    """Size-weighted mid, leaning toward the thin side (Stoikov 2018, top-of-book form)."""
    if top is None or top.bid is None or top.ask is None:
        return None
    total = top.bid_vol + top.ask_vol
    if total <= 0:
        return (top.bid + top.ask) / 2.0
    return (top.bid * top.ask_vol + top.ask * top.bid_vol) / total


def imbalance(top: Top | None) -> float:
    """(bid size - ask size) / total, in [-1, 1]; 0 when a side is missing."""
    if top is None or top.bid is None or top.ask is None:
        return 0.0
    total = top.bid_vol + top.ask_vol
    return (top.bid_vol - top.ask_vol) / total if total > 0 else 0.0


def near_touch_imbalance(book, band: float) -> float:
    """Imbalance using only levels within `band` of mid.

    One team's two days of Optibook recordings: near-touch imbalance correlated 0.22 to 0.29 with
    the next 5 s move, the full book only 0.01 to 0.10. Re-measure it on the day.
    """
    if book is None or not book.bids or not book.asks:
        return 0.0
    center = (book.bids[0].price + book.asks[0].price) / 2.0
    bid_volume = sum(level.volume for level in book.bids if center - level.price <= band)
    ask_volume = sum(level.volume for level in book.asks if level.price - center <= band)
    total = bid_volume + ask_volume
    return (bid_volume - ask_volume) / total if total > 0 else 0.0


def wall_mid(book) -> float | None:
    """Mid of the largest resting level on each side, robust to small pennying quotes."""
    if book is None or not book.bids or not book.asks:
        return None
    wall_bid = max(book.bids, key=lambda level: level.volume)
    wall_ask = max(book.asks, key=lambda level: level.volume)
    return (wall_bid.price + wall_ask.price) / 2.0


def _strip_side(levels, own: Mapping[float, int], tick: float) -> Sequence[Level]:
    remaining = []
    for level in levels or []:
        ours = own.get(snap(level.price, tick), 0)
        volume = level.volume - ours
        if volume > 0:
            remaining.append(Level(level.price, volume))
    return remaining


def external_book(book, own_orders: Iterable, tick: float = 0.1) -> Book:
    """The book without our resting volume, so we never price off or penny ourselves."""
    bids: dict = {}
    asks: dict = {}
    for order in own_orders:
        side = bids if order.side == "bid" else asks
        key = snap(order.price, tick)
        side[key] = side.get(key, 0) + order.volume
    if book is None:
        return Book([], [])
    return Book(_strip_side(book.bids, bids, tick), _strip_side(book.asks, asks, tick))


def index_value(
    prices: Mapping[str, float], weights: Mapping[str, float], divisor: float = 1000.0
) -> float | None:
    """Weighted index level; None when any constituent price is missing."""
    total = 0.0
    for name, weight in weights.items():
        price = prices.get(name)
        if price is None:
            return None
        total += weight * price
    return total / divisor


def etf_fair(index: float, multiplier: float = 0.25, cash: float = 2.5) -> float:
    return cash + multiplier * index


def future_fair(spot: float, rate: float, tau_years: float) -> float:
    return spot * math.exp(rate * tau_years)


def _d1_d2(spot: float, strike: float, tau: float, rate: float, vol: float):
    root = vol * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * tau) / root
    return d1, d1 - root


def bs_price(kind: str, spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    """European Black-Scholes; intrinsic value at or after expiry or at zero vol."""
    if tau <= 0.0 or vol <= 0.0:
        forward_strike = strike * math.exp(-rate * max(tau, 0.0))
        payoff = spot - forward_strike if kind == "call" else forward_strike - spot
        return max(payoff, 0.0)
    d1, d2 = _d1_d2(spot, strike, tau, rate, vol)
    discounted = strike * math.exp(-rate * tau)
    if kind == "call":
        return spot * _NORMAL.cdf(d1) - discounted * _NORMAL.cdf(d2)
    return discounted * _NORMAL.cdf(-d2) - spot * _NORMAL.cdf(-d1)


def bs_delta(kind: str, spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    if tau <= 0.0 or vol <= 0.0:
        itm = spot > strike
        if kind == "call":
            return 1.0 if itm else 0.0
        return 0.0 if itm else -1.0
    d1, _ = _d1_d2(spot, strike, tau, rate, vol)
    return _NORMAL.cdf(d1) if kind == "call" else _NORMAL.cdf(d1) - 1.0


def implied_vol(
    kind: str, price: float, spot: float, strike: float, tau: float, rate: float
) -> float | None:
    """Bisection on [0.001, 5]; None when the price is outside the no-arbitrage range."""
    low, high = 0.001, 5.0
    if tau <= 0.0 or price <= bs_price(kind, spot, strike, tau, rate, low):
        return None
    if price >= bs_price(kind, spot, strike, tau, rate, high):
        return None
    for _ in range(60):
        middle = (low + high) / 2.0
        if bs_price(kind, spot, strike, tau, rate, middle) < price:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0
