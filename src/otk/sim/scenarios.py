"""Scenario presets covering every format seen in public Optibook editions.

Formats and constants come from public evidence on past Optibook events. The numeric bot parameters are
illustrative; only live captures on the day reveal the real ones (`scripts/analyze_capture.py`).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from otk.sim.bots import (
    LiquidityBot,
    MeanRevertingFair,
    MomentumChaser,
    NewsProcess,
    NoiseTaker,
    PennyTeam,
    RandomWalk,
    Sniper,
)
from otk.sim.exchange import SimMarket

INDEX_WEIGHTS: dict[str, float] = {
    "ASML": 908.06, "AAPL": 129.24, "SAP": 124.78, "TSLA": 2245.39, "NVDA": 953.21,
}
INDEX_INSTRUMENTS = (
    "ASML", "AAPL", "SAP", "TSLA", "NVDA", "ASML_DUAL", "SAP_DUAL", "OB5X_ETF",
    "OB5X_202703_F", "ASML_202703_F", "ASML_202703_100C", "ASML_202703_100P",
)


def _market(instruments: Sequence[str], start: float = 100.0, **kwargs) -> SimMarket:
    market = SimMarket(instruments=list(instruments), **kwargs)
    return market


def quiet_market(
    seed: int = 0,
    instruments: Iterable[str] = ("A", "B"),
    sigma: float = 0.02,
    noise_rate: float = 1.0,
) -> SimMarket:
    """Independent stocks, a ladder per stock, uninformed flow."""
    rng = random.Random(seed)
    ids = list(instruments)
    market = _market(ids)
    market.fair = {iid: 100.0 for iid in ids}
    market.actors.append(RandomWalk(rng, sigma))
    for iid in ids:
        market.actors.append(LiquidityBot(f"mm_{iid}", iid, half_spread_ticks=3, rng=rng))
    market.actors.append(NoiseTaker("noise", ids, rng, rate_per_s=noise_rate))
    return market


def dual_listing(seed: int = 0, bases: Iterable[str] = ("ASML", "SAP"), sigma: float = 0.03) -> SimMarket:
    """Liquid primary, thin dual that lags it and wanders: the classic first Optibook exercise."""
    rng = random.Random(seed)
    names = list(bases)
    ids = names + [f"{base}_DUAL" for base in names]
    market = _market(ids)
    market.fair = {base: 100.0 for base in names}
    market.actors.append(RandomWalk(rng, sigma))
    for base in names:
        market.actors.append(LiquidityBot(f"mm_{base}", base, half_spread_ticks=2, rng=rng))
        market.actors.append(
            LiquidityBot(
                f"mm_{base}_DUAL", f"{base}_DUAL", half_spread_ticks=4, levels=2, size=50,
                refresh_s=2.0, lag_s=3.0, noise_ticks=1.5, rng=rng,
            )
        )
    market.actors.append(NoiseTaker("noise", ids, rng, rate_per_s=2.0))
    return market


def news_stocks(
    seed: int = 0,
    instruments: Iterable[str] = ("NVDA", "ING", "SAN", "PFE", "CSCO"),
    sigma: float = 0.02,
    news_rate: float = 0.03,
    jump_fraction: float = 0.02,
) -> SimMarket:
    """Five uncorrelated stocks with a social feed: the hackaTUM and TUM 2025 format."""
    rng = random.Random(seed)
    ids = list(instruments)
    market = _market(ids)
    market.fair = {iid: 100.0 for iid in ids}
    market.actors.append(RandomWalk(rng, sigma))
    market.actors.append(NewsProcess(rng, ids, rate_per_s=news_rate, jump_fraction=jump_fraction))
    for iid in ids:
        market.actors.append(LiquidityBot(f"mm_{iid}", iid, half_spread_ticks=3, rng=rng))
    market.actors.append(NoiseTaker("noise", ids, rng, rate_per_s=2.0))
    market.actors.append(Sniper("sniper", ids, rng=rng))
    return market


def index_complex(seed: int = 0, sigma: float = 0.02, stock_tick: float | None = None) -> SimMarket:
    """Stocks, duals, ETF, futures and options: the 2026 academy and FutureFocus universe.

    `stock_tick` gives stocks and duals a finer grid than the rest, as live dashboards have shown.
    """
    rng = random.Random(seed)
    fine = {iid: stock_tick for iid in INDEX_INSTRUMENTS if "_20" not in iid and "_ETF" not in iid}
    market = _market(
        INDEX_INSTRUMENTS, index_weights=dict(INDEX_WEIGHTS), tick_sizes=fine if stock_tick else None
    )
    market.fair = {name: 100.0 for name in INDEX_WEIGHTS}
    market.actors.append(RandomWalk(rng, sigma))
    for iid in INDEX_INSTRUMENTS:
        thin = iid.endswith(("_DUAL", "_ETF")) or "_2027" in iid
        market.actors.append(
            LiquidityBot(
                f"mm_{iid}", iid, half_spread_ticks=5 if thin else 2, levels=2,
                size=50 if thin else 200, refresh_s=2.0 if thin else 1.0, rng=rng,
            )
        )
    market.actors.append(NoiseTaker("noise", list(INDEX_INSTRUMENTS), rng, rate_per_s=4.0))
    return market


def passive_field(seed: int = 0, bases: Iterable[str] = ("ASML", "SAP")) -> SimMarket:
    """Passive quoters anchor the book, so prices wander and return: reversion pays against them."""
    rng = random.Random(seed)
    names = list(bases)
    ids = names + [f"{base}_DUAL" for base in names]
    market = _market(ids)
    market.fair = {base: 100.0 for base in names}
    market.actors.append(MeanRevertingFair(rng, dict(market.fair), half_life_s=60.0, sigma_per_sqrt_s=0.08))
    for iid in ids:
        market.actors.append(LiquidityBot(f"mm_{iid}", iid, half_spread_ticks=1, levels=2, rng=rng))
        market.actors.append(PennyTeam(f"team_{iid}", iid))
    market.actors.append(NoiseTaker("noise", ids, rng, rate_per_s=2.0))
    return market


def adverse_field(market: SimMarket, seed: int = 0, teams_per_instrument: int = 2) -> SimMarket:
    """Add competitors and informed flow: the conservative preset for any headline number."""
    rng = random.Random(seed + 10_000)
    stocks = [iid for iid in market.instruments if "_20" not in iid]
    market.actors.append(Sniper("sniper_adverse", stocks, rate_per_s=6.0, rng=rng))
    for iid in stocks:
        market.actors.append(MomentumChaser(f"chaser_{iid}", iid, rng))
        for number in range(teams_per_instrument):
            market.actors.append(PennyTeam(f"team{number}_{iid}", iid))
    return market
