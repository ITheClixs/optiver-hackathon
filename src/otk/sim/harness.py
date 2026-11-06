"""Runs strategies against a simulated session on virtual time."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from otk.engine import Engine, EngineConfig
from otk.instruments import Universe
from otk.params import INSTRUMENTS_KEY, ParamStore, fine_tick_overrides
from otk.sim.exchange import LaggedExchange, SimExchange, SimMarket


def make_engine(
    market: SimMarket,
    team: str,
    strategies: list,
    config: EngineConfig | None = None,
    params: ParamStore | None = None,
    recorder=None,
    lag_s: float = 0.0,
) -> Engine:
    exchange = LaggedExchange(market, team, lag_s) if lag_s else SimExchange(market, team)
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    if params is None:
        fine = fine_tick_overrides(universe)
        params = ParamStore(base={INSTRUMENTS_KEY: fine} if fine else None)
    return Engine(exchange, universe, strategies, config, params, recorder, clock=lambda: market.now)


@dataclass(frozen=True)
class SimResult:
    team: str
    seed: int
    settled_pnl: float
    marked_pnl: float
    max_drawdown: float
    pnl_path: Sequence[float]
    disconnects: int
    limit_breaches: int
    rate_breaches: int
    delta_hacks: int
    peak_write_rate: int
    writes: int
    errors: int


def run_session(
    market: SimMarket,
    engines: dict[str, Engine],
    seconds: float,
    dt: float = 0.1,
    warmup_s: float = 5.0,
    seed: int = 0,
) -> dict[str, SimResult]:
    """Alternate market steps with one tick per engine. Sample PnL once per simulated second."""
    for _ in range(int(warmup_s / dt)):
        market.advance(dt)
    paths: dict[str, list[float]] = {team: [] for team in engines}
    steps_per_sample = max(round(1.0 / dt), 1)
    for step in range(int(seconds / dt)):
        market.advance(dt)
        for engine in engines.values():
            engine.tick()
        if step % steps_per_sample == 0:
            for team in engines:
                paths[team].append(market.marked_pnl(team))
    return {team: _result(market, team, engines[team], paths[team], seed) for team in engines}


def _max_drawdown(path: Sequence[float]) -> float:
    peak, worst = float("-inf"), 0.0
    for value in path:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def _result(
    market: SimMarket, team: str, engine: Engine, path: Sequence[float], seed: int
) -> SimResult:
    stats = market.stats[team]
    return SimResult(
        team, seed, market.settled_pnl(team), market.marked_pnl(team), _max_drawdown(path),
        tuple(path), stats.disconnects, stats.limit_breaches, stats.rate_breaches,
        stats.delta_hacks, stats.peak_write_rate, stats.writes, engine.errors,
    )


def run_seeds(
    scenario: Callable[[int], SimMarket],
    strategies: Callable[[], list],
    seeds: Sequence[int],
    seconds: float = 120.0,
    config: EngineConfig | None = None,
    lag_s: float = 0.0,
) -> list[SimResult]:
    results = []
    for seed in seeds:
        market = scenario(seed)
        engine = make_engine(market, "us", strategies(), config, lag_s=lag_s)
        results.append(run_session(market, {"us": engine}, seconds, seed=seed)["us"])
    return results
