#!/usr/bin/env python3
"""Rehearse every kill switch on the simulator, in seconds. Run it before the event and after any change.

Prints one line per switch with what it should do and whether it did. Nothing here logs in.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.engine import Engine, EngineConfig
from otk.instruments import Universe
from otk.params import ParamStore, write_params_atomic
from otk.sim.exchange import SimExchange, SimMarket
from otk.strategies import MarketMaker

STEP_S = 0.5


class Book:
    """A two-sided wall that can fall or freeze, with a little trading so marks stay current."""

    def __init__(self, per_step: float = 0.0, stop_after: float | None = None) -> None:
        self.level = 100.0
        self.per_step = per_step
        self.stop_after = stop_after

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        market.wipe_orders("wall", "A")
        if self.stop_after is None or now < self.stop_after:
            self.level -= self.per_step
        wall = SimExchange(market, "wall", bot=True)
        wall.insert_order("A", price=round(self.level - 0.1, 1), volume=500, side="bid", order_type="limit")
        wall.insert_order("A", price=round(self.level + 0.1, 1), volume=500, side="ask", order_type="limit")
        SimExchange(market, "noise", bot=True).insert_order(
            "A", price=round(self.level - 0.1, 1), volume=1, side="ask", order_type="ioc"
        )


def setup(actor: Book, run_dir: Path, **params) -> tuple:
    market = SimMarket(instruments=["A"])
    market.actors.append(actor)
    market.advance(STEP_S)
    exchange = SimExchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    config = EngineConfig(kill_file=str(run_dir / "KILL"), flatten_file=str(run_dir / "FLATTEN"))
    store = ParamStore(run_dir / "params.json", base=params)
    engine = Engine(exchange, universe, [MarketMaker(["A"])], config, store, clock=lambda: market.now)
    exchange.insert_order("A", price=101.0, volume=30, side="bid", order_type="ioc")
    return market, engine, exchange


def steps(market: SimMarket, engine: Engine, count: int) -> None:
    for _ in range(count):
        market.advance(STEP_S)
        engine.tick()


def flat_and_quiet(exchange: SimExchange) -> bool:
    return exchange.get_positions()["A"] == 0 and not exchange.get_outstanding_orders("A")


def drill_max_loss(run_dir: Path) -> bool:
    market, engine, exchange = setup(Book(0.2, stop_after=5.0), run_dir, kill_max_loss=5, kill_cooloff_s=10)
    steps(market, engine, 8)
    stopped = engine.halted and flat_and_quiet(exchange)
    steps(market, engine, 30)
    return stopped and not engine.halted and bool(exchange.get_outstanding_orders("A"))


def drill_drawdown_scaled(run_dir: Path) -> bool:
    market, engine, exchange = setup(Book(0.5), run_dir, kill_drawdown_frac=0.001, kill_cooloff_s=1000)
    steps(market, engine, 10)
    return engine.halted and flat_and_quiet(exchange)


def drill_latch(run_dir: Path) -> bool:
    market, engine, exchange = setup(Book(0.5), run_dir, kill_max_loss=2, kill_cooloff_s=1, kill_max_trips=2)
    steps(market, engine, 12)
    first_trip = engine.kill.trips == 1 and not engine.kill.latched
    exchange.insert_order("A", price=200.0, volume=30, side="bid", order_type="ioc")
    steps(market, engine, 12)
    return first_trip and engine.kill.latched and engine.halted


def drill_kill_file(run_dir: Path) -> bool:
    (run_dir / "KILL").write_text("stop")
    market, engine, exchange = setup(Book(), run_dir)
    steps(market, engine, 4)
    pulled = engine.halted and not exchange.get_outstanding_orders("A") and exchange.get_positions()["A"] == 30
    (run_dir / "KILL").unlink()
    steps(market, engine, 4)
    return pulled and not engine.halted


def drill_flatten_file(run_dir: Path) -> bool:
    (run_dir / "FLATTEN").write_text("stop")
    market, engine, exchange = setup(Book(), run_dir)
    steps(market, engine, 4)
    result = engine.halted and flat_and_quiet(exchange)
    (run_dir / "FLATTEN").unlink()
    return result


def drill_params_off(run_dir: Path) -> bool:
    market, engine, exchange = setup(Book(), run_dir)
    steps(market, engine, 2)
    write_params_atomic(run_dir / "params.json", {"enabled": 0})
    steps(market, engine, 3)
    stopped = engine.halted and not exchange.get_outstanding_orders("A")
    (run_dir / "params.json").unlink()
    return stopped


def drill_stale_data(run_dir: Path) -> bool:
    actor = Book(0.05)
    market, engine, exchange = setup(actor, run_dir, kill_stale_s=3)
    steps(market, engine, 4)
    market.actors.remove(actor)
    steps(market, engine, 16)
    stopped = engine.halted and not exchange.get_outstanding_orders("A")
    market.actors.append(actor)
    steps(market, engine, 4)
    return stopped and not engine.halted and exchange.get_positions()["A"] != 0


def drill_rejects(run_dir: Path) -> bool:
    market, engine, exchange = setup(Book(), run_dir, kill_reject_limit=3, kill_pause_s=2)
    steps(market, engine, 2)
    engine.manager.rejects += 10
    steps(market, engine, 2)
    stopped = engine.halted and not exchange.get_outstanding_orders("A")
    steps(market, engine, 12)
    return stopped and not engine.halted


DRILLS: dict[str, tuple[str, Callable[[Path], bool]]] = {
    "session max loss": ("flatten, cool off, resume smaller", drill_max_loss),
    "drawdown scaled to notional": ("flatten", drill_drawdown_scaled),
    "repeated losses": ("latch until restart", drill_latch),
    "touch run/KILL": ("pull, keep position, resume on rm", drill_kill_file),
    "touch run/FLATTEN": ("pull and close out", drill_flatten_file),
    "params enabled=0": ("pull", drill_params_off),
    "stale market data": ("pull, resume when data flows", drill_stale_data),
    "reject storm": ("pull, resume after a pause", drill_rejects),
}


def run_all() -> dict[str, bool]:
    results = {}
    for name, (_expected, drill) in DRILLS.items():
        with tempfile.TemporaryDirectory() as folder:
            results[name] = drill(Path(folder))
    return results


def main() -> int:
    results = run_all()
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name:<30} {DRILLS[name][0]}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
