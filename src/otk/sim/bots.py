"""Simulated participants. Behaviours follow what public Optibook repos and guides describe.

Each actor exposes `act(market, now, dt)`. All randomness comes from the injected `random.Random`,
so a seed fully determines a session.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Sequence

from otk.fair import external_book, top_of_book
from otk.instruments import parse_instrument_id
from otk.sim.exchange import SimExchange, SimMarket, SocialMediaFeed


def _poisson_events(rng: random.Random, rate_per_s: float, dt: float) -> int:
    threshold = math.exp(-rate_per_s * dt)
    count, product = 0, rng.random()
    while product > threshold:
        count += 1
        product *= rng.random()
    return count


class RandomWalk:
    """Fair value per underlying: Gaussian steps in price units per sqrt(second)."""

    def __init__(self, rng: random.Random, sigma_per_sqrt_s: float) -> None:
        self.rng = rng
        self.sigma = sigma_per_sqrt_s

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        for name in sorted(market.fair):
            step = self.rng.gauss(0.0, self.sigma * math.sqrt(dt))
            market.fair[name] = max(market.tick * 10, market.fair[name] + step)


class MeanRevertingFair:
    """Fair value pulled back to an anchor: the regime where fading moves pays."""

    def __init__(
        self, rng: random.Random, anchors: dict[str, float], half_life_s: float, sigma_per_sqrt_s: float
    ) -> None:
        self.rng = rng
        self.anchors = anchors
        self.half_life_s = half_life_s
        self.sigma = sigma_per_sqrt_s

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        pull = 1.0 - 0.5 ** (dt / self.half_life_s)
        for name, anchor in sorted(self.anchors.items()):
            step = self.rng.gauss(0.0, self.sigma * math.sqrt(dt))
            market.fair[name] += pull * (anchor - market.fair[name]) + step


class LiquidityBot:
    """Large ladder around fair; refreshes on a period, so it goes stale between refreshes."""

    def __init__(
        self,
        name: str,
        instrument_id: str,
        half_spread_ticks: int = 2,
        levels: int = 3,
        size: int = 200,
        refresh_s: float = 1.0,
        lag_s: float = 0.0,
        noise_ticks: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        self.name = name
        self.iid = instrument_id
        self.half_spread_ticks = half_spread_ticks
        self.levels = levels
        self.size = size
        self.refresh_s = refresh_s
        self.lag_s = lag_s
        self.noise_ticks = noise_ticks
        self.rng = rng or random.Random(0)
        self._next = 0.0
        self._history: deque[tuple[float, float]] = deque()
        self._session: SimExchange | None = None

    def _lagged_value(self, market: SimMarket, now: float) -> float | None:
        value = market.value(self.iid)
        if value is None:
            return None
        self._history.append((now, value))
        while len(self._history) > 1 and now - self._history[1][0] >= self.lag_s:
            self._history.popleft()
        return self._history[0][1] if self.lag_s > 0 else value

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        value = self._lagged_value(market, now)
        if value is None or now < self._next:
            return
        self._next = now + self.refresh_s
        session = self._session or SimExchange(market, self.name, bot=True)
        self._session = session
        session.delete_orders(self.iid)
        tick = market.tick_of(self.iid)
        center = value + self.rng.gauss(0.0, self.noise_ticks * tick)
        center_ticks = round(center / tick)
        for level in range(self.levels):
            offset = (self.half_spread_ticks + level) * tick
            for side, price in (("bid", center_ticks * tick - offset), ("ask", center_ticks * tick + offset)):
                if price > tick:
                    session.insert_order(
                        self.iid, price=price, volume=self.size, side=side, order_type="limit"
                    )


class NoiseTaker:
    """Uninformed aggressor: the flow a market maker earns the spread from."""

    def __init__(
        self,
        name: str,
        instruments: Sequence[str],
        rng: random.Random,
        rate_per_s: float = 1.0,
        sizes: Sequence[int] = (5, 10, 25, 50),
    ) -> None:
        self.name = name
        self.instruments = list(instruments)
        self.rng = rng
        self.rate_per_s = rate_per_s
        self.sizes = list(sizes)

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        session = SimExchange(market, self.name, bot=True)
        for _ in range(_poisson_events(self.rng, self.rate_per_s, dt)):
            iid = self.rng.choice(self.instruments)
            book = market.book_snapshot(iid)
            side = self.rng.choice(("bid", "ask"))
            levels = book.asks if side == "bid" else book.bids
            if levels:
                session.insert_order(
                    iid, price=levels[0].price, volume=self.rng.choice(self.sizes), side=side,
                    order_type="ioc",
                )


class Sniper:
    """Informed aggressor: lifts any resting price that is stale relative to fair."""

    def __init__(
        self, name: str, instruments: Sequence[str], rate_per_s: float = 4.0,
        edge_ticks: float = 1.0, size: int = 25, rng: random.Random | None = None,
    ) -> None:
        self.name = name
        self.instruments = list(instruments)
        self.rate_per_s = rate_per_s
        self.edge_ticks = edge_ticks
        self.size = size
        self.rng = rng or random.Random(1)

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        if _poisson_events(self.rng, self.rate_per_s, dt) == 0:
            return
        session = SimExchange(market, self.name, bot=True)
        for iid in self.instruments:
            value = market.value(iid)
            top = top_of_book(market.book_snapshot(iid))
            if value is None or top is None:
                continue
            edge = self.edge_ticks * market.tick_of(iid)
            if top.ask is not None and top.ask < value - edge:
                session.insert_order(iid, price=top.ask, volume=self.size, side="bid", order_type="ioc")
            if top.bid is not None and top.bid > value + edge:
                session.insert_order(iid, price=top.bid, volume=self.size, side="ask", order_type="ioc")


class MomentumChaser:
    """Buys after up-moves, sells after down-moves: positive feedback that overshoots."""

    def __init__(
        self, name: str, instrument_id: str, rng: random.Random,
        lookback_s: float = 2.0, threshold_ticks: float = 3.0, rate_per_s: float = 1.0, size: int = 30,
    ) -> None:
        self.name = name
        self.iid = instrument_id
        self.rng = rng
        self.lookback_s = lookback_s
        self.threshold_ticks = threshold_ticks
        self.rate_per_s = rate_per_s
        self.size = size
        self._mids: deque[tuple[float, float]] = deque()

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        top = top_of_book(market.book_snapshot(self.iid))
        if top is None or top.bid is None or top.ask is None:
            return
        self._mids.append((now, (top.bid + top.ask) / 2.0))
        while now - self._mids[0][0] > self.lookback_s:
            self._mids.popleft()
        move = self._mids[-1][1] - self._mids[0][1]
        if abs(move) < self.threshold_ticks * market.tick_of(self.iid):
            return
        if _poisson_events(self.rng, self.rate_per_s, dt) == 0:
            return
        side, price = ("bid", top.ask) if move > 0 else ("ask", top.bid)
        SimExchange(market, self.name, bot=True).insert_order(
            self.iid, price=price, volume=self.size, side=side, order_type="ioc"
        )


class PennyTeam:
    """Another contestant: small two-sided quotes one tick inside the external touch."""

    def __init__(self, name: str, instrument_id: str, size: int = 5, refresh_s: float = 0.5) -> None:
        self.name = name
        self.iid = instrument_id
        self.size = size
        self.refresh_s = refresh_s
        self._next = 0.0
        self._session: SimExchange | None = None

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        if now < self._next:
            return
        self._next = now + self.refresh_s
        session = self._session or SimExchange(market, self.name, bot=True)
        self._session = session
        own = list(session.get_outstanding_orders(self.iid).values())
        top = top_of_book(external_book(market.book_snapshot(self.iid), own, market.tick_of(self.iid)))
        session.delete_orders(self.iid)
        if top is None or top.bid is None or top.ask is None:
            return
        tick = market.tick_of(self.iid)
        if top.ask - top.bid > 2.5 * tick:
            session.insert_order(self.iid, price=top.bid + tick, volume=self.size, side="bid", order_type="limit")
            session.insert_order(self.iid, price=top.ask - tick, volume=self.size, side="ask", order_type="limit")


class NewsProcess:
    """Social-feed posts: relevant ones jump fair value persistently; others are noise."""

    GOOD = ("beats expectations", "wins a major contract", "reports record revenue", "raises guidance")
    BAD = ("faces a regulatory probe", "misses earnings", "recalls its products", "loses a key customer")
    NOISE = ("A recent study found CEOs only read business books.", "Markets await the weekend.")

    def __init__(
        self, rng: random.Random, bases: Sequence[str], rate_per_s: float = 0.03,
        jump_fraction: float = 0.02, relevant_fraction: float = 0.5, delay_s: float = 0.0,
    ) -> None:
        self.rng = rng
        self.bases = list(bases)
        self.rate_per_s = rate_per_s
        self.jump_fraction = jump_fraction
        self.relevant_fraction = relevant_fraction
        self.delay_s = delay_s
        self._pending: list[tuple[float, str, float]] = []

    def act(self, market: SimMarket, now: float, dt: float) -> None:
        for _ in range(_poisson_events(self.rng, self.rate_per_s, dt)):
            self._emit(market)
        due = [item for item in self._pending if item[0] <= now]
        self._pending = [item for item in self._pending if item[0] > now]
        for _, base, factor in due:
            market.fair[base] *= factor

    def _emit(self, market: SimMarket) -> None:
        stamp = market.timestamp()
        if self.rng.random() > self.relevant_fraction:
            market.news.append(SocialMediaFeed(stamp, self.rng.choice(self.NOISE), None))
            return
        base = self.rng.choice(self.bases)
        up = self.rng.random() < 0.5
        phrase = self.rng.choice(self.GOOD if up else self.BAD)
        size = self.jump_fraction * self.rng.uniform(0.5, 1.5)
        factor = 1.0 + size if up else 1.0 - size
        market.news.append(SocialMediaFeed(stamp, f"@Wire: {base} {phrase}. #{base}", None))
        self._pending.append((market.now + self.delay_s, base, factor))


def base_of(instrument_id: str) -> str:
    return parse_instrument_id(instrument_id).base
