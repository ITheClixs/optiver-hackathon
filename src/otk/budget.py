"""Write budget: Optibook disconnects a client above 25 updates per second (guide §4).

Inserts, deletes and amends count. Reads appear to be served from a locally pushed cache, which
`scripts/probe_exchange.py` verifies on the day. High-value actions pre-empt market-making updates:
hedges and cancels may use the whole budget, takes half the reserve, and quotes stop before it, so a
quote storm cannot block a hedge, a cancel, or a news or arb trade.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from enum import IntEnum

EXCHANGE_MAX_UPDATES_PER_SECOND = 25


class Priority(IntEnum):
    QUOTE = 0
    TAKE = 1
    RISK = 2


class WriteBudget:
    """Sliding one-second window, so the cap holds for every window rather than per calendar second."""

    def __init__(
        self,
        max_per_second: int = 22,
        reserve: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 < max_per_second <= EXCHANGE_MAX_UPDATES_PER_SECOND:
            raise ValueError(f"max_per_second must be in 1..25, got {max_per_second}")
        if not 0 <= reserve < max_per_second:
            raise ValueError(f"reserve must be in 0..{max_per_second - 1}, got {reserve}")
        self._max = max_per_second
        self._reserve = reserve
        self._clock = clock
        self._sent: deque[float] = deque()

    def _expire(self, now: float) -> None:
        while self._sent and now - self._sent[0] >= 1.0:
            self._sent.popleft()

    def available(self, priority: Priority = Priority.QUOTE) -> int:
        self._expire(self._clock())
        if priority == Priority.RISK:
            cap = self._max
        elif priority == Priority.TAKE:
            cap = self._max - self._reserve // 2
        else:
            cap = self._max - self._reserve
        return max(0, cap - len(self._sent))

    def try_acquire(self, priority: Priority = Priority.QUOTE) -> bool:
        if self.available(priority) <= 0:
            return False
        self._sent.append(self._clock())
        return True

    def now(self) -> float:
        return self._clock()

    def used_last_second(self) -> int:
        self._expire(self._clock())
        return len(self._sent)
