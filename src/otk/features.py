"""O(1) online estimators for the trading loop. Runtime code: stdlib only, bounded memory."""

from __future__ import annotations

import math


class Ewma:
    """Exponentially weighted mean and variance with a half-life in observations."""

    def __init__(self, half_life: float) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.alpha = 1.0 - 0.5 ** (1.0 / half_life)
        self.mean: float | None = None
        self.var = 0.0

    def update(self, value: float) -> float:
        if self.mean is None:
            self.mean = value
            return value
        delta = value - self.mean
        self.mean += self.alpha * delta
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta)
        return self.mean

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def z(self, value: float) -> float:
        if self.mean is None or self.var <= 0.0:
            return 0.0
        return (value - self.mean) / self.std


class TimedZ:
    """Exponentially weighted mean and variance with a half-life in seconds, for irregular sampling."""

    __slots__ = ("first", "last", "mean", "var")

    def __init__(self) -> None:
        self.mean: float | None = None
        self.var = 0.0
        self.first = 0.0
        self.last = 0.0

    def update_z(self, now: float, value: float, half_life: float, warmup_s: float) -> float | None:
        """Absorb `value` and return its z-score, or None until warmed up."""
        if self.mean is None:
            self.mean, self.first, self.last = value, now, now
            return None
        weight = 1.0 - 0.5 ** ((now - self.last) / half_life)
        self.last = now
        delta = value - self.mean
        self.mean += weight * delta
        self.var = (1.0 - weight) * (self.var + weight * delta * delta)
        if now - self.first < warmup_s or self.var <= 0.0:
            return None
        # The variance starts at zero, so early on it reads low and every z-score looks extreme.
        settled = 1.0 - 0.5 ** ((now - self.first) / half_life)
        if settled <= 0.0:
            return None
        return (value - self.mean) / math.sqrt(self.var / settled)


class ChangeVol:
    """Stdev of per-update changes in a series, updated only when the series has moved or a tick passed."""

    def __init__(self, half_life: float = 50.0) -> None:
        self._ewma = Ewma(half_life)
        self._last: float | None = None

    def update(self, value: float) -> float:
        if self._last is not None:
            change = value - self._last
            alpha = self._ewma.alpha
            self._ewma.var = (1.0 - alpha) * self._ewma.var + alpha * change * change
        self._last = value
        return self.std

    @property
    def std(self) -> float:
        return math.sqrt(self._ewma.var)
