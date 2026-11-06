"""Pre-trade arithmetic for Optibook's pre-emptive limits (guide §4).

The exchange rejects an order and disconnects the client if any conceivable fill sequence could
breach a limit, so capacity assumes every resting order on that side fills, and resting orders on
the other side never help. All values default to the published limits; the event can change them.
"""

from __future__ import annotations

from typing import NamedTuple


class Limits(NamedTuple):
    position: int = 100
    outstanding: int = 200
    margin: int = 3  # lots kept free: position and order updates arrive separately


def buy_capacity(position: int, open_buy: int, open_sell: int, limits: Limits) -> int:
    room = limits.position - limits.margin - position - open_buy
    return max(0, min(room, limits.outstanding - open_buy - open_sell))


def sell_capacity(position: int, open_buy: int, open_sell: int, limits: Limits) -> int:
    room = limits.position - limits.margin + position - open_sell
    return max(0, min(room, limits.outstanding - open_buy - open_sell))


def clip_order(
    side: str, volume: int, position: int, open_buy: int, open_sell: int, limits: Limits
) -> int:
    """Largest volume of `volume` that cannot breach a limit; 0 means do not send."""
    if side == "bid":
        return min(volume, buy_capacity(position, open_buy, open_sell, limits))
    if side == "ask":
        return min(volume, sell_capacity(position, open_buy, open_sell, limits))
    raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")


class DeltaWatch:
    """Fires before the soft delta limit charges us (10% hacked out at 2% worse than mid)."""

    def __init__(self, soft: float = 100.0, grace_s: float = 10.0, act_after_s: float = 4.0) -> None:
        if act_after_s >= grace_s:
            raise ValueError("act_after_s must be shorter than the exchange grace period")
        self.soft = soft
        self.grace_s = grace_s
        self.act_after_s = act_after_s
        self._since: dict[str, float | None] = {}

    def update(self, group: str, delta: float, now: float) -> bool:
        if abs(delta) < self.soft:
            self._since[group] = None
            return False
        since = self._since.get(group)
        if since is None:
            self._since[group] = now
            return False
        return now - since >= self.act_after_s


class DrawdownStop:
    """Latching kill switch on marked PnL; only an operator clears it."""

    def __init__(self, max_drawdown: float) -> None:
        self.max_drawdown = max_drawdown
        self._peak: float | None = None
        self.tripped = False

    def update(self, pnl: float) -> bool:
        self._peak = pnl if self._peak is None else max(self._peak, pnl)
        if self._peak - pnl > self.max_drawdown:
            self.tripped = True
        return self.tripped

    def reset(self) -> None:
        self._peak = None
        self.tripped = False
