"""What a strategy may ask for. Strategies never touch the exchange; the engine executes intents."""

from __future__ import annotations

from typing import NamedTuple


class QuoteIntent(NamedTuple):
    """Desired resting two-sided quote. A None price means no order on that side."""

    instrument_id: str
    bid: float | None
    bid_size: int
    ask: float | None
    ask_size: int
    source: str = ""


class LadderIntent(NamedTuple):
    """Several resting levels per side, best first, as ((price, size), ...)."""

    instrument_id: str
    bids: tuple
    asks: tuple
    source: str = ""


class Take(NamedTuple):
    """IOC order. `hedge` marks risk-reducing flow, which may use the reserved write budget."""

    instrument_id: str
    side: str
    price: float
    volume: int
    hedge: bool = False
    source: str = ""
    reduce_only: bool = False  # an exit: never more than the position, whatever has not reported yet


class Pull(NamedTuple):
    """Remove every resting order on the instrument and suppress quotes this tick."""

    instrument_id: str
    source: str = ""


Intent = QuoteIntent | LadderIntent | Take | Pull
