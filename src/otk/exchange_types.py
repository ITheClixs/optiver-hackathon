"""What an exchange hands back, shared by the simulator and the replay. The real client has its own, of the same shape."""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple


class PriceVolume(NamedTuple):
    price: float
    volume: int


class PriceBook(NamedTuple):
    timestamp: datetime
    instrument_id: str
    bids: list[PriceVolume]
    asks: list[PriceVolume]


class TradeTick(NamedTuple):
    timestamp: datetime
    instrument_id: str
    price: float
    volume: int
    aggressor_side: str
    buyer: str
    seller: str
    trade_id: int


class Trade(NamedTuple):
    timestamp: datetime
    order_id: int
    trade_id: int
    instrument_id: str
    price: float
    volume: int
    side: str


class OrderStatus(NamedTuple):
    order_id: int
    instrument_id: str
    price: float
    volume: int
    side: str


class InsertOrderResponse(NamedTuple):
    success: bool
    order_id: int
    error_reason: str


class ActionResponse(NamedTuple):
    success: bool
    error_reason: str


class SocialMediaFeed(NamedTuple):
    timestamp: datetime
    post: str
    meta_data: dict | None


class SimInstrument(NamedTuple):
    instrument_id: str
    tick_size: float
    interest_rate: float | None = None
    paused: bool = False
