"""Instrument discovery from Optibook's id conventions (guide §3) and exchange metadata."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import NamedTuple

from otk.fair import bs_delta

_OPTION = re.compile(r"^(?P<base>.+)_(?P<ym>\d{6})_(?P<strike>\d+(?:\.\d+)?)(?P<kind>[CP])$")
_FUTURE = re.compile(r"^(?P<base>.+)_(?P<ym>\d{6})_F$")
SECONDS_PER_YEAR = 365.0 * 24 * 3600
DEFAULT_TICK = 0.1
_TICK_CANDIDATES = (0.1, 0.05, 0.01, 0.005, 0.001)  # coarsest first


class Kind(Enum):
    STOCK = "stock"
    DUAL = "dual"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"


_TYPE_WORDS = (("OPTION", Kind.OPTION), ("FUTURE", Kind.FUTURE), ("ETF", Kind.ETF))
_DERIVED = frozenset({Kind.OPTION, Kind.FUTURE, Kind.ETF})


class IndexSpec(NamedTuple):
    """An index as the client describes it: level = sum(weight * price) / divisor."""

    weights: Mapping[str, float]
    divisor: float


class InstrumentInfo(NamedTuple):
    instrument_id: str
    kind: Kind
    base: str
    expiry: datetime | None = None
    strike: float | None = None
    option_kind: str | None = None
    tick: float = 0.1
    rate: float | None = None
    index: IndexSpec | None = None
    vol: float | None = None
    etf_multiplier: float | None = None
    etf_cash: float | None = None


def third_friday(year: int, month: int) -> datetime:
    """Optibook convention: expiry on the third Friday of the month at 12:00 UTC."""
    first = datetime(year, month, 1, 12, tzinfo=timezone.utc)
    to_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=to_friday + 14)


def _expiry_from(match: re.Match[str]) -> datetime:
    stamp = match.group("ym")
    return third_friday(int(stamp[:4]), int(stamp[4:]))


def parse_instrument_id(instrument_id: str) -> InstrumentInfo:
    option = _OPTION.match(instrument_id)
    if option:
        return InstrumentInfo(
            instrument_id,
            Kind.OPTION,
            option.group("base"),
            _expiry_from(option),
            float(option.group("strike")),
            "call" if option.group("kind") == "C" else "put",
        )
    future = _FUTURE.match(instrument_id)
    if future:
        return InstrumentInfo(instrument_id, Kind.FUTURE, future.group("base"), _expiry_from(future))
    if instrument_id.endswith("_DUAL"):
        return InstrumentInfo(instrument_id, Kind.DUAL, instrument_id[: -len("_DUAL")])
    if instrument_id.endswith("_ETF"):
        return InstrumentInfo(instrument_id, Kind.ETF, instrument_id[: -len("_ETF")])
    if instrument_id.endswith("_BASKET"):
        return InstrumentInfo(instrument_id, Kind.ETF, instrument_id[: -len("_BASKET")])
    return InstrumentInfo(instrument_id, Kind.STOCK, instrument_id)


def _number(value) -> float | None:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return float(value) if is_number else None


def _kind_from_client(instrument, parsed: InstrumentInfo) -> Kind:
    if parsed.kind is not Kind.STOCK:
        return parsed.kind
    text = str(getattr(instrument, "instrument_type", "")).upper()
    return next((kind for word, kind in _TYPE_WORDS if word in text), Kind.STOCK)


def _option_kind_from_client(instrument, parsed: InstrumentInfo) -> str | None:
    text = str(getattr(instrument, "option_kind", "") or "").lower()
    return "call" if "call" in text else "put" if "put" in text else parsed.option_kind


def _expiry_from_client(instrument, parsed: InstrumentInfo) -> datetime | None:
    expiry = getattr(instrument, "expiry", None)
    if not isinstance(expiry, datetime):
        return parsed.expiry
    return expiry.replace(tzinfo=timezone.utc) if expiry.tzinfo is None else expiry


def _index_from_client(instrument) -> IndexSpec | None:
    weights = getattr(instrument, "index_constituents", None)
    divisor = _number(getattr(instrument, "index_divisor", None))
    if not isinstance(weights, Mapping) or not weights or not divisor or divisor <= 0:
        return None
    numbers = {name: _number(weight) for name, weight in weights.items()}
    return None if None in numbers.values() else IndexSpec(numbers, divisor)


def from_exchange_instrument(instrument) -> InstrumentInfo:
    """What the id says, then what the client says, which wins. A thin client gives only the id."""
    parsed = parse_instrument_id(instrument.instrument_id)
    kind = _kind_from_client(instrument, parsed)
    info = parsed._replace(
        kind=kind,
        tick=getattr(instrument, "tick_size", None) or parsed.tick,
        expiry=_expiry_from_client(instrument, parsed),
        strike=getattr(instrument, "strike", None) or parsed.strike,
        option_kind=_option_kind_from_client(instrument, parsed),
        rate=getattr(instrument, "interest_rate", None),
    )
    if kind not in _DERIVED:
        return info
    underlying = getattr(instrument, "base_instrument_id", None) or getattr(instrument, "index_id", None)
    return info._replace(
        base=str(underlying) if underlying else parsed.base,
        index=_index_from_client(instrument),
        vol=_number(getattr(instrument, "index_volatility", None)),
        etf_multiplier=_number(getattr(instrument, "etf_multiplier", None)),
        etf_cash=_number(getattr(instrument, "etf_cash_comp", None)),
    )


def theo_ready(info: InstrumentInfo, listed: Iterable[str]) -> bool:
    """Whether a theoretical value can be worked out. If not, a future or ETF quotes off its own book, and an option stays out."""
    if info.kind is Kind.OPTION and None in (info.strike, info.expiry, info.option_kind):
        return False
    if info.kind is Kind.FUTURE and info.expiry is None:
        return False
    index_known = info.index is not None and set(info.index.weights) <= set(listed)
    if info.kind is Kind.ETF and info.etf_multiplier is None:
        index_known = False
    return index_known or (info.base != info.instrument_id and info.base in listed)


class Universe:
    """Every tradable instrument, grouped by the underlying that shares its delta."""

    def __init__(self, infos: Iterable[InstrumentInfo]) -> None:
        self.by_id: dict[str, InstrumentInfo] = {info.instrument_id: info for info in infos}
        self.groups: dict[str, list[str]] = {}
        for info in self.by_id.values():
            self.groups.setdefault(info.base, []).append(info.instrument_id)

    @classmethod
    def from_instruments(cls, instruments: Mapping[str, object]) -> Universe:
        return cls(
            from_exchange_instrument(item)
            for item in instruments.values()
            if not getattr(item, "expired", False)
        )

    def tick_of(self, instrument_id: str) -> float:
        info = self.by_id.get(instrument_id)
        return info.tick if info else DEFAULT_TICK

    def ids_of(self, *kinds: Kind) -> list[str]:
        return sorted(i for i, info in self.by_id.items() if info.kind in kinds)

    def dual_pairs(self) -> list[tuple]:
        return sorted(
            (info.base, iid)
            for iid, info in self.by_id.items()
            if info.kind is Kind.DUAL and info.base in self.by_id
        )


def _on_grid(price: float, tick: float) -> bool:
    return abs(price / tick - round(price / tick)) < 1e-6


def _event_prices(event: Mapping) -> Iterable[float]:
    if event.get("k") == "book":
        return [level[0] for side in ("bids", "asks") for level in event.get(side, ())]
    if event.get("k") in ("trade", "named_trade"):
        return [event["px"]]
    return ()


def infer_ticks(events: Sequence[Mapping]) -> dict[str, float]:
    """Tick per instrument: recorded in the capture if available, else the coarsest grid that fits.

    Tick sizes differ by instrument on live dashboards, for example 0.01 on stocks and 0.1 on ETFs.
    """
    position: dict[str, int] = {}
    recorded: dict[str, float] = {}
    for event in events:
        if event.get("k") == "meta":
            recorded.update(event.get("ticks", {}))
        iid = event.get("iid")
        for price in _event_prices(event):
            index = position.setdefault(iid, 0)
            while index < len(_TICK_CANDIDATES) - 1 and not _on_grid(price, _TICK_CANDIDATES[index]):
                index += 1
            position[iid] = index
    return {**{iid: _TICK_CANDIDATES[index] for iid, index in position.items()}, **recorded}


def years_to_expiry(expiry: datetime, now: datetime) -> float:
    return max((expiry - now).total_seconds(), 0.0) / SECONDS_PER_YEAR


def delta_per_lot(
    info: InstrumentInfo, spot: float, vol: float, rate: float, now: datetime
) -> float:
    """Delta of one lot against its underlying.

    ETF is 1.0 per lot here: conservative for the exchange's soft delta limit, whose treatment of
    the 0.25 multiplier is unconfirmed. `scripts/probe_exchange.py` lists this as an open question.
    """
    if info.kind is not Kind.OPTION:
        return 1.0
    if spot <= 0 or math.isclose(spot, 0.0) or None in (info.strike, info.expiry, info.option_kind):
        return 0.0
    tau = years_to_expiry(info.expiry, now)
    return bs_delta(
        info.option_kind, spot, info.strike, tau, rate if info.rate is None else info.rate, vol if info.vol is None else info.vol
    )
