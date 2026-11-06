"""What kind of event is this? The main algo's first decision, made from what the exchange lists.

The scenario is not known until the briefing, and public editions fall into a few families: a dual
listing alone, several stocks with duals and a news feed, baskets or ETFs against their stocks, and the
full board with futures and options. Instruments are named differently each time (`X_DUAL`, `X_A`/`X_B`,
`X_NEW_COUNTRY`, `X_US`/`X_EU`), so pairs are detected rather than assumed. `plan_event` turns that into
a card saying what runs and what the team must still supply, so nobody has to work it out under pressure.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

from otk.fair import top_of_book
from otk.instruments import Kind, Universe, parse_instrument_id
from otk.params import INSTRUMENTS_KEY, REFERENCE_TICK
from otk.strategies import default_stack

_SECOND_LISTING = ("_NEW_COUNTRY", "_NEW", "_FOREIGN", "_INTL", "_ADR")
_REGION_ORDER = ("_US", "_EU", "_UK", "_ASIA", "_JP", "_HK")
FULL_BOARD_INSTRUMENTS = 12
PAIR_SIZE = 2
LIQUID_LEG_SPREAD_TICKS = 4.0


OPTION_START_SIZE = 2  # no option price has been checked on a real book yet


class EventPlan(NamedTuple):
    name: str
    pairs: tuple
    strategies: tuple
    params: Mapping
    missing: tuple
    notes: tuple


def _is_plain_stock(instrument_id: str) -> bool:
    return parse_instrument_id(instrument_id).kind in (Kind.STOCK, Kind.DUAL)


def _named_pairs(ids: Sequence[str]) -> list[tuple[str, str]]:
    present = set(ids)
    pairs: list[tuple[str, str]] = []
    for iid in sorted(ids):
        if iid.endswith("_DUAL") and iid[: -len("_DUAL")] in present:
            pairs.append((iid[: -len("_DUAL")], iid))
        if iid.endswith("_A") and iid[:-2] + "_B" in present:
            pairs.append((iid, iid[:-2] + "_B"))
        pairs.extend((iid, iid + suffix) for suffix in _SECOND_LISTING if iid + suffix in present)
        for position, region in enumerate(_REGION_ORDER):
            if iid.endswith(region):
                stem = iid[: -len(region)]
                pairs.extend(
                    (iid, stem + other)
                    for other in _REGION_ORDER[position + 1 :]
                    if stem + other in present
                )
    return pairs


def detect_pairs(
    instrument_ids: Iterable[str], groups: Mapping[str, str | None] | None = None
) -> list[tuple[str, str]]:
    """(primary, second listing). Each instrument joins one pair. Names first, then shared groups."""
    ids = list(instrument_ids)
    pairs = _named_pairs(ids)
    used = {iid for pair in pairs for iid in pair}
    members: dict[str, list[str]] = defaultdict(list)
    for iid, group in (groups or {}).items():
        if group is not None and iid not in used:
            members[group].append(iid)
    for names in members.values():
        if len(names) == PAIR_SIZE and all(_is_plain_stock(iid) for iid in names):
            pairs.append(tuple(sorted(names)))
    return sorted(pairs)


def choose_liquid_leg(pair: tuple[str, str], books: Mapping[str, object]) -> tuple[str, str]:
    """Hedges go on the primary, so it must be the liquid one: tighter spread, then deeper."""

    def quality(iid: str) -> tuple | None:
        top = top_of_book(books.get(iid))
        if top is None or top.bid is None or top.ask is None:
            return None
        return (top.ask - top.bid, -(top.bid_vol + top.ask_vol))

    first, second = quality(pair[0]), quality(pair[1])
    if first is not None and second is not None and second < first:
        return pair[1], pair[0]
    return pair


def apply_pairs(universe: Universe, pairs: Iterable[tuple[str, str]]) -> Universe:
    """A second listing shares its primary's delta, however the exchange spelled the two names. Derivatives are never one."""

    def is_listing(iid: str) -> bool:
        info = universe.by_id.get(iid)
        return info is not None and info.kind in (Kind.STOCK, Kind.DUAL)

    second = {dual: primary for primary, dual in pairs if is_listing(primary) and is_listing(dual)}
    infos = []
    for iid, info in universe.by_id.items():
        if iid in second:
            infos.append(info._replace(kind=Kind.DUAL, base=second[iid]))
        elif iid in second.values():
            infos.append(info._replace(kind=Kind.STOCK, base=iid))
        else:
            infos.append(info)
    return Universe(infos)


def _format_name(universe: Universe) -> str:
    if universe.ids_of(Kind.FUTURE, Kind.OPTION):
        return "full_board"
    if universe.dual_pairs():
        return "dual_listing"
    return "basket" if universe.ids_of(Kind.ETF) else "stocks"


def _missing(universe: Universe, event: Mapping) -> tuple[str, ...]:
    items = []
    derived = [universe.by_id[iid] for iid in universe.ids_of(Kind.ETF, Kind.FUTURE, Kind.OPTION)]
    needs_index = any(
        info.index is None and (info.kind is Kind.ETF or info.base not in universe.by_id) for info in derived
    )
    if needs_index and not event.get("index_weights"):
        items.append("index_weights: until given, ETF and index products quote off their own book only")
    if any(info.kind is Kind.OPTION and info.vol is None for info in derived) and not event.get("option_vol"):
        items.append("option_vol: stock options stay out until it is set; get the volatility at the briefing")
    if "position_limit" not in event:
        items.append("position_limit: using 100; set it from the challenge text")
    if not event.get("news_training_csv") and not event.get("llm_url"):
        items.append("news model: keyword rules only; add news_training_csv or llm_url if the event has a feed")
    return tuple(items)


def _notes(universe: Universe, has_news_feed: bool) -> tuple[str, ...]:
    notes = [
        "news feed seen: trading headlines on both listings"
        if has_news_feed
        else "no news feed seen yet: the news strategy stays idle at no cost until a post arrives"
    ]
    if universe.dual_pairs():
        notes.append("if a market closes, the dual quotes off its own book and hedges on what is open")
    if universe.ids_of(Kind.OPTION):
        notes.append(f"options start at size {OPTION_START_SIZE}, and stop while our price sits outside their books: see `option check` in the log")
    if any(info.index is not None for info in universe.by_id.values()):
        notes.append("index products priced from the client's constituents, unless index_weights is set")
    notes.append("Autopilot turns reversion on only where paper trades on this book paid after costs")
    return tuple(notes)


def plan_event(universe: Universe, event: Mapping | None = None, has_news_feed: bool = False) -> EventPlan:
    settings = event or {}
    if not universe.dual_pairs():
        universe = apply_pairs(universe, detect_pairs(universe.by_id))
    stack = default_stack(
        sorted(universe.by_id),
        index_weights=settings.get("index_weights"),
        classifier=object(),  # planning only: the news strategy is always part of the main algo
        vol=settings.get("option_vol"),
        universe=universe,
    )
    params: dict = {}
    if len(universe.by_id) >= FULL_BOARD_INSTRUMENTS:
        params["reprice_ticks"] = 2.0  # 25 writes per second go further with less chasing
    # On real books the liquid leg's touch lost money per fill while the thin leg's earned, and quoting
    # the liquid leg one tick wider added PnL in four of five held-out hours. On finer ticks we have no
    # evidence, so Autopilot's fill quality decides there.
    wider = {
        primary: {"base_spread_ticks": LIQUID_LEG_SPREAD_TICKS}
        for primary, _ in universe.dual_pairs()
        if universe.tick_of(primary) >= REFERENCE_TICK
    }
    small = {iid: {"size": OPTION_START_SIZE} for iid in universe.ids_of(Kind.OPTION)}
    if wider or small:
        params[INSTRUMENTS_KEY] = {**wider, **small}
    return EventPlan(
        _format_name(universe),
        tuple(universe.dual_pairs()),
        tuple(getattr(item, "name", type(item).__name__) for item in stack),
        params,
        _missing(universe, settings),
        _notes(universe, has_news_feed),
    )


def render_plan(plan: EventPlan) -> list[str]:
    lines = [f"FORMAT: {plan.name}"]
    if plan.pairs:
        lines.append("pairs (primary, second listing): " + ", ".join(f"{a}/{b}" for a, b in plan.pairs))
    lines.append("running: " + ", ".join(plan.strategies))
    if plan.params:
        lines.append(f"preset: {dict(plan.params)}")
    lines.extend(f"needs: {item}" for item in plan.missing)
    lines.extend(f"note: {item}" for item in plan.notes)
    return lines
