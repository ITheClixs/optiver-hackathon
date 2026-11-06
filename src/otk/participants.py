"""Who is who on the tape, from the dashboard's named trades and leaderboard.

Answers the questions the PnL plan depends on: which names overpay and are worth quoting into, which
are informed and should be leaned with, who pays us and who hurts us, and what the leading teams do.
Markouts are signed so that negative means the aggressor overpaid. Runs offline on a capture and live
in the feed process, never in the trading loop. Stdlib only.

Estimated PnL counts only trades this process saw, so a late start makes it wrong. The leaderboard's
PnL is authoritative. Bot or team comes from leaderboard membership, and a board limited to the top 50
would label lower teams as bots.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from otk.instruments import DEFAULT_TICK, infer_ticks
from otk.opponents import Welford

MIN_TARGET_TRADES = 30
MIN_TARGET_TICKS = 0.5
TARGET_T = 2.0
ROLE_SHARE = 0.9
MIN_ROLE_TRADES = 5
MIN_COUNTERPARTY_TRADES = 5
MAX_PENDING = 50_000
TOP_N = 10
PNL_WINDOW_S = 300.0


def normalise_side(raw: object) -> str:
    """Any spelling of an aggressor side as 'bid' (a buy), 'ask' (a sell), or '' if unknown."""
    text = str(raw).lower() if raw is not None else ""
    if "bid" in text or "buy" in text:
        return "bid"
    if "ask" in text or "sell" in text:
        return "ask"
    return ""


def infer_aggressor(price: float, bid: float | None, ask: float | None) -> str:
    """A print at the ask was a buy, at the bid a sell. Inside the spread stays unknown."""
    if ask is not None and price >= ask:
        return "bid"
    if bid is not None and price <= bid:
        return "ask"
    return ""


def _t_stat(scores: Welford) -> float:
    var = scores.s2 / (scores.n - 1) if scores.n > 1 else 0.0
    if var <= 1e-12:
        return math.copysign(math.inf, scores.mean) if scores.mean else 0.0
    return scores.mean / math.sqrt(var / scores.n)


class _Pending(NamedTuple):
    due: float
    price: float
    aggressor: str
    aggressor_sign: int
    us_sign: int
    counterparty: str


class _Name:
    __slots__ = (
        "aggressive", "cash", "counterparties", "instruments", "passive", "positions", "sizes",
        "trades", "volume",
    )

    def __init__(self) -> None:
        self.volume = 0
        self.trades = 0
        self.aggressive = 0
        self.passive = 0
        self.sizes: Counter = Counter()
        self.instruments: Counter = Counter()
        self.counterparties: Counter = Counter()
        self.positions: dict[str, int] = defaultdict(int)
        self.cash = 0.0


class ParticipantBook:
    def __init__(
        self,
        horizons_s: Sequence[float] = (5.0, 30.0),
        me: str | None = None,
        tick_of: Callable[[str], float] = lambda _instrument_id: DEFAULT_TICK,
    ) -> None:
        self.horizons_s = tuple(horizons_s)
        self.me = me
        self.tick_of = tick_of
        self._names: dict[str, _Name] = defaultdict(_Name)
        self._mids: dict[str, float] = {}
        self._quoted: set[str] = set()
        self._tops: dict[str, tuple] = {}
        self._pending: dict[str, dict[float, deque]] = defaultdict(
            lambda: {horizon: deque() for horizon in self.horizons_s}
        )
        self._marks: dict[tuple, dict[float, Welford]] = defaultdict(lambda: defaultdict(Welford))
        self._aggressive_sizes: dict[tuple, Counter] = defaultdict(Counter)
        self._aggressive_volume: Counter = Counter()
        self._ours: dict[str, dict[float, Welford]] = defaultdict(lambda: defaultdict(Welford))
        self._ours_volume: Counter = Counter()
        self._flow: dict[str, dict[str, Counter]] = defaultdict(
            lambda: {"taker": Counter(), "maker": Counter()}
        )
        self._teams: set[str] = set()
        self._board: dict[str, dict] = {}
        self._board_points: dict[str, deque] = defaultdict(deque)
        self._first: float | None = None
        self._last = 0.0

    # --- input --------------------------------------------------------------------------------

    def note(self, now: float) -> None:
        self._first = now if self._first is None else self._first
        self._last = max(self._last, now)

    def on_top(self, instrument_id: str, now: float, bid: float | None, ask: float | None) -> None:
        self.note(now)
        self._tops[instrument_id] = (bid, ask)
        if bid is not None and ask is not None:
            self.on_mid(instrument_id, now, (bid + ask) / 2.0)

    def on_mid(self, instrument_id: str, now: float, value: float) -> None:
        self.note(now)
        self._quoted.add(instrument_id)
        self._settle(instrument_id, now, value)

    def _settle(self, instrument_id: str, now: float, value: float) -> None:
        self._mids[instrument_id] = value
        tick = self.tick_of(instrument_id)
        for horizon, queue in self._pending[instrument_id].items():
            while queue and queue[0].due <= now:
                item = queue.popleft()
                move = (value - item.price) / tick
                if item.aggressor:
                    key = (item.aggressor, instrument_id)
                    self._marks[key][horizon].add(item.aggressor_sign * move)
                if item.us_sign:
                    self._ours[item.counterparty][horizon].add(item.us_sign * move)

    def on_trade(
        self, now: float, instrument_id: str, price: float, volume: int, buyer: str, seller: str,
        aggressor: str = "",
    ) -> None:
        self.note(now)
        if instrument_id not in self._quoted:
            # No book seen: the last price stands in for the mid, with bid-ask bounce in the markouts.
            self._settle(instrument_id, now, price)
        side = normalise_side(aggressor) or infer_aggressor(
            price, *self._tops.get(instrument_id, (None, None))
        )
        self._book(buyer, seller, instrument_id, price, volume)
        self._book_side(side, buyer, seller, instrument_id, volume)
        self._queue(now, instrument_id, price, buyer, seller, side)

    def _book(self, buyer: str, seller: str, iid: str, price: float, volume: int) -> None:
        for name, sign, other in ((buyer, 1, seller), (seller, -1, buyer)):
            row = self._names[name]
            row.volume += volume
            row.trades += 1
            row.sizes[volume] += 1
            row.instruments[iid] += volume
            row.counterparties[other] += volume
            row.positions[iid] += sign * volume
            row.cash -= sign * price * volume

    def _book_side(self, side: str, buyer: str, seller: str, iid: str, volume: int) -> None:
        if not side:
            return
        taker, maker = (buyer, seller) if side == "bid" else (seller, buyer)
        self._names[taker].aggressive += volume
        self._names[maker].passive += volume
        self._flow[iid]["taker"][taker] += volume
        self._flow[iid]["maker"][maker] += volume
        self._aggressive_sizes[(taker, iid)][volume] += 1
        self._aggressive_volume[(taker, iid)] += volume

    def _queue(
        self, now: float, iid: str, price: float, buyer: str, seller: str, side: str
    ) -> None:
        taker, taker_sign = "", 0
        if side:
            taker, taker_sign = (buyer, 1) if side == "bid" else (seller, -1)
        us_sign, counterparty = 0, ""
        if self.me == buyer:
            us_sign, counterparty = 1, seller
        elif self.me == seller:
            us_sign, counterparty = -1, buyer
        if not taker and not us_sign:
            return
        for horizon, queue in self._pending[iid].items():
            if len(queue) < MAX_PENDING:
                queue.append(_Pending(now + horizon, price, taker, taker_sign, us_sign, counterparty))

    def on_leaderboard(self, now: float, rows: Iterable[Mapping]) -> None:
        self.note(now)
        for row in rows:
            team = row.get("team")
            if not team:
                continue
            self._teams.add(team)
            self._board[team] = dict(row)
            pnl = row.get("pnl")
            if pnl is None:
                continue
            points = self._board_points[team]
            points.append((now, pnl))
            while len(points) > 2 and now - points[0][0] > PNL_WINDOW_S:
                points.popleft()

    # --- report -------------------------------------------------------------------------------

    def _kind(self, name: str) -> str:
        if name in self._teams:
            return "team"
        return "bot" if self._teams else "unknown"

    @staticmethod
    def _role(row: _Name) -> str:
        sided = row.aggressive + row.passive
        if row.trades < MIN_ROLE_TRADES or not sided:
            return "unknown"
        share = row.aggressive / sided
        if share >= ROLE_SHARE:
            return "pure_taker"
        return "pure_maker" if share <= 1.0 - ROLE_SHARE else "two_way"

    def _est_pnl(self, row: _Name) -> float:
        marks = sum(position * self._mids.get(iid, 0.0) for iid, position in row.positions.items())
        return round(row.cash + marks, 2)

    def _participant(self, name: str, row: _Name) -> dict:
        sided = row.aggressive + row.passive
        return {
            "kind": self._kind(name),
            "role": self._role(row),
            "volume": row.volume,
            "trades": row.trades,
            "aggressive_volume": row.aggressive,
            "passive_volume": row.passive,
            "aggressor_share": round(row.aggressive / sided, 3) if sided else None,
            "modal_size": row.sizes.most_common(1)[0][0] if row.sizes else None,
            "instruments": [iid for iid, _ in row.instruments.most_common(5)],
            "counterparties": [other for other, _ in row.counterparties.most_common(5)],
            "positions": {iid: pos for iid, pos in row.positions.items() if pos},
            "est_pnl": self._est_pnl(row),
        }

    def _target_row(self, name: str, iid: str, horizons: Mapping) -> dict:
        sizes = self._aggressive_sizes[(name, iid)]
        total = sum(sizes.values()) or 1
        modal, count = sizes.most_common(1)[0] if sizes else (None, 0)
        return {
            "name": name,
            "instrument": iid,
            "kind": self._kind(name),
            "modal_size": modal,
            "modal_share": round(count / total, 3),
            "trades": total,
            "volume": self._aggressive_volume[(name, iid)],
            "markout_ticks": {str(h): scores.stats() for h, scores in horizons.items()},
        }

    def _targets(self) -> dict:
        overpayers, informed = [], []
        primary = self.horizons_s[0]
        for (name, iid), horizons in self._marks.items():
            scores = horizons.get(primary)
            if scores is None or scores.n < MIN_TARGET_TRADES:
                continue
            if abs(scores.mean) < MIN_TARGET_TICKS or abs(_t_stat(scores)) < TARGET_T:
                continue
            row = self._target_row(name, iid, horizons)
            (informed if scores.mean > 0 else overpayers).append((scores.mean * row["volume"], row))
        overpayers.sort(key=lambda pair: pair[0])
        informed.sort(key=lambda pair: -pair[0])
        return {
            "overpayers": [row for _, row in overpayers[:TOP_N]],
            "informed": [row for _, row in informed[:TOP_N]],
        }

    def _me(self) -> dict:
        if self.me is None:
            return {}
        primary = self.horizons_s[0]
        rows = []
        for name, horizons in self._ours.items():
            scores = horizons.get(primary)
            if scores is None or scores.n < MIN_COUNTERPARTY_TRADES:
                continue
            rows.append(
                {
                    "name": name,
                    "kind": self._kind(name),
                    "our_markout_ticks": round(scores.mean, 3),
                    "n": scores.n,
                }
            )
        return {
            "name": self.me,
            "pays_us": sorted(
                (r for r in rows if r["our_markout_ticks"] > 0), key=lambda r: -r["our_markout_ticks"]
            )[:TOP_N],
            "hurts_us": sorted(
                (r for r in rows if r["our_markout_ticks"] < 0), key=lambda r: r["our_markout_ticks"]
            )[:TOP_N],
        }

    def _pnl_rate(self, team: str) -> float | None:
        points = self._board_points.get(team)
        if not points or len(points) < 2 or points[-1][0] <= points[0][0]:
            return None
        return round((points[-1][1] - points[0][1]) * 60.0 / (points[-1][0] - points[0][0]), 2)

    def _teams_report(self) -> list[dict]:
        rows = []
        for team, board in self._board.items():
            row = self._names.get(team)
            sided = (row.aggressive + row.passive) if row else 0
            rows.append(
                {
                    **board,
                    "pnl_per_min": self._pnl_rate(team),
                    "passive_share": round(row.passive / sided, 3) if sided else None,
                    "modal_size": row.sizes.most_common(1)[0][0] if row and row.sizes else None,
                    "top_instruments": [iid for iid, _ in row.instruments.most_common(3)] if row else [],
                    "role": self._role(row) if row else "unknown",
                }
            )
        return sorted(rows, key=lambda r: -(r.get("pnl") or 0.0))

    def _instruments_report(self) -> dict:
        report = {}
        for iid, flow in self._flow.items():
            by_kind: Counter = Counter()
            for name, volume in flow["taker"].items():
                by_kind[f"{self._kind(name)}_taker"] += volume
            for name, volume in flow["maker"].items():
                by_kind[f"{self._kind(name)}_maker"] += volume
            team_total = by_kind["team_taker"] + by_kind["team_maker"]
            report[iid] = {
                "volume": sum(flow["taker"].values()),
                "by_kind": dict(by_kind),
                "team_passive_share": round(by_kind["team_maker"] / team_total, 3) if team_total else None,
                "top_takers": [name for name, _ in flow["taker"].most_common(5)],
                "top_makers": [name for name, _ in flow["maker"].most_common(5)],
            }
        return report

    def report(self) -> dict:
        return {
            "duration_s": round(self._last - (self._first or 0.0), 3),
            "participants": {name: self._participant(name, row) for name, row in self._names.items()},
            "targets": self._targets(),
            "me": self._me(),
            "teams": self._teams_report(),
            "instruments": self._instruments_report(),
        }


def _before_connect(book: ParticipantBook, event: Mapping) -> bool:
    """The feed resends about 500 old trades when it connects, so their markouts and roles say nothing about now.

    The server's trade time is compared with the first event this book saw, on our clock: a laptop clock off by N
    seconds moves the boundary by N seconds. A trade with no numeric server time is kept.
    """
    server_time = event.get("time")
    first = book._first
    is_number = isinstance(server_time, (int, float)) and not isinstance(server_time, bool)
    return is_number and first is not None and server_time < first


def apply_event(book: ParticipantBook, now: float, event: Mapping) -> None:
    """Feed one normalised event: from a capture offline, or from the feed process live."""
    book.note(now)
    kind = event.get("k")
    if kind == "book":
        bids, asks = event.get("bids"), event.get("asks")
        book.on_top(event["iid"], now, bids[0][0] if bids else None, asks[0][0] if asks else None)
    elif kind == "leaderboard":
        book.on_leaderboard(now, event.get("rows", ()))
    elif kind == "named_trade" or (kind == "trade" and event.get("buyer") and event.get("seller")):
        if _before_connect(book, event):
            return
        book.on_trade(
            now, event["iid"], event["px"], event["vol"], event["buyer"], event["seller"],
            event.get("aggr", ""),
        )


def report_from_events(
    events: Iterable[Mapping],
    me: str | None = None,
    tick: float | None = None,
    horizons_s: Sequence[float] = (5.0, 30.0),
) -> dict:
    rows = list(events)
    if tick is None:
        ticks = infer_ticks(rows)
        tick_of = lambda instrument_id: ticks.get(instrument_id, DEFAULT_TICK)
    else:
        tick_of = lambda _instrument_id: tick
    book = ParticipantBook(horizons_s, me, tick_of)
    for event in rows:
        apply_event(book, float(event["t"]), event)
    return book.report()


def read_named_trades_csv(path: str | Path) -> Iterator[dict]:
    """Rows of captured_at,time,ticker,price,volume,buyer,seller,aggressor, as named_trade events.

    The shape a DOM scraper of the dashboard writes, which is the fallback if the feed cannot connect.
    """
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                captured = datetime.fromisoformat(row["captured_at"])
                clock = datetime.strptime(row["time"], "%H:%M:%S").time()
                yield {
                    "t": datetime.combine(captured.date(), clock, captured.tzinfo).timestamp(),
                    "k": "named_trade",
                    "iid": row["ticker"],
                    "px": float(row["price"]),
                    "vol": int(float(row["volume"])),
                    "buyer": row["buyer"],
                    "seller": row["seller"],
                    "aggr": normalise_side(row.get("aggressor")),
                }
            except (KeyError, TypeError, ValueError):
                continue
