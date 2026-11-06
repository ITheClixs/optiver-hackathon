#!/usr/bin/env python3
"""Last resort: a bot that needs only this file and the organisers' client. No tokens, no network, no config.

Use it only when `optibook_bot.py` cannot run at all. It shares no code with
the main bot on purpose, so whatever broke that cannot break this. Run one at a time: a second login kicks the first.

    nohup python emergency_bot.py > emergency_stdout.log 2>&1 &
    touch run/KILL      # stop and pull quotes; rm to resume
    touch run/FLATTEN   # close every position; rm to resume

What it does, from replay evidence on real Optibook books:
  * A listed pair: takes a stale quote on the thin leg, priced off the liquid leg's mid, with IOC orders only. Nothing
    rests, so there is no self-trade, no resting capacity to count, and little of the write limit used. Net delta over
    the pair stays under the soft limit.
  * No pair: two passive quotes on each instrument, away from the touch, leaning against inventory.

Edit the block below after the briefing, if it differs. The defaults are the published limits.
"""

# ruff: noqa: UP006, UP035, UP045  - old-style typing on purpose: this file must start on whatever Python the devenv has
import argparse
import collections
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

# ======================================================================================================
LIMIT = 100  # position limit per instrument
MARGIN = 10  # stop this far short of it
CLIP = 20  # most lots in one IOC
EDGE_TICKS = 1.0  # how far past the liquid mid the thin leg must be
QUOTE_SIZE = 5  # lots per passive quote, when nothing pairs
QUOTE_TICKS = 2.0  # passive quote distance from mid
LEAN_TICKS = 2.0  # how far a full position moves the quotes
MAX_WRITES_PER_S = 10  # of the 25 allowed
PAIRS: List[Tuple[str, str]] = []  # for example [("ASML", "ASML_DUAL")] if the names do not show it, liquid leg first
# ======================================================================================================

LOOP_S = 0.05
TAKE_GAP_S = 0.1
REQUOTE_GAP_S = 1.0
FLATTEN_GAP_S = 0.5
STATUS_S = 10.0
MAX_BACKOFF_S = 30.0
HEARTBEAT_FRESH_S = 20.0
ROUNDING = 1e-9
SUFFIXES = (("_DUAL", ""), ("_B", "_A"), ("_2", ""), ("_US", "_EU"))


class Settings(NamedTuple):
    run_dir: Path = Path("run")
    limit: int = LIMIT
    margin: int = MARGIN
    clip: int = CLIP
    edge_ticks: float = EDGE_TICKS
    quote_size: int = QUOTE_SIZE
    quote_ticks: float = QUOTE_TICKS
    lean_ticks: float = LEAN_TICKS
    max_writes_per_s: int = MAX_WRITES_PER_S
    pairs: Optional[List[Tuple[str, str]]] = None


class Top(NamedTuple):
    bid: float
    bid_vol: int
    ask: float
    ask_vol: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def read_top(book) -> Optional[Top]:
    """Both sides present and not crossed, or None. Sides can be None, empty or junk on this exchange."""
    bids, asks = getattr(book, "bids", None), getattr(book, "asks", None)
    if not bids or not asks:
        return None
    top = Top(bids[0].price, int(bids[0].volume), asks[0].price, int(asks[0].volume))
    return top if top.bid < top.ask else None


def find_pairs(ids) -> List[Tuple[str, str]]:
    present = set(ids)
    found = []
    for iid in sorted(present):
        for second, first in SUFFIXES:
            if iid.endswith(second):
                stem = iid[: -len(second)] + first
                if stem in present and stem != iid:
                    found.append((stem, iid))
    return found


def liquid_first(pair: Tuple[str, str], books: Dict) -> Tuple[str, str]:
    """Tighter spread now, then deeper. If a book is missing, keep the order given."""

    def quality(iid: str):
        top = read_top(books.get(iid))
        return None if top is None else (top.ask - top.bid, -(top.bid_vol + top.ask_vol))

    first, second = quality(pair[0]), quality(pair[1])
    return (pair[1], pair[0]) if first is not None and second is not None and second < first else pair


class Budget:
    """A sliding one-second window, so the cap holds in every second rather than per calendar second."""

    def __init__(self, per_second: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_second, self.clock = per_second, clock
        self.sent: collections.deque = collections.deque()

    def _forget_old(self) -> None:
        now = self.clock()
        while self.sent and now - self.sent[0] >= 1.0:
            self.sent.popleft()

    def left(self) -> int:
        self._forget_old()
        return self.per_second - len(self.sent)

    def take(self) -> bool:
        if self.left() <= 0:
            return False
        self.sent.append(self.clock())
        return True


def _tick(info) -> float:
    return getattr(info, "tick_size", None) or 0.1


class Trader:
    def __init__(self, client, settings: Optional[Settings] = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.client, self.settings, self.clock = client, settings or Settings(), clock
        self.budget = Budget(self.settings.max_writes_per_s, clock)
        self.last_take: Dict[str, float] = {}
        self.last_quote: Dict[str, Tuple[float, float, int]] = {}
        self.last_flatten = float("-inf")
        self.spreads: Dict[Tuple[str, str], List[float]] = {}
        self.pulled = False
        self.takes = self.quotes = self.rejects = 0

    def _insert(self, iid: str, price: float, volume: int, side: str, order_type: str) -> None:
        response = self.client.insert_order(iid, price=price, volume=volume, side=side, order_type=order_type)
        if not getattr(response, "success", True):
            self.rejects += 1
        elif order_type == "ioc":
            self.takes += 1
        else:
            self.quotes += 1

    def _pull(self, ids) -> None:
        for iid in ids:
            if self.budget.take():
                self.client.delete_orders(iid)
                self.last_quote.pop(iid, None)

    def _flatten(self, ids, books: Dict, positions: Dict[str, int]) -> None:
        if self.clock() - self.last_flatten < FLATTEN_GAP_S:
            return
        self.last_flatten = self.clock()
        for iid in ids:
            top, position = read_top(books.get(iid)), positions.get(iid, 0)
            if top is None or not position or not self.budget.take():
                continue
            if position > 0:
                self._insert(iid, top.bid, min(position, top.bid_vol, self.settings.clip), "ask", "ioc")
            else:
                self._insert(iid, top.ask, min(-position, top.ask_vol, self.settings.clip), "bid", "ioc")

    def _take(self, liquid: Top, thin: Top, iid: str, tick: float, position: int, net: int) -> None:
        """An IOC at the worst price that still has the edge: it fills what is there and the rest disappears."""
        if self.clock() - self.last_take.get(iid, float("-inf")) < TAKE_GAP_S:
            return
        edge = self.settings.edge_ticks * tick
        room = self.settings.limit - self.settings.margin
        buy = math.floor((liquid.mid - edge) / tick + ROUNDING) * tick
        sell = math.ceil((liquid.mid + edge) / tick - ROUNDING) * tick
        if thin.ask <= buy + ROUNDING:
            side, price, volume = "bid", buy, min(self.settings.clip, room - position, room - net)
        elif thin.bid >= sell - ROUNDING:
            side, price, volume = "ask", sell, min(self.settings.clip, room + position, room + net)
        else:
            return
        if volume > 0 and self.budget.take():
            self._insert(iid, round(price, 8), volume, side, "ioc")
            self.last_take[iid] = self.clock()

    def _quote_is_current(self, iid: str, top: Top, tick: float, position: int) -> bool:
        done = self.last_quote.get(iid)
        if done is None or done[2] != position:
            return False
        return self.clock() - done[0] < REQUOTE_GAP_S or abs(top.mid - done[1]) < tick - ROUNDING

    def _quote(self, iid: str, top: Top, tick: float, position: int) -> None:
        if self._quote_is_current(iid, top, tick, position) or self.budget.left() < 3:
            return
        settings = self.settings
        self.budget.take()
        self.client.delete_orders(iid)  # first, so nothing of ours can cross what comes next
        fair = top.mid - position / settings.limit * settings.lean_ticks * tick
        bid = min(math.floor((fair - settings.quote_ticks * tick) / tick + ROUNDING) * tick, top.ask - tick)
        ask = max(math.ceil((fair + settings.quote_ticks * tick) / tick - ROUNDING) * tick, top.bid + tick)
        room = settings.limit - settings.margin
        buy, sell = min(settings.quote_size, room - position), min(settings.quote_size, room + position)
        if buy > 0 and self.budget.take():
            self._insert(iid, round(bid, 8), buy, "bid", "limit")
        if sell > 0 and self.budget.take():
            self._insert(iid, round(ask, 8), sell, "ask", "limit")
        self.last_quote[iid] = (self.clock(), top.mid, position)

    def _legs(self, pair: Tuple[str, str], first: Top, second: Top, books: Dict) -> Tuple[str, str]:
        """The tighter leg on average so far, so that one odd moment cannot swap them and double the delta."""
        seen = self.spreads.setdefault(pair, [0.0, 0.0])
        seen[0] += first.ask - first.bid
        seen[1] += second.ask - second.bid
        if seen[0] == seen[1]:
            return liquid_first(pair, books)
        return pair if seen[0] < seen[1] else (pair[1], pair[0])

    def _trade_pairs(self, pairs, instruments: Dict, ids, books: Dict, positions: Dict[str, int]) -> None:
        for pair in pairs:
            if pair[0] not in ids or pair[1] not in ids:
                continue
            first, second = read_top(books[pair[0]]), read_top(books[pair[1]])
            if first is None or second is None:
                continue
            liquid_id, thin_id = self._legs(pair, first, second, books)
            liquid, thin = (first, second) if liquid_id == pair[0] else (second, first)
            position, net = positions.get(thin_id, 0), positions.get(pair[0], 0) + positions.get(pair[1], 0)
            self._take(liquid, thin, thin_id, _tick(instruments[thin_id]), position, net)

    def step(self) -> None:
        instruments = self.client.get_instruments() or {}
        ids = [iid for iid, info in instruments.items() if not getattr(info, "paused", False)]
        books = {iid: self.client.get_last_price_book(iid) for iid in ids}
        positions = self.client.get_positions() or {}
        flatten = (self.settings.run_dir / "FLATTEN").exists()
        if flatten or (self.settings.run_dir / "KILL").exists():
            if not self.pulled:
                self._pull(ids)
                self.pulled = True
            if flatten:
                self._flatten(ids, books, positions)
            return
        self.pulled = False
        pairs = self.settings.pairs if self.settings.pairs is not None else PAIRS or find_pairs(instruments)
        if pairs:
            self._trade_pairs(pairs, instruments, ids, books, positions)
            return
        for iid in ids:
            top = read_top(books[iid])
            if top is not None:
                self._quote(iid, top, _tick(instruments[iid]), positions.get(iid, 0))


def other_bot_running(run_dir: Path) -> Optional[str]:
    """Why not to start, or None. A second login disconnects the first and deletes its orders."""
    try:
        age = time.time() - (run_dir / "heartbeat").stat().st_mtime
    except OSError:
        return None
    return f"the main bot looks still running: {run_dir}/heartbeat is {age:.0f} s old" if age < HEARTBEAT_FRESH_S else None


def _claim_session(run_dir: Path) -> None:
    """The main bot checks these two files, so it will refuse to log in beside this one."""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "LIVE.lock").write_text(f"{os.getpid()}\n")
        (run_dir / "heartbeat").touch()
    except OSError as exc:
        print(f"could not mark the session as taken: {exc}", flush=True)


def _beat(run_dir: Path) -> None:
    try:
        (run_dir / "heartbeat").touch()
    except OSError:
        pass


def default_client():
    from optibook.synchronous_client import Exchange

    # For a devenv where no arguments cannot connect. The same variable as the main bot, and never in a file.
    args = json.loads(os.environ.get("OPTIBOOK_CLIENT_ARGS") or "{}")
    return Exchange(**args)


def _connect(factory: Callable[[], object]):
    client = factory()
    connect = getattr(client, "connect", None)
    if connect is not None:
        connect()
    return client


def _report(trader: Trader, log: Callable[[str], None]) -> None:
    try:
        positions = {iid: pos for iid, pos in (trader.client.get_positions() or {}).items() if pos}
        pnl = trader.client.get_pnl() if hasattr(trader.client, "get_pnl") else None
    except Exception as exc:  # noqa: BLE001 - a report must not stop the trading
        positions, pnl = str(exc), None
    log(f"pnl {pnl} positions {positions} takes {trader.takes} quotes {trader.quotes} rejects {trader.rejects}")


def run(
    factory: Callable[[], object] = default_client,
    settings: Optional[Settings] = None,
    seconds: Optional[float] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> None:
    """Never gives up: any failure means a new client, after a growing pause so as not to duel a teammate's login."""
    settings = settings or Settings()
    _claim_session(settings.run_dir)
    deadline = clock() + seconds if seconds is not None else float("inf")
    trader, backoff, next_status = None, 1.0, clock()
    while clock() < deadline:
        try:
            if trader is None or not trader.client.is_connected():
                trader = Trader(_connect(factory), settings, clock)
                log(f"connected; instruments {sorted(trader.client.get_instruments() or {})}")
            trader.step()
            backoff = 1.0
            if clock() >= next_status:
                _report(trader, log)
                _beat(settings.run_dir)
                next_status = clock() + STATUS_S
            sleep(LOOP_S)
        except Exception as exc:  # noqa: BLE001 - staying up is the whole job of this file
            log(f"{type(exc).__name__}: {exc}. Retrying in {backoff:.0f} s with a new client.")
            trader = None
            sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, help="stop after this long; default runs until killed")
    parser.add_argument("--run-dir", default="run")
    parser.add_argument("--force", action="store_true", help="start even if the main bot's heartbeat is fresh")
    args = parser.parse_known_args()[0]  # a notebook puts its own arguments on sys.argv
    run_dir = Path(args.run_dir)
    busy = other_bot_running(run_dir)
    if busy and not args.force:
        print(f"Refusing to start: {busy}. Stop it first, or use --force.", flush=True)
        return 2

    def log(message: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)

    try:
        run(settings=Settings(run_dir=run_dir), seconds=args.seconds, log=log)
    except KeyboardInterrupt:
        log("stopped by hand. Resting orders stay until the exchange removes them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
