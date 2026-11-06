"""Which strategy earned what. Without it nobody can say which card to keep, so every fill must carry its strategy."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from otk.budget import WriteBudget
from otk.engine import Engine
from otk.fair import Book, Level
from otk.instruments import Universe
from otk.intents import LadderIntent, QuoteIntent, Take
from otk.main import write_status
from otk.monitor import UNKNOWN_STRATEGY, LiveMonitor
from otk.params import ParamStore
from otk.quote_manager import MAX_TAGS, QuoteManager
from otk.recorder import JsonlRecorder, read_capture
from otk.risk import Limits
from otk.sim.exchange import SimExchange
from otk.sim.harness import make_engine, run_session
from otk.sim.scenarios import dual_listing, quiet_market
from otk.strategies import MarketMaker, default_stack


class Exchange:
    def __init__(self):
        self.next_id = 0
        self.orders = {}

    def insert_order(self, instrument_id, price, volume, side, order_type):
        self.next_id += 1
        if order_type == "limit":
            self.orders[self.next_id] = SimpleNamespace(order_id=self.next_id, instrument_id=instrument_id, price=price, volume=volume, side=side)
        return SimpleNamespace(success=True, order_id=self.next_id, error_reason="")

    def get_positions(self):
        return {}

    def get_outstanding_orders(self, instrument_id):
        return {key: order for key, order in self.orders.items() if order.instrument_id == instrument_id}

    def delete_order(self, _instrument_id, order_id):
        self.orders.pop(order_id, None)

    def delete_orders(self, instrument_id):
        for key in [key for key, order in self.orders.items() if order.instrument_id == instrument_id]:
            del self.orders[key]

    def amend_order(self, _instrument_id, order_id, volume):
        self.orders[order_id].volume = volume


def manager(exchange=None):
    return QuoteManager(exchange or Exchange(), WriteBudget(), Limits(), 0.1)


def test_a_resting_order_remembers_the_strategy_that_asked_for_it():
    exchange = Exchange()
    quotes = manager(exchange)
    quotes.sync(QuoteIntent("ASML", 99.0, 5, 101.0, 5, "market_maker"), {})
    assert len(exchange.orders) == 2
    assert {quotes.strategy_of(order_id) for order_id in exchange.orders} == {"market_maker"}


def test_a_ladder_rung_remembers_it_too():
    exchange = Exchange()
    quotes = manager(exchange)
    quotes.sync(LadderIntent("ASML", ((99.0, 5), (98.0, 5)), (), "market_maker"), {})
    assert len(exchange.orders) == 2
    assert {quotes.strategy_of(order_id) for order_id in exchange.orders} == {"market_maker"}


def test_an_ioc_remembers_it_too():
    exchange = Exchange()
    quotes = manager(exchange)
    quotes.take(Take("ASML", "bid", 100.0, 5, False, "dual_listing"), {})
    assert quotes.strategy_of(exchange.next_id) == "dual_listing"


def test_an_order_it_never_sent_has_no_strategy():
    assert manager().strategy_of(12345) == ""
    assert manager().strategy_of(None) == ""


def test_the_memory_of_old_orders_stays_bounded_and_keeps_the_newest():
    quotes = manager()
    for order_id in range(MAX_TAGS + 10):
        quotes._tag(order_id, "market_maker")
    assert len(quotes._tags) <= MAX_TAGS
    assert quotes.strategy_of(MAX_TAGS + 9) == "market_maker"
    assert quotes.strategy_of(0) == ""


def test_tagging_does_not_touch_what_strategies_are_told_they_hold():
    """`source_positions` drives real decisions: a quote must not start counting as a directional strategy's view."""
    exchange = Exchange()
    quotes = manager(exchange)
    quotes.sync(QuoteIntent("ASML", 99.0, 5, 101.0, 5, "gap_reversion"), {})
    order_id = next(iter(exchange.orders))
    quotes.note_fills({"ASML": [SimpleNamespace(order_id=order_id, volume=5, side="bid", price=99.0)]})
    assert quotes.source_positions == {}


# --- the monitor -----------------------------------------------------------------------------------


def book(bid, ask):
    return Book((Level(bid, 10),), (Level(ask, 10),))


def fill(order_id, side, price, volume):
    return SimpleNamespace(order_id=order_id, side=side, price=price, volume=volume)


def state(now, books, fills=None):
    tops = {iid: SimpleNamespace(bid=b.bids[0].price, bid_vol=10, ask=b.asks[0].price, ask_vol=10) for iid, b in books.items()}
    return SimpleNamespace(now=now, books=books, top=tops.get, new_ticks={}, new_fills=fills or {})


def make_monitor(tags=None, group_of=None):
    tags = tags or {}
    return LiveMonitor(0.1, group_of=group_of, strategy_of=lambda order_id: tags.get(order_id, ""))


def test_pnl_by_strategy_is_cash_plus_what_it_holds_at_the_mid():
    monitor = make_monitor({1: "market_maker", 2: "hedger"})
    monitor.tick(state(0.0, {"ASML": book(99.0, 101.0)}, {"ASML": [fill(1, "bid", 99.0, 10), fill(2, "ask", 101.0, 4)]}))
    monitor.tick(state(1.0, {"ASML": book(101.0, 103.0)}))
    report = monitor.by_strategy()
    assert report["market_maker"]["pnl"] == 30.0  # bought 10 at 99, mid now 102
    assert report["hedger"]["pnl"] == -4.0  # sold 4 at 101, mid now 102
    assert report["market_maker"]["fills"] == 1
    assert report["market_maker"]["lots"] == 10


def test_a_fill_with_no_known_order_is_counted_rather_than_lost():
    monitor = make_monitor()
    monitor.tick(state(0.0, {"ASML": book(99.0, 101.0)}, {"ASML": [fill(7, "bid", 99.0, 1)]}))
    assert monitor.by_strategy()[UNKNOWN_STRATEGY]["lots"] == 1


def test_a_client_whose_fills_carry_no_order_id_still_counts():
    monitor = make_monitor({1: "market_maker"})
    plain = SimpleNamespace(side="bid", price=99.0, volume=2)
    monitor.tick(state(0.0, {"ASML": book(99.0, 101.0)}, {"ASML": [plain]}))
    assert monitor.by_strategy()[UNKNOWN_STRATEGY]["lots"] == 2


def test_pnl_by_instrument_carries_its_group_so_a_hedged_pair_can_be_judged_as_one():
    monitor = make_monitor({1: "dual_listing", 2: "hedger"}, group_of=lambda iid: "ASML")
    fills = {"ASML_DUAL": [fill(1, "bid", 99.0, 5)], "ASML": [fill(2, "ask", 101.0, 5)]}
    monitor.tick(state(0.0, {"ASML": book(99.0, 101.0), "ASML_DUAL": book(99.0, 101.0)}, fills))
    report = monitor.by_instrument()
    assert report["ASML_DUAL"]["pnl"] == 5.0
    assert report["ASML"]["pnl"] == 5.0
    assert report["ASML_DUAL"]["group"] == report["ASML"]["group"] == "ASML"


def test_markout_is_scored_per_strategy_after_the_horizon_in_ticks():
    monitor = make_monitor({1: "market_maker"})
    monitor.tick(state(0.0, {"ASML": book(99.0, 101.0)}, {"ASML": [fill(1, "bid", 99.0, 10)]}))
    assert monitor.by_strategy()["market_maker"]["markout_5s_ticks"] is None
    monitor.tick(state(6.0, {"ASML": book(99.5, 101.5)}))
    assert monitor.by_strategy()["market_maker"]["markout_5s_ticks"] == 15.0  # mid 100.5 against 99, at a 0.1 tick


def test_a_monitor_left_unattached_still_works():
    monitor = LiveMonitor(0.1)
    monitor.tick(state(0.0, {"ASML": book(99.0, 101.0)}, {"ASML": [fill(1, "bid", 99.0, 1)]}))
    assert monitor.by_strategy()[UNKNOWN_STRATEGY]["fills"] == 1
    assert monitor.by_instrument()["ASML"]["group"] == "ASML"


# --- the capture and the status --------------------------------------------------------------------


def test_the_capture_records_each_fills_strategy(tmp_path):
    path = tmp_path / "capture.jsonl.gz"
    recorder = JsonlRecorder(path)
    recorder.strategy_of = {3: "hedger"}.get
    recorder.tick(SimpleNamespace(books={}, new_ticks={}, new_fills={"ASML": [fill(3, "ask", 100.0, 2), fill(4, "bid", 99.0, 1)]}))
    recorder.close()
    events = [event for event in read_capture(path) if event["k"] == "fill"]
    assert events[0]["src"] == "hedger"
    assert events[0]["vol"] == 2
    assert events[1]["src"] == ""


def make_sim_engine(market, monitor, strategies):
    return make_engine(market, "us", strategies, recorder=monitor)


def test_the_status_file_says_it_all_and_the_split_adds_up_on_a_real_session(tmp_path):
    market = dual_listing(seed=3)
    probe = SimExchange(market, "probe")
    probe.connect()
    universe = Universe.from_instruments(probe.get_instruments())
    ids = sorted(universe.by_id)
    monitor = LiveMonitor(universe.tick_of, group_of=lambda iid: universe.by_id[iid].base)
    engine: Engine = make_sim_engine(market, monitor, default_stack(ids, universe=universe))
    monitor.strategy_of = engine.manager.strategy_of
    result = run_session(market, {"us": engine}, seconds=120.0)["us"]
    path = tmp_path / "status.json"
    write_status(path, engine, 0.0, monitor)
    status = json.loads(path.read_text())

    split = status["by_strategy"]
    assert split, "a two-minute session on a dual market trades"
    assert UNKNOWN_STRATEGY not in split, "every fill came from an order we tagged"
    assert sum(row["lots"] for row in split.values()) > 0
    total = sum(row["pnl"] for row in split.values())
    assert abs(total - result.marked_pnl) <= max(5.0, abs(result.marked_pnl) * 0.05)
    assert abs(sum(row["pnl"] for row in status["by_instrument"].values()) - total) < 0.1
    assert {row["group"] for row in status["by_instrument"].values()} <= {"ASML", "SAP"}


def test_the_status_says_what_is_in_force_apart_from_autopilot(tmp_path):
    market = dual_listing(seed=4)
    store = ParamStore(base={"size": 5})
    store.set_overlay({"size": 9})
    engine = make_engine(market, "us", [], params=store)
    path = tmp_path / "status.json"
    write_status(path, engine, 0.0)
    assert json.loads(path.read_text())["baseline"]["size"] == 5


def test_the_status_is_still_written_when_the_client_cannot_price_a_position(tmp_path):
    """The client says no PnL can be calculated with a position and no tick, and ticks start at connect: just after any restart."""
    market = dual_listing(seed=4)
    engine = make_engine(market, "us", [], params=ParamStore(base={"size": 5}))

    def cannot_price(*_args, **_kwargs):
        raise RuntimeError("no PnL can be calculated")

    engine.exchange.get_pnl = cannot_price
    path = tmp_path / "status.json"
    write_status(path, engine, 0.0)
    status = json.loads(path.read_text())
    assert status["pnl"] is None
    assert "positions" in status and status["telemetry_errors"] >= 1


def test_the_capture_leaves_our_own_quotes_out_of_its_books(tmp_path):
    """Otherwise a replay sees them as someone else's liquidity, and could take them."""
    market = quiet_market(instruments=("ASML",), seed=1)
    path = tmp_path / "capture.jsonl.gz"
    recorder = JsonlRecorder(path, book_every_s=0.0, clock=lambda: market.now)
    engine = make_engine(market, "us", [MarketMaker(["ASML"])], recorder=recorder)
    for _ in range(50):
        market.advance(0.1)
        engine.tick()
    ours = engine.manager.resting("ASML")
    assert ours, "the bot is quoting"
    public = SimExchange(market, "probe").get_last_price_book("ASML")
    public_volume = {round(level.price, 6): level.volume for level in (*public.bids, *public.asks)}
    recorder.close()
    last = [event for event in read_capture(path) if event["k"] == "book"][-1]
    recorded = {round(price, 6): volume for price, volume in (*last["bids"], *last["asks"])}
    for order in ours:
        price = round(order.price, 6)
        assert public_volume[price] >= order.volume, "our quotes are in the public book"
        assert recorded.get(price, 0) == public_volume[price] - order.volume, "and not in the recorded one"


def test_status_without_a_monitor_is_still_valid(tmp_path):
    market = dual_listing(seed=4)
    engine = make_engine(market, "us", [], params=ParamStore())
    path = tmp_path / "status.json"
    write_status(path, engine, 0.0)
    status = json.loads(path.read_text())
    assert status["by_strategy"] == {}
    assert status["by_instrument"] == {}
