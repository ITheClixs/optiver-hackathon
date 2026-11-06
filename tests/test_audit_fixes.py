"""What a pre-event audit found: each test is a way the bot would have idled, or taken risk it was told not to."""

import json
import sys
import types
from pathlib import Path

import pytest

from otk import adapter
from otk.fair import Top
from otk.instruments import Universe
from otk.intents import Take
from otk.main import EXIT_LISTING_CHANGED, supervise
from otk.quoter import QuoteParams, compute_quotes
from otk.sim.exchange import SimExchange, SimMarket
from otk.sim.harness import make_engine
from otk.strategies import DualListingArb, MarketMaker, TheoTaker
from otk.updater import risk_cuts

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import set_params


def market_with(books):
    market = SimMarket(instruments=list(books))
    maker = SimExchange(market, "wall", bot=True)
    for iid, (bid, ask) in books.items():
        maker.insert_order(iid, price=bid, volume=500, side="bid", order_type="limit")
        maker.insert_order(iid, price=ask, volume=500, side="ask", order_type="limit")
    return market


def advance(market, engine, steps=4, dt=0.3):
    for _ in range(steps):
        market.advance(dt)
        engine.tick()


def listing_with(engine, **extra):
    listed = {**engine.exchange.get_instruments(), **extra}
    engine.exchange.get_instruments = lambda: listed


def listed_item(**flags):
    return types.SimpleNamespace(**{"paused": False, "expired": False, **flags})


# --- the freeze compares against what is really running -------------------------------------------


@pytest.mark.parametrize(
    "base, new, expected",
    [
        ({"size": 20}, {"size": 15}, {"size": 15}),
        ({"size": 5}, {"size": 8}, {}),
        ({"position_limit": 300}, {"position_limit": 200}, {"position_limit": 200}),
        ({"position_limit": 50}, {"position_limit": 80}, {}),
        ({"lag_enabled": 1}, {"lag_enabled": 0}, {"lag_enabled": 0}),
        ({"lag_enabled": 1}, {"lag_clip": 0}, {"lag_clip": 0}),
        ({}, {"ap_enabled": 0}, {"ap_enabled": 0}),
    ],
)
def test_past_the_freeze_a_cut_is_judged_against_the_event_settings(base, new, expected):
    assert risk_cuts({}, new, base) == expected


@pytest.mark.parametrize("key", ["kill_max_loss", "kill_loss_per_min", "kill_drawdown"])
def test_past_the_freeze_a_stop_that_was_off_can_be_set_and_tightened_but_not_loosened(key):
    assert risk_cuts({}, {key: 500}, {}) == {key: 500}
    assert risk_cuts({key: 500}, {key: 400}, {}) == {key: 400}
    assert risk_cuts({key: 500}, {key: 900}, {}) == {key: 500}
    assert risk_cuts({key: 500}, {key: 0}, {}) == {key: 500}, "0 turns a stop off, which is no cut"


def test_a_published_value_outranks_the_event_settings_as_the_thing_to_cut_from():
    assert risk_cuts({"size": 10}, {"size": 12}, {"size": 20}) == {"size": 10}


def test_a_per_instrument_cut_is_judged_against_the_event_settings_too():
    base = {"size": 20, "instruments": {"AMD": {"size": 30}}}
    assert risk_cuts({}, {"instruments": {"AMD": {"size": 25}}}, base) == {"instruments": {"AMD": {"size": 25}}}
    assert risk_cuts({}, {"instruments": {"NVDA": {"size": 25}}}, base) == {}, "NVDA runs at the event's 20"
    assert risk_cuts({}, {"instruments": {"NVDA": {"size": 15}}}, base) == {"instruments": {"NVDA": {"size": 15}}}


# --- a broken connection must not idle for ever ---------------------------------------------------


def test_a_connection_error_with_the_client_claiming_to_be_connected_leads_to_a_reconnect():
    market = market_with({"A": (99.0, 101.0)})
    engine = make_engine(market, "us", [MarketMaker(["A"])])

    def broken(_iid):
        raise ConnectionResetError("reset")

    engine.exchange.get_last_price_book = broken
    disconnects = []
    engine.exchange.disconnect = lambda: disconnects.append(1)
    for _ in range(40):
        engine.tick()
    assert engine.errors > 0
    assert disconnects, "the same error of any other type drops the session and rebuilds it"


# --- the listing ----------------------------------------------------------------------------------


def test_an_expired_instrument_is_treated_as_closed():
    market = market_with({"A": (99.0, 101.0)})
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.tick()
    listing_with(engine, A=listed_item(expired=True))
    assert engine._paused() == frozenset({"A"})


def test_a_new_listing_stops_the_child_so_the_parent_rebuilds_it():
    market = market_with({"A": (99.0, 101.0)})
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.tick()
    listing_with(engine, NEW=listed_item())
    with pytest.raises(SystemExit) as stopped:
        engine.tick()
    assert stopped.value.code == EXIT_LISTING_CHANGED


def test_a_bot_that_started_on_an_empty_listing_restarts_when_one_appears():
    market = market_with({"A": (99.0, 101.0)})
    engine = make_engine(market, "us", [])
    engine.universe = Universe([])
    engine.exchange.get_instruments = dict
    engine.tick()
    listing_with(engine, A=listed_item())
    with pytest.raises(SystemExit):
        engine.tick()


def test_a_listing_that_is_already_expired_does_not_restart_anything():
    market = market_with({"A": (99.0, 101.0)})
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.tick()
    listing_with(engine, OLD=listed_item(expired=True))
    engine.tick()


def test_an_instrument_the_event_left_out_on_purpose_does_not_restart_anything():
    """The event can name a subset, and a restart would leave it out again, for ever."""
    market = market_with({"A": (99.0, 101.0), "B": (49.0, 51.0)})
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.universe = Universe(info for iid, info in engine.universe.by_id.items() if iid == "A")
    engine.tick()
    engine.tick()


def test_the_parent_restarts_a_child_that_found_a_new_listing_without_calling_it_a_crash(tmp_path):
    counter = tmp_path / "n"
    script = tmp_path / "child.py"
    script.write_text(
        "import pathlib, sys\n"
        f"p = pathlib.Path({str(counter)!r})\n"
        "n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n"
        f"sys.exit({EXIT_LISTING_CHANGED} if n == 0 else 0)\n"
    )

    class Updater:
        crashes = 0

        def command(self, base):
            return base

        def action(self, _now):
            return None

        def on_crash(self, _now):
            Updater.crashes += 1
            return False

    code = supervise(
        [sys.executable, str(script)], tmp_path / "beat", sleep=lambda _s: None, restart_pause_s=0.0, poll_s=0.01, updater=Updater()
    )
    assert code == 0
    assert counter.read_text() == "2"
    assert Updater.crashes == 0


# --- client arguments -----------------------------------------------------------------------------


def fake_client(monkeypatch):
    seen = {}

    class Exchange:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    module = types.ModuleType("optibook.synchronous_client")
    module.Exchange = Exchange
    monkeypatch.setitem(sys.modules, "optibook", types.ModuleType("optibook"))
    monkeypatch.setitem(sys.modules, "optibook.synchronous_client", module)
    return seen


def test_with_nothing_set_the_client_gets_no_arguments(monkeypatch):
    seen = fake_client(monkeypatch)
    monkeypatch.delenv(adapter.CLIENT_ARGS_ENV, raising=False)
    adapter.optibook_client_factory()()
    assert seen == {}


def test_client_arguments_come_from_the_environment(monkeypatch):
    seen = fake_client(monkeypatch)
    monkeypatch.setenv(adapter.CLIENT_ARGS_ENV, json.dumps({"host": "h", "info_port": 7001}))
    adapter.optibook_client_factory()()
    assert seen == {"host": "h", "info_port": 7001}


@pytest.mark.parametrize("bad", ["not json", "[1, 2]"])
def test_bad_client_arguments_stop_with_a_message_that_does_not_repeat_them(monkeypatch, bad):
    fake_client(monkeypatch)
    monkeypatch.setenv(adapter.CLIENT_ARGS_ENV, bad)
    with pytest.raises(SystemExit) as stopped:
        adapter.optibook_client_factory()()
    assert adapter.CLIENT_ARGS_ENV in str(stopped.value)
    assert bad not in str(stopped.value), "it may hold a password"


# --- quotes and takes -----------------------------------------------------------------------------


def test_a_quote_never_goes_through_the_book_when_fair_is_far_from_it():
    params = QuoteParams(base_spread_ticks=2, max_deviation_ticks=15)
    top = Top(99.9, 100, 100.1, 100)
    quote = compute_quotes(102.0, 0, 100, 0.0, 0.0, params, 0.1, top)
    assert quote.bid < 100.1 - 1e-9, quote
    assert quote.ask > 99.9 + 1e-9, quote


def test_a_pair_of_different_stocks_is_not_arbitraged():
    """`instrument_group` can label two unrelated stocks: their gap is no edge, and it never closes."""
    market = market_with({"A": (149.9, 150.1), "B": (189.9, 190.1)})
    engine = make_engine(market, "us", [DualListingArb([("A", "B")])])
    advance(market, engine)
    assert engine.exchange.get_positions().get("A", 0) == 0
    assert engine.exchange.get_positions().get("B", 0) == 0


def test_a_real_dual_gap_is_still_arbitraged():
    market = market_with({"A": (100.5, 100.7), "A_DUAL": (99.9, 100.1)})
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")])])
    advance(market, engine)
    assert engine.exchange.get_positions().get("A_DUAL", 0) > 0


def test_a_theo_far_from_the_book_is_not_taken():
    market = market_with({"A": (9.9, 10.1)})
    engine = make_engine(market, "us", [TheoTaker(["A"], lambda _state, _iid: 40.0)])
    advance(market, engine)
    assert engine.exchange.get_positions().get("A", 0) == 0


def test_a_theo_near_the_book_is_still_taken():
    market = market_with({"A": (9.9, 10.1)})
    engine = make_engine(market, "us", [TheoTaker(["A"], lambda _state, _iid: 10.5)])
    advance(market, engine)
    assert engine.exchange.get_positions().get("A", 0) > 0


def test_a_take_uses_the_position_at_send_time_not_at_the_start_of_the_tick():
    market = market_with({"A": (99.0, 101.0)})
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.exchange.get_positions = lambda: {"A": 90}
    room = engine.manager._take_volume(Take("A", "bid", 101.0, 40, False, "t"), {"A": 0})
    assert room <= engine.manager.limits.position - 90


# --- only A publishes -----------------------------------------------------------------------------


def test_a_job_cannot_apply_parameters_to_a_file_the_bot_never_reads(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OTK_ROLE", "B")
    path = tmp_path / "params.json"
    assert set_params.main(['{"size": 5}', "--path", str(path)]) == 2
    assert not path.exists()
    assert "publish" in capsys.readouterr().err


def test_a_job_can_still_list_the_parameters(monkeypatch):
    monkeypatch.setenv("OTK_ROLE", "X")
    assert set_params.main(["--list"]) == 0


def test_a_person_can_still_apply_parameters(monkeypatch, tmp_path):
    monkeypatch.delenv("OTK_ROLE", raising=False)
    path = tmp_path / "params.json"
    assert set_params.main(['{"size": 5}', "--path", str(path)]) == 0
    assert json.loads(path.read_text()) == {"size": 5}
