"""The last-resort bot must run from one file, on very little of the client, and stay inside the limits that end a session."""

import ast
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import emergency_bot as bot


def level(price, volume):
    return SimpleNamespace(price=price, volume=volume)


def book(bid=None, ask=None, bid_vol=50, ask_vol=50):
    return SimpleNamespace(
        bids=[level(bid, bid_vol)] if bid is not None else [],
        asks=[level(ask, ask_vol)] if ask is not None else [],
    )


class Client:
    """Only what the bot may rely on. Anything else raises, so a new dependency shows up here."""

    def __init__(self, books=None, positions=None, tick=0.1, paused=()):
        self.books = books or {}
        self.positions = positions or {}
        self.tick, self.paused = tick, set(paused)
        self.sent = []
        self.cancelled = []
        self.connected = True

    def connect(self):
        self.connected = True

    def is_connected(self):
        return self.connected

    def get_instruments(self):
        return {iid: SimpleNamespace(tick_size=self.tick, paused=iid in self.paused) for iid in self.books}

    def get_last_price_book(self, iid):
        return self.books.get(iid)

    def get_positions(self):
        return dict(self.positions)

    def insert_order(self, iid, *, price, volume, side, order_type):
        self.sent.append(SimpleNamespace(iid=iid, price=price, volume=volume, side=side, order_type=order_type))
        return SimpleNamespace(success=True, order_id=len(self.sent), error_reason=None)

    def delete_orders(self, iid):
        self.cancelled.append(iid)


def stepped(client, tmp_path, now=100.0, **settings):
    trader = bot.Trader(client, bot.Settings(run_dir=tmp_path, **settings), clock=lambda: now)
    trader.step()
    return trader


# --- pairs ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        (["ASML", "ASML_DUAL"], [("ASML", "ASML_DUAL")]),
        (["SAP_A", "SAP_B"], [("SAP_A", "SAP_B")]),
        (["PHILIPS_A", "PHILIPS_B", "SOLO"], [("PHILIPS_A", "PHILIPS_B")]),
        (["SOLO", "OTHER"], []),
    ],
)
def test_pairs_are_read_from_the_names(ids, expected):
    assert bot.find_pairs(ids) == expected


def test_the_tighter_leg_is_the_liquid_one():
    books = {"A": book(99.0, 101.0), "A_DUAL": book(99.9, 100.1)}
    assert bot.liquid_first(("A", "A_DUAL"), books) == ("A_DUAL", "A")


# --- taking -----------------------------------------------------------------------------------------


def test_it_buys_a_thin_leg_offer_below_the_liquid_mid(tmp_path):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9, ask_vol=30)})
    stepped(client, tmp_path)
    assert [(o.iid, o.side, o.price, o.volume, o.order_type) for o in client.sent] == [("A_DUAL", "bid", 100.0, 20, "ioc")]


def test_it_sells_a_thin_leg_bid_above_the_liquid_mid(tmp_path):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(100.3, 100.9)})
    stepped(client, tmp_path)
    assert [(o.side, o.price, o.order_type) for o in client.sent] == [("ask", 100.2, "ioc")]


def test_no_edge_no_order(tmp_path):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.9, 100.3)})
    stepped(client, tmp_path)
    assert client.sent == []


def test_it_never_touches_the_liquid_leg_so_no_order_can_rest(tmp_path):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9)})
    stepped(client, tmp_path)
    assert {o.iid for o in client.sent} == {"A_DUAL"} and {o.order_type for o in client.sent} == {"ioc"}


@pytest.mark.parametrize(("position", "expected"), [(80, 10), (89, 1), (90, 0), (95, 0)])
def test_size_stops_short_of_the_position_limit(tmp_path, position, expected):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9)}, {"A_DUAL": position})
    stepped(client, tmp_path)
    assert sum(o.volume for o in client.sent) == expected


def test_net_delta_over_the_pair_stays_under_the_soft_limit(tmp_path):
    """At least 100 unhedged deltas for over 10 s costs money, so the pair's net counts as well as the leg's own position."""
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9)}, {"A_DUAL": 10, "A": 75})
    stepped(client, tmp_path)
    assert sum(o.volume for o in client.sent) == 5


def test_one_odd_moment_does_not_swap_the_legs(tmp_path):
    """Trading both legs would double the delta, so the liquid leg is the tighter one on average, not at this instant."""
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 100.5)})
    now = [100.0]
    trader = bot.Trader(client, bot.Settings(run_dir=tmp_path), clock=lambda: now[0])
    for _ in range(20):
        trader.step()
        now[0] += 0.05
    client.books = {"A": book(99.0, 101.0), "A_DUAL": book(100.0, 100.2)}
    now[0] += 1.0
    trader.step()
    assert {o.iid for o in client.sent} <= {"A_DUAL"}


@pytest.mark.parametrize("thin", [None, book(), book(99.0), book(ask=99.0), book(100.0, 99.0)])
def test_a_broken_book_is_skipped(tmp_path, thin):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": thin})
    client.books["A_DUAL"] = thin
    stepped(client, tmp_path)
    assert client.sent == []


def test_a_paused_instrument_is_left_alone(tmp_path):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9)}, paused=["A_DUAL"])
    stepped(client, tmp_path)
    assert client.sent == []


def test_it_does_not_repeat_itself_before_the_book_can_update(tmp_path):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9)})
    now = [100.0]
    trader = bot.Trader(client, bot.Settings(run_dir=tmp_path), clock=lambda: now[0])
    trader.step()
    now[0] += 0.01
    trader.step()
    assert len(client.sent) == 1
    now[0] += 1.0
    trader.step()
    assert len(client.sent) == 2


# --- quoting when nothing pairs ---------------------------------------------------------------------


def test_an_unpaired_instrument_gets_two_passive_quotes(tmp_path):
    client = Client({"SOLO": book(99.0, 101.0)})
    stepped(client, tmp_path)
    sides = {o.side: o for o in client.sent}
    assert client.cancelled == ["SOLO"]
    assert sides["bid"].order_type == sides["ask"].order_type == "limit"
    assert sides["bid"].price < 100.0 < sides["ask"].price and sides["bid"].price < 101.0 and sides["ask"].price > 99.0


def test_a_tight_book_cannot_make_a_quote_cross(tmp_path):
    client = Client({"SOLO": book(99.9, 100.0)})
    stepped(client, tmp_path)
    sides = {o.side: o for o in client.sent}
    assert sides["bid"].price < 100.0 and sides["ask"].price > 99.9 and sides["bid"].price < sides["ask"].price


def test_quotes_lean_away_from_inventory_and_stop_at_the_limit(tmp_path):
    client = Client({"SOLO": book(99.0, 101.0)}, {"SOLO": 90})
    stepped(client, tmp_path)
    assert {o.side for o in client.sent} == {"ask"}
    flat = Client({"SOLO": book(99.0, 101.0)})
    stepped(flat, tmp_path)
    assert client.sent[0].price < next(o.price for o in flat.sent if o.side == "ask")


def test_quotes_are_not_rewritten_while_the_price_stands_still(tmp_path):
    client = Client({"SOLO": book(99.0, 101.0)})
    now = [100.0]
    trader = bot.Trader(client, bot.Settings(run_dir=tmp_path), clock=lambda: now[0])
    trader.step()
    written = len(client.sent)
    now[0] += 5.0
    trader.step()
    assert len(client.sent) == written
    client.books["SOLO"] = book(100.0, 102.0)
    trader.step()
    assert len(client.sent) == written * 2


def test_pairs_present_means_no_resting_quotes_anywhere(tmp_path):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.9, 100.3), "SOLO": book(99.0, 101.0)})
    stepped(client, tmp_path)
    assert client.sent == []


def test_listed_pairs_override_the_names(tmp_path):
    client = Client({"X": book(100.0, 100.2), "Y": book(99.5, 99.9)})
    stepped(client, tmp_path, pairs=[("X", "Y")])
    assert [(o.iid, o.order_type) for o in client.sent] == [("Y", "ioc")]


# --- write limit ------------------------------------------------------------------------------------


def test_the_write_budget_holds_in_any_second():
    now = [0.0]
    budget = bot.Budget(10, clock=lambda: now[0])
    assert sum(budget.take() for _ in range(50)) == 10
    now[0] = 0.5
    assert not budget.take()
    now[0] = 1.01
    assert budget.take()


def test_writes_stop_when_the_budget_is_gone(tmp_path):
    books = {f"S{i}": book(99.0, 101.0) for i in range(40)}
    client = Client(books)
    trader = stepped(client, tmp_path)
    assert len(client.sent) + len(client.cancelled) <= trader.settings.max_writes_per_s


# --- kill switches ----------------------------------------------------------------------------------


def test_kill_pulls_quotes_and_sends_nothing_until_it_is_removed(tmp_path):
    client = Client({"SOLO": book(99.0, 101.0)})
    (tmp_path / "KILL").touch()
    trader = stepped(client, tmp_path)
    assert client.sent == [] and client.cancelled == ["SOLO"]
    trader.step()
    assert client.cancelled == ["SOLO"]
    (tmp_path / "KILL").unlink()
    trader.step()
    assert client.sent


def test_flatten_closes_at_the_touch_within_limits(tmp_path):
    client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9, bid_vol=7)}, {"A_DUAL": 30, "A": -5})
    (tmp_path / "FLATTEN").touch()
    stepped(client, tmp_path)
    orders = {o.iid: o for o in client.sent}
    assert (orders["A_DUAL"].side, orders["A_DUAL"].price, orders["A_DUAL"].volume, orders["A_DUAL"].order_type) == ("ask", 99.5, 7, "ioc")
    assert (orders["A"].side, orders["A"].price, orders["A"].volume) == ("bid", 100.2, 5)


# --- staying up -------------------------------------------------------------------------------------


def run_with(factory, tmp_path, seconds=3.0):
    now = [1000.0]

    def sleep(duration):
        now[0] += max(duration, 0.01)

    bot.run(factory, bot.Settings(run_dir=tmp_path), seconds=seconds, clock=lambda: now[0], sleep=sleep, log=lambda _message: None)


def test_a_new_client_replaces_one_that_failed(tmp_path):
    made = []

    def factory():
        client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9)})
        if not made:
            client.get_positions = lambda: (_ for _ in ()).throw(RuntimeError("gone"))
        made.append(client)
        return client

    run_with(factory, tmp_path)
    assert len(made) >= 2 and made[-1].sent


def test_a_dropped_connection_builds_a_new_client(tmp_path):
    made = []

    def factory():
        client = Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9)})
        client.connected = True
        if not made:
            client.is_connected = lambda: False
        made.append(client)
        return client

    run_with(factory, tmp_path)
    assert len(made) >= 2


def test_it_keeps_trying_when_the_exchange_is_not_there_yet(tmp_path):
    tries = []

    def factory():
        tries.append(1)
        if len(tries) < 4:
            raise ConnectionError("not open yet")
        return Client({"A": book(100.0, 100.2), "A_DUAL": book(99.5, 99.9)})

    now = [0.0]
    naps = []

    def sleep(duration):
        naps.append(duration)
        now[0] += max(duration, 0.01)

    bot.run(factory, bot.Settings(run_dir=tmp_path), seconds=30.0, clock=lambda: now[0], sleep=sleep, log=lambda _message: None)
    assert len(tries) >= 4
    assert naps[:3] == [1.0, 2.0, 4.0]


# --- one session ------------------------------------------------------------------------------------


def test_it_refuses_while_the_main_bot_is_alive(tmp_path):
    (tmp_path / "heartbeat").touch()
    assert "still running" in bot.other_bot_running(tmp_path)


def test_an_old_heartbeat_does_not_block_it(tmp_path):
    (tmp_path / "heartbeat").touch()
    old = time.time() - 60
    os.utime(tmp_path / "heartbeat", (old, old))
    assert bot.other_bot_running(tmp_path) is None


def test_no_run_folder_does_not_block_it(tmp_path):
    assert bot.other_bot_running(tmp_path / "missing") is None


def _ticking():
    state = [0.0]

    def clock():
        state[0] += 0.1
        return state[0]

    return clock


def test_it_marks_the_session_as_taken_so_the_main_bot_will_not_start_beside_it(tmp_path):
    settings = bot.Settings(run_dir=tmp_path)
    bot.run(lambda: Client({}), settings, seconds=0.5, clock=_ticking(), sleep=lambda _s: None, log=lambda _m: None)
    assert (tmp_path / "LIVE.lock").read_text().strip() == str(os.getpid())
    assert (tmp_path / "heartbeat").exists()


# --- one file, old Pythons --------------------------------------------------------------------------


def test_only_the_standard_library_at_the_top_level():
    tree = ast.parse((ROOT / "scripts" / "emergency_bot.py").read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    assert top <= set(sys.stdlib_module_names)
    assert "otk" not in top and "optibook" not in top


def test_the_source_parses_as_python_3_8():
    ast.parse((ROOT / "scripts" / "emergency_bot.py").read_text(), feature_version=(3, 8))


def test_a_notebook_s_own_arguments_do_not_stop_it(tmp_path):
    cell = (
        "import sys\n"
        "sys.argv = ['ipykernel_launcher.py', '-f', '/tmp/kernel.json', '--seconds', '0.5']\n"
        f"exec(compile(open({str(ROOT / 'scripts' / 'emergency_bot.py')!r}).read(), '<cell>', 'exec'), {{'__name__': '__main__'}})\n"
    )
    (tmp_path / "cell.py").write_text(cell)
    done = subprocess.run([sys.executable, "cell.py"], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr


def test_it_starts_alone_with_an_empty_environment_and_explains_a_missing_client(tmp_path):
    script = ROOT / "scripts" / "emergency_bot.py"
    done = subprocess.run(
        [sys.executable, "-I", str(script), "--seconds", "1"],
        cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=30, check=False,
    )
    assert done.returncode == 0
    assert "optibook" in done.stdout
    assert "Traceback" not in done.stdout + done.stderr
