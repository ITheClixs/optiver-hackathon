"""The replayed path: enter just before each known sweep, close a few seconds after, and stand down on a new path."""

from types import SimpleNamespace

from otk.fair import Top
from otk.intents import Pull, Take
from otk.params import defaults
from otk.schedule import Leg, ScheduleTrader

LEG = Leg(100.0, "NVDA", -1, 90)


class State:
    def __init__(self, now, params, mids, held=None, ticks=None):
        self.now = now
        self.params = params
        self._mids = mids
        self.new_ticks = ticks or {}
        self.source_positions = {("schedule", iid): lots for iid, lots in (held or {}).items()}

    def raw_top(self, iid):
        mid = self._mids.get(iid)
        return None if mid is None else Top(mid - 0.05, 100, mid + 0.05, 100)

    def tick_of(self, _iid):
        return 0.1

    def position_of(self, source, iid):
        return self.source_positions.get((source, iid), 0)


def params(**changes):
    return {**defaults(), "sched_enabled": 1, **changes}


def first_trade(trade_id=1):
    return {"NVDA": [SimpleNamespace(trade_id=trade_id)]}


def started(trader, at=1000.0, mids=None):
    trader.on_tick(State(at, params(), mids or {"NVDA": 120.0}, ticks=first_trade()))
    return trader


def takes(intents):
    return [i for i in intents if isinstance(i, Take)]


def test_the_first_trade_of_a_round_sets_t0():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    assert trader.t0 == 1000.0


def test_a_late_trade_id_does_not_start_a_round():
    trader = ScheduleTrader(["NVDA"], (LEG,))
    trader.on_tick(State(1000.0, params(), {"NVDA": 120.0}, ticks=first_trade(trade_id=5000)))
    assert trader.t0 is None


def test_it_sells_just_before_a_down_sweep_and_pulls_the_quotes():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    intents = trader.on_tick(State(1000.0 + 99.8, params(), {"NVDA": 120.0}))
    [take] = takes(intents)
    assert take.instrument_id == "NVDA" and take.side == "ask" and take.volume == 90
    assert take.price < 120.0 - 0.05  # reaches past the touch
    assert any(isinstance(i, Pull) and i.instrument_id == "NVDA" for i in intents)


def test_it_does_nothing_when_switched_off():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    assert trader.on_tick(State(1099.8, params(sched_enabled=0), {"NVDA": 120.0})) == []


def test_it_buys_back_to_flat_after_the_exit_time():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    trader.on_tick(State(1099.8, params(), {"NVDA": 120.0}))
    [take] = takes(trader.on_tick(State(1107.0, params(), {"NVDA": 105.0}, held={"NVDA": -90})))
    assert take.side == "bid" and take.volume == 90 and take.reduce_only


def test_size_scales_down_for_a_first_live_round():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    [take] = takes(trader.on_tick(State(1099.8, params(sched_scale=0.5), {"NVDA": 120.0})))
    assert take.volume == 45


def test_two_consecutive_legs_against_us_mean_a_new_path_and_it_stands_down():
    legs = (Leg(100.0, "NVDA", -1, 90), Leg(200.0, "NVDA", -1, 90), Leg(300.0, "NVDA", -1, 90))
    trader = started(ScheduleTrader(["NVDA"], legs))
    for at in (100.0, 200.0):
        trader.on_tick(State(1000.0 + at - 0.2, params(), {"NVDA": 120.0}))
        trader.on_tick(State(1000.0 + at + 7.0, params(), {"NVDA": 125.0}))  # went up, not down
    assert not takes(trader.on_tick(State(1299.8, params(), {"NVDA": 125.0})))


def test_a_flat_leg_is_not_counted_against_the_path():
    legs = (Leg(100.0, "NVDA", -1, 90), Leg(200.0, "NVDA", -1, 90), Leg(300.0, "NVDA", -1, 90))
    trader = started(ScheduleTrader(["NVDA"], legs))
    for at in (100.0, 200.0):
        trader.on_tick(State(1000.0 + at - 0.2, params(), {"NVDA": 120.0}))
        trader.on_tick(State(1000.0 + at + 7.0, params(), {"NVDA": 120.1}))  # one tick: noise
    assert takes(trader.on_tick(State(1299.8, params(), {"NVDA": 120.1})))


def test_each_leg_is_sent_once_however_long_fills_take():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    assert takes(trader.on_tick(State(1099.75, params(), {"NVDA": 120.0})))
    assert not takes(trader.on_tick(State(1099.85, params(), {"NVDA": 120.0})))


def test_nothing_is_entered_after_the_sweep():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    assert not takes(trader.on_tick(State(1100.5, params(), {"NVDA": 105.0})))


def test_a_pinned_epoch_t0_is_converted_to_the_engine_clock(monkeypatch):
    monkeypatch.setattr("otk.schedule.time.time", lambda: 1_800_000_000.0)
    trader = ScheduleTrader(["NVDA"], (LEG,))
    sent = trader.on_tick(State(5000.0, params(sched_t0=1_800_000_000.0 - 99.8), {"NVDA": 120.0}))
    assert abs(trader.t0 - (5000.0 - 99.8)) < 1e-6
    assert takes(sent)


def test_stocks_not_listed_are_left_out():
    trader = ScheduleTrader(["AMD"], (LEG,))
    assert trader.schedule == ()


def test_a_held_position_is_closed_even_when_switched_off():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    [take] = takes(trader.on_tick(State(1500.0, params(sched_enabled=0), {"NVDA": 120.0}, held={"NVDA": -30})))
    assert take.side == "bid" and take.volume == 30 and take.reduce_only


def test_a_reset_forgets_the_old_round_start():
    trader = started(ScheduleTrader(["NVDA"], (LEG,)))
    trader.on_reset()
    assert trader.t0 is None
    assert takes(trader.on_tick(State(1099.8, params(), {"NVDA": 120.0}))) == []


def test_a_pin_from_the_last_round_does_not_stop_the_next_round_starting(monkeypatch):
    """At 20:10 the old pin blocked auto t0, and the clock had to be pinned by hand mid-round."""
    monkeypatch.setattr("otk.schedule.time.time", lambda: 1_800_000_000.0)
    trader = ScheduleTrader(["NVDA"])
    pin = params(sched_t0=1_800_000_000.0 - 50.0)
    trader.on_tick(State(5000.0, pin, {"NVDA": 120.0}))
    assert trader.t0 is not None
    trader.on_reset()
    trader.on_tick(State(5100.0, pin, {"NVDA": 120.0}, ticks=first_trade()))
    assert trader.t0 == 5100.0


def test_a_new_pin_after_a_reset_is_honoured(monkeypatch):
    monkeypatch.setattr("otk.schedule.time.time", lambda: 1_800_000_000.0)
    trader = ScheduleTrader(["NVDA"])
    trader.on_tick(State(5000.0, params(sched_t0=1_800_000_000.0 - 50.0), {"NVDA": 120.0}))
    trader.on_reset()
    trader.on_tick(State(5100.0, params(sched_t0=1_800_000_000.0 - 7.0), {"NVDA": 120.0}))
    assert trader.t0 == 5100.0 - 7.0
