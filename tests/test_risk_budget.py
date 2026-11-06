"""Exchange-limit arithmetic: a breach means rejection and a forced disconnect (Optibook guide §4)."""

import pytest

from otk.budget import Priority, WriteBudget
from otk.risk import (
    DeltaWatch,
    DrawdownStop,
    Limits,
    buy_capacity,
    clip_order,
    sell_capacity,
)


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    "position,open_buy,open_sell,expected",
    [
        (0, 0, 0, 100),
        (90, 0, 0, 10),
        (90, 10, 0, 0),
        (-100, 0, 0, 200),
        (100, 0, 0, 0),
        (110, 0, 0, 0),  # never negative, even from an already breached view
    ],
)
def test_buy_capacity_assumes_every_resting_buy_fills(position, open_buy, open_sell, expected):
    assert buy_capacity(position, open_buy, open_sell, Limits(margin=0)) == expected


def test_sell_capacity_mirrors_buy_capacity():
    assert sell_capacity(90, 0, 0, Limits(margin=0)) == 190
    assert sell_capacity(-90, 0, 10, Limits(margin=0)) == 0


def test_resting_sells_never_release_buy_capacity():
    # An opposite resting order might never fill, so it cannot fund this side.
    assert buy_capacity(100, 0, 50, Limits(margin=0)) == 0


def test_outstanding_cap_counts_both_sides():
    limits = Limits(margin=0)
    assert buy_capacity(0, 100, 100, limits) == 0
    assert sell_capacity(50, 90, 90, limits) == 20  # position room is 60, outstanding room 20


def test_margin_shrinks_position_capacity_only():
    assert buy_capacity(0, 0, 0, Limits(position=100, margin=5)) == 95
    assert sell_capacity(0, 0, 0, Limits(position=100, margin=5)) == 95


def test_clip_order_returns_zero_when_no_capacity():
    assert clip_order("bid", 10, 100, 0, 0, Limits(margin=0)) == 0
    assert clip_order("bid", 10, 95, 0, 0, Limits(margin=0)) == 5
    assert clip_order("ask", 10, 95, 0, 0, Limits(margin=0)) == 10


def test_clip_order_rejects_unknown_side():
    with pytest.raises(ValueError):
        clip_order("buy", 10, 0, 0, 0, Limits())


def test_budget_never_exceeds_cap_in_any_one_second_window():
    clock = FakeClock()
    budget = WriteBudget(max_per_second=20, reserve=4, clock=clock)
    sent = []
    for step in range(5000):
        clock.now = step * 0.001
        if budget.try_acquire(Priority.RISK):
            sent.append(clock.now)
    for idx, start in enumerate(sent):
        window = [t for t in sent[idx:idx + 40] if t < start + 1.0]
        assert len(window) <= 20


def test_quotes_cannot_use_the_reserve():
    clock = FakeClock()
    budget = WriteBudget(max_per_second=10, reserve=4, clock=clock)
    assert sum(budget.try_acquire(Priority.QUOTE) for _ in range(20)) == 6
    assert sum(budget.try_acquire(Priority.RISK) for _ in range(20)) == 4


def test_budget_refills_as_window_slides():
    clock = FakeClock()
    budget = WriteBudget(max_per_second=5, reserve=0, clock=clock)
    assert sum(budget.try_acquire(Priority.QUOTE) for _ in range(10)) == 5
    clock.now = 1.01
    assert budget.try_acquire(Priority.QUOTE)


def test_budget_rejects_a_cap_above_the_exchange_limit():
    with pytest.raises(ValueError):
        WriteBudget(max_per_second=26)


def test_delta_watch_fires_before_the_exchange_grace_period_ends():
    watch = DeltaWatch(soft=100.0, grace_s=10.0, act_after_s=4.0)
    assert not watch.update("ASML", 120.0, now=0.0)
    assert not watch.update("ASML", 120.0, now=3.9)
    assert watch.update("ASML", 120.0, now=4.1)


def test_delta_watch_resets_when_exposure_falls():
    watch = DeltaWatch(soft=100.0, grace_s=10.0, act_after_s=4.0)
    watch.update("ASML", 120.0, now=0.0)
    watch.update("ASML", 50.0, now=2.0)
    assert not watch.update("ASML", 120.0, now=5.0)


def test_delta_watch_tracks_groups_and_signs_independently():
    watch = DeltaWatch(soft=100.0, grace_s=10.0, act_after_s=1.0)
    watch.update("A", -150.0, now=0.0)
    assert not watch.update("B", 150.0, now=2.0)
    assert watch.update("A", -150.0, now=2.0)


def test_drawdown_stop_latches_from_peak():
    stop = DrawdownStop(max_drawdown=50.0)
    assert not stop.update(100.0)
    assert not stop.update(60.0)
    assert stop.update(49.0)
    assert stop.update(500.0)  # latched until an operator resets it
    stop.reset()
    assert not stop.update(500.0)
