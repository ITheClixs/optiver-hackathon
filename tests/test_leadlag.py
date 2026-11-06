"""Some stocks copy another a moment later. The bot learns which from the book, and leans on the lag only when told to."""

import random
from types import SimpleNamespace

import pytest

from otk.fair import Top
from otk.intents import Pull, Take
from otk.leadlag import KNOWN_LINKS, LagAdjustedFair, LagLink, LeadLag, learn_links
from otk.params import defaults


def market(seconds=900, lag=2, beta=0.4, seed=1):
    """A leader on a random walk with jumps, a follower copying it `lag` seconds later, and an unrelated stock."""
    rng = random.Random(seed)
    leader, other = [200.0], [80.0]
    for _ in range(seconds):
        jump = rng.choice([-4.0, 4.0]) if rng.random() < 0.02 else 0.0
        leader.append(leader[-1] + rng.gauss(0, 0.1) + jump)
        other.append(other[-1] + rng.gauss(0, 0.3))
    follower = [90.0 + beta * (leader[max(0, i - lag)] - 200.0) for i in range(len(leader))]
    return {"LEAD": leader, "COPY": follower, "ALONE": other}


def test_it_finds_the_pair_the_direction_and_the_ratio():
    links = learn_links(market())
    assert set(links) == {"COPY"}
    assert links["COPY"].leader == "LEAD"
    assert abs(links["COPY"].beta - 0.4) < 0.02


def test_unrelated_stocks_make_no_pair():
    rng = random.Random(2)
    history = {name: [100.0] for name in "ABCDEFGH"}
    for _ in range(900):
        for series in history.values():
            series.append(series[-1] + rng.gauss(0, 0.3))
    assert learn_links(history) == {}


def test_too_little_history_makes_no_pair():
    short = {name: series[:200] for name, series in market().items()}
    assert learn_links(short) == {}


def test_two_stocks_moving_together_with_no_lag_name_no_leader():
    moves = market(lag=0)
    assert learn_links(moves) == {}


def test_a_stock_joins_only_one_pair():
    moves = market()
    moves["COPY2"] = [value * 2 for value in moves["COPY"]]
    links = learn_links(moves)
    used = [link.leader for link in links.values()] + list(links)
    assert len(used) == len(set(used))


class State:
    def __init__(self, params=None):
        self.now = 0.0
        self.tops = {}
        self.positions = {}
        self.source_positions = {}
        self.params = {**defaults(), **(params or {})}

    def top(self, iid):
        return self.tops.get(iid)

    def position(self, iid):
        return self.positions.get(iid, 0)

    def tick_of(self, _iid):
        return 0.1


def quoted(center, half=0.3):
    return Top(round(center - half, 1), 1000, round(center + half, 1), 1000)


def warmed_up(params=None, seconds=900):
    """A LeadLag that has watched the market long enough to know the pair, ending with the two in line."""
    moves = market(seconds=seconds)
    state = State(params)
    strategy = LeadLag(["ALONE", "COPY", "LEAD"])
    for second in range(seconds + 1):
        state.now = float(second)
        state.tops = {name: quoted(series[second]) for name, series in moves.items()}
        strategy.on_tick(state)
    assert "COPY" in strategy.links
    for extra in range(1, 200):  # let the slow mean settle with both legs still
        state.now = float(seconds + extra)
        strategy.on_tick(state)
    strategy._last_take.clear()
    return strategy, state, moves


def jump_leader(state, moves, seconds, size):
    """The leader jumps and the follower has not moved yet."""
    state.now += 1.0
    state.tops = {
        "LEAD": quoted(moves["LEAD"][seconds] + size),
        "COPY": quoted(moves["COPY"][seconds]),
        "ALONE": quoted(moves["ALONE"][seconds]),
    }


def test_off_by_default_it_learns_but_never_trades():
    strategy, state, moves = warmed_up()
    assert state.params["lag_enabled"] == 0
    jump_leader(state, moves, 900, 8.0)
    assert strategy.on_tick(state) == []
    assert strategy.shift(state, "COPY") == 0.0


def test_a_leader_jump_buys_the_stale_follower():
    strategy, state, moves = warmed_up({"lag_enabled": 1})
    jump_leader(state, moves, 900, 8.0)
    intents = strategy.on_tick(state)
    assert len(intents) == 1 and isinstance(intents[0], Take)
    assert intents[0].instrument_id == "COPY" and intents[0].side == "bid"
    assert intents[0].price == state.tops["COPY"].ask
    assert intents[0].volume == state.params["lag_clip"]
    assert intents[0].source == "lead_lag"


def test_a_leader_drop_sells_it():
    strategy, state, moves = warmed_up({"lag_enabled": 1})
    jump_leader(state, moves, 900, -8.0)
    intents = strategy.on_tick(state)
    assert intents[0].side == "ask" and intents[0].price == state.tops["COPY"].bid


def test_a_move_that_does_not_beat_the_spread_is_left_alone():
    strategy, state, moves = warmed_up({"lag_enabled": 1})
    jump_leader(state, moves, 900, 0.5)  # 0.2 on the follower, against a half spread of 0.3
    assert strategy.on_tick(state) == []


def test_it_stops_at_its_position_cap_and_clips_to_it():
    strategy, state, moves = warmed_up({"lag_enabled": 1, "lag_max_position": 80})
    jump_leader(state, moves, 900, 8.0)
    state.positions["COPY"] = 80
    assert strategy.on_tick(state) == []
    strategy, state, moves = warmed_up({"lag_enabled": 1, "lag_max_position": 80})
    jump_leader(state, moves, 900, 8.0)
    state.positions["COPY"] = 70
    assert strategy.on_tick(state)[0].volume == 10


def test_it_waits_for_fills_to_report_before_taking_again():
    strategy, state, moves = warmed_up({"lag_enabled": 1})
    jump_leader(state, moves, 900, 8.0)
    assert strategy.on_tick(state)
    state.now += 0.1
    assert strategy.on_tick(state) == []


def test_the_quoting_fair_leans_with_the_leader_only_when_on():
    strategy, state, moves = warmed_up({"lag_enabled": 1})
    jump_leader(state, moves, 900, 8.0)
    strategy.on_tick(state)
    assert strategy.shift(state, "COPY") > 2.0
    assert strategy.shift(state, "LEAD") == 0.0
    fair = LagAdjustedFair(lambda _state, _iid: 100.0, strategy)
    assert fair(state, "COPY") == 100.0 + strategy.shift(state, "COPY")
    assert fair(state, "LEAD") == 100.0
    assert LagAdjustedFair(lambda _state, _iid: None, strategy)(state, "COPY") is None


def test_a_missing_book_does_not_stop_it():
    strategy, state, _moves = warmed_up({"lag_enabled": 1})
    state.tops["COPY"] = None
    state.now += 1.0
    assert strategy.on_tick(state) == []
    state.tops["COPY"] = SimpleNamespace(bid=None, bid_vol=None, ask=None, ask_vol=None)
    state.now += 1.0
    assert strategy.on_tick(state) == []


def test_the_stock_stack_has_it_and_a_dual_listing_stack_does_not():
    from otk.strategies import default_stack

    names = [item.name for item in default_stack(["AMD", "AVGO", "NBIS"])]
    assert "lead_lag" in names and names.index("lead_lag") < names.index("market_maker")
    names = [item.name for item in default_stack(["ASML", "ASML_DUAL"])]
    assert "lead_lag" not in names


def test_it_says_in_the_log_which_pairs_it_learned(caplog):
    with caplog.at_level("INFO", logger="otk.leadlag"):
        warmed_up()
    assert "COPY follows LEAD" in caplog.text
    assert caplog.text.count("COPY follows LEAD") == 1, "once, not every pass"


def test_the_taker_can_run_without_the_quote_shift():
    """exp-002 found the shift raised peak inventory and drawdown, so the take stands on its own."""
    assert defaults()["lag_take_enabled"] == 0 and defaults()["lag_sweep_ticks"] == 0
    strategy, state, moves = warmed_up({"lag_take_enabled": 1})
    jump_leader(state, moves, 900, 8.0)
    intents = strategy.on_tick(state)
    assert len(intents) == 1 and intents[0].side == "bid"
    assert intents[0].price == state.tops["COPY"].ask, "with no sweep it takes at the touch"
    assert strategy.shift(state, "COPY") == 0.0


def test_a_sweep_reaches_past_the_touch_by_at_most_its_ticks():
    strategy, state, moves = warmed_up({"lag_take_enabled": 1, "lag_sweep_ticks": 5})
    jump_leader(state, moves, 900, 8.0)  # the follower should rise about 3.2, so half the move is far beyond 5 ticks
    take = strategy.on_tick(state)[0]
    assert take.price == pytest.approx(state.tops["COPY"].ask + 0.5)


def test_a_sweep_never_reaches_past_half_the_expected_move():
    strategy, state, moves = warmed_up({"lag_take_enabled": 1, "lag_sweep_ticks": 50})
    jump_leader(state, moves, 900, 8.0)
    take = strategy.on_tick(state)[0]
    expected = -strategy._residual["COPY"]
    ask = state.tops["COPY"].ask
    assert ask + 1.0 < take.price <= ask + expected / 2 + 1e-9
    assert abs(round(take.price / 0.1) - take.price / 0.1) < 1e-6, "on the tick grid"


def test_a_sweep_sells_down_through_the_bids_too():
    strategy, state, moves = warmed_up({"lag_take_enabled": 1, "lag_sweep_ticks": 5})
    jump_leader(state, moves, 900, -8.0)
    take = strategy.on_tick(state)[0]
    assert take.side == "ask" and take.price == pytest.approx(state.tops["COPY"].bid - 0.5)


def test_a_sweep_keeps_the_clip_and_the_position_cap():
    strategy, state, moves = warmed_up({"lag_take_enabled": 1, "lag_sweep_ticks": 5, "lag_clip": 20, "lag_max_position": 80})
    jump_leader(state, moves, 900, 8.0)
    state.positions["COPY"] = 70
    assert strategy.on_tick(state)[0].volume == 10


def test_a_sweep_fills_through_the_levels_and_never_trades_with_us():
    from otk.budget import WriteBudget
    from otk.quote_manager import QuoteManager
    from otk.risk import Limits
    from otk.sim.exchange import SimExchange, SimMarket

    market = SimMarket(instruments=["F"])
    wall = SimExchange(market, "wall", bot=True)
    for price, volume in ((100.1, 20), (100.2, 30), (100.3, 1000)):
        wall.insert_order("F", price=price, volume=volume, side="ask", order_type="limit")
    us = SimExchange(market, "us")
    us.connect()
    us.insert_order("F", price=100.4, volume=5, side="ask", order_type="limit")  # our own stale quote inside the reach
    manager = QuoteManager(us, WriteBudget(25, 0, clock=lambda: market.now), Limits(margin=0), tick=market.tick)
    manager.take(Take("F", "bid", 100.5, 60, False, "lead_lag"), {"F": 0})
    assert us.is_connected()
    assert market.stats["us"].self_trades == 0
    assert us.get_positions()["F"] == 60


def in_line(state, moves, seconds, after):
    """Both legs back where the pair says they belong."""
    state.now += after
    state.tops = {name: quoted(series[seconds]) for name, series in moves.items()}


def pulls(intents):
    return [i.instrument_id for i in intents if isinstance(i, Pull)]


def test_the_guard_is_off_by_default():
    assert defaults()["lag_guard_s"] == 0
    strategy, state, moves = warmed_up({"lag_take_enabled": 1})
    jump_leader(state, moves, 900, 8.0)
    assert pulls(strategy.on_tick(state)) == []


def test_a_leader_jump_pulls_the_followers_quotes_for_the_guard_time():
    """We were swept as a maker on every follower jump: stand aside until the follower has caught up."""
    strategy, state, moves = warmed_up({"lag_guard_s": 0.5})
    jump_leader(state, moves, 900, 8.0)
    assert pulls(strategy.on_tick(state)) == ["COPY"]
    in_line(state, moves, 900, 0.3)
    assert pulls(strategy.on_tick(state)) == ["COPY"], "still inside the guard time"
    in_line(state, moves, 900, 0.3)
    assert pulls(strategy.on_tick(state)) == []


def test_the_guard_pulls_only_the_follower_and_needs_no_quote_shift():
    strategy, state, moves = warmed_up({"lag_guard_s": 0.5})
    jump_leader(state, moves, 900, 8.0)
    strategy.on_tick(state)
    assert strategy.shift(state, "COPY") == 0.0
    assert "LEAD" not in pulls(strategy.on_tick(state))


def test_the_guard_and_the_take_work_together():
    strategy, state, moves = warmed_up({"lag_take_enabled": 1, "lag_sweep_ticks": 5, "lag_guard_s": 0.5})
    jump_leader(state, moves, 900, 8.0)
    intents = strategy.on_tick(state)
    assert pulls(intents) == ["COPY"]
    assert [type(i).__name__ for i in intents].count("Take") == 1


def test_a_small_move_does_not_pull():
    strategy, state, moves = warmed_up({"lag_guard_s": 0.5})
    jump_leader(state, moves, 900, 0.5)
    assert pulls(strategy.on_tick(state)) == []


def hedges(intents):
    return [i for i in intents if isinstance(i, Take) and i.source == "lead_lag_hedge"]


def take_filled(state, lots, after=0.1):
    """Lead-lag's own takes on the follower filled this many lots in all, as the quote manager attributes them."""
    state.now += after
    state.source_positions[("lead_lag", "COPY")] = lots


def test_the_hedge_is_off_by_default():
    assert defaults()["lag_hedge"] == 0
    strategy, state, _moves = warmed_up({"lag_take_enabled": 1})
    take_filled(state, 20)
    assert hedges(strategy.on_tick(state)) == []


def test_a_filled_take_is_hedged_on_the_leader_at_the_learned_ratio():
    """One rival team hedged 366 of 369 follower takes at once and stayed flat, which judging rewards."""
    strategy, state, _moves = warmed_up({"lag_take_enabled": 1, "lag_hedge": 1})
    ratio = strategy.links["COPY"].beta
    take_filled(state, 20)
    hedge = hedges(strategy.on_tick(state))
    assert len(hedge) == 1
    assert (hedge[0].instrument_id, hedge[0].side, hedge[0].hedge) == ("LEAD", "ask", True)
    assert hedge[0].volume == round(ratio * 20)
    assert hedge[0].price == state.tops["LEAD"].bid
    take_filled(state, 20)
    assert hedges(strategy.on_tick(state)) == [], "each filled lot is hedged once"


def test_small_fills_add_up_before_they_are_hedged():
    strategy, state, _moves = warmed_up({"lag_take_enabled": 1, "lag_hedge": 1})
    ratio = strategy.links["COPY"].beta
    sent = 0
    for lots in range(1, 6):
        take_filled(state, lots)
        sent += sum(h.volume for h in hedges(strategy.on_tick(state)))
    assert sent == round(ratio * 5)


def test_a_take_that_reverses_is_hedged_back():
    strategy, state, _moves = warmed_up({"lag_take_enabled": 1, "lag_hedge": 1})
    take_filled(state, 20)
    strategy.on_tick(state)
    take_filled(state, 0)
    hedge = hedges(strategy.on_tick(state))
    assert hedge and hedge[0].side == "bid" and hedge[0].price == state.tops["LEAD"].ask


def test_after_a_disconnect_forgets_the_takes_nothing_is_hedged_back():
    strategy, state, _moves = warmed_up({"lag_take_enabled": 1, "lag_hedge": 1})
    take_filled(state, 20)
    strategy.on_tick(state)
    state.source_positions.clear()  # the quote manager forgets attributions after a disconnect
    state.now += 0.1
    assert hedges(strategy.on_tick(state)) == []
    take_filled(state, 10)
    assert sum(h.volume for h in hedges(strategy.on_tick(state))) == round(strategy.links["COPY"].beta * 10)


def test_a_hedge_waits_for_a_leader_book():
    strategy, state, moves = warmed_up({"lag_take_enabled": 1, "lag_hedge": 1})
    take_filled(state, 20)
    state.tops["LEAD"] = None
    assert hedges(strategy.on_tick(state)) == []
    state.tops["LEAD"] = quoted(moves["LEAD"][900])
    state.now += 0.1
    assert sum(h.volume for h in hedges(strategy.on_tick(state))) == round(strategy.links["COPY"].beta * 20)


def test_a_sweep_keeps_the_edge_at_its_deepest_level():
    """A small gap: the reach stops where what is left of the move still beats the half spread and the edge."""
    strategy, state, moves = warmed_up({"lag_take_enabled": 1, "lag_sweep_ticks": 50})
    jump_leader(state, moves, 900, 1.5)
    intents = [i for i in strategy.on_tick(state) if isinstance(i, Take)]
    assert intents
    residual = abs(strategy._residual["COPY"])
    needed = 0.3 + 0.1  # half the 0.6 spread, plus one tick of edge
    assert intents[0].price - state.tops["COPY"].ask <= residual - needed + 1e-9


def test_a_follower_that_stays_off_is_guarded_only_once_until_it_is_back_in_line():
    """A pair that stays apart, as a false pair would, must not keep the follower unquoted."""
    strategy, state, moves = warmed_up({"lag_guard_s": 0.5})
    jump_leader(state, moves, 900, 8.0)
    assert pulls(strategy.on_tick(state)) == ["COPY"]
    for _ in range(4):  # still off, 0.3 s at a time
        state.now += 0.3
        last = pulls(strategy.on_tick(state))
    assert last == [], "the guard ended 0.5 s after it began"
    in_line(state, moves, 900, 0.3)
    strategy.on_tick(state)
    jump_leader(state, moves, 900, 8.0)
    assert pulls(strategy.on_tick(state)) == ["COPY"], "a new jump after the follower came back guards again"


def test_switching_the_hedge_on_does_not_hedge_what_was_taken_while_it_was_off():
    strategy, state, _moves = warmed_up({"lag_take_enabled": 1})
    take_filled(state, 20)
    strategy.on_tick(state)
    state.params = {**state.params, "lag_hedge": 1}
    state.now += 0.1
    assert hedges(strategy.on_tick(state)) == [], "the 20 lots taken while off stay unhedged"
    take_filled(state, 25)
    assert sum(h.volume for h in hedges(strategy.on_tick(state))) == round(strategy.links["COPY"].beta * 5)


def fresh(params):
    """No pairs learned yet, as for the first 300 s after every start."""
    state = State(params)
    strategy = LeadLag(["ALONE", "COPY", "LEAD"])
    state.tops = {"LEAD": quoted(200.0), "COPY": quoted(90.0), "ALONE": quoted(80.0)}
    strategy.on_tick(state)
    assert strategy.links == {}
    return strategy, state


def test_the_warm_up_guard_is_off_by_default():
    assert defaults()["lag_warmup_ticks"] == 0
    strategy, state = fresh({"lag_guard_s": 2.5})
    state.now += 1.0
    state.tops["LEAD"] = quoted(188.0)
    assert pulls(strategy.on_tick(state)) == []


def test_before_any_pair_is_learned_a_jump_pulls_every_other_stock():
    """18:13: NVDA gapped 78 s after a restart, and about 220 lots were dumped into our unguarded SNDK bids."""
    strategy, state = fresh({"lag_warmup_ticks": 10, "lag_guard_s": 2.5})
    state.now += 1.0
    state.tops["LEAD"] = quoted(198.8)  # 12 ticks
    assert sorted(pulls(strategy.on_tick(state))) == ["ALONE", "COPY"]
    state.now += 2.0
    assert sorted(pulls(strategy.on_tick(state))) == ["ALONE", "COPY"], "still inside lag_guard_s"
    state.now += 1.0
    assert pulls(strategy.on_tick(state)) == []


def test_a_move_under_the_warm_up_threshold_pulls_nothing():
    strategy, state = fresh({"lag_warmup_ticks": 10, "lag_guard_s": 2.5})
    state.now += 1.0
    state.tops["LEAD"] = quoted(199.5)  # 5 ticks
    assert pulls(strategy.on_tick(state)) == []


def test_once_the_pairs_are_learned_the_warm_up_guard_stands_down():
    strategy, state, moves = warmed_up({"lag_warmup_ticks": 10, "lag_guard_s": 2.5})
    state.now += 1.0
    state.tops = {**state.tops, "ALONE": quoted(moves["ALONE"][900] + 3.0)}  # an unpaired stock jumps
    assert pulls(strategy.on_tick(state)) == []


def test_a_reset_forgets_prices_but_keeps_the_pairs():
    lag = LeadLag(["LEAD", "COPY"])
    lag.links = {"COPY": SimpleNamespace(leader="LEAD", beta=0.4)}
    lag._mean["COPY"] = 12.3
    lag._history["LEAD"].extend([200.0] * 50)
    lag.on_reset()
    assert lag._mean == {} and not lag._history["LEAD"]
    assert "COPY" in lag.links
    lag._learn(10_000.0)  # too little new history: the pairs stay
    assert "COPY" in lag.links


def open_state(now, leader, follower):
    values = {**defaults(), "lag_take_enabled": 1, "lag_edge_ticks": 1, "lag_clip": 30}
    tops = {
        "LEAD": Top(round(leader - 0.1, 1), 3000, round(leader + 0.1, 1), 3000),
        "COPY": Top(round(follower - 0.1, 1), 1000, round(follower + 0.1, 1), 1000),
    }
    return SimpleNamespace(
        now=now, params=values, top=tops.get, position=lambda iid: 0, tick_of=lambda iid: 0.1, source_positions={},
    )


def seeded_state(now, tops, **params):
    values = {**defaults(), "lag_seed": 1, "lag_take_enabled": 1, "lag_edge_ticks": 1, "lag_clip": 30, **params}
    return SimpleNamespace(
        now=now, params=values, top=tops.get, position=lambda iid: 0, tick_of=lambda iid: 0.1, source_positions={},
    )


def paired_after_a_reset():
    lag = LeadLag(["LEAD", "COPY"])
    lag.links = {"COPY": LagLink("LEAD", 1.0, 1.0)}
    lag.on_reset()
    return lag


def test_a_new_round_opening_on_a_thin_book_is_not_traded():
    """At 20:10:30 the baseline came from the pre-open book, and the open sold 60 CRWV in a second."""
    lag = paired_after_a_reset()
    assert lag.on_tick(open_state(100.0, 75.6, 140.7)) == []
    assert lag.on_tick(open_state(106.5, 76.2, 141.4)) == []
    assert lag.on_tick(open_state(106.6, 76.2, 140.7)) == []
    assert lag._mean == {}


def test_the_baseline_starts_after_the_open_so_a_real_lag_still_trades():
    lag = paired_after_a_reset()
    lag.on_tick(open_state(100.0, 75.6, 140.7))
    lag.on_tick(open_state(116.0, 76.2, 141.3))
    takes = [i for i in lag.on_tick(open_state(117.0, 74.2, 141.3)) if isinstance(i, Take)]
    assert [(t.instrument_id, t.side) for t in takes] == [("COPY", "ask")]


def test_a_start_with_no_reset_does_not_wait():
    lag = LeadLag(["LEAD", "COPY"])
    lag.links = {"COPY": LagLink("LEAD", 1.0, 1.0)}
    lag._next_learn = 10_000.0
    lag.on_tick(open_state(100.0, 75.6, 140.7))
    assert "COPY" in lag._mean
def stocks():
    return ["NVDA", "SNDK", "AVGO", "NBIS", "MU", "CRWV", "AMD", "VRT"]


def calm(**moved):
    prices = {"NVDA": 125.0, "SNDK": 54.0, "AVGO": 200.0, "NBIS": 92.0, "MU": 75.0, "CRWV": 140.0, "AMD": 170.0, "VRT": 66.0}
    prices.update(moved)
    return {iid: Top(round(p - 0.1, 1), 10, round(p + 0.1, 1), 10) for iid, p in prices.items()}


def test_seeding_is_off_unless_asked_for():
    lag = LeadLag(stocks())
    lag.on_tick(seeded_state(0.0, calm(), lag_seed=0))
    assert lag.links == {}


def test_seeded_pairs_hold_from_the_first_tick():
    lag = LeadLag(stocks())
    lag.on_tick(seeded_state(0.0, calm()))
    assert lag.links == KNOWN_LINKS


def test_a_seeded_follower_is_taken_on_the_first_leader_jump_after_a_start():
    """The big drop lands 78 s into a round, before anything could be learned."""
    lag = LeadLag(stocks())
    lag.on_tick(seeded_state(0.0, calm()))
    intents = lag.on_tick(seeded_state(1.0, calm(NVDA=110.0)))
    takes = [i for i in intents if isinstance(i, Take)]
    assert [(t.instrument_id, t.side) for t in takes] == [("SNDK", "ask")]


def test_seeds_skip_stocks_that_are_not_listed():
    lag = LeadLag(["NVDA", "SNDK", "AMD"])
    lag.on_tick(seeded_state(0.0, calm()))
    assert set(lag.links) == {"SNDK"}


def test_relearning_does_not_forget_the_slow_mean_of_a_seeded_pair():
    lag = LeadLag(stocks())
    lag.on_tick(seeded_state(0.0, calm()))
    mean = lag._mean["SNDK"]
    lag.on_tick(seeded_state(31.0, calm()))
    assert lag.links["SNDK"] == KNOWN_LINKS["SNDK"]
    assert lag._mean["SNDK"] == pytest.approx(mean)


def test_a_learned_ratio_does_not_displace_a_seed():
    lag = LeadLag(stocks())
    lag.links = {"SNDK": LagLink("NVDA", 0.2, 0.95)}
    lag.on_tick(seeded_state(0.0, calm()))
    assert lag.links["SNDK"].beta == KNOWN_LINKS["SNDK"].beta
