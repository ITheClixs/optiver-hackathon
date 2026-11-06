"""A wrong volatility makes every option look mispriced at once, and a wrong limit ends in rejects. Neither may run unchecked."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from otk.engine import Engine, EngineConfig
from otk.fair import Top
from otk.formats import plan_event
from otk.instruments import Universe, from_exchange_instrument
from otk.main import engine_config, option_vol_from
from otk.optioncheck import CHECK_MIN_SAMPLES as MIN_SAMPLES
from otk.optioncheck import CHECK_WINDOW_S as WINDOW_S
from otk.optioncheck import OptionMarketCheck
from otk.params import ParamStore, merge_params
from otk.risk import Limits
from otk.sim.exchange import SimExchange, SimMarket
from otk.strategies import OptionTheo, default_stack

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 6, 16, tzinfo=timezone.utc)
EXPIRY = datetime(2027, 3, 19, 12)


def listed(iid, kind="InstrumentType.STOCK", **extra):
    return SimpleNamespace(instrument_id=iid, tick_size=0.1, instrument_type=kind, expired=False, **extra)


def option(iid, base="AAA", strike=100.0, kind="OptionKind.CALL", **extra):
    return listed(
        iid, "InstrumentType.STOCK_OPTION", base_instrument_id=base, strike=strike, expiry=EXPIRY, option_kind=kind, **extra
    )


def index_option(iid="IDX_C"):
    return listed(
        iid, "InstrumentType.INDEX_OPTION", index_id="IDX", index_constituents={"AAA": 1.0}, index_divisor=1.0, strike=100.0,
        expiry=EXPIRY, option_kind="OptionKind.CALL", index_volatility=0.3,
    )


def board(*instruments):
    return Universe.from_instruments({item.instrument_id: item for item in instruments})


def stocks():
    return listed("AAA"), listed("BBB")


def quoted_by(stack, name):
    return {iid for item in stack if item.name == name for iid in item.instruments}


class State:
    def __init__(self, universe):
        self.universe, self.now, self.books = universe, 0.0, {}

    def top(self, iid):
        return self.books.get(iid)

    def external(self, _iid):
        return None


def book(center, half=0.5):
    return Top(center - half, 100, center + half, 100)


def checked(prices):
    """A check over a fixed theo. `prices` holds what theo says now, so a test can move it."""
    return OptionMarketCheck(lambda _state, iid: prices.get(iid), {"AAA_C": "AAA", "AAA_P": "AAA", "BBB_C": "BBB"})


def state_with(**books):
    state = State(board(*stocks(), option("AAA_C"), option("AAA_P"), option("BBB_C", base="BBB")))
    state.books = {iid: book(center) for iid, center in books.items()}
    return state


def run(check, state, seconds, iid="AAA_C"):
    last = None
    for _ in range(seconds):
        state.now += 1.0
        last = check(state, iid)
    return last


def test_no_volatility_means_stock_options_stay_out():
    universe = board(*stocks(), option("AAA_C"))
    stack = default_stack(sorted(universe.by_id), universe=universe, vol=None)
    assert quoted_by(stack, "theo_quoter") == set()
    assert "AAA_C" not in quoted_by(stack, "market_maker")


def test_with_no_volatility_it_does_not_price_a_stock_option():
    universe = board(*stocks(), option("AAA_C"))
    state = State(universe)
    state.books = {"AAA": book(100.0)}
    assert OptionTheo(vol=None, calendar_now=lambda: NOW)(state, "AAA_C") is None


def test_an_index_option_needs_no_guess_because_the_client_gives_its_volatility():
    universe = board(*stocks(), option("AAA_C"), index_option())
    stack = default_stack(sorted(universe.by_id), universe=universe, vol=None)
    assert quoted_by(stack, "theo_quoter") == {"IDX_C"}


def test_a_stock_with_options_is_still_hedged_with_no_volatility_set():
    universe = board(*stocks(), option("AAA_C"))
    stack = default_stack(sorted(universe.by_id), universe=universe, vol=None)
    hedger = next(item for item in stack if item.name == "hedger")
    assert hedger.vol > 0


def test_a_volatility_must_be_a_positive_number():
    assert option_vol_from({"option_vol": 1.5}) == 1.5
    assert option_vol_from({"option_vol": "1.5"}) == 1.5
    assert option_vol_from({}) is None
    assert option_vol_from({"option_vol": 0}) is None
    assert option_vol_from({"option_vol": -1}) is None
    assert option_vol_from({"option_vol": "high"}) is None
    assert option_vol_from({"option_vol": True}) is None


def test_no_preset_ships_a_volatility_guess():
    for path in (ROOT / "config" / "presets").glob("*.json"):
        assert "option_vol" not in json.loads(path.read_text()), path.name


def test_options_are_shut_until_the_check_has_watched_long_enough():
    check = checked({"AAA_C": 10.0})
    state = state_with(AAA_C=10.0)
    assert run(check, state, MIN_SAMPLES - 1) is None
    assert run(check, state, 2) == 10.0


def test_a_price_outside_the_book_for_most_of_the_window_shuts_them():
    prices = {"AAA_C": 10.0}
    check = checked(prices)
    state = state_with(AAA_C=10.0)
    assert run(check, state, int(WINDOW_S)) == 10.0
    prices["AAA_C"] = 15.0
    assert run(check, state, int(WINDOW_S)) is None


def test_the_stale_seconds_after_a_jump_are_the_edge_and_do_not_shut_them():
    prices = {"AAA_C": 10.0}
    check = checked(prices)
    state = state_with(AAA_C=10.0)
    run(check, state, int(WINDOW_S))
    prices["AAA_C"] = 15.0
    assert run(check, state, 5) == 15.0


def test_it_opens_again_once_the_prices_agree():
    prices = {"AAA_C": 15.0}
    check = checked(prices)
    state = state_with(AAA_C=10.0)
    assert run(check, state, int(WINDOW_S)) is None
    prices["AAA_C"] = 10.0
    assert run(check, state, int(WINDOW_S)) == 10.0


def test_one_underlying_does_not_shut_another():
    check = checked({"AAA_C": 15.0, "BBB_C": 10.0})
    state = state_with(AAA_C=10.0, BBB_C=10.0)
    run(check, state, int(WINDOW_S))
    assert check(state, "AAA_C") is None
    assert run(check, state, int(WINDOW_S), "BBB_C") == 10.0


def test_empty_books_keep_them_shut():
    check = checked({"AAA_C": 10.0})
    assert run(check, state_with(), int(WINDOW_S) * 2) is None


def test_anything_but_an_option_passes_straight_through():
    assert checked({"AAA_F": 12.0})(state_with(), "AAA_F") == 12.0


def test_the_log_says_when_and_why_it_stopped(caplog):
    check = checked({"AAA_C": 15.0})
    state = state_with(AAA_C=10.0)
    with caplog.at_level("WARNING", logger="otk.optioncheck"):
        run(check, state, int(WINDOW_S) * 2)
    assert "AAA" in caplog.text
    assert caplog.text.count("option_vol") == 1, "once, not every second"


def option_engine(vol):
    """The real engine. The option's book sits where a volatility of 0.25 would put it, at about 7."""
    instruments = {item.instrument_id: item for item in (*stocks(), option("AAA_C"))}
    market = SimMarket(instruments=list(instruments))

    class Exchange(SimExchange):
        def get_instruments(self):
            return instruments

    for iid, center in (("AAA", 100.0), ("BBB", 50.0), ("AAA_C", 7.0)):
        maker = SimExchange(market, "maker", bot=True)
        maker.insert_order(iid, price=round(center - 1.5, 1), volume=100, side="bid", order_type="limit")
        maker.insert_order(iid, price=round(center + 1.5, 1), volume=100, side="ask", order_type="limit")
    exchange = Exchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    stack = default_stack(sorted(universe.by_id), universe=universe, vol=vol)
    engine = Engine(exchange, universe, stack, None, ParamStore(), None, clock=lambda: market.now)
    for _ in range(int(WINDOW_S) + 5):
        market.now += 1.0
        engine.tick()
    return exchange


def test_a_wrong_volatility_never_trades_an_option_on_the_real_engine():
    exchange = option_engine(vol=3.0)
    assert exchange.get_positions().get("AAA_C", 0) == 0
    assert exchange.get_outstanding_orders("AAA_C") == {}


def test_the_right_volatility_quotes_it():
    exchange = option_engine(vol=0.25)
    assert {order.side for order in exchange.get_outstanding_orders("AAA_C").values()} == {"bid", "ask"}


def test_options_start_small_and_the_team_can_raise_them():
    plan = plan_event(board(*stocks(), option("AAA_C")), {"option_vol": 0.25})
    assert plan.params["instruments"]["AAA_C"]["size"] == 2
    merged = merge_params(plan.params, {"instruments": {"AAA_C": {"size": 5}}})
    assert merged["instruments"]["AAA_C"]["size"] == 5


def test_the_card_says_what_happens_without_a_volatility():
    plan = plan_event(board(*stocks(), option("AAA_C")), {"position_limit": 100})
    assert any("option_vol" in item and "stay out" in item for item in plan.missing)
    assert any("size 2" in note for note in plan.notes)


def test_the_card_asks_for_the_position_limit_until_it_is_set():
    assert any("position_limit" in item for item in plan_event(board(*stocks())).missing)
    assert not any("position_limit" in item for item in plan_event(board(*stocks()), {"position_limit": 300}).missing)


def limit_engine(tmp_path, base, config):
    market = SimMarket(instruments=["AAA"])
    exchange = SimExchange(market, "us")
    exchange.connect()
    universe = Universe([from_exchange_instrument(listed("AAA"))])
    path = tmp_path / "params.json"
    return Engine(exchange, universe, [], config, ParamStore(path, base=base), None, clock=lambda: market.now), path


def test_the_live_config_follows_a_published_limit(tmp_path):
    assert engine_config({"position_limit": 100}, tmp_path, None).follow_limit_param


def test_a_published_limit_reaches_the_gate_with_no_restart(tmp_path):
    config = EngineConfig(limits=Limits(position=100), follow_limit_param=True)
    engine, path = limit_engine(tmp_path, {"position_limit": 100}, config)
    engine.tick()
    assert engine.manager.limits.position == 100
    path.write_text(json.dumps({"position_limit": 300}))
    engine.tick()
    assert engine.manager.limits.position == 300
    path.write_text(json.dumps({"position_limit": 50}))
    engine.tick()
    assert engine.manager.limits.position == 50


def test_a_limit_already_published_when_the_bot_restarts_still_counts(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"position_limit": 50}))
    config = EngineConfig(limits=Limits(position=100), follow_limit_param=True)
    engine, _path = limit_engine(tmp_path, {"position_limit": 100}, config)
    engine.tick()
    assert engine.manager.limits.position == 50


def test_a_gate_set_in_a_replay_or_sim_config_is_left_alone(tmp_path):
    engine, _path = limit_engine(tmp_path, None, EngineConfig(limits=Limits(position=10)))
    engine.tick()
    assert engine.manager.limits.position == 10
