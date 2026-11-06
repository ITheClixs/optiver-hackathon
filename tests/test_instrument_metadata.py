"""The client says what each instrument is. The bot must believe it, whatever the instrument is called.

The enums and fields below copy the event's published `optibook.common_types`.
"""

from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace

import pytest

from otk.engine import Engine
from otk.fair import Top, bs_price, future_fair
from otk.formats import apply_pairs, detect_pairs, plan_event
from otk.instruments import (
    Kind,
    Universe,
    delta_per_lot,
    from_exchange_instrument,
    theo_ready,
    years_to_expiry,
)
from otk.params import ParamStore
from otk.sim.exchange import SimExchange, SimMarket
from otk.strategies import IndexTheo, OptionTheo, default_stack

NOW = datetime(2026, 11, 6, 16, tzinfo=timezone.utc)
EXPIRY = datetime(2027, 3, 19, 12)
WEIGHTS = {"AAA": 2.0, "BBB": 3.0}


class InstrumentType(Enum):
    STOCK = 1
    STOCK_OPTION = 2
    STOCK_FUTURE = 3
    INDEX_TRACKING_ETF = 4
    INDEX_OPTION = 5
    INDEX_FUTURE = 6


class OptionKind(Enum):
    PUT = 1
    CALL = 2


def listed(iid, kind=InstrumentType.STOCK, **extra):
    return SimpleNamespace(instrument_id=iid, tick_size=0.1, instrument_type=kind, expired=False, **extra)


def stock_option(iid="AAA_CALL_MAR", base="AAA", option_kind=OptionKind.CALL, **extra):
    fields = {"expiry": EXPIRY, "strike": 100.0, "base_instrument_id": base, "option_kind": option_kind, **extra}
    kind = fields.pop("instrument_type", InstrumentType.STOCK_OPTION)
    return listed(iid, kind, **fields)


def index_fields(**extra):
    return {"index_id": "TECHX", "index_constituents": dict(WEIGHTS), "index_divisor": 10.0, **extra}


def index_etf(iid="TRACKER", **extra):
    return listed(iid, InstrumentType.INDEX_TRACKING_ETF, etf_multiplier=0.5, etf_cash_comp=1.0, **index_fields(**extra))


def index_future(iid="TECHX_MAR"):
    return listed(iid, InstrumentType.INDEX_FUTURE, expiry=EXPIRY, interest_rate=0.05, **index_fields())


def index_option(iid="TECHX_C50"):
    return listed(
        iid, InstrumentType.INDEX_OPTION, expiry=EXPIRY, strike=50.0, option_kind=OptionKind.CALL, interest_rate=0.05,
        index_volatility=1.5, **index_fields(),
    )


def board(*instruments):
    return Universe.from_instruments({item.instrument_id: item for item in instruments})


def stocks():
    return listed("AAA"), listed("BBB")


class State:
    """Books at the prices given, with equal size, so the microprice is the mid."""

    def __init__(self, universe, prices):
        self.universe, self.prices = universe, prices

    def top(self, iid):
        price = self.prices.get(iid)
        return None if price is None else Top(price - 0.1, 100, price + 0.1, 100)

    def external(self, _iid):
        return None


def option_theo(vol=0.25):
    return OptionTheo(vol=vol, rate=0.03, calendar_now=lambda: NOW)


def quoted_by(stack, name):
    return {iid for item in stack if item.name == name for iid in item.instruments}


def test_an_option_is_an_option_whatever_it_is_called():
    info = from_exchange_instrument(stock_option())
    assert (info.kind, info.base, info.strike, info.option_kind) == (Kind.OPTION, "AAA", 100.0, "call")
    assert info.expiry == EXPIRY.replace(tzinfo=timezone.utc)
    assert from_exchange_instrument(stock_option(option_kind=OptionKind.PUT)).option_kind == "put"


def test_a_future_is_a_future_whatever_it_is_called():
    info = from_exchange_instrument(listed("AAA_MAR", InstrumentType.STOCK_FUTURE, expiry=EXPIRY, base_instrument_id="AAA"))
    assert (info.kind, info.base) == (Kind.FUTURE, "AAA")
    assert from_exchange_instrument(index_future()).kind is Kind.FUTURE


def test_the_type_may_arrive_as_text():
    info = from_exchange_instrument(stock_option(instrument_type="InstrumentType.STOCK_OPTION", option_kind="OptionKind.PUT"))
    assert (info.kind, info.option_kind) == (Kind.OPTION, "put")


def test_a_stock_stays_a_stock_and_a_thin_client_still_parses_by_name():
    assert from_exchange_instrument(listed("AAA")).kind is Kind.STOCK
    info = from_exchange_instrument(SimpleNamespace(instrument_id="AAA_202703_100C", tick_size=0.1))
    assert (info.kind, info.base, info.strike, info.option_kind) == (Kind.OPTION, "AAA", 100.0, "call")


def test_a_named_option_takes_the_clients_facts_over_its_name():
    info = from_exchange_instrument(
        stock_option("AAA_202703_100C", base="ZZZ", strike=120.0, option_kind=OptionKind.PUT)
    )
    assert (info.base, info.strike, info.option_kind) == ("ZZZ", 120.0, "put")


def test_an_expiry_that_is_not_a_date_is_ignored():
    assert from_exchange_instrument(stock_option(expiry="soon")).expiry is None


@pytest.mark.parametrize(
    "fields",
    [
        {"index_constituents": {}},
        {"index_constituents": None},
        {"index_constituents": {"AAA": "two"}},
        {"index_divisor": 0},
        {"index_divisor": None},
    ],
)
def test_unusable_index_facts_are_dropped(fields):
    assert from_exchange_instrument(index_etf(**fields)).index is None


def test_a_stock_option_is_priced_from_its_underlying():
    universe = board(*stocks(), stock_option())
    info = universe.by_id["AAA_CALL_MAR"]
    expected = bs_price("call", 110.0, 100.0, years_to_expiry(info.expiry, NOW), 0.03, 0.25)
    assert option_theo()(State(universe, {"AAA": 110.0}), "AAA_CALL_MAR") == pytest.approx(expected)


def test_an_option_with_no_underlying_book_has_no_price():
    universe = board(*stocks(), stock_option())
    assert option_theo()(State(universe, {}), "AAA_CALL_MAR") is None


def test_an_etf_is_priced_from_the_clients_index_with_no_weights_in_the_event():
    universe = board(*stocks(), index_etf())
    level = (2.0 * 10.0 + 3.0 * 20.0) / 10.0
    theo = option_theo()(State(universe, {"AAA": 10.0, "BBB": 20.0}), "TRACKER")
    assert theo == pytest.approx(1.0 + 0.5 * level)


def test_index_products_use_the_clients_rate_and_volatility():
    universe = board(*stocks(), index_future(), index_option())
    state = State(universe, {"AAA": 10.0, "BBB": 20.0})
    tau = years_to_expiry(universe.by_id["TECHX_MAR"].expiry, NOW)
    assert option_theo()(state, "TECHX_MAR") == pytest.approx(future_fair(8.0, 0.05, tau))
    assert option_theo(vol=3.0)(state, "TECHX_C50") == pytest.approx(bs_price("call", 8.0, 50.0, tau, 0.05, 1.5))


def test_a_missing_constituent_price_gives_no_price():
    universe = board(*stocks(), index_etf())
    assert option_theo()(State(universe, {"AAA": 10.0}), "TRACKER") is None


def test_weights_in_the_event_outrank_the_client_and_keep_its_divisor_and_etf_terms():
    universe = board(*stocks(), index_etf())
    theo = IndexTheo({"AAA": 1.0}, calendar_now=lambda: NOW)
    assert theo(State(universe, {"AAA": 10.0, "BBB": 20.0}), "TRACKER") == pytest.approx(1.0 + 0.5 * 1.0)


def test_weights_in_the_event_work_for_an_index_that_is_not_called_ob5x():
    etf = listed("TRACKER", InstrumentType.INDEX_TRACKING_ETF, index_id="TECHX")
    universe = board(*stocks(), etf)
    theo = IndexTheo({"AAA": 100.0}, calendar_now=lambda: NOW)
    assert theo(State(universe, {"AAA": 10.0}), "TRACKER") == pytest.approx(2.5 + 0.25 * 1.0)


def test_the_stack_quotes_them_on_theo_and_only_stocks_on_the_book():
    universe = board(*stocks(), stock_option(), index_etf(), index_future(), index_option())
    stack = default_stack(sorted(universe.by_id), universe=universe, vol=3.0)
    assert quoted_by(stack, "theo_quoter") == {"AAA_CALL_MAR", "TRACKER", "TECHX_MAR", "TECHX_C50"}
    assert quoted_by(stack, "market_maker") == {"AAA", "BBB"}
    assert quoted_by(stack, "lead_lag") == {"AAA", "BBB"}


def test_an_option_it_cannot_price_is_left_alone():
    universe = board(*stocks(), stock_option(strike=None), stock_option("ORPHAN", base=None))
    stack = default_stack(sorted(universe.by_id), universe=universe)
    assert quoted_by(stack, "theo_quoter") == set()
    assert quoted_by(stack, "market_maker") == {"AAA", "BBB"}
    assert not theo_ready(universe.by_id["AAA_CALL_MAR"], universe.by_id)
    assert not theo_ready(universe.by_id["ORPHAN"], universe.by_id)


def test_an_etf_with_a_constituent_not_listed_quotes_off_its_own_book():
    universe = board(listed("AAA"), index_etf())
    stack = default_stack(sorted(universe.by_id), universe=universe)
    assert "TRACKER" not in quoted_by(stack, "theo_quoter")
    assert "TRACKER" in quoted_by(stack, "market_maker")


def test_a_derivative_with_no_underlying_is_never_hedged_on_itself():
    universe = board(*stocks(), listed("LONELY_MAR", InstrumentType.STOCK_FUTURE, expiry=EXPIRY))
    stack = default_stack(sorted(universe.by_id), universe=universe)
    assert not any("LONELY_MAR" in item.hedge_on.values() for item in stack if item.name == "hedger")


def test_a_derivative_never_becomes_a_second_listing():
    universe = board(*stocks(), stock_option("OPT_A"), stock_option("OPT_B"))
    applied = apply_pairs(universe, detect_pairs(universe.by_id))
    assert applied.by_id["OPT_A"].kind is Kind.OPTION
    assert applied.by_id["OPT_B"].kind is Kind.OPTION
    assert applied.dual_pairs() == []


def test_a_real_pair_of_stocks_still_pairs():
    universe = board(listed("AAA"), listed("AAA_DUAL"))
    assert apply_pairs(universe, detect_pairs(universe.by_id)).dual_pairs() == [("AAA", "AAA_DUAL")]


def test_the_card_stops_asking_for_what_the_client_gave():
    plan = plan_event(board(*stocks(), index_etf(), index_future(), index_option()))
    assert plan.name == "full_board"
    assert not any("index_weights" in item for item in plan.missing)
    assert not any("option_vol" in item for item in plan.missing)
    assert any("client" in note for note in plan.notes)


def test_the_card_still_asks_when_the_client_did_not_give_it():
    etf = listed("TRACKER", InstrumentType.INDEX_TRACKING_ETF, index_id="TECHX")
    plan = plan_event(board(*stocks(), etf, stock_option()))
    assert any("index_weights" in item for item in plan.missing)
    assert any("option_vol" in item for item in plan.missing)


def test_delta_uses_the_clients_volatility_and_rate():
    info = board(*stocks(), index_option()).by_id["TECHX_C50"]
    client = delta_per_lot(info, 50.0, 1.5, 0.05, NOW)
    assert delta_per_lot(info, 50.0, 3.0, 0.9, NOW) == pytest.approx(client)


def test_delta_of_an_option_missing_its_terms_does_not_crash():
    info = from_exchange_instrument(stock_option(strike=None, expiry=None))
    assert delta_per_lot(info, 50.0, 0.25, 0.03, NOW) == 0.0


def test_the_real_engine_quotes_a_board_with_names_it_has_never_seen():
    instruments = {item.instrument_id: item for item in (*stocks(), index_etf(), stock_option("TICKER77", strike=10.0))}
    market = SimMarket(instruments=list(instruments))

    class Exchange(SimExchange):
        def get_instruments(self):
            return instruments

    for iid, center in (("AAA", 10.0), ("BBB", 20.0), ("TRACKER", 5.0), ("TICKER77", 0.6)):
        maker = SimExchange(market, "maker", bot=True)
        maker.insert_order(iid, price=round(center - 0.5, 1), volume=100, side="bid", order_type="limit")
        maker.insert_order(iid, price=round(center + 0.5, 1), volume=100, side="ask", order_type="limit")
    exchange = Exchange(market, "us")
    exchange.connect()
    universe = Universe.from_instruments(exchange.get_instruments())
    stack = default_stack(sorted(universe.by_id), universe=universe)
    engine = Engine(exchange, universe, stack, None, ParamStore(), None, clock=lambda: market.now)
    for _ in range(25):  # an option waits for the check that its price agrees with the market
        market.now += 1.0
        engine.tick()
    assert quoted_by(stack, "theo_quoter") == {"TRACKER", "TICKER77"}
    tracker = exchange.get_outstanding_orders("TRACKER").values()
    assert {order.side for order in tracker} == {"bid", "ask"}
    assert max(o.price for o in tracker if o.side == "bid") < 5.0 < min(o.price for o in tracker if o.side == "ask")
    assert {order.side for order in exchange.get_outstanding_orders("TICKER77").values()} == {"bid", "ask"}
