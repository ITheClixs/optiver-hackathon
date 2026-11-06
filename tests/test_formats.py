"""The main algo's first decision: what kind of event is this? Instrument lists are from past events."""

from types import SimpleNamespace

import pytest

from otk.formats import apply_pairs, choose_liquid_leg, detect_pairs, plan_event, render_plan
from otk.instruments import Kind, Universe, parse_instrument_id
from otk.strategies import default_stack


def universe(*ids):
    return Universe(parse_instrument_id(iid) for iid in ids)


FIVE = ("NVDA", "ING", "SAN", "PFE", "CSCO")
OB5X = (
    "ASML", "AAPL", "SAP", "TSLA", "NVDA", "ASML_DUAL", "SAP_DUAL", "OB5X_ETF", "OB5X_202703_F",
    "ASML_202703_F", "ASML_202703_100C", "ASML_202703_100P",
)


@pytest.mark.parametrize(
    "ids,pairs",
    [
        (("ASML", "ASML_DUAL"), [("ASML", "ASML_DUAL")]),
        (("PHILIPS_A", "PHILIPS_B"), [("PHILIPS_A", "PHILIPS_B")]),
        (("SMALL_CHIPS", "SMALL_CHIPS_NEW_COUNTRY"), [("SMALL_CHIPS", "SMALL_CHIPS_NEW_COUNTRY")]),
        (("SEMIS_ETF_US", "SEMIS_ETF_EU", "NVDA", "AMD"), [("SEMIS_ETF_US", "SEMIS_ETF_EU")]),
        (FIVE, []),
        (("A_B", "C_D"), []),
    ],
)
def test_pairs_are_found_however_the_event_names_them(ids, pairs):
    assert detect_pairs(ids) == pairs


def test_a_shared_group_makes_a_pair_when_names_do_not():
    groups = {"ALPHA": "G1", "BETA": "G1", "GAMMA": "G2"}
    assert detect_pairs(("ALPHA", "BETA", "GAMMA"), groups) == [("ALPHA", "BETA")]


def test_a_group_with_derivatives_is_not_a_dual_listing():
    ids = ("NVDA", "NVDA_202703_F", "NVDA_202703_100C")
    assert detect_pairs(ids, dict.fromkeys(ids, "NVDA")) == []


def test_the_liquid_leg_becomes_primary_from_the_live_book():
    books = {
        "PHILIPS_A": SimpleNamespace(bids=[SimpleNamespace(price=99.0, volume=5)], asks=[SimpleNamespace(price=101.0, volume=5)]),
        "PHILIPS_B": SimpleNamespace(bids=[SimpleNamespace(price=99.9, volume=500)], asks=[SimpleNamespace(price=100.1, volume=500)]),
    }
    assert choose_liquid_leg(("PHILIPS_A", "PHILIPS_B"), books) == ("PHILIPS_B", "PHILIPS_A")


def test_an_empty_book_leaves_the_pair_as_named():
    assert choose_liquid_leg(("A", "A_DUAL"), {"A": None, "A_DUAL": None}) == ("A", "A_DUAL")


def test_applying_pairs_groups_the_listings_for_delta_and_arb():
    applied = apply_pairs(universe("PHILIPS_A", "PHILIPS_B"), [("PHILIPS_A", "PHILIPS_B")])
    assert applied.by_id["PHILIPS_B"].kind is Kind.DUAL
    assert applied.by_id["PHILIPS_B"].base == "PHILIPS_A"
    assert applied.dual_pairs() == [("PHILIPS_A", "PHILIPS_B")]
    assert sorted(applied.groups["PHILIPS_A"]) == ["PHILIPS_A", "PHILIPS_B"]


@pytest.mark.parametrize(
    "ids,name",
    [
        (("PHILIPS_A", "PHILIPS_B"), "dual_listing"),
        (FIVE + tuple(f"{s}_DUAL" for s in FIVE), "dual_listing"),
        (FIVE, "stocks"),
        (("TECH_BASKET", "AAA", "BBB"), "basket"),
        (("SEMIS_ETF_US", "SEMIS_ETF_EU", "NVDA", "AMD", "ASML"), "dual_listing"),
        (OB5X, "full_board"),
    ],
)
def test_the_format_is_named_from_what_is_listed(ids, name):
    assert plan_event(universe(*ids)).name == name


def test_the_plan_says_which_strategies_run_for_dual_listing():
    plan = plan_event(universe("PHILIPS_A", "PHILIPS_B"))
    assert plan.pairs == (("PHILIPS_A", "PHILIPS_B"),)
    assert {"dual_listing", "gap_reversion", "hedger", "news_trader"} <= set(plan.strategies)


def test_the_plan_lists_what_the_team_must_still_supply():
    plan = plan_event(universe(*OB5X))
    assert any("index_weights" in item for item in plan.missing)
    assert not any("index_weights" in item for item in plan_event(universe(*OB5X), {"index_weights": {"ASML": 1.0}}).missing)


def test_a_full_board_saves_write_budget():
    assert plan_event(universe(*OB5X)).params["reprice_ticks"] > 1
    assert "reprice_ticks" not in plan_event(universe("A", "A_DUAL")).params


def test_the_liquid_leg_quotes_wider_where_real_books_support_it():
    plan = plan_event(universe("ASML", "ASML_DUAL"))
    assert plan.params["instruments"]["ASML"]["base_spread_ticks"] == 4.0
    assert "ASML_DUAL" not in plan.params["instruments"]


def test_a_fine_tick_gets_no_preset_because_we_have_no_evidence_there():
    fine = Universe(
        [parse_instrument_id("ASML")._replace(tick=0.01), parse_instrument_id("ASML_DUAL")._replace(tick=0.01)]
    )
    assert "instruments" not in plan_event(fine).params


def test_a_news_feed_changes_the_plan_and_its_notes():
    assert "news" in plan_event(universe(*FIVE), has_news_feed=True).notes[0]


def test_render_is_a_card_a_person_can_read_in_seconds():
    text = "\n".join(render_plan(plan_event(universe(*OB5X))))
    assert "full_board" in text
    assert "index_weights" in text


def test_the_stack_trades_a_pair_the_exchange_named_its_own_way():
    applied = apply_pairs(universe("PHILIPS_A", "PHILIPS_B"), [("PHILIPS_A", "PHILIPS_B")])
    stack = default_stack(sorted(applied.by_id), universe=applied)
    names = [getattr(item, "name", "") for item in stack]
    assert "dual_listing" in names
    assert "gap_reversion" in names


def test_an_etf_without_weights_is_still_quoted_off_its_own_book():
    ids = ("OB5X_ETF", "AAA", "BBB")
    stack = default_stack(list(ids))
    quoted = {iid for item in stack for iid in getattr(item, "instruments", [])}
    assert "OB5X_ETF" in quoted


def test_exchange_metadata_names_an_etf_the_id_does_not():
    from otk.instruments import from_exchange_instrument

    info = from_exchange_instrument(
        SimpleNamespace(instrument_id="WEIRD", tick_size=0.1, instrument_type="InstrumentType.INDEX_TRACKING_ETF", index_id="IDX")
    )
    assert info.kind is Kind.ETF
    assert info.base == "IDX"


def test_a_basket_suffix_reads_as_an_etf():
    assert parse_instrument_id("TECH_BASKET").kind is Kind.ETF
