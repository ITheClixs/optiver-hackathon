"""End-to-end strategy behaviour on the simulator. Assertions target mechanics and sign, not size."""

import statistics

import pytest

from otk.news import KeywordClassifier, NaiveBayesNews, NewsSignal, parse_labelled_rows
from otk.params import ParamStore
from otk.sim.exchange import SimExchange, SimMarket
from otk.sim.harness import make_engine, run_seeds, run_session
from otk.sim.scenarios import adverse_field, dual_listing, index_complex, news_stocks, quiet_market
from otk.strategies import (
    DualListingArb,
    Flattener,
    Hedger,
    IndexTheo,
    MarketMaker,
    NewsTrader,
    OptionTheo,
    TheoQuoter,
    TheoTaker,
    default_stack,
)


def rest(market, iid, side, price, volume, team="bot"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def clean(result):
    assert result.disconnects == 0, result
    assert result.limit_breaches == 0, result
    assert result.rate_breaches == 0, result
    assert result.errors == 0, result


def test_market_maker_quotes_inside_the_ladder_and_earns_on_noise_flow():
    results = run_seeds(
        lambda seed: quiet_market(seed, ("A", "B"), sigma=0.01),
        lambda: [MarketMaker(["A", "B"])],
        seeds=range(5),
        seconds=120,
    )
    for result in results:
        clean(result)
        assert result.delta_hacks == 0
    assert statistics.median(r.settled_pnl for r in results) > 0


def test_market_maker_never_quotes_against_itself():
    market = quiet_market(1, ("A",))
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    for _ in range(50):
        market.advance(0.1)
    for _ in range(100):
        market.advance(0.1)
        engine.tick()
    assert market.stats["us"].self_trades == 0


def test_market_maker_pulls_on_a_jump_and_waits_out_the_cooldown():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.0, 100)
    rest(market, "A", "ask", 101.0, 100)
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A")
    market.wipe_orders("bot", "A")
    rest(market, "A", "bid", 89.0, 100)
    rest(market, "A", "ask", 91.0, 100)
    market.advance(0.1)
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A") == {}
    market.advance(5.0)
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A")


def test_market_maker_with_one_sided_book_stays_out():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.0, 100)
    engine = make_engine(market, "us", [MarketMaker(["A"])])
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A") == {}


def test_dual_arb_takes_a_crossed_pair_and_stays_hedged():
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.8, 200)
    rest(market, "A", "ask", 100.0, 200)
    rest(market, "A_DUAL", "bid", 100.5, 20)
    rest(market, "A_DUAL", "ask", 101.0, 20)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)])
    engine.tick()
    positions = engine.exchange.get_positions()
    assert positions["A_DUAL"] < 0 and positions["A"] == -positions["A_DUAL"]
    assert engine.exchange.get_pnl({"A": 100.0, "A_DUAL": 100.0}) > 0


def test_dual_arb_ignores_a_gap_below_the_required_edge():
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.8, 200)
    rest(market, "A", "ask", 100.0, 200)
    rest(market, "A_DUAL", "bid", 100.0, 20)
    rest(market, "A_DUAL", "ask", 100.3, 20)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")], quote=False)])
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 0


def test_dual_quotes_follow_the_liquid_primary_not_the_stale_dual():
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 109.9, 200)
    rest(market, "A", "ask", 110.1, 200)
    rest(market, "A_DUAL", "bid", 99.0, 20)
    rest(market, "A_DUAL", "ask", 121.0, 20)
    engine = make_engine(market, "us", [DualListingArb([("A", "A_DUAL")])])
    engine.tick()
    prices = [o.price for o in engine.exchange.get_outstanding_orders("A_DUAL").values()]
    assert prices and all(105.0 < price < 115.0 for price in prices)


def test_dual_listing_stack_is_profitable_and_hedged_on_the_simulator():
    results = run_seeds(
        dual_listing,
        lambda: [
            DualListingArb([("ASML", "ASML_DUAL"), ("SAP", "SAP_DUAL")]),
            Hedger(hedge_on={"ASML": "ASML", "SAP": "SAP"}),
        ],
        seeds=range(5),
        seconds=120,
    )
    for result in results:
        clean(result)
    assert statistics.median(r.settled_pnl for r in results) > 0


def test_hedger_flattens_net_delta_on_the_liquid_leg():
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.8, 500)
    rest(market, "A", "ask", 100.0, 500)
    rest(market, "A_DUAL", "ask", 100.0, 40)
    tight = ParamStore(base={"hedge_threshold": 10, "hedge_to": 0})
    engine = make_engine(market, "us", [Hedger(hedge_on={"A": "A"})], params=tight)
    engine.exchange.insert_order("A_DUAL", price=100.0, volume=40, side="bid", order_type="ioc")
    engine.tick()
    positions = engine.exchange.get_positions()
    assert abs(positions["A"] + positions["A_DUAL"]) <= 10


def test_hedger_does_not_trade_below_threshold():
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.8, 500)
    rest(market, "A", "ask", 100.0, 500)
    rest(market, "A_DUAL", "ask", 100.0, 5)
    engine = make_engine(market, "us", [Hedger(hedge_on={"A": "A"})])
    engine.exchange.insert_order("A_DUAL", price=100.0, volume=5, side="bid", order_type="ioc")
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 0


def test_hedger_waits_for_positions_to_settle_before_hedging_again():
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.8, 5)
    rest(market, "A_DUAL", "ask", 100.0, 60)
    engine = make_engine(market, "us", [Hedger(hedge_on={"A": "A"}, min_interval_s=1.0)])
    engine.exchange.insert_order("A_DUAL", price=100.0, volume=60, side="bid", order_type="ioc")
    engine.tick()
    before = market.stats["us"].writes
    engine.tick()
    assert market.stats["us"].writes == before


def test_flattener_reduces_positions_at_the_deadline():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.0, 500)
    rest(market, "A", "ask", 100.0, 500)
    engine = make_engine(
        market, "us", [Flattener(flatten_at=1.0), MarketMaker(["A"])]
    )
    engine.exchange.insert_order("A", price=100.0, volume=30, side="bid", order_type="ioc")
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 30
    market.advance(2.0)
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 0
    assert engine.exchange.get_outstanding_orders("A") == {}


def test_index_theo_matches_the_guide_formulas():
    theo = IndexTheo(weights={"A": 1000.0, "B": 2000.0})
    market = SimMarket(instruments=["A", "B", "OB5X_ETF", "OB5X_202703_F"])
    for iid, price in (("A", 10.0), ("B", 20.0)):
        rest(market, iid, "bid", price - 0.1, 100)
        rest(market, iid, "ask", price + 0.1, 100)
    seen = {}

    class Spy:
        name = "spy"

        def on_tick(self, state):
            seen["etf"] = theo(state, "OB5X_ETF")
            seen["fut"] = theo(state, "OB5X_202703_F")
            return []

    make_engine(market, "us", [Spy()]).tick()
    assert seen["etf"] == pytest.approx(2.5 + 0.25 * 50.0)
    assert seen["fut"] > 50.0


def test_theo_taker_lifts_only_mispriced_quotes():
    market = SimMarket(instruments=["A", "A_202703_F"])
    rest(market, "A", "bid", 99.9, 100)
    rest(market, "A", "ask", 100.1, 100)
    rest(market, "A_202703_F", "bid", 90.0, 10)
    rest(market, "A_202703_F", "ask", 95.0, 10)
    theo = OptionTheo(vol=0.25, rate=0.03)
    engine = make_engine(market, "us", [TheoTaker(["A_202703_F"], theo)])
    engine.tick()
    assert engine.exchange.get_positions()["A_202703_F"] == 10


def test_index_stack_runs_clean_on_the_full_universe():
    def stack():
        weights = {"ASML": 908.06, "AAPL": 129.24, "SAP": 124.78, "TSLA": 2245.39, "NVDA": 953.21}
        theo = IndexTheo(weights=weights, vol=0.25)
        derivatives = ["OB5X_ETF", "OB5X_202703_F", "ASML_202703_F", "ASML_202703_100C", "ASML_202703_100P"]
        return [
            Hedger(hedge_on={"ASML": "ASML", "SAP": "SAP", "OB5X": "OB5X_202703_F"}),
            DualListingArb([("ASML", "ASML_DUAL"), ("SAP", "SAP_DUAL")]),
            TheoTaker(derivatives, theo),
            TheoQuoter(derivatives, theo),
            MarketMaker(["AAPL", "TSLA", "NVDA"]),
        ]

    for result in run_seeds(index_complex, stack, seeds=range(2), seconds=60):
        clean(result)
        assert result.peak_write_rate <= 25


def test_default_stack_builds_from_discovery_and_runs_under_adverse_field():
    def scenario(seed):
        return adverse_field(index_complex(seed), seed)

    probe = scenario(0)
    universe_ids = probe.instruments
    results = run_seeds(scenario, lambda: default_stack(universe_ids), seeds=range(2), seconds=60)
    for result in results:
        clean(result)


NEWS_GOOD = ["beats", "record", "wins", "raises"]
NEWS_BAD = ["probe", "misses", "recalls", "loses"]


def classifier(ids):
    return KeywordClassifier({iid: [iid] for iid in ids}, NEWS_GOOD, NEWS_BAD)


def test_keyword_classifier_maps_target_and_direction():
    clf = classifier(["NVDA", "PFE"])
    assert clf.classify("@Wire: NVDA beats expectations").instrument_id == "NVDA"
    assert clf.classify("@Wire: NVDA beats expectations").direction == 1
    assert clf.classify("@Wire: PFE faces a regulatory probe").direction == -1
    assert clf.classify("A study found CEOs read books").instrument_id is None


def test_keyword_classifier_is_unsure_with_conflicting_words():
    signal = classifier(["NVDA"]).classify("NVDA beats but faces a probe")
    assert signal.direction == 0


def test_naive_bayes_learns_target_and_direction_from_labelled_rows():
    header = ["SocialMediaFeed", "NVDA", "PFE"]
    lines = [
        ["NVIDIA chip demand surges to a record", "0.03", "0"],
        ["NVIDIA graphics win lifts the stock", "0.02", "0"],
        ["NVIDIA chip recall hurts sales badly", "-0.03", "0"],
        ["NVIDIA faces graphics lawsuit and falls", "-0.02", "0"],
        ["Pfizer vaccine approval lifts the stock", "0", "0.03"],
        ["Pfizer drug win surges to a record", "0", "0.02"],
        ["Pfizer recall hurts sales badly", "0", "-0.03"],
        ["Pfizer lawsuit and falls", "0", "-0.02"],
        ["CEOs only read business books", "0", "0"],
        ["Markets await the weekend", "0", "0"],
    ]
    rows = parse_labelled_rows(header, lines)
    model = NaiveBayesNews.fit(rows)
    signal = model.classify("NVIDIA chip surges to a record")
    assert signal.instrument_id == "NVDA" and signal.direction == 1
    signal = model.classify("Pfizer recall and lawsuit")
    assert signal.instrument_id == "PFE" and signal.direction == -1
    assert model.classify("CEOs read business books").instrument_id is None


def test_news_trader_ducks_then_goes_directional_then_exits_on_z_flip():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.9, 500)
    rest(market, "A", "ask", 100.1, 500)
    clf = classifier(["A"])
    engine = make_engine(market, "us", [NewsTrader(clf), MarketMaker(["A"])])
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A")
    market.news.append(market_news(market, "A beats expectations"))
    engine.tick()
    assert engine.exchange.get_positions()["A"] > 50
    assert engine.exchange.get_outstanding_orders("A") == {}

    # Price rises, then the z-score flips below its mean: directional mode hands back to quoting.
    for price in (105.0, 110.0, 110.0, 110.0):
        market.wipe_orders("bot", "A")
        rest(market, "A", "bid", price - 0.1, 500)
        rest(market, "A", "ask", price + 0.1, 500)
        market.advance(1.1)
        engine.tick()
    assert engine.strategies[0]._direction.get("A") == 1
    market.wipe_orders("bot", "A")
    rest(market, "A", "bid", 95.0, 500)
    rest(market, "A", "ask", 95.2, 500)
    market.advance(1.1)
    engine.tick()
    assert "A" not in engine.strategies[0]._direction


def test_news_trader_low_confidence_only_ducks():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.9, 500)
    rest(market, "A", "ask", 100.1, 500)

    class Unsure:
        def classify(self, _text):
            return NewsSignal("A", 1, 0.55)

    engine = make_engine(market, "us", [NewsTrader(Unsure()), MarketMaker(["A"])])
    engine.tick()
    market.news.append(market_news(market, "A maybe"))
    engine.tick()
    assert engine.exchange.get_positions()["A"] == 0
    assert engine.exchange.get_outstanding_orders("A") == {}


def test_news_trader_ignores_noise():
    market = SimMarket(instruments=["A"])
    rest(market, "A", "bid", 99.9, 500)
    rest(market, "A", "ask", 100.1, 500)
    engine = make_engine(market, "us", [NewsTrader(classifier(["A"])), MarketMaker(["A"])])
    engine.tick()
    market.news.append(market_news(market, "Markets await the weekend"))
    engine.tick()
    assert engine.exchange.get_outstanding_orders("A")


def test_news_handler_outperforms_plain_market_making_on_news_flow():
    ids = ["NVDA", "ING", "SAN", "PFE", "CSCO"]

    def plain():
        return [MarketMaker(ids)]

    def with_news():
        return [NewsTrader(classifier(ids)), MarketMaker(ids)]

    def scenario(seed):
        return news_stocks(seed, news_rate=0.1, jump_fraction=0.03)

    baseline = run_seeds(scenario, plain, seeds=range(6), seconds=120)
    handled = run_seeds(scenario, with_news, seeds=range(6), seconds=120)
    for result in handled:
        clean(result)
    assert statistics.median(r.settled_pnl for r in handled) > statistics.median(
        r.settled_pnl for r in baseline
    )


def market_news(market, text):
    from otk.sim.exchange import SocialMediaFeed

    return SocialMediaFeed(market.timestamp(), text, None)


def test_two_teams_trade_the_same_session():
    market = quiet_market(3, ("A",))
    first = make_engine(market, "one", [MarketMaker(["A"])])
    second = make_engine(market, "two", [MarketMaker(["A"])])
    results = run_session(market, {"one": first, "two": second}, 60)
    assert set(results) == {"one", "two"}
    clean(results["one"])
    clean(results["two"])
