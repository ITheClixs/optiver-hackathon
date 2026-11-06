"""Each mechanism named in the hackaTUM 2025 write-up (3rd highest PnL) by fabianbaiertum.

Source: https://github.com/fabianbaiertum/Optiver-hackaTUM. The review is in
a public HackaTUM 2025 write-up. Tests cover behaviour; earning is for the live evidence to show.
"""

import json

import pytest

from otk.budget import Priority, WriteBudget
from otk.intents import QuoteIntent
from otk.news import (
    CombinedClassifier,
    KeywordClassifier,
    LlmClassifier,
    NewsSignal,
    default_aliases,
    default_classifier,
    http_completer,
    parse_llm_reply,
)
from otk.params import ParamStore, write_params_atomic
from otk.quote_manager import QuoteManager
from otk.quoter import QuoteParams, compute_quotes
from otk.risk import Limits
from otk.sim.exchange import SimExchange, SimMarket, SocialMediaFeed
from otk.sim.harness import make_engine
from otk.strategies import (
    CrossListingFair,
    GapReversion,
    MarketMaker,
    NewsTrader,
    micro_fair,
)

FIVE = ["NVDA", "ING", "SAN", "PFE", "CSCO"]


def rest(market, iid, side, price, volume, team="wall"):
    SimExchange(market, team, bot=True).insert_order(
        iid, price=price, volume=volume, side=side, order_type="limit"
    )


def book(market, iid, bid, ask, volume=500, team="wall"):
    rest(market, iid, "bid", bid, volume, team)
    rest(market, iid, "ask", ask, volume, team)


def store(tmp_path, **params):
    path = tmp_path / "params.json"
    write_params_atomic(path, params)
    return ParamStore(path)


def advance(market, engine, steps=6, dt=0.3):
    for _ in range(steps):
        market.advance(dt)
        engine.tick()


def publish(market, text="post"):
    market.news.append(SocialMediaFeed(market.timestamp(), text, None))


# --- "any subset of the 5 stocks ... direction ... confidence" -------------------------------------


class Fixed:
    def __init__(self, *signals):
        self.signals = list(signals)

    def classify_all(self, _text):
        return list(self.signals)

    def classify(self, _text):
        return self.signals[0]


def news_market(*names):
    market = SimMarket(instruments=list(names))
    for name in names:
        book(market, name, 99.9, 100.1)
    return market


def test_one_post_can_move_several_stocks_in_different_directions():
    market = news_market("NVDA", "ING", "SAN")
    signals = Fixed(NewsSignal("NVDA", 1, 0.9), NewsSignal("ING", -1, 0.9))
    engine = make_engine(market, "us", [NewsTrader(signals)])
    engine.tick()
    publish(market)
    advance(market, engine)
    positions = engine.exchange.get_positions()
    assert positions["NVDA"] > 50
    assert positions["ING"] < -50
    assert positions["SAN"] == 0


def test_exposure_follows_confidence():
    def position(confidence):
        market = news_market("NVDA")
        engine = make_engine(market, "us", [NewsTrader(Fixed(NewsSignal("NVDA", 1, confidence)))])
        engine.tick()
        publish(market)
        advance(market, engine)
        return engine.exchange.get_positions()["NVDA"]

    assert position(0.76) < position(0.95)


def test_keywords_find_every_named_stock():
    classifier = KeywordClassifier(default_aliases(FIVE), ["beats"], ["misses"])
    found = classifier.classify_all("Nvidia and Pfizer both beats on revenue")
    assert {signal.instrument_id for signal in found} == {"NVDA", "PFE"}
    assert all(signal.direction == 1 for signal in found)
    assert classifier.classify_all("A quiet day in the markets") == []


def test_company_names_are_known_for_common_tickers():
    aliases = default_aliases(["NVDA", "SAN", "PFE", "XYZ", "NVDA_DUAL"])
    assert "nvidia" in aliases["NVDA"]
    assert "santander" in aliases["SAN"]
    assert aliases["XYZ"] == {"xyz"}
    assert "NVDA_DUAL" not in aliases, "a second listing shares its primary's name"


def test_a_rumour_only_ducks_but_a_plain_report_trades():
    classifier = default_classifier(FIVE)
    assert classifier.classify("Pfizer faces a product recall.").confidence >= 0.75
    hedged = classifier.classify("Rumor has it Pfizer might face a recall.")
    assert hedged.instrument_id == "PFE"
    assert hedged.confidence < 0.75, "ducks without trading"


def test_a_model_and_rules_cover_for_each_other_and_veto_on_disagreement():
    up, down, nothing = NewsSignal("A", 1, 0.9), NewsSignal("A", -1, 0.9), NewsSignal(None, 0, 0.0)

    class Says:
        def __init__(self, signal):
            self.signal = signal

        def classify(self, _text):
            return self.signal

    assert CombinedClassifier(Says(up), Says(nothing)).classify("x") == up
    assert CombinedClassifier(Says(nothing), Says(down)).classify("x") == down
    assert CombinedClassifier(Says(up), Says(down)).classify("x").direction == 0
    assert CombinedClassifier(Says(nothing), Says(nothing)).classify("x") == nothing
    assert CombinedClassifier(Says(NewsSignal("A", 1, 0.5)), Says(nothing)).classify("x").instrument_id == "A"


def test_rules_read_a_typical_headline_correctly():
    classifier = default_classifier(FIVE)
    assert classifier.classify("Pfizer faces backlash over closure of regional office.") == NewsSignal("PFE", -1, pytest.approx(0.9))
    assert classifier.classify("NVIDIA's contributions have given the stock a boost.").direction == 1
    assert classifier.classify("Johnson & Johnson faces lawsuits over product safety.").instrument_id is None


# --- the LLM: classifier off-thread, with a reply that cannot be trusted ---------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"impacts": [{"instrument": "NVDA", "direction": 1, "confidence": 0.9}]}', [("NVDA", 1, 0.9)]),
        ('Sure!\n```json\n{"impacts": [{"instrument": "PFE", "direction": -1, "confidence": 0.8}]}\n```', [("PFE", -1, 0.8)]),
        ('{"impacts": []}', []),
        ('{"impacts": [{"instrument": "ZZZ", "direction": 1, "confidence": 0.9}]}', []),
        ('{"impacts": [{"instrument": "NVDA", "direction": 7, "confidence": 0.9}]}', []),
        ('{"impacts": [{"instrument": "NVDA", "direction": 1, "confidence": "high"}]}', []),
        ('{"impacts": [{"instrument": "NVDA", "direction": 1, "confidence": 9}]}', [("NVDA", 1, 1.0)]),
        ("no json here", []),
        ('{"impacts": "nope"}', []),
        ("[1, 2]", []),
    ],
)
def test_llm_replies_are_validated_before_they_can_trade(reply, expected):
    signals = parse_llm_reply(reply, FIVE)
    assert [(s.instrument_id, s.direction, s.confidence) for s in signals] == expected


def test_the_prompt_names_the_instruments_and_the_post():
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return '{"impacts": [{"instrument": "ING", "direction": -1, "confidence": 0.7}]}'

    signals = LlmClassifier(complete, FIVE).classify_all("ING fined")
    assert signals == [NewsSignal("ING", -1, 0.7)]
    assert "ING" in prompts[0] and "ING fined" in prompts[0]


def test_http_completer_speaks_both_api_styles(monkeypatch):
    sent = []

    class Reply:
        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        sent.append((request.full_url, json.loads(request.data), dict(request.header_items()), timeout))
        if "anthropic" in request.full_url:
            return Reply({"content": [{"type": "text", "text": "hi-a"}]})
        return Reply({"choices": [{"message": {"content": "hi-o"}}]})

    monkeypatch.setattr("otk.news.urlopen", fake_urlopen)
    monkeypatch.setenv("MY_KEY", "secret")
    assert http_completer("https://x/v1/chat/completions", "m", "MY_KEY")("p") == "hi-o"
    assert http_completer("https://api.anthropic.com/v1/messages", "m", "MY_KEY", style="anthropic")("p") == "hi-a"
    assert sent[0][1]["messages"][0]["content"] == "p"
    assert sent[1][2]["X-api-key"] == "secret"
    assert sent[0][3] <= 5.0


# --- "if we predicted wrong, detect it after a short delay and close out" --------------------------


class Moving:
    """The book moves against the news verdict after the post."""

    def __init__(self, per_step):
        self.level = 100.0
        self.per_step = per_step

    def act(self, market, now, dt):
        market.wipe_orders("wall", "NVDA")
        self.level += self.per_step
        book(market, "NVDA", round(self.level - 0.1, 1), round(self.level + 0.1, 1))


def test_a_wrong_news_call_is_cut_after_a_short_delay():
    market = SimMarket(instruments=["NVDA"])
    actor = Moving(0.0)
    market.actors.append(actor)
    market.advance(0.3)
    trader = NewsTrader(Fixed(NewsSignal("NVDA", 1, 0.9)))
    engine = make_engine(market, "us", [trader])
    engine.tick()
    publish(market)
    advance(market, engine, steps=2)
    assert engine.exchange.get_positions()["NVDA"] > 50
    actor.per_step = -0.5
    advance(market, engine, steps=12)
    assert engine.exchange.get_positions()["NVDA"] == 0, "closed out, not left to the market maker"
    assert not trader.holding_instrument("NVDA")


def test_a_right_news_call_is_not_cut():
    market = SimMarket(instruments=["NVDA"])
    actor = Moving(0.0)
    market.actors.append(actor)
    market.advance(0.3)
    engine = make_engine(market, "us", [NewsTrader(Fixed(NewsSignal("NVDA", 1, 0.9)))])
    engine.tick()
    publish(market)
    advance(market, engine, steps=2)
    actor.per_step = 0.5
    advance(market, engine, steps=8)
    assert engine.exchange.get_positions()["NVDA"] > 50


# --- "if only microprices diverged, nudge the theoretical prices" ----------------------------------


def nudge_engine(tmp_path, primary_size=(500, 500), dual_size=(500, 10), **params):
    market = SimMarket(instruments=["A", "A_DUAL"])
    rest(market, "A", "bid", 99.8, primary_size[0])
    rest(market, "A", "ask", 100.2, primary_size[1])
    rest(market, "A_DUAL", "bid", 99.8, dual_size[0])
    rest(market, "A_DUAL", "ask", 100.2, dual_size[1])
    settings = store(tmp_path, imbalance_ticks=0, skew_ticks=0, **params)
    fair = CrossListingFair(micro_fair, {"A": "A_DUAL"})
    engine = make_engine(market, "us", [MarketMaker(["A"], fair=fair, use_imbalance=False)], params=settings)
    return market, engine


def test_a_microprice_lean_on_the_other_listing_nudges_our_fair(tmp_path):
    market, engine = nudge_engine(tmp_path, dual_nudge=0.5)
    market.advance(0.1)
    engine.tick()
    leaning = {o.side: o.price for o in engine.exchange.get_outstanding_orders("A").values()}
    market, engine = nudge_engine(tmp_path, dual_nudge=0)
    market.advance(0.1)
    engine.tick()
    flat = {o.side: o.price for o in engine.exchange.get_outstanding_orders("A").values()}
    assert leaning["ask"] > flat["ask"] or leaning["bid"] > flat["bid"]


def test_the_nudge_is_bounded_in_ticks(tmp_path):
    fair = CrossListingFair(lambda _state, _iid: 100.0, {"A": "A_DUAL"})

    class State:
        now = 0.0
        params = ParamStore().current

        @staticmethod
        def top(iid):
            from otk.fair import Top

            return Top(100.0, 10, 100.2, 10) if iid == "A" else Top(150.0, 900, 150.2, 1)

        @staticmethod
        def tick_of(_iid):
            return 0.1

    assert fair(State, "A") <= 100.0 + State.params["max_alpha_ticks"] * 0.1 + 1e-9
    assert fair(State, "OTHER") == 100.0


# --- "volume per side based on inventory and strength of signals" ---------------------------------


def test_size_leans_with_the_imbalance_signal():
    parameters = QuoteParams(size=10, signal_size=0.5)
    quote = compute_quotes(100.0, 0, 100, 0.0, 0.8, parameters, 0.1)
    assert quote.bid_size > 10 > quote.ask_size


def test_inventory_still_decides_when_both_apply():
    parameters = QuoteParams(size=10, signal_size=0.5)
    quote = compute_quotes(100.0, 80, 100, 0.0, 0.8, parameters, 0.1)
    assert quote.ask_size > quote.bid_size or quote.bid_size <= 4


def test_no_signal_leaves_size_alone():
    quote = compute_quotes(100.0, 0, 100, 0.0, 0.0, QuoteParams(size=10, signal_size=0.5), 0.1)
    assert (quote.bid_size, quote.ask_size) == (10, 10)


# --- "increased execution priority for mean-reverting trades" --------------------------------------


def test_a_stat_arb_trade_that_cuts_inventory_goes_at_risk_priority():
    strategy = GapReversion([("A", "A_DUAL")])

    class State:
        now = 0.0
        params = ParamStore(base={"gap_enabled": 1}).current

        @staticmethod
        def top(iid):
            from otk.fair import Top

            return Top(99.9, 500, 100.1, 500) if iid == "A" else Top(100.9, 500, 101.1, 500)

        @staticmethod
        def position(iid):
            return {"A": -40, "A_DUAL": 40}[iid]

        @staticmethod
        def position_of(_source, iid):
            return {"A": -40, "A_DUAL": 40}[iid]

        @staticmethod
        def tick_of(_iid):
            return 0.1

    strategy._stats["A_DUAL"] = _Frozen(z=3.0, mean=0.0)
    takes = strategy._trade(State, "A", "A_DUAL")
    assert takes
    assert all(take.hedge for take in takes), "this trade reduces what we hold"


class _Frozen:
    def __init__(self, z, mean):
        self.z = z
        self.mean = mean

    def update_z(self, *_args):
        return self.z


# --- "centralize rate-limit management so high-value actions pre-empt market-making updates" -------


def test_takes_can_use_budget_quotes_cannot():
    budget = WriteBudget(22, 6, clock=lambda: 0.0)
    for _ in range(16):
        assert budget.try_acquire(Priority.QUOTE)
    assert not budget.try_acquire(Priority.QUOTE), "quotes stop at 16"
    assert budget.try_acquire(Priority.TAKE), "a take still goes"
    for _ in range(2):
        budget.try_acquire(Priority.TAKE)
    assert not budget.try_acquire(Priority.TAKE), "takes stop at 19"
    assert budget.try_acquire(Priority.RISK), "hedges and cancels still go"


def test_the_stalest_quote_is_fixed_first_when_budget_is_short():
    market = SimMarket(instruments=["A", "B", "C"])
    for name in ("A", "B", "C"):
        book(market, name, 99.0, 101.0)
    us = SimExchange(market, "us")
    us.connect()
    budget = WriteBudget(25, 0, clock=lambda: market.now)
    manager = QuoteManager(us, budget, Limits(margin=0))
    for name in ("A", "B", "C"):
        manager.sync(QuoteIntent(name, 99.5, 5, 100.5, 5), {})
    market.advance(5.0)
    near = QuoteIntent("A", 99.6, 5, 100.6, 5)
    far = QuoteIntent("B", 95.0, 5, 96.0, 5)
    same = QuoteIntent("C", 99.5, 5, 100.5, 5)
    assert manager.urgency(far) > manager.urgency(near) > manager.urgency(same)
    assert manager.urgency(same) == 0


def test_a_missing_quote_counts_as_urgent():
    market = SimMarket(instruments=["A"])
    us = SimExchange(market, "us")
    us.connect()
    manager = QuoteManager(us, WriteBudget(25, 0, clock=lambda: market.now), Limits(margin=0))
    assert manager.urgency(QuoteIntent("A", 99.5, 5, 100.5, 5)) > 0


def test_small_reprices_wait_when_the_budget_is_nearly_spent():
    market = SimMarket(instruments=["A"])
    book(market, "A", 99.0, 101.0)
    us = SimExchange(market, "us")
    us.connect()
    budget = WriteBudget(10, 0, clock=lambda: market.now)
    manager = QuoteManager(us, budget, Limits(margin=0))
    manager.sync(QuoteIntent("A", 99.5, 5, None, 0), {})
    resting = list(us.get_outstanding_orders("A"))
    market.advance(2.0)
    for _ in range(9):
        budget.try_acquire(Priority.QUOTE)
    manager.sync(QuoteIntent("A", 99.6, 5, None, 0), {})
    assert list(us.get_outstanding_orders("A")) == resting, "one tick is not worth a scarce write"
    market.advance(2.0)
    manager.sync(QuoteIntent("A", 99.6, 5, None, 0), {})
    assert list(us.get_outstanding_orders("A")) != resting, "with budget free it reprices"
