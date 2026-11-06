"""News classification. Keyword and Naive Bayes run inline; an LLM only ever runs off-thread.

The feed shape comes from the public Optibook news assignment (`poll_new_social_media_feeds`,
`.post`, labelled `training.csv` with one return column per stock). Whether Friday has news at all
is unconfirmed.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple
from urllib.request import Request, urlopen

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "will", "with"]
)


def tokens(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOP]


class NewsSignal(NamedTuple):
    instrument_id: str | None
    direction: int  # +1 price up, -1 down, 0 relevant but direction unknown
    confidence: float


NO_SIGNAL = NewsSignal(None, 0, 0.0)


# Company names for tickers seen in public editions, so a headline finds its stock with no setup.
COMPANY_NAMES: dict[str, tuple[str, ...]] = {
    "NVDA": ("nvidia",),
    "ING": ("ing",),
    "SAN": ("santander",),
    "PFE": ("pfizer",),
    "CSCO": ("cisco",),
    "ASML": ("asml",),
    "AAPL": ("apple",),
    "SAP": ("sap",),
    "TSLA": ("tesla",),
    "AMD": ("amd",),
    "AMZN": ("amazon",),
    "JPM": ("jpmorgan", "jpm"),
    "XOM": ("exxon", "exxonmobil"),
    "NVO": ("novo", "nordisk"),
    "MSFT": ("microsoft",),
    "GOOGL": ("google", "alphabet"),
    "META": ("meta", "facebook"),
    "PHILIPS": ("philips",),
}
POSITIVE_WORDS = (
    "beats", "beat", "record", "surges", "surge", "soars", "wins", "raises", "approval", "upgrade",
    "boost", "boosts", "breakthrough", "promising", "secures", "expansion", "exceed", "exceeds",
    "positive", "strong", "profit", "profits", "gains", "rally", "launch", "deal", "rise", "rises", "rising",
    "growth", "upgraded", "optimism", "optimistic", "welcome", "bullish", "success", "successful", "outperform",
)
NEGATIVE_WORDS = (
    "misses", "miss", "probe", "recall", "recalls", "lawsuit", "lawsuits", "falls", "backlash",
    "downgrade", "cuts", "plunges", "faces", "facing", "dip", "delay", "delays", "setback", "concerns",
    "issues", "fine", "fined", "loses", "weak", "losses", "fraud", "challenge", "challenges", "plague", "plagues",
    "blunder", "wary", "decline", "declines", "drop", "drops", "slowdown", "looms", "bearish", "shortage",
    "disruption", "scandal", "breach", "worries", "pressure",
)
# In labelled public headlines, 48% of posts with these words moved the stock, against 83% of plain ones.
HEDGE_WORDS = frozenset(
    ("rumor", "rumors", "rumored", "rumour", "rumours", "unconfirmed", "whispers", "speculation", "might", "could",
     "allegedly", "possibly", "possible", "minor", "slight", "slightly")
)
HEDGE_PENALTY = 0.15


LONG_TICKER = 4


def _words(text: str) -> set[str]:
    """Lower-case words, with "NVIDIA's" also read as "nvidia"."""
    found = set(_WORD.findall(text.lower()))
    return found | {word[:-2] for word in found if word.endswith("'s")}


def default_aliases(instrument_ids: Iterable[str]) -> dict[str, set[str]]:
    """Words naming each stock: its ticker, and its company when known. Second listings share them."""
    aliases: dict[str, set[str]] = {}
    for iid in instrument_ids:
        if "_20" in iid or iid.endswith(("_DUAL", "_ETF", "_BASKET")):
            continue
        names = set(COMPANY_NAMES.get(iid, ()))
        # Short tickers are ordinary words too: "san" must not match San Francisco.
        if not names or len(iid) >= LONG_TICKER:
            names.add(iid.lower())
        aliases[iid] = names
    return aliases


def default_classifier(instrument_ids: Iterable[str]) -> KeywordClassifier:
    return KeywordClassifier(default_aliases(instrument_ids), POSITIVE_WORDS, NEGATIVE_WORDS)


class KeywordClassifier:
    """Zero-training fallback, usable the moment the feed appears."""

    def __init__(
        self,
        aliases: Mapping[str, Sequence[str]],
        positive: Iterable[str],
        negative: Iterable[str],
    ) -> None:
        self.aliases = {iid: {a.lower() for a in names} for iid, names in aliases.items()}
        self.positive = {word.lower() for word in positive}
        self.negative = {word.lower() for word in negative}

    @staticmethod
    def _confidence(words: set[str], up: int, down: int) -> float:
        """Hedged wording cuts it below the trading threshold, unless the post says more than once."""
        base = min(0.8 + 0.05 * abs(up - down), 0.95)
        return base - HEDGE_PENALTY if words & HEDGE_WORDS else base

    def classify(self, text: str) -> NewsSignal:
        words = _words(text)  # no stop filter: a ticker may be a stop word
        targets = [iid for iid, names in self.aliases.items() if names & words]
        if len(targets) != 1:
            return NO_SIGNAL
        up, down = len(words & self.positive), len(words & self.negative)
        if up == down:
            return NewsSignal(targets[0], 0, 0.5)
        direction = 1 if up > down else -1
        return NewsSignal(targets[0], direction, self._confidence(words, up, down))


    def classify_all(self, text: str) -> list[NewsSignal]:
        """Every stock the post names, all the same way: a post may affect any subset of them."""
        words = _words(text)
        targets = [iid for iid, names in self.aliases.items() if names & words]
        if not targets:
            return []
        up, down = len(words & self.positive), len(words & self.negative)
        if up == down:
            return [NewsSignal(iid, 0, 0.5) for iid in targets]
        direction = 1 if up > down else -1
        confidence = self._confidence(words, up, down)
        return [NewsSignal(iid, direction, confidence) for iid in targets]


LLM_PROMPT = """You score one social media post for a trading system. The tradable stocks are: {names}.
Decide which of these stocks the post is about, if any, and whether it pushes each price up or down.
Posts about other companies, general chatter and jokes affect nothing. Rumours count for less than facts.
Reply with JSON only, and nothing else, in exactly this shape:
{{"impacts": [{{"instrument": "<one of the stocks>", "direction": 1, "confidence": 0.9}}]}}
direction is 1 for up and -1 for down. confidence runs from 0 to 1. Use an empty list if nothing applies.

Post: {post}"""


def parse_llm_reply(reply: str, instruments: Iterable[str]) -> list[NewsSignal]:
    """Anything a model says is checked before it can trade. What does not parse means no trade."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(reply[start : end + 1])
    except ValueError:
        return []
    impacts = parsed.get("impacts") if isinstance(parsed, dict) else None
    if not isinstance(impacts, list):
        return []
    known = set(instruments)
    signals = []
    for item in impacts:
        if not isinstance(item, dict) or item.get("instrument") not in known:
            continue
        direction, confidence = item.get("direction"), item.get("confidence")
        if direction not in (1, -1) or isinstance(confidence, bool):
            continue
        if not isinstance(confidence, (int, float)):
            continue
        signals.append(NewsSignal(item["instrument"], int(direction), max(0.0, min(1.0, float(confidence)))))
    return signals


class LlmClassifier:
    """Slow but better at meaning. It only ever runs through `AsyncClassifier`, off the trading thread."""

    def __init__(self, complete: Callable[[str], str], instruments: Iterable[str]) -> None:
        self.complete = complete
        self.instruments = list(instruments)

    def classify_all(self, text: str) -> list[NewsSignal]:
        names = ", ".join(self.instruments)
        return parse_llm_reply(self.complete(LLM_PROMPT.format(names=names, post=text)), self.instruments)


LLM_TIMEOUT_S = 5.0


def http_completer(
    url: str, model: str, key_env: str | None = None, style: str = "openai", timeout_s: float = LLM_TIMEOUT_S
) -> Callable[[str], str]:
    """Calls whatever LLM the event provides, over plain HTTP, so the single file needs no packages.

    The key is read from the environment and never written down. `style` is "openai" for any
    chat-completions endpoint, or "anthropic" for the Messages API.
    """

    def complete(prompt: str) -> str:
        key = os.environ.get(key_env, "") if key_env else ""
        message = [{"role": "user", "content": prompt}]
        if style == "anthropic":
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            body = {"model": model, "max_tokens": 300, "messages": message}
        else:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            body = {"model": model, "messages": message, "temperature": 0}
        request = Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}
        )
        with urlopen(request, timeout=timeout_s) as response:
            reply = json.loads(response.read())
        if style == "anthropic":
            return "".join(part.get("text", "") for part in reply["content"])
        return reply["choices"][0]["message"]["content"]

    return complete


class LabelledRow(NamedTuple):
    text: str
    target: str | None
    direction: int


def parse_labelled_rows(header: Sequence[str], lines: Iterable[Sequence[str]]) -> list[LabelledRow]:
    """Rows of text plus one return column per stock; the largest nonzero return labels the row."""
    stock_names = list(header[1:])
    rows = []
    for line in lines:
        returns = [float(value or 0.0) for value in line[1:]]
        largest = max(range(len(returns)), key=lambda idx: abs(returns[idx]))
        if returns[largest] == 0.0:
            rows.append(LabelledRow(line[0], None, 0))
        else:
            rows.append(LabelledRow(line[0], stock_names[largest], 1 if returns[largest] > 0 else -1))
    return rows


def _log_posteriors(
    words: Sequence[str],
    prior: Mapping[str, int],
    counts: Mapping[str, Mapping[str, int]],
    vocab: int,
) -> dict[str, float]:
    total = sum(prior.values())
    scores = {}
    for label, count in prior.items():
        table = counts[label]
        denominator = sum(table.values()) + vocab
        score = math.log(count / total)
        for word in words:
            score += math.log((table.get(word, 0) + 1) / denominator)
        scores[label] = score
    return scores


def _softmax_top(scores: Mapping[str, float]) -> tuple[str, float]:
    peak = max(scores.values())
    weights = {label: math.exp(score - peak) for label, score in scores.items()}
    best = max(weights, key=weights.get)
    return best, weights[best] / sum(weights.values())


class NaiveBayesNews:
    """Two multinomial models: which stock (or none), then which direction. Trains in milliseconds."""

    NONE = "__none__"

    def __init__(self) -> None:
        self.target_prior: dict[str, int] = defaultdict(int)
        self.target_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.direction_prior: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.direction_counts: dict[str, dict[int, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.vocab: set = set()

    @classmethod
    def fit(cls, rows: Iterable[LabelledRow]) -> NaiveBayesNews:
        model = cls()
        for row in rows:
            words = tokens(row.text)
            model.vocab.update(words)
            label = row.target or cls.NONE
            model.target_prior[label] += 1
            for word in words:
                model.target_counts[label][word] += 1
                if row.target:
                    model.direction_counts["*"][row.direction][word] += 1
            if row.target:
                model.direction_prior["*"][row.direction] += 1
        return model

    def classify(self, text: str) -> NewsSignal:
        words = [word for word in tokens(text) if word in self.vocab]
        if not words or not self.target_prior:
            return NO_SIGNAL
        scores = _log_posteriors(words, self.target_prior, self.target_counts, len(self.vocab))
        target, target_p = _softmax_top(scores)
        if target == self.NONE:
            return NO_SIGNAL
        prior = self.direction_prior["*"]
        if len(prior) < 2:
            return NewsSignal(target, 0, target_p)
        direction_scores = _log_posteriors(
            words, prior, self.direction_counts["*"], len(self.vocab)
        )
        direction, direction_p = _softmax_top(direction_scores)
        return NewsSignal(target, int(direction), target_p * direction_p)


class CombinedClassifier:
    """A trained model first, rules for what it passes over. When both are sure and disagree, no trade.

    On 1,182 labelled headlines the model was more careful and the rules caught more, so together
    they cover more than either at about the same direction accuracy.
    """

    def __init__(self, model, rules, min_confidence: float = 0.75) -> None:
        self.model = model
        self.rules = rules
        self.min_confidence = min_confidence

    def _sure(self, signal: NewsSignal) -> bool:
        return bool(signal.instrument_id and signal.direction and signal.confidence >= self.min_confidence)

    def classify(self, text: str) -> NewsSignal:
        first, second = self.model.classify(text), self.rules.classify(text)
        both_sure = self._sure(first) and self._sure(second)
        if both_sure and first.instrument_id == second.instrument_id and first.direction != second.direction:
            return NewsSignal(first.instrument_id, 0, 0.5)
        if self._sure(first):
            return first
        return second if self._sure(second) else (first if first.instrument_id else second)


class AsyncClassifier:
    """Runs a slow classifier, such as an LLM, on a worker thread.

    `submit` never blocks the trading loop. `poll` returns results that finished since the last
    call. Failures are returned as (text, None) so the caller can log them.
    """

    def __init__(self, classify: Callable[[str], NewsSignal], max_pending: int = 64) -> None:
        self._classify = classify
        self._in: queue.Queue[str | None] = queue.Queue(maxsize=max_pending)
        self._out: queue.Queue[tuple[str, NewsSignal | None]] = queue.Queue()
        self.dropped = 0
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self) -> None:
        while True:
            text = self._in.get()
            if text is None:
                return
            try:
                self._out.put((text, self._classify(text)))
            except Exception:  # noqa: BLE001 - a model failure must never reach the trading loop
                self._out.put((text, None))

    def submit(self, text: str) -> bool:
        try:
            self._in.put_nowait(text)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def poll(self) -> list[tuple[str, NewsSignal | None]]:
        results = []
        while True:
            try:
                results.append(self._out.get_nowait())
            except queue.Empty:
                return results

    def close(self) -> None:
        self._in.put(None)
        self._thread.join(timeout=2.0)
