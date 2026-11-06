"""The one predictive model we run live: online ridge with few features, O(n^2) per update.

Evidence: 30 to 90 minutes gives only about 900 to 2,700 independent 2 s labels, far
below what XGBoost or deep models need, and no such model is documented winning a live-exchange
format. So the model only nudges fair value by a bounded number of ticks, and it switches itself
off whenever its out-of-sample error is not better than predicting no move.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

from otk.fair import imbalance, near_touch_imbalance


class OnlineRidge:
    """Recursive least squares with a forgetting factor."""

    def __init__(self, n_features: int, ridge: float = 1.0, forget: float = 0.999) -> None:
        self.n = n_features
        self.forget = forget
        self.weights = [0.0] * n_features
        self._p = [[1.0 / ridge if i == j else 0.0 for j in range(n_features)] for i in range(n_features)]

    def predict(self, x: Sequence[float]) -> float:
        return sum(w * value for w, value in zip(self.weights, x))

    def update(self, x: Sequence[float], y: float) -> None:
        px = [sum(row[j] * x[j] for j in range(self.n)) for row in self._p]
        denominator = self.forget + sum(x[i] * px[i] for i in range(self.n))
        gain = [value / denominator for value in px]
        error = y - self.predict(x)
        self.weights = [w + g * error for w, g in zip(self.weights, gain)]
        self._p = [
            [(self._p[i][j] - gain[i] * px[j]) / self.forget for j in range(self.n)]
            for i in range(self.n)
        ]


class Prequential:
    """Decayed squared error of the model versus the zero predictor, scored before each update."""

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay
        self.model_error = 0.0
        self.zero_error = 0.0
        self.count = 0

    def add(self, prediction: float, actual: float) -> None:
        self.model_error = self.model_error * self.decay + (actual - prediction) ** 2
        self.zero_error = self.zero_error * self.decay + actual * actual
        self.count += 1

    @property
    def skill(self) -> float:
        """Out-of-sample R squared against predicting no move; 0 before there is any error."""
        return 1.0 - self.model_error / self.zero_error if self.zero_error > 0 else 0.0


def book_features(state, instrument_id: str) -> list[float]:
    """Bounded and causal: near-touch imbalance, top imbalance, intercept."""
    book = state.external(instrument_id)
    band = state.params["near_touch_ticks"] * state.tick_size
    return [near_touch_imbalance(book, band), imbalance(state.top(instrument_id)), 1.0]


class AlphaFair:
    """Fair-value source: a base fair plus a bounded predicted move.

    A label completes only after `horizon_s`, so training never sees the future, and each pending
    label is scored before it is learned, which makes the skill estimate honest.
    """

    def __init__(
        self,
        base: Callable,
        features: Callable = book_features,
        n_features: int = 3,
        horizon_s: float = 2.0,
        warmup: int = 300,
        min_skill: float = 0.01,
        sample_every_s: float = 0.25,
    ) -> None:
        self.base = base
        self.features = features
        self.n_features = n_features
        self.horizon_s = horizon_s
        self.warmup = warmup
        self.min_skill = min_skill
        self.sample_every_s = sample_every_s
        self._models: dict[str, OnlineRidge] = {}
        self._scores: dict[str, Prequential] = {}
        self._pending: dict[str, deque] = {}
        self._last_sample: dict[str, float] = {}

    def _learn(self, iid: str, now: float, fair: float, model: OnlineRidge, score: Prequential) -> None:
        pending = self._pending[iid]
        while pending and now - pending[0][0] >= self.horizon_s:
            _, x, then = pending.popleft()
            actual = fair - then
            score.add(model.predict(x), actual)
            model.update(x, actual)

    def skill(self, instrument_id: str) -> float:
        score = self._scores.get(instrument_id)
        return score.skill if score else 0.0

    def __call__(self, state, instrument_id: str) -> float | None:
        fair = self.base(state, instrument_id)
        if fair is None:
            return None
        model = self._models.setdefault(instrument_id, OnlineRidge(self.n_features))
        score = self._scores.setdefault(instrument_id, Prequential())
        self._pending.setdefault(instrument_id, deque())
        self._learn(instrument_id, state.now, fair, model, score)
        x = self.features(state, instrument_id)
        if state.now - self._last_sample.get(instrument_id, float("-inf")) >= self.sample_every_s:
            self._pending[instrument_id].append((state.now, x, fair))
            self._last_sample[instrument_id] = state.now
        if score.count < self.warmup or score.skill < self.min_skill:
            return fair
        cap = state.params["max_alpha_ticks"] * state.tick_size
        return fair + max(-cap, min(cap, model.predict(x)))
