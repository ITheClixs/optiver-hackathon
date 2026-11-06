"""An option's price rests on a volatility. If ours is wrong, every option looks mispriced at once, and the bot would keep trading on it.

So options trade only while our price agrees with the market: inside each option's book, for most of the last 30 s. A real
mispricing is rare and brief, and an underlying's jump leaves options stale for seconds, which is the edge. A wrong volatility
stays. They start shut, and shut again by themselves, per underlying. No book to compare with also means shut.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable, Mapping

logger = logging.getLogger("otk.optioncheck")

CHECK_SAMPLE_S = 1.0
CHECK_WINDOW_S = 30.0
CHECK_MIN_SAMPLES = 20
CHECK_MAX_OUTSIDE = 0.5


class OptionMarketCheck:
    """A fair value source. Gives `theo`'s price for an option only while the check passes, and for anything else always."""

    def __init__(self, theo: Callable, base_of: Mapping[str, str]) -> None:
        self.theo = theo
        self.base_of = dict(base_of)
        self._options: dict[str, list[str]] = defaultdict(list)
        for iid, base in self.base_of.items():
            self._options[base].append(iid)
        self._samples: dict[str, deque] = defaultdict(deque)
        self._next_sample: dict[str, float] = {}
        self._open: dict[str, bool] = {}

    def __call__(self, state, instrument_id: str) -> float | None:
        base = self.base_of.get(instrument_id)
        if base is None or self._agrees(state, base):
            return self.theo(state, instrument_id)
        return None

    def _share_outside(self, state, base: str) -> float | None:
        """Of the options with a two-sided book, the share whose price is outside it. None if there is nothing to compare."""
        compared = outside = 0
        for iid in self._options[base]:
            top, price = state.top(iid), self.theo(state, iid)
            if top is None or top.bid is None or top.ask is None or price is None:
                continue
            compared += 1
            outside += not top.bid <= price <= top.ask
        return outside / compared if compared else None

    def _agrees(self, state, base: str) -> bool:
        if state.now >= self._next_sample.get(base, float("-inf")):
            self._next_sample[base] = state.now + CHECK_SAMPLE_S
            self._sample(state, base)
        return self._open.get(base, False)

    def _sample(self, state, base: str) -> None:
        samples = self._samples[base]
        share = self._share_outside(state, base)
        if share is not None:
            samples.append((state.now, share))
        while samples and samples[0][0] < state.now - CHECK_WINDOW_S:
            samples.popleft()
        watched = len(samples) >= CHECK_MIN_SAMPLES
        outside = sum(value for _at, value in samples) / len(samples) if samples else 0.0
        self._report(base, watched and outside < CHECK_MAX_OUTSIDE, watched, outside)

    def _report(self, base: str, agrees: bool, watched: bool, outside: float) -> None:
        if base not in self._open and not watched:
            return  # still watching for the first time: nothing to say yet
        if self._open.get(base) != agrees:
            if agrees:
                logger.info("option check: %s options may trade, our price agrees with their books", base)
            else:
                logger.warning(
                    "option check: %s options stopped, our price was outside their books %.0f%% of the last %.0f s. Check option_vol",
                    base, outside * 100, CHECK_WINDOW_S,
                )
        self._open[base] = agrees
