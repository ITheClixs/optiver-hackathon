"""Decision rules for short samples. Stdlib only, so they also run in the devenv.

A shared live exchange cannot be A/B tested in parallel on one account, so variants are compared
in alternating time slices and only same-block differences count. Per-second Sharpe overstates
significance because PnL is autocorrelated, hence block bootstraps rather than a t-test on seconds.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import NamedTuple

EULER_GAMMA = 0.5772156649015329
SIGN_FLIP_EXACT_UP_TO = 16
MIN_PAIRS = 6
ADOPTION_P = 0.10
MIN_EFFECT = 0.5
MIN_UNDERLYINGS = 3
SAME_SIGN_SHARE = 0.6
STRONG_DEFLATED_SHARPE = 0.95
WEAK_DEFLATED_SHARPE = 0.5
RISK_MEASURES = ("peak_inventory", "drawdown", "rejects")


def block_bootstrap_ci(
    series: Sequence[float], block: int = 10, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI of the mean using a moving block bootstrap."""
    if not series:
        raise ValueError("series must be non-empty")
    rng = random.Random(seed)
    size = len(series)
    block = max(1, min(block, size))
    starts = size - block + 1
    blocks_needed = math.ceil(size / block)
    means = []
    for _ in range(n_boot):
        total = 0.0
        count = 0
        for _ in range(blocks_needed):
            start = rng.randrange(starts)
            for value in series[start : start + block]:
                total += value
                count += 1
        means.append(total / count)
    means.sort()
    return means[int(alpha / 2 * n_boot)], means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]


def pnl_stats(path: Sequence[float]) -> dict:
    """Summary of a cumulative PnL path sampled once per second."""
    if len(path) < 2:
        return {"n": 0, "total": 0.0, "max_drawdown": 0.0, "sharpe_per_step": 0.0}
    steps = [path[i + 1] - path[i] for i in range(len(path) - 1)]
    peak, worst = path[0], 0.0
    for value in path:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    sd = statistics.pstdev(steps)
    return {
        "n": len(steps),
        "total": path[-1] - path[0],
        "max_drawdown": worst,
        "sharpe_per_step": statistics.mean(steps) / sd if sd > 0 else 0.0,
    }


def _paired_differences(slices: Sequence[Mapping], first: str, second: str) -> list[float]:
    """Walk the schedule and pair adjacent slices of the two variants."""
    differences = []
    pending: Mapping | None = None
    for item in slices:
        if item["variant"] not in (first, second):
            pending = None
            continue
        if pending is not None and pending["variant"] != item["variant"]:
            a, b = (item, pending) if item["variant"] == first else (pending, item)
            differences.append(a["pnl"] - b["pnl"])
            pending = None
        else:
            pending = item
    return differences


def compare_variants(slices: Sequence[Mapping], first: str, second: str, seed: int = 0) -> dict:
    """Mean PnL difference of `first` over `second` per block, with a bootstrap CI."""
    differences = _paired_differences(slices, first, second)
    if not differences:
        return {"n_pairs": 0, "diff_mean": None, "ci_low": None, "ci_high": None}
    low, high = block_bootstrap_ci(differences, block=1, seed=seed)
    return {
        "n_pairs": len(differences),
        "diff_mean": statistics.mean(differences),
        "ci_low": low,
        "ci_high": high,
    }


# --- the adoption rule ------------------------------------------------


def sign_flip_p(values: Sequence[float], draws: int = 20_000, seed: int = 0) -> float:
    """Two-sided paired test: how often random signs give a sum this large. Exact up to 16 values."""
    observed = abs(sum(values)) - 1e-12
    if observed <= 0:
        return 1.0
    if len(values) <= SIGN_FLIP_EXACT_UP_TO:
        sums = [0.0]
        for value in values:
            sums = [total + value for total in sums] + [total - value for total in sums]
        return sum(abs(total) >= observed for total in sums) / len(sums)
    rng = random.Random(seed)
    hits = sum(abs(sum(value if rng.random() < 0.5 else -value for value in values)) >= observed for _ in range(draws))
    return (hits + 1) / (draws + 1)


def deflated_sharpe_normal(sharpe: float, periods: int, trials: int) -> float:
    """Deflated Sharpe with normal returns, (Bailey and López de Prado). Stdlib, so it runs in the bot's parent."""
    if periods < 3:
        return 0.0
    normal = NormalDist()
    benchmark = 0.0
    if trials > 1:
        tail = (1 - EULER_GAMMA) * normal.inv_cdf(1 - 1 / trials) + EULER_GAMMA * normal.inv_cdf(1 - 1 / (trials * math.e))
        benchmark = tail / math.sqrt(periods - 1)
    return normal.cdf((sharpe - benchmark) * math.sqrt(periods - 1) / math.sqrt(1 + sharpe * sharpe / 2))


class Adoption(NamedTuple):
    verdict: str
    reasons: tuple[str, ...]
    stats: dict


def _clean_pairs(slices: Sequence[Mapping], control: str, challenger: str) -> list[tuple[Mapping, Mapping]]:
    """Only same-block differences count, and a slice that cannot be trusted takes its partner with it."""
    blocks: dict[int, dict[str, Mapping]] = {}
    for item in slices:
        if item.get("valid"):
            blocks.setdefault(item["block"], {})[item["variant"]] = item
    return [(block[control], block[challenger]) for _, block in sorted(blocks.items()) if control in block and challenger in block]


def _worse_risk(pairs: Sequence[tuple[Mapping, Mapping]]) -> list[str]:
    found = []
    for key in RISK_MEASURES:
        differences = [second.get(key, 0.0) - first.get(key, 0.0) for first, second in pairs]
        if statistics.mean(differences) > 0 and sign_flip_p(differences) < ADOPTION_P:
            found.append(f"{key} is worse")
    return found


def _underlyings_disagree(pairs: Sequence[tuple[Mapping, Mapping]], mean: float) -> str | None:
    """With one or two underlyings this says nothing, so it applies from three."""
    names = sorted({name for pair in pairs for item in pair for name in item.get("by_group", {})})
    if len(names) < MIN_UNDERLYINGS:
        return None
    agree = 0
    for name in names:
        moved = sum(second.get("by_group", {}).get(name, 0.0) - first.get("by_group", {}).get(name, 0.0) for first, second in pairs)
        agree += moved * mean > 0
    if agree / len(names) >= SAME_SIGN_SHARE:
        return None
    return f"only {agree} of {len(names)} underlyings moved the same way"


def adoption_rule(slices: Sequence[Mapping], control: str, challenger: str, trials: int) -> Adoption:
    """Adopt the challenger only if every test holds. Anything else leaves the control in place.

    The metric is PnL per slice. The research preferred markout, but measured on 54 real hours its signal to noise per
    slice was 0.14 against 0.57 for PnL (measured in a rehearsal).
    """
    pairs = _clean_pairs(slices, control, challenger)
    if len(pairs) < MIN_PAIRS:
        return Adoption("NO CHANGE", (f"only {len(pairs)} clean pairs, need {MIN_PAIRS}",), {"pairs": len(pairs)})
    differences = [second["pnl"] - first["pnl"] for first, second in pairs]
    mean, spread = statistics.mean(differences), statistics.stdev(differences)
    half = len(differences) // 2
    sigma = math.sqrt((statistics.variance(p[0]["pnl"] for p in pairs) + statistics.variance(p[1]["pnl"] for p in pairs)) / 2)
    sharpe = mean / spread if spread > 0 else (math.inf if mean > 0 else 0.0)
    dsr = 1.0 if math.isinf(sharpe) else deflated_sharpe_normal(sharpe, len(differences), trials)
    p_value = sign_flip_p(differences)
    effect = mean / sigma if sigma > 0 else (math.inf if mean > 0 else 0.0)
    stats = {
        "pairs": len(pairs), "mean": round(mean, 4), "sd": round(spread, 4), "p": round(p_value, 4),
        "effect": round(effect, 3) if math.isfinite(effect) else None, "deflated_sharpe": round(dsr, 3), "trials": trials,
    }
    reasons = []
    if mean <= 0:
        reasons.append(f"{challenger} did not earn more: {mean:+.2f} a slice")
    if p_value >= ADOPTION_P:
        reasons.append(f"paired p {p_value:.3f} is not below {ADOPTION_P}")
    if statistics.mean(differences[:half]) * mean <= 0 or statistics.mean(differences[half:]) * mean <= 0:
        reasons.append("not the same sign in both halves")
    disagree = _underlyings_disagree(pairs, mean)
    if disagree:
        reasons.append(disagree)
    reasons.extend(_worse_risk(pairs))
    if effect <= MIN_EFFECT:
        reasons.append(f"effect {effect:.2f} of a slice's spread is not above {MIN_EFFECT}")
    if reasons:
        return Adoption("NO CHANGE", tuple(reasons), stats)
    if dsr >= STRONG_DEFLATED_SHARPE:
        return Adoption("ADOPT", (), stats)
    if dsr >= WEAK_DEFLATED_SHARPE:
        return Adoption("ADOPT IN PRACTICE", (f"deflated Sharpe {dsr:.2f} with {trials} trials is weak",), stats)
    return Adoption("NO CHANGE", (f"deflated Sharpe {dsr:.2f} does not survive {trials} trials",), stats)


def no_harm_rule(slices: Sequence[Mapping], control: str, challenger: str) -> Adoption:
    """For a change replay already picked. Twenty minutes cannot show it is better, only that it is not plainly worse."""
    pairs = _clean_pairs(slices, control, challenger)
    if len(pairs) < MIN_PAIRS:
        return Adoption("NOT ENOUGH", (f"only {len(pairs)} clean pairs, need {MIN_PAIRS}",), {"pairs": len(pairs)})
    differences = [second["pnl"] - first["pnl"] for first, second in pairs]
    mean, p_value = statistics.mean(differences), sign_flip_p(differences)
    stats = {"pairs": len(pairs), "mean": round(mean, 4), "p": round(p_value, 4)}
    earned_less = [f"{challenger} earned less: {mean:+.2f} a slice, paired p {p_value:.3f}"] if mean < 0 and p_value < ADOPTION_P else []
    reasons = (*earned_less, *_worse_risk(pairs))
    return Adoption("HARM", reasons, stats) if reasons else Adoption("NO HARM", (), stats)


def decision_line(decision: Adoption | Mapping, control: str, challenger: str) -> str:
    """The one line everyone quotes. Takes the rule's answer, or that answer read back from a result file."""
    verdict, reasons, stats = decision if isinstance(decision, Adoption) else (decision["verdict"], decision["reasons"], decision["stats"])
    numbers = ", ".join(f"{key} {stats[key]}" for key in ("pairs", "mean", "p", "effect", "deflated_sharpe") if stats.get(key) is not None)
    why = f" {'; '.join(reasons)}." if reasons else ""
    if verdict == "ADOPT":
        return f"DECISION: ADOPT {challenger} over {control}. {numbers}"
    if verdict == "ADOPT IN PRACTICE":
        return f"DECISION: ADOPT IN PRACTICE {challenger} over {control}, not for the final.{why} {numbers}"
    if verdict == "NO HARM":
        return f"DECISION: NO HARM from {challenger} against {control}. {numbers}"
    if verdict in ("HARM", "NOT ENOUGH"):
        return f"DECISION: {verdict}, {control} stays.{why} {numbers}"
    return f"DECISION: NO CHANGE, {control} stays.{why} {numbers}"


class Decision(NamedTuple):
    verdict: str
    size_fraction: float
    reasons: tuple[str, ...]


def decide(
    pbo: float | None,
    dsr: float | None,
    t: int,
    min_btl: int,
    ci_low: float | None,
    breaches: int,
) -> Decision:
    """GO, GO-REDUCED (half size) or NO-GO.

    Missing statistics do not block a mechanism with no tuning: it runs reduced until a live
    switchback supports it. Any limit breach, disconnect or hack-out in testing blocks it outright.
    """
    if breaches > 0:
        return Decision("NO-GO", 0.0, (f"{breaches} limit or session breaches",))
    if pbo is not None and pbo > 0.5:
        return Decision("NO-GO", 0.0, (f"PBO {pbo:.2f} above 0.5",))
    if dsr is not None and dsr < 0.5:
        return Decision("NO-GO", 0.0, (f"DSR {dsr:.2f} below 0.5",))
    soft: list[str] = []
    if pbo is None or dsr is None:
        soft.append("no overfitting statistics")
    else:
        if pbo > 0.2:
            soft.append(f"PBO {pbo:.2f} above 0.2")
        if dsr < 0.95:
            soft.append(f"DSR {dsr:.2f} below 0.95")
        if t < min_btl:
            soft.append(f"sample {t} shorter than MinBTL {min_btl}")
    if ci_low is None:
        soft.append("no live same-window comparison")
    elif ci_low <= 0:
        soft.append(f"same-window CI lower bound {ci_low:.3f} not above 0")
    if soft:
        return Decision("GO-REDUCED", 0.5, tuple(soft))
    return Decision("GO", 1.0, ())
