"""Backtest-overfitting statistics. Research only: uses numpy, never imported by the live loop.

* PBO via CSCV: Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting". Probability that the best in-sample configuration ranks below the median
  out-of-sample, over all symmetric half/half block splits.
* PSR and DSR: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
* MinBTL: Bailey, Borwein, Lopez de Prado & Zhu (2014), "Pseudo-Mathematics and Financial
  Charlatanism".

All Sharpe ratios here are per period (per row), not annualised.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def sharpe(returns: np.ndarray, axis: int = 0) -> np.ndarray | float:
    values = np.asarray(returns, float)
    mean, sd = values.mean(axis=axis), values.std(axis=axis, ddof=1)
    out = np.divide(mean, sd, out=np.zeros_like(np.asarray(mean, float)), where=sd > 0)
    return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_combos: int
    prob_oos_loss: float
    degradation_slope: float


def _block_sums(matrix: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray, int]:
    rows, configs = matrix.shape
    per = rows // n_blocks
    if per < 2:
        raise ValueError(f"need at least {2 * n_blocks} rows for {n_blocks} blocks, got {rows}")
    blocks = matrix[rows - per * n_blocks :].reshape(n_blocks, per, configs)
    return blocks.sum(axis=1), (blocks**2).sum(axis=1), per


def _perf(sums: np.ndarray, squares: np.ndarray, count: int) -> np.ndarray:
    mean = sums / count
    var = np.maximum(squares / count - mean**2, 0.0) * count / (count - 1)
    sd = np.sqrt(var)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


def _split_masks(n_blocks: int, max_combos: int, seed: int) -> np.ndarray:
    half = n_blocks // 2
    if math.comb(n_blocks, half) <= max_combos:
        combos = itertools.combinations(range(n_blocks), half)
        return np.array([np.isin(np.arange(n_blocks), c) for c in combos], dtype=float)
    rng = np.random.default_rng(seed)
    return np.array(
        [
            np.isin(np.arange(n_blocks), rng.choice(n_blocks, half, replace=False))
            for _ in range(max_combos)
        ],
        dtype=float,
    )


def cscv_pbo(
    matrix: np.ndarray, n_blocks: int = 16, max_combos: int = 20_000, seed: int = 0
) -> PBOResult:
    """`matrix` is (T steps x N configs) of per-step PnL for every configuration tried."""
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    values = np.nan_to_num(np.asarray(matrix, float))
    sums, squares, per = _block_sums(values, n_blocks)
    masks = _split_masks(n_blocks, max_combos, seed)
    half = per * (n_blocks // 2)
    in_sample = _perf(masks @ sums, masks @ squares, half)
    out_sample = _perf((1 - masks) @ sums, (1 - masks) @ squares, half)
    best = np.argmax(in_sample, axis=1)
    rows = np.arange(len(masks))
    chosen_oos = out_sample[rows, best]
    below = (out_sample < chosen_oos[:, None]).sum(axis=1)
    ties = (out_sample == chosen_oos[:, None]).sum(axis=1)
    omega = (below + 0.5 * (ties - 1) + 1) / (values.shape[1] + 1)
    logits = np.log(omega / (1 - omega))
    chosen_is = in_sample[rows, best]
    slope = float(np.polyfit(chosen_is, chosen_oos, 1)[0]) if np.std(chosen_is) > 0 else 0.0
    return PBOResult(
        float(np.mean(logits <= 0)), logits, len(masks), float(np.mean(chosen_oos < 0)), slope
    )


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """E[max Sharpe] among n unskilled strategies whose Sharpe estimates have variance sr_var."""
    if n_trials <= 1:
        return 0.0
    z = _NORMAL.inv_cdf
    tail = (1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_var) * tail


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark), adjusting for skewness, kurtosis and length."""
    values = np.asarray(returns, float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.std(values) == 0:
        return 0.0
    observed = sharpe(values)
    standard = (values - values.mean()) / values.std()
    skew, kurt = float(np.mean(standard**3)), float(np.mean(standard**4))
    denominator = math.sqrt(max(1 - skew * observed + (kurt - 1) / 4 * observed**2, 1e-12))
    return _NORMAL.cdf((observed - sr_benchmark) * math.sqrt(len(values) - 1) / denominator)


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_var: float) -> float:
    """PSR against the Sharpe that the best of n_trials pure-noise strategies would show."""
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sr_var))


def min_backtest_length(n_trials: int, sr_per_period: float) -> int:
    """Rows needed so the best of n_trials noise strategies is unlikely to reach this Sharpe."""
    if n_trials <= 1 or sr_per_period <= 0:
        return 1
    return math.ceil(2 * math.log(n_trials) / sr_per_period**2)
