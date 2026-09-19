"""The three statistics the evaluation needs: an interval, an exact test, and
a paired resampling test. Nothing here judges a result; `pvalue.py` keeps that
job. This module only describes how well a system was measured.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

from .agreement import bootstrap_ci

__all__ = ["wilson", "mcnemar_exact", "paired_bootstrap", "bootstrap_ci"]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """The Wilson score interval for a proportion k / n.

    Used for recall on small subsets, such as one damage family, where a
    normal interval can fall outside [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((centre - spread) / denom, (centre + spread) / denom)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs b and c.

    b and c are the counts of gold results one system alone found. Agreement
    on the rest of the gold set carries no information about which system
    finds more, so only the discordant pairs enter the test.
    """
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def paired_bootstrap(units: Sequence, stat_a: Callable[[Sequence], float],
                      stat_b: Callable[[Sequence], float], n: int = 2000,
                      seed: int = 0) -> Tuple[float, Tuple[float, float], float]:
    """Bootstrap the difference stat_a(units) - stat_b(units).

    Units are resampled together, so a unit that favours one system keeps its
    paired counterpart. Returns the observed difference, a 95% percentile
    interval over the resampled differences, and a two-sided p-value: twice
    the smaller share of resampled differences on the near-zero side, capped
    at 1.
    """
    units = list(units)
    diff = stat_a(units) - stat_b(units)
    m = len(units)
    if m == 0:
        return diff, (diff, diff), 1.0

    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        sample = [units[j] for j in idx]
        diffs[i] = stat_a(sample) - stat_b(sample)

    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    share_le = float(np.mean(diffs <= 0))
    share_ge = float(np.mean(diffs >= 0))
    p = min(1.0, 2 * min(share_le, share_ge))
    return diff, (lo, hi), p
