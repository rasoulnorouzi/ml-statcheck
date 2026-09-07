"""Recompute a p-value from a test statistic, and compare it with the reported one.

This module is deterministic mathematics. No model output ever reaches a verdict
here. A confident wrong verdict is worse than no tool at all, so this file is
the one place in the project where every branch is explicit.

The same behaviour must exist in JavaScript and in R. Keep the logic simple and
free of Python-only tricks, so the other two ports can mirror it line by line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scipy import stats

# Test names, normalised to lower case. Everything else is rejected rather
# than guessed at.
TESTS = ("t", "f", "r", "z", "chi2", "q")

# statcheck reports two kinds of problem.
CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"     # reported and computed p disagree
DECISION_ERROR = "decision_error"  # they disagree about significance
UNDECIDABLE = "undecidable"        # not enough information to judge


@dataclass
class Result:
    """One reported statistical result, as extracted from text."""

    test_type: str
    statistic: float
    df1: Optional[float] = None
    df2: Optional[float] = None
    p_operator: Optional[str] = None   # "=", "<" or ">"
    p_value: Optional[float] = None
    one_tailed: bool = False


@dataclass
class Check:
    """The verdict for one result."""

    verdict: str
    computed_p: Optional[float]
    reported_p: Optional[float]
    reason: str = ""


def compute_p(test_type: str, statistic: float,
              df1: Optional[float] = None, df2: Optional[float] = None,
              one_tailed: bool = False) -> Optional[float]:
    """Return the p-value implied by a test statistic, or None if it cannot be found.

    The p-value is two-tailed by default, which is what statcheck assumes.
    """
    t = (test_type or "").strip().lower()
    if statistic is None:
        return None

    try:
        if t == "t":
            if df1 is None:
                return None
            p = 2 * stats.t.sf(abs(statistic), df1)
        elif t == "f":
            if df1 is None or df2 is None:
                return None
            # An F test is one-tailed by construction. The `one_tailed` flag
            # refers to the hypothesis, not to this distribution, so it is not
            # applied here.
            return float(stats.f.sf(statistic, df1, df2))
        elif t == "r":
            if df1 is None:
                return None
            r = float(statistic)
            if abs(r) >= 1:
                return None
            # Convert the correlation to a t statistic on df1 degrees of freedom.
            tval = r * ((df1 / (1 - r * r)) ** 0.5)
            p = 2 * stats.t.sf(abs(tval), df1)
        elif t == "z":
            p = 2 * stats.norm.sf(abs(statistic))
        elif t in ("chi2", "q"):
            if df1 is None:
                return None
            return float(stats.chi2.sf(statistic, df1))
        else:
            return None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None

    if one_tailed:
        p = p / 2
    return float(p)


def _decimals(text: str) -> int:
    """Count the digits after the decimal point in a number written as text."""
    s = str(text)
    return len(s.split(".", 1)[1]) if "." in s else 0


def is_significant(p: float, alpha: float = 0.05, p_equal_alpha_sig: bool = True) -> bool:
    """Say whether a p-value counts as significant at `alpha`."""
    return p <= alpha if p_equal_alpha_sig else p < alpha


def check(result: Result, alpha: float = 0.05,
          p_equal_alpha_sig: bool = True,
          reported_p_text: Optional[str] = None) -> Check:
    """Compare a reported p-value with the value implied by the statistic.

    `reported_p_text` is the p-value exactly as written, for example ".03".
    It is used to learn how many decimals were reported, so the comparison
    allows for the rounding the author applied.
    """
    computed = compute_p(result.test_type, result.statistic,
                         result.df1, result.df2, result.one_tailed)

    if computed is None:
        return Check(UNDECIDABLE, None, result.p_value,
                     "the statistic and degrees of freedom do not give a p-value")
    if result.p_value is None or result.p_operator not in ("=", "<", ">"):
        return Check(UNDECIDABLE, computed, result.p_value,
                     "no reported p-value to compare against")

    reported = float(result.p_value)
    op = result.p_operator

    if op == "=":
        # The author rounded. A reported .03 stands for any value that rounds
        # to .03 at the same number of decimals.
        nd = _decimals(reported_p_text if reported_p_text is not None else result.p_value)
        tol = 0.5 * (10 ** -nd) if nd > 0 else 0.5
        agrees = abs(computed - reported) <= tol
    elif op == "<":
        agrees = computed < reported
    else:
        agrees = computed > reported

    if agrees:
        return Check(CONSISTENT, computed, reported)

    # The values disagree. A disagreement that also flips the conclusion is
    # reported separately, because it changes what the paper claims.
    if op == "=":
        reported_sig = is_significant(reported, alpha, p_equal_alpha_sig)
    elif op == "<":
        reported_sig = reported <= alpha
    else:
        reported_sig = False
    computed_sig = is_significant(computed, alpha, p_equal_alpha_sig)

    if reported_sig != computed_sig:
        return Check(DECISION_ERROR, computed, reported,
                     "the reported and computed p-values disagree about significance")
    return Check(INCONSISTENT, computed, reported,
                 "the reported and computed p-values disagree")
