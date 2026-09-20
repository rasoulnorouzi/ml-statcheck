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


#: What each kind of test needs before a p-value can be recomputed.
#: A z test needs no degrees of freedom, and an F test needs two.
REQUIRED_PARTS = {
    "t": ("df1",), "r": ("df1",), "chi2": ("df1",), "q": ("df1",),
    "f": ("df1", "df2"), "z": (),
}

#: How to name each missing part to a reader.
#:
#: The wording says what the tool observed, and never why. The tool read the
#: text and did not find the part. It cannot know whether the author omitted it,
#: whether the PDF conversion destroyed it, or whether the extraction failed.
#: Saying "the paper does not report a p-value" claims the first, and a reader
#: would act on that claim.
PART_NAMES = {
    "statistic": "no test statistic",
    "df1": "no degrees of freedom",
    "df2": "no second degrees of freedom",
    "p_value": "no p-value",
    "p_operator": "no operator before the p-value",
}


def missing_parts(result: "Result") -> tuple:
    """Name every part the check needs and the result does not carry.

    More than one published result in five carries no p-value at all, measured
    over 323 results of this corpus. Nothing can recover a number the author
    did not print, so the tool says which part is absent instead of returning a
    bare "undecidable". A reader can then see whether the paper is incomplete
    or the tool failed.
    """
    absent = []
    if result.statistic is None:
        absent.append("statistic")
    for part in REQUIRED_PARTS.get((result.test_type or "").strip().lower(),
                                   ("df1",)):
        if getattr(result, part) is None:
            absent.append(part)
    if result.p_value is None:
        absent.append("p_value")
    elif result.p_operator not in ("=", "<", ">"):
        absent.append("p_operator")
    return tuple(absent)


def describe_missing(parts) -> str:
    """Write the missing parts as a sentence a reader can act on.

    The sentence reports an observation, not a cause. "no p-value found beside
    this result" is what happened. Whether the author omitted it, the font
    destroyed it, or the extraction missed it is a separate question, and the
    quote beside the result is what answers it.
    """
    if not parts:
        return ""
    names = [PART_NAMES.get(p, p) for p in parts]
    if len(names) == 1:
        return f"{names[0]} found beside this result"
    return (", ".join(names[:-1]) + f" and {names[-1]} found beside this "
            f"result")


@dataclass
class Check:
    """The verdict for one result."""

    verdict: str
    computed_p: Optional[float]
    reported_p: Optional[float]
    reason: str = ""
    #: The parts the check needed and the result did not carry. Empty when the
    #: result was complete, whatever the verdict.
    missing: tuple = ()


def compute_p(test_type: str, statistic: float,
              df1: Optional[float] = None, df2: Optional[float] = None,
              one_tailed: bool = False) -> Optional[float]:
    """Return the p-value implied by a test statistic, or None if it cannot be found.

    The p-value is two-tailed by default, which is what statcheck assumes.
    """
    t = (test_type or "").strip().lower()
    if statistic is None:
        return None
    # scipy returns nan, not an error, for a non-positive degree of freedom,
    # and nan would then pass through the comparison as a number.
    if (df1 is not None and df1 <= 0) or (df2 is not None and df2 <= 0):
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
    return None if p != p else float(p)     # nan never leaves this function


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

    absent = missing_parts(result)

    if computed is None:
        # Name the part that is absent. When every part is present the fault is
        # the value itself, such as a correlation at or beyond 1.
        reason = (describe_missing(absent) if absent else
                  "the statistic and its degrees of freedom give no p-value")
        return Check(UNDECIDABLE, None, result.p_value, reason, absent)
    if result.p_value is None or result.p_operator not in ("=", "<", ">"):
        return Check(UNDECIDABLE, computed, result.p_value,
                     describe_missing(absent) or
                     "there is no reported p-value to compare against",
                     absent)

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
