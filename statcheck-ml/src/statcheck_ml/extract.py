"""The pattern-based extractor. This is the baseline, not the product.

It reproduces what the R package `statcheck` finds: a test statistic reported
in APA style, with its degrees of freedom and its p-value.

Two rules govern this file.

1. It keeps its faults. Improvement is measured against this baseline, so the
   baseline must stay faithful rather than become better. If the R version
   misses a case, this version misses it too.
2. It is never the fallback for the model. Mixing the two would make the
   comparison meaningless.

Parity against R is not yet proved, because R is not installed. Until it is,
treat this as a close port and not a verified one. `PLAN.md` phase 2a records
this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Optional

# A number, with an optional sign and an optional leading decimal point.
_NUM = r"-?\d*\.?\d+"

# One pattern for each family of test. Named groups carry the parts out.
# Spacing is permissive, because typesetting varies between publishers.
PATTERNS = {
    "t": re.compile(
        rf"\bt\s*\(\s*(?P<df1>{_NUM})\s*\)\s*(?P<sop>[=<>])\s*(?P<stat>{_NUM})"
        rf"\s*,\s*p\s*(?P<pop>[=<>])\s*(?P<p>{_NUM})", re.I),
    "f": re.compile(
        rf"\bF\s*\(\s*(?P<df1>{_NUM})\s*,\s*(?P<df2>{_NUM})\s*\)\s*(?P<sop>[=<>])\s*(?P<stat>{_NUM})"
        rf"\s*,\s*p\s*(?P<pop>[=<>])\s*(?P<p>{_NUM})", re.I),
    "r": re.compile(
        rf"\br\s*\(\s*(?P<df1>{_NUM})\s*\)\s*(?P<sop>[=<>])\s*(?P<stat>{_NUM})"
        rf"\s*,\s*p\s*(?P<pop>[=<>])\s*(?P<p>{_NUM})", re.I),
    "z": re.compile(
        rf"\bz\s*(?P<sop>[=<>])\s*(?P<stat>{_NUM})"
        rf"\s*,\s*p\s*(?P<pop>[=<>])\s*(?P<p>{_NUM})", re.I),
    "chi2": re.compile(
        r"\b(?:χ\s*2|χ2|chi2|X2|c2)\s*\(\s*(?P<df1>" + _NUM + r")"
        r"(?:\s*,\s*N\s*[=<>]\s*(?P<n>[\d,]+))?\s*\)\s*(?P<sop>[=<>])\s*(?P<stat>" + _NUM + r")"
        r"\s*,\s*p\s*(?P<pop>[=<>])\s*(?P<p>" + _NUM + r")", re.I),
    "q": re.compile(
        rf"\bQ(?:w|b)?\s*\(\s*(?P<df1>{_NUM})\s*\)\s*(?P<sop>[=<>])\s*(?P<stat>{_NUM})"
        rf"\s*,\s*p\s*(?P<pop>[=<>])\s*(?P<p>{_NUM})", re.I),
}


@dataclass
class Extraction:
    """One result found by the pattern, with where it was found."""

    test_type: str
    statistic: str
    df1: Optional[str]
    df2: Optional[str]
    n: Optional[str]
    p_operator: str
    p_value: str
    start: int
    end: int
    raw: str

    def as_dict(self) -> dict:
        return {
            "test_type": self.test_type, "statistic": self.statistic,
            "df1": self.df1, "df2": self.df2, "n": self.n,
            "p_operator": self.p_operator, "p_value": self.p_value,
            "quote": self.raw, "block_span": [self.start, self.end],
        }


def extract(text: str) -> List[Extraction]:
    """Find every result the pattern can read, ordered by position.

    Whitespace inside a result is not normalised first. That is deliberate:
    the offsets must point into the text the caller passed in.
    """
    found: List[Extraction] = []
    for name, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            g = m.groupdict()
            found.append(Extraction(
                test_type=name,
                statistic=g["stat"],
                df1=g.get("df1"),
                df2=g.get("df2"),
                n=g.get("n"),
                p_operator=g["pop"],
                p_value=g["p"],
                start=m.start(),
                end=m.end(),
                raw=m.group(0),
            ))
    found.sort(key=lambda e: (e.start, e.end))

    # A z pattern can also match inside a longer result. Drop any extraction
    # that sits wholly inside another one.
    kept: List[Extraction] = []
    for e in found:
        if any(o is not e and o.start <= e.start and e.end <= o.end for o in found):
            continue
        kept.append(e)
    return kept


def iter_extractions(texts) -> Iterator[tuple]:
    """Yield (index, Extraction) for a sequence of texts."""
    for i, text in enumerate(texts):
        for e in extract(text):
            yield i, e
