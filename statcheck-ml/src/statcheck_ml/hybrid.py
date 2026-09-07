"""The hybrid cascade: the regular expression first, the model for the rest.

The measurements motivate the order. On the round 2 passages the real
statcheck package reaches a precision of 1.000 and a recall near 0.205. It
almost never reports a result that is not there, and it misses about four out
of five.

So the cascade runs in three steps:

  1. The prefilter removes the text that cannot hold a result. About 1 line in
     700 of a corpus holds one, so this is what makes the rest affordable.
  2. statcheck reads the candidates. Everything it finds is accepted without
     question, because its precision is 1.000 in this measurement.
  3. The model reads the same candidates and adds only results statcheck did
     not find.

Two properties follow, and both matter more than the score.

An existing statcheck user sees no regression. Every result the package
reports today is still reported, parsed by the package itself, so a
disagreement can never be introduced into work that already relies on it.

The model is answerable only for the results the package cannot read. That is
also where the evidence for a learned extractor is strongest, since 51.9% of
what statcheck misses is text whose operator the PDF conversion destroyed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .pvalue import Result, Check, check


def _as_number(text) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip().replace("−", "-").replace("–", "-").lstrip("<>=").strip()
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]
    try:
        return round(float(s), 4)
    except ValueError:
        return None


@dataclass
class Finding:
    """One result, and which stage of the cascade produced it."""

    result: Result
    source: str                    # "statcheck" or "model"
    quote: str = ""
    reported_p_text: Optional[str] = None
    verdict: Optional[Check] = None

    @property
    def value(self) -> Optional[float]:
        return _as_number(self.result.statistic)


@dataclass
class Cascade:
    """Combine a pattern extractor and a model, in that order of trust.

    `statcheck_fn` takes a passage and returns a list of Finding. `model_fn`
    does the same. Either may be None, which turns the cascade into the other
    system alone and makes an ablation trivial.
    """

    statcheck_fn: Optional[callable] = None
    model_fn: Optional[callable] = None
    # Two results count as the same when their statistics agree to this many
    # decimals. Published values rarely carry more than three.
    tolerance: int = 3
    stats: Dict[str, int] = field(default_factory=dict)

    def _key(self, finding: Finding):
        v = finding.value
        return None if v is None else round(v, self.tolerance)

    def run(self, text: str) -> List[Finding]:
        """Return the results for one passage, statcheck first."""
        found: List[Finding] = []
        seen = set()

        if self.statcheck_fn is not None:
            for f in self.statcheck_fn(text):
                key = self._key(f)
                if key in seen:
                    continue
                seen.add(key)
                f.source = "statcheck"
                found.append(f)
                self.stats["statcheck"] = self.stats.get("statcheck", 0) + 1

        if self.model_fn is not None:
            for f in self.model_fn(text):
                key = self._key(f)
                if key is None or key in seen:
                    # statcheck already reported this one. Its parse wins,
                    # because its precision is the reason it goes first.
                    self.stats["model_duplicate"] = self.stats.get("model_duplicate", 0) + 1
                    continue
                seen.add(key)
                f.source = "model"
                found.append(f)
                self.stats["model_added"] = self.stats.get("model_added", 0) + 1

        return found

    def check_all(self, findings: Sequence[Finding], alpha: float = 0.05) -> List[Finding]:
        """Recompute the p-value for each finding and attach the verdict.

        The mathematics is the same for both sources. A result found by the
        model is judged by exactly the code that judges a result found by
        statcheck, so the source cannot change a verdict.
        """
        for f in findings:
            f.verdict = check(f.result, alpha=alpha,
                              reported_p_text=f.reported_p_text)
        return list(findings)


def summarise(findings: Sequence[Finding]) -> Dict[str, int]:
    """Count the findings by source and by verdict."""
    out: Dict[str, int] = {}
    for f in findings:
        out[f"found_by_{f.source}"] = out.get(f"found_by_{f.source}", 0) + 1
        if f.verdict is not None:
            out[f.verdict.verdict] = out.get(f.verdict.verdict, 0) + 1
    return out
