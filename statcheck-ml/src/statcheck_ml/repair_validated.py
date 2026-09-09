"""Repair damaged operators, and let the arithmetic choose the mapping.

The simple repair in `repair.py` assumes that a suspect character between the
degrees of freedom and a number is an equals sign. That assumption is usually
right and occasionally wrong, and when it is wrong it corrupts the whole
document, because one mapping governs every occurrence in the file.

This module removes the assumption. It tries every plausible mapping and keeps
the one the mathematics supports.

The test is independent of any annotator, which is what makes it worth the
extra work. A test statistic and its degrees of freedom determine the p-value.
If a mapping is right, the recomputed p-values agree with the reported ones
most of the time. Published work puts the true rate of disagreement near 10%,
so a mapping that produces 80% disagreement is wrong regardless of how
plausible it looked.

This is the only check in the project that no language model took part in.
"""
from __future__ import annotations

import itertools
import re
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pvalue import compute_p

#: The rules live in `spec/repair.json`, so that the Python, JavaScript and R
#: ports read one definition instead of three copies.
SPEC_PATH = Path(__file__).parent / "spec" / "repair.json"
_SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _suspect_class(spec: dict) -> str:
    """Build the character class of a possible damaged operator.

    The control characters are what the normalisation stage produces, whatever
    engine read the PDF. Without that stage this class would only match text
    from PyMuPDF, and the R port would repair nothing.
    """
    parts = []
    for low, high in spec["suspect_characters"]["control_range"]:
        parts.append(f"\\x{low:02x}-\\x{high:02x}")
    for ch in spec["suspect_characters"]["literal"]:
        parts.append("\\\\" if ch == "\\" else re.escape(ch))
    return "[" + "".join(parts) + "]"


SUSPECT_CLASS = _suspect_class(_SPEC)

# A complete result whose two operators may both be damaged.
RESULT = re.compile(
    _SPEC["result_pattern"].replace("SUSPECT", SUSPECT_CLASS),
    re.IGNORECASE,
)

OPERATORS = tuple(_SPEC["operators"])
MAX_SUSPECTS = _SPEC["max_suspects"]
MIN_TESTABLE = _SPEC["min_testable_results"]


def _num(text: str) -> Optional[float]:
    try:
        s = text.strip()
        if s.startswith("."):
            s = "0" + s
        elif s.startswith("-."):
            s = "-0" + s[1:]
        return float(s)
    except (ValueError, AttributeError):
        return None


def candidates(text: str) -> List[dict]:
    """Every complete result in the text, with its two operator characters."""
    out = []
    for m in RESULT.finditer(text):
        test, df1, df2, sop, stat, pop, pval = m.groups()
        out.append({
            "test": test.lower(), "df1": _num(df1), "df2": _num(df2),
            "stat_op": sop, "stat": _num(stat),
            "p_op": pop, "p": _num(pval),
        })
    return out


def _agrees(row: dict, p_operator: str) -> Optional[bool]:
    """Does the recomputed p-value agree with the reported one?"""
    test = row["test"]
    test = {"c2": "chi2", "v2": "chi2", "x2": "chi2", "χ2": "chi2"}.get(test, test)
    computed = compute_p(test, row["stat"], row["df1"], row["df2"])
    if computed is None or row["p"] is None:
        return None
    if p_operator == "=":
        # Allow for the rounding the author applied.
        return abs(computed - row["p"]) <= max(0.5 * 10 ** -_decimals(row["p"]), 1e-6)
    if p_operator == "<":
        return computed < row["p"]
    return computed > row["p"]


def _decimals(value: float) -> int:
    s = f"{value:.10f}".rstrip("0")
    return len(s.split(".", 1)[1]) if "." in s else 0


def score_mapping(rows: List[dict], mapping: Dict[str, str]) -> Tuple[int, int]:
    """Return how many results agree, and how many were testable."""
    agree = testable = 0
    for row in rows:
        sop = mapping.get(row["stat_op"], row["stat_op"])
        pop = mapping.get(row["p_op"], row["p_op"])
        if sop != "=":
            # A test statistic is reported with an equals sign. A mapping that
            # makes it an inequality is not worth testing further.
            continue
        verdict = _agrees(row, pop)
        if verdict is None:
            continue
        testable += 1
        agree += bool(verdict)
    return agree, testable


def infer_validated(text: str, min_testable: int = MIN_TESTABLE) -> Tuple[Dict[str, str], dict]:
    """Choose the mapping the arithmetic supports best.

    Returns the mapping and a small report. When too few results can be tested,
    the mapping is empty and the caller should fall back to the simple rule.
    """
    rows = candidates(text)
    suspects = sorted({r["stat_op"] for r in rows} | {r["p_op"] for r in rows}
                      - set(OPERATORS))
    suspects = [c for c in suspects if c not in OPERATORS]
    if not rows or not suspects or len(suspects) > MAX_SUSPECTS:
        return {}, {"reason": "nothing to infer", "results": len(rows)}

    best, best_score = None, (-1.0, 0)
    tried = []
    for combo in itertools.product(OPERATORS, repeat=len(suspects)):
        mapping = dict(zip(suspects, combo))
        agree, testable = score_mapping(rows, mapping)
        if testable < min_testable:
            continue
        rate = agree / testable
        tried.append({"mapping": {repr(k): v for k, v in mapping.items()},
                      "agreement": rate, "testable": testable})
        if (rate, testable) > best_score:
            best, best_score = mapping, (rate, testable)

    if best is None:
        return {}, {"reason": "too few testable results", "results": len(rows)}

    return best, {
        "chosen": {repr(k): v for k, v in best.items()},
        "agreement": best_score[0],
        "testable": best_score[1],
        "alternatives": sorted(tried, key=lambda t: -t["agreement"])[:3],
    }


def repair_validated(text: str) -> Tuple[str, dict]:
    """Repair the text with the mapping the arithmetic chose."""
    mapping, info = infer_validated(text)
    if not mapping:
        from .repair import repair
        fixed, simple_map, n = repair(text)
        info["fallback"] = "simple rule"
        info["chosen"] = {repr(k): v for k, v in simple_map.items()}
        info["replacements"] = n
        return fixed, info

    # Replace only in the two positions an operator can occupy.
    def fix(m):
        whole = m.group(0)
        for ch, op in mapping.items():
            whole = whole.replace(ch, op)
        return whole

    fixed = RESULT.sub(fix, text)
    info["replacements"] = sum(text.count(c) for c in mapping)
    return fixed, info
