"""Helpers for `pipeline/09_evaluate.py`: matching, scoring, and the small
statistics that compare systems on the holdout.

Nothing here reads a checkpoint or an ONNX graph. This module turns spans and
CSV rows into counts, and counts into precision, recall and F1. The value
match is the one the version-1 evaluator used: statistics are compared as
numbers rounded to four decimal places, not as text.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .align import as_number
from .data import normalise
from .labels import ENTITY_OPERATOR, tags_to_spans
from .pvalue import CONSISTENT, DECISION_ERROR, UNDECIDABLE, Result, check
from .stats import wilson

# ---------------------------------------------------------------- I/O -----

def read_json(path) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read(), strict=False)


def read_jsonl(path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line, strict=False))
    return rows


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo=None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ------------------------------------------------------- damage family ----

CONTROL_CLASS = "[" + "".join(
    chr(c) for c in list(range(1, 9)) + [11, 12] + list(range(14, 32))) + "]"

# Ordered: the first pattern that matches names the family. One result can
# carry two kinds of damage, and the first entry names the one that does the
# most harm.
DAMAGE_FAMILIES = [
    ("decimal point lost", re.compile(r"=\s*\d{3,}\s*,\s*p\s*[<>=]\s*\d{3,}")),
    ("control character", re.compile(CONTROL_CLASS)),
    ("fraction sign for =", re.compile("¼")),
    ("backslash for <", re.compile(r"\\")),
    ("letter b for <", re.compile(r"\)\s*b\s*[-.0-9]|\bp\s*b\s*\.?[0-9]")),
    ("letter N for >", re.compile(r"\bp\s*N\s*\.?[0-9]")),
    ("chi-square symbol lost",
     re.compile(r"\b(v2|c2|x2)\s*\(|(?<![0-9A-Za-z])2\s*\(\s*\d")),
    ("letter p or ! for operator",
     re.compile(r"\)\s*p\s*[-.0-9]|\bp\s*!\s*\.?[0-9]")),
]


def damage_family(quote: str) -> str:
    for name, pattern in DAMAGE_FAMILIES:
        if pattern.search(quote or ""):
            return name
    return "other damage"


# -------------------------------------------------------------- gold ------

def extract_gold(windows: Sequence[dict]):
    """From the holdout rows: gold values per window, their metadata, gold
    block spans per window, windows grouped by document, and the gold
    results `as_number` could not parse at all (excluded from value matching
    entirely, so a system can neither find them nor be blamed for missing
    them).
    """
    gold_by_window: Dict[str, set] = defaultdict(set)
    gold_meta: Dict[Tuple[str, float], dict] = {}
    gold_spans_by_window: Dict[str, set] = defaultdict(set)
    doc_to_windows: Dict[str, list] = defaultdict(list)
    gold_unparseable: List[dict] = []

    for w in windows:
        wid = w["window_id"]
        doc_to_windows[w.get("source_doc")].append(wid)
        for res in w.get("results") or []:
            v = as_number(res.get("statistic"))
            if v is None:
                gold_unparseable.append({"window_id": wid, "statistic": res.get("statistic")})
                continue
            gold_by_window[wid].add(v)
            block_span = tuple(res["block_span"]) if res.get("block_span") else None
            if block_span:
                gold_spans_by_window[wid].add(block_span)
            fam = damage_family(res.get("quote") or "") if res.get("damaged") else None
            gold_meta.setdefault((wid, v), {
                "damaged": bool(res.get("damaged")),
                "test_type": (res.get("test_type") or "").lower(),
                "checkable": bool(res.get("checkable")),
                "family": fam,
                "tier": res.get("tier"),
                "block_span": block_span,
            })
    return (dict(gold_by_window), gold_meta, dict(gold_spans_by_window), dict(doc_to_windows),
           gold_unparseable)


# ---------------------------------------------------------- statcheck -----

def load_statcheck(path) -> Tuple[Dict[str, set], List[dict]]:
    """Read a statcheck CSV. Returns {window_id: {value, ...}} and the rows."""
    by_window: Dict[str, set] = defaultdict(set)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
            v = as_number(row.get("test_value"))
            if v is not None:
                by_window[row["window_id"]].add(v)
    return dict(by_window), rows


def _csv_degrees_of_freedom(row: dict, test_type: str) -> Tuple[Optional[float], Optional[float]]:
    """Read df1/df2 the way statcheck's own CSV stores them.

    An F test uses both columns. Every other test needs one degree-of-freedom
    number, and statcheck writes it in whichever column applies to that test
    (df1 for chi-square, df2 for t) and leaves the other NA. Reading both and
    keeping whichever is not empty works for every single-df test without a
    table of which column belongs to which.
    """
    df1_raw, df2_raw = as_number(row.get("df1")), as_number(row.get("df2"))
    if test_type == "f":
        return df1_raw, df2_raw
    return (df1_raw if df1_raw is not None else df2_raw), None


def printed_decimals(row: dict, key: str) -> str:
    """The number as the paper printed it, taken from statcheck's `raw` match.

    Both rounding rules count decimals, and the CSV holds parsed numbers: a
    printed 5.10 is written 5.1, and the comparison would then allow more
    room than the paper does. The raw text still has the digits.
    """
    raw = (row.get("raw") or "").replace("\n", " ")
    value = (row.get(key) or "").strip()
    if not value:
        return value
    try:
        wanted = float(value)
    except ValueError:
        return value
    # The token that equals the number, not the first token that contains its
    # digits: the 0.9 inside 10.9 would otherwise pass for a reported .21.
    for token in re.findall(r"\d*\.\d+|\d+", raw):
        try:
            if float(token) == wanted:
                return token
        except ValueError:
            continue
    return value


def verdict_agreement(rows: Iterable[dict]) -> dict:
    """Compare this project's p-value math with statcheck's own error flag,
    on the results statcheck itself reports. Checks the arithmetic, not the
    extraction: the one part of the pipeline that is not machine-labelled.
    """
    agree = disagree = undecidable = 0
    for row in rows:
        stat = as_number(row.get("test_value"))
        if stat is None:
            continue
        test_type = (row.get("test_type") or "").strip().lower()
        df1, df2 = _csv_degrees_of_freedom(row, test_type)
        res = Result(
            test_type=test_type,
            statistic=stat,
            df1=df1,
            df2=df2,
            p_operator=(row.get("p_comp") or "").strip() or None,
            p_value=as_number(row.get("reported_p")),
        )
        ours = check(res, reported_p_text=printed_decimals(row, "reported_p"),
                     statistic_text=printed_decimals(row, "test_value"))
        if ours.computed_p is None or ours.verdict == UNDECIDABLE:
            # No p to compare: statcheck writes `ns` as the operator and NA as
            # the value, and reports no error. That is not a disagreement.
            undecidable += 1
            continue
        theirs = str(row.get("error", "")).strip().upper() in ("TRUE", "1")
        theirs_decision = str(row.get("decision_error", "")).strip().upper() in ("TRUE", "1")
        if ((ours.verdict != CONSISTENT) == theirs
                and (ours.verdict == DECISION_ERROR) == theirs_decision):
            agree += 1
        else:
            disagree += 1
    return {"agree": agree, "disagree": disagree, "undecidable": undecidable}


# ------------------------------------------------------- model spans ------

def group_spans(text: str, spans: Sequence[tuple]):
    """Turn a flat span list into (parts, part_spans) pairs, one per result.

    A new result starts at a TEST tag, or at a second STAT tag with no TEST
    between them, matching the boundary the annotators used.
    """
    results, current, current_spans = [], None, []
    for start, end, label in sorted(spans):
        if label == "TEST" or (label == "STAT" and current and "STAT" in current):
            if current:
                results.append((current, current_spans))
            current, current_spans = {}, []
        if current is None:
            current, current_spans = {}, []
        current.setdefault(label, text[start:end])
        current_spans.append((start, end))
    if current:
        results.append((current, current_spans))
    return results


def build_result(parts: dict) -> Tuple[Optional[Result], Optional[str]]:
    stat = as_number(parts.get("STAT"))
    if stat is None:
        return None, None
    operator = next((ENTITY_OPERATOR[k] for k in parts if k.startswith("POP_")), None)
    return Result(
        test_type=(parts.get("TEST") or "").strip().lower() or "t",
        statistic=stat,
        df1=as_number(parts.get("DF1")),
        df2=as_number(parts.get("DF2")),
        p_operator=operator,
        p_value=as_number(parts.get("PVAL")),
    ), parts.get("PVAL")


def model_predictions(tagger, windows: Sequence[dict]) -> Dict[str, list]:
    """Return {window_id: [(Result, p_text, block_span), ...]} for one tagger."""
    out: Dict[str, list] = {}
    for w in windows:
        text = normalise(w["text"])
        tags = tagger.tag_text(text)
        built = []
        for parts, part_spans in group_spans(text, tags_to_spans(tags)):
            res, ptext = build_result(parts)
            if res is None:
                continue
            starts = [s for s, _ in part_spans]
            ends = [e for _, e in part_spans]
            built.append((res, ptext, (min(starts), max(ends))))
        out[w["window_id"]] = built
    return out


# -------------------------------------------------------------- score -----

def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}


def score_system(found_by_window: Dict[str, set], gold_by_window: Dict[str, set],
                 gold_meta: dict, window_ids: Sequence[str]) -> Dict[str, dict]:
    """Score one system, overall and by subset, matching on statistic value."""
    buckets: Dict[str, list] = defaultdict(lambda: [0, 0, 0])
    for wid in window_ids:
        gold = gold_by_window.get(wid, set())
        got = found_by_window.get(wid, set())
        for v in got & gold:
            meta = gold_meta.get((wid, v), {})
            buckets["overall"][0] += 1
            buckets["damaged" if meta.get("damaged") else "undamaged"][0] += 1
            buckets[f"test:{meta.get('test_type') or 'unknown'}"][0] += 1
            if meta.get("family"):
                buckets[f"damage:{meta['family']}"][0] += 1
            if meta.get("checkable"):
                buckets["checkable"][0] += 1
        for _ in got - gold:
            buckets["overall"][1] += 1
        for v in gold - got:
            meta = gold_meta.get((wid, v), {})
            buckets["overall"][2] += 1
            buckets["damaged" if meta.get("damaged") else "undamaged"][2] += 1
            buckets[f"test:{meta.get('test_type') or 'unknown'}"][2] += 1
            if meta.get("family"):
                buckets[f"damage:{meta['family']}"][2] += 1
            if meta.get("checkable"):
                buckets["checkable"][2] += 1
    return {k: prf(*v) for k, v in buckets.items()}


def score_span_strict(found_spans_by_window: Dict[str, set],
                      gold_spans_by_window: Dict[str, set],
                      window_ids: Sequence[str]) -> dict:
    """Predicted block span equal to the gold block span, exactly."""
    tp = fp = fn = 0
    for wid in window_ids:
        gold = gold_spans_by_window.get(wid, set())
        got = found_spans_by_window.get(wid, set())
        tp += len(got & gold)
        fp += len(got - gold)
        fn += len(gold - got)
    return prf(tp, fp, fn)


# ----------------------------------------------------------- bootstrap ----

def bootstrap_prf(doc_ids: Sequence[str], doc_to_windows: Dict[str, list],
                  found_by_window: Dict[str, set], gold_by_window: Dict[str, set],
                  gold_meta: dict, n: int = 2000, seed: int = 0,
                  buckets: Sequence[str] = ("overall", "damaged")) -> dict:
    """95% percentile CI for p, r, f1, resampling whole documents.

    One draw of the loop stands in for `bootstrap_ci`, called once per metric:
    both use the same percentile method over `numpy.random.default_rng(seed)`,
    but a single pass here computes p, r and f1 together instead of redoing
    the resample for each one.
    """
    docs = list(doc_ids)
    m = len(docs)
    samples = {b: {"p": [], "r": [], "f1": []} for b in buckets}
    if m == 0:
        return {b: {k: (0.0, 0.0) for k in ("p", "r", "f1")} for b in buckets}

    rng = np.random.default_rng(seed)
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        window_ids = [wid for j in idx for wid in doc_to_windows.get(docs[j], [])]
        scored = score_system(found_by_window, gold_by_window, gold_meta, window_ids)
        for b in buckets:
            s = scored.get(b, {"p": 0.0, "r": 0.0, "f1": 0.0})
            samples[b]["p"].append(s["p"])
            samples[b]["r"].append(s["r"])
            samples[b]["f1"].append(s["f1"])

    return {b: {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
               for k, v in samples[b].items()} for b in buckets}


# ---------------------------------------------------------- mcnemar -------

def hit_map(found_by_window: Dict[str, set], gold_by_window: Dict[str, set]) -> dict:
    """{(window_id, value): True/False} — whether a system found each gold result."""
    out = {}
    for wid, values in gold_by_window.items():
        got = found_by_window.get(wid, set())
        for v in values:
            out[(wid, v)] = v in got
    return out


def mcnemar_pair(found_a: Dict[str, set], found_b: Dict[str, set],
                 gold_by_window: Dict[str, set]) -> Tuple[int, int]:
    """b = gold results only A found, c = gold results only B found."""
    ha = hit_map(found_a, gold_by_window)
    hb = hit_map(found_b, gold_by_window)
    b = sum(1 for k in ha if ha[k] and not hb[k])
    c = sum(1 for k in ha if hb[k] and not ha[k])
    return b, c


# ------------------------------------------------------ family recall -----

def family_recall(found_by_window: Dict[str, set], gold_by_window: Dict[str, set],
                  gold_meta: dict) -> dict:
    n = Counter()
    k = Counter()
    for wid, values in gold_by_window.items():
        got = found_by_window.get(wid, set())
        for v in values:
            fam = gold_meta.get((wid, v), {}).get("family")
            if not fam:
                continue
            n[fam] += 1
            if v in got:
                k[fam] += 1
    out = {}
    for fam in n:
        r = k[fam] / n[fam] if n[fam] else 0.0
        out[fam] = {"k": k[fam], "n": n[fam], "r": r, "wilson": list(wilson(k[fam], n[fam]))}
    return out
