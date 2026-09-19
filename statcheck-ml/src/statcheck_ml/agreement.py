"""Inter-rater agreement: labels, spans, and result fields.

Measured coarse to fine: window level (does a window hold a result), result
level (do raters point at the same span), field level (for a result every
rater found, did they read it the same way). Nothing here judges a result;
it measures how much the raters agree, which caps how well a distilled model
can agree with them.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from statcheck_ml.align import as_number

CATEGORICAL_FIELDS = ["test_type", "p_operator", "damaged"]
NUMERIC_FIELDS = ["statistic", "df1", "df2", "n", "p_value"]

def cohen_kappa(a: list, b: list) -> float:
    """Chance-corrected agreement between two raters over the same items."""
    if len(a) != len(b):
        raise ValueError("cohen_kappa needs two sequences of equal length")
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b), key=lambda v: (v is None, str(v)))
    index = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        confusion[index[x], index[y]] += 1
    po = float(np.trace(confusion)) / n
    pa, pb = confusion.sum(axis=1) / n, confusion.sum(axis=0) / n
    pe = float((pa * pb).sum())
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)

def fleiss_kappa(table: List[List[int]]) -> float:
    """Chance-corrected agreement across raters and categories.

    `table[i][j]` is how many raters put subject `i` in category `j`; every
    subject needs the same rater count.
    """
    mat = np.array(table, dtype=float)
    n_subjects = mat.shape[0]
    if n_subjects == 0:
        return float("nan")
    per_subject = mat.sum(axis=1)
    if not np.allclose(per_subject, per_subject[0]):
        raise ValueError("fleiss_kappa needs the same rater count on every subject")
    n = per_subject[0]
    p_e = float(((mat.sum(axis=0) / (n_subjects * n)) ** 2).sum())
    p_bar = float((((mat ** 2).sum(axis=1) - n) / (n * (n - 1))).mean())
    return 1.0 if p_e == 1.0 else (p_bar - p_e) / (1 - p_e)

def krippendorff_alpha(ratings: List[list], level: str = "nominal") -> float:
    """Agreement across any number of raters. `None` means missing."""
    import krippendorff as _krippendorff

    label_ids: Dict[object, float] = {}

    def to_id(value):
        if value is None:
            return np.nan
        return label_ids.setdefault(value, float(len(label_ids)))

    data = np.array([[to_id(v) for v in row] for row in ratings], dtype=float)
    return float(_krippendorff.alpha(reliability_data=data, level_of_measurement=level))

def _span_tuple(span) -> Optional[Tuple[int, int]]:
    return None if span is None else (span[0], span[1])

def match_results(a: List[dict], b: List[dict], mode: str) -> List[Tuple[int, int]]:
    """Greedy one-to-one matching of two raters' results in one window.

    `strict` needs equal spans; `lenient` needs the overlap to cover at
    least half of the shorter span. With no span on either side, matching
    falls back to the normalised statistic value, compared by magnitude: a
    damaged minus sign can make one rater write -0.23 and another 0.23 for
    the same result.
    """
    scored: List[Tuple[float, int, int]] = []
    for i, ra in enumerate(a):
        sa = _span_tuple(ra.get("span"))
        for j, rb in enumerate(b):
            sb = _span_tuple(rb.get("span"))
            if sa is not None and sb is not None:
                if mode == "strict":
                    if sa != sb:
                        continue
                    score = max(1, sa[1] - sa[0])
                elif mode == "lenient":
                    overlap = max(0, min(sa[1], sb[1]) - max(sa[0], sb[0]))
                    shorter = min(sa[1] - sa[0], sb[1] - sb[0])
                    if shorter <= 0 or overlap < 0.5 * shorter:
                        continue
                    score = overlap
                else:
                    raise ValueError(f"unknown mode: {mode}")
            else:
                na, nb = as_number(ra.get("statistic")), as_number(rb.get("statistic"))
                if na is None or nb is None or abs(na) != abs(nb):
                    continue
                score = 1.0
            scored.append((score, i, j))
    scored.sort(key=lambda t: -t[0])
    used_a: set = set()
    used_b: set = set()
    pairs: List[Tuple[int, int]] = []
    for _, i, j in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    pairs.sort()
    return pairs

def pairwise_f1(a: Dict[str, list], b: Dict[str, list], mode: str) -> float:
    """F1 between two raters' results, pooled over every shared window.

    All-empty on both sides returns 1.0: two raters who find nothing agree.
    """
    keys = set(a.keys()) | set(b.keys())
    tp = fp = fn = 0
    for k in keys:
        la, lb = a.get(k, []), b.get(k, [])
        pairs = match_results(la, lb, mode=mode)
        tp += len(pairs)
        fp += len(la) - len(pairs)
        fn += len(lb) - len(pairs)
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bootstrap_ci(units: Sequence, statistic: Callable[[list], float], n: int = 1000,
                  seed: int = 0, level: float = 0.95) -> Tuple[float, float]:
    """A percentile bootstrap interval for `statistic` over `units`."""
    rng = np.random.default_rng(seed)
    units = list(units)
    m = len(units)
    if m == 0:
        return float("nan"), float("nan")
    values = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        values[i] = statistic([units[j] for j in idx])
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))

def _common_window_ids(raters: Dict[str, Dict[str, dict]]) -> List[str]:
    sets = [set(records.keys()) for records in raters.values()]
    return sorted(set.intersection(*sets)) if sets else []

def window_agreement(raters: Dict[str, Dict[str, dict]]) -> dict:
    """Agreement on whether a window holds a result at all."""
    names = list(raters.keys())
    window_ids = _common_window_ids(raters)
    labels = {name: [bool(raters[name][w]["contains_result"]) for w in window_ids]
              for name in names}
    agree = sum(1 for w in window_ids
                if len({raters[name][w]["contains_result"] for name in names}) == 1)
    raw_pct = agree / len(window_ids) if window_ids else float("nan")
    cohen = {f"{names[i]}-{names[j]}": cohen_kappa(labels[names[i]], labels[names[j]])
             for i in range(len(names)) for j in range(i + 1, len(names))}
    table = [[sum(1 for name in names if raters[name][w]["contains_result"] is False),
              sum(1 for name in names if raters[name][w]["contains_result"] is True)]
             for w in window_ids]
    fleiss = fleiss_kappa(table) if window_ids else float("nan")
    alpha = (krippendorff_alpha([labels[name] for name in names], level="nominal")
             if window_ids else float("nan"))
    return {"raw_pct": raw_pct, "cohen": cohen, "fleiss": fleiss, "alpha": alpha}

def result_agreement(raters: Dict[str, Dict[str, dict]], mode: str) -> dict:
    """Span-level F1 between every pair of raters, pooled over windows."""
    names = list(raters.keys())
    window_ids = _common_window_ids(raters)
    results = {name: {w: raters[name][w].get("results", []) for w in window_ids}
               for name in names}
    pairwise = {f"{names[i]}-{names[j]}": pairwise_f1(results[names[i]], results[names[j]], mode=mode)
                for i in range(len(names)) for j in range(i + 1, len(names))}
    mean = sum(pairwise.values()) / len(pairwise) if pairwise else 1.0
    return {"pairwise": pairwise, "mean": mean}

def field_agreement(raters: Dict[str, Dict[str, dict]]) -> dict:
    """Agreement on the content of results every rater found.

    A result counts only when every rater has a lenient span match for it,
    linked transitively (A matched B, B matched C puts all three in one
    group). Fields are compared only across these grouped results.
    """
    names = list(raters.keys())
    window_ids = _common_window_ids(raters)
    n_raters = len(names)
    groups: List[Dict[str, dict]] = []
    for w in window_ids:
        results = {name: raters[name][w].get("results", []) for name in names}
        nodes = [(name, i) for name in names for i in range(len(results[name]))]
        parent = {node: node for node in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                na, nb = names[i], names[j]
                for ia, ib in match_results(results[na], results[nb], mode="lenient"):
                    ra, rb = find((na, ia)), find((nb, ib))
                    if ra != rb:
                        parent[rb] = ra
        clusters = defaultdict(list)
        for node in nodes:
            clusters[find(node)].append(node)
        for members in clusters.values():
            if len(members) == n_raters and len({m[0] for m in members}) == n_raters:
                groups.append({name: results[name][idx] for name, idx in members})

    n_matched = len(groups)
    out: dict = {"n_results_all_raters": n_matched}
    for field in CATEGORICAL_FIELDS:
        out[field] = (krippendorff_alpha([[g[name].get(field) for g in groups] for name in names],
                                          level="nominal") if n_matched else float("nan"))
    for field in NUMERIC_FIELDS:
        if n_matched == 0:
            out[field] = float("nan")
            continue
        exact = sum(1 for g in groups
                    if len({as_number(g[name].get(field)) for name in names}) == 1)
        out[field] = exact / n_matched
    return out
