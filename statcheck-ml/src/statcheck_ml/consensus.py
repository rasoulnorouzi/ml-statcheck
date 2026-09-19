"""Consensus across three raters, for one window at a time.

Three raters label the same window, blind to each other. Two of three
agreeing on a result is enough to keep it without a human look. A result
only one rater found, or a result on which the three raters write three
different values for one field, goes to the adjudicator instead. Nothing
here judges whether a result is correct; it only decides which results
need a fourth opinion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from statcheck_ml.agreement import match_results, NUMERIC_FIELDS
from statcheck_ml.align import as_number

NINE_FIELDS = ["test_type", "statistic", "df1", "df2", "n",
               "p_operator", "p_value", "quote", "damaged"]
VOTED_FIELDS = NINE_FIELDS + ["confidence"]

Node = Tuple[str, int]


def _norm_ws(text) -> Optional[str]:
    return None if text is None else " ".join(str(text).split())


def _field_key(field: str, candidate: dict):
    """The value used to compare `field` across candidates."""
    value = candidate.get(field)
    if field in NUMERIC_FIELDS:
        return as_number(value)
    if field == "quote":
        return _norm_ws(value)
    return value


def _span_tuple(span) -> Optional[tuple]:
    return None if span is None else tuple(span)


def _cluster(raters: Dict[str, List[dict]]) -> List[List[Node]]:
    """Union-find clustering of results, transitively, by lenient span match.

    A merge that would put two results from the same rater in one cluster
    is skipped, so every cluster keeps at most one result per rater.
    """
    names = list(raters.keys())
    nodes: List[Node] = [(name, i) for name in names for i in range(len(raters[name]))]
    parent: Dict[Node, Node] = {n: n for n in nodes}
    members: Dict[Node, Dict[str, Node]] = {n: {n[0]: n} for n in nodes}

    def find(x: Node) -> Node:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(u: Node, v: Node) -> None:
        ru, rv = find(u), find(v)
        if ru == rv:
            return
        if set(members[ru]) & set(members[rv]):
            return  # would duplicate a rater in one cluster; skip the merge
        parent[rv] = ru
        members[ru].update(members[rv])
        del members[rv]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            for ia, ib in match_results(raters[na], raters[nb], mode="lenient"):
                union((na, ia), (nb, ib))

    # Group by root in the order `nodes` was built, not by set iteration:
    # a set of tuples hashes strings, and string hashing is randomised per
    # process, which would make cluster order (and so dispute numbering)
    # change from one run to the next on identical input.
    groups: Dict[Node, List[Node]] = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return [sorted(g, key=lambda n: names.index(n[0])) for g in groups.values()]


def _majority_span(ordered: List[dict]) -> Optional[list]:
    spans = [_span_tuple(c.get("span")) for c in ordered]
    counts = Counter(s for s in spans if s is not None)
    if counts:
        best, best_count = counts.most_common(1)[0]
        if best_count >= 2:
            return list(best)
    for s in spans:
        if s is not None:
            return list(s)
    return None


def _resolve_field(field: str, ordered: List[dict]) -> Tuple[object, int, bool]:
    """Vote on one field. Returns (raw value, vote count, is a 3-way conflict)."""
    keys = [_field_key(field, c) for c in ordered]
    counts = Counter(keys)
    best_key, best_count = counts.most_common(1)[0]
    if best_count >= 2:
        rep = next(c for c, k in zip(ordered, keys) if k == best_key)
        return rep.get(field), best_count, False
    if len(ordered) == 3:
        return None, 1, True  # three raters, three different values: a dispute
    return ordered[0].get(field), 1, False  # a two-way tie: the first rater wins


def consensus(raters: Dict[str, List[dict]]) -> dict:
    """2-of-3 consensus over one window's results, from three raters.

    `raters` maps a rater name to that rater's list of result dicts for
    this window. Returns `{"kept": [...], "disputes": [...]}`.
    """
    kept: List[dict] = []
    disputes: List[dict] = []

    for cluster in _cluster(raters):
        ordered = [dict(raters[name][idx]) for name, idx in cluster]
        cluster_names = [name for name, _ in cluster]

        if len(ordered) == 1:
            disputes.append({"kind": "singleton", "field": None,
                              "candidates": [dict(ordered[0])], "raters": cluster_names})
            continue

        # A cluster of two or three is kept outright. When three raters wrote
        # three different values for one field, the first such field (nine-
        # key order) is left `None` here and raised as a dispute below; the
        # rest of the record is unaffected.
        votes: Dict[str, int] = {}
        values: Dict[str, object] = {}
        conflict_field: Optional[str] = None
        for field in NINE_FIELDS:
            value, count, is_conflict = _resolve_field(field, ordered)
            if is_conflict and conflict_field is None:
                conflict_field = field
            votes[field] = count
            values[field] = value

        conf_value, conf_count, _ = _resolve_field("confidence", ordered)
        votes["confidence"] = conf_count
        values["confidence"] = conf_value

        # The quote follows the span, not the vote: whichever candidate's own
        # span is the chosen span keeps its literal text; otherwise the
        # first. Skipped when quote itself is the pending field.
        span = _majority_span(ordered)
        if conflict_field != "quote":
            span_owner = next((c for c in ordered
                                if _span_tuple(c.get("span")) == _span_tuple(span)), ordered[0])
            values["quote"] = span_owner.get("quote")

        unanimous = (conflict_field is None and len(cluster_names) == 3
                     and all(votes[f] == 3 for f in VOTED_FIELDS))

        record = dict(values)
        record["span"] = span
        record["raters"] = cluster_names
        record["votes"] = votes
        record["tier"] = "unanimous" if unanimous else "majority"
        if conflict_field is not None:
            record["pending_field"] = conflict_field
        kept.append(record)

        if conflict_field is not None:
            disputes.append({"kind": "field_conflict", "field": conflict_field,
                              "candidates": [dict(c) for c in ordered], "raters": cluster_names})

    return {"kept": kept, "disputes": disputes}
