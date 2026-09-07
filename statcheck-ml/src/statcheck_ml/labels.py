"""The tag set, and the conversion between spans and tags.

One design choice is worth stating. The operator is not a separate output of
the model. It is part of the tag itself: `POP_EQ`, `POP_LT` and `POP_GT`.

The reason is the corpus. In 198 documents of 3100 the conversion writes a
control character where the operator belongs, and 35.5% of results of that
shape are affected. Marking the position of such a character says nothing,
because the character carries no meaning. Naming the operator in the tag makes
the model choose from the surrounding characters, which is the only way to
recover it.

This also keeps the model to one output. A window-level class is not needed
either: a window holds a result when the model tags one, so the class follows
from the spans.

Tags use the BIOES scheme. Most spans here are one to four characters long, so
the explicit End and Single tags carry real information.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# The parts of a reported result.
ENTITIES = [
    "TEST",    # the test name, such as t or F
    "STAT",    # the value of the test statistic
    "DF1",     # the first degrees of freedom
    "DF2",     # the second degrees of freedom
    "N",       # the sample size written inside the parentheses
    "PVAL",    # the reported p-value
    "POP_EQ",  # the operator before the p-value, an equals sign
    "POP_LT",  # the operator, a less-than sign
    "POP_GT",  # the operator, a greater-than sign
]

OUTSIDE = "O"
_PREFIXES = ("B", "I", "E", "S")

TAGS: List[str] = [OUTSIDE] + [f"{p}-{e}" for e in ENTITIES for p in _PREFIXES]
TAG_TO_ID: Dict[str, int] = {t: i for i, t in enumerate(TAGS)}
ID_TO_TAG: Dict[int, str] = {i: t for t, i in TAG_TO_ID.items()}

OPERATOR_ENTITY = {"=": "POP_EQ", "<": "POP_LT", ">": "POP_GT"}
ENTITY_OPERATOR = {v: k for k, v in OPERATOR_ENTITY.items()}


def part_label(name: str, operator: str | None = None) -> str | None:
    """Map a dataset part name to a tag entity.

    `POP` becomes one of three entities, chosen by the operator the annotator
    named rather than by the character in the text, which may be damaged.
    """
    if name != "POP":
        return name if name in ENTITIES else None
    return OPERATOR_ENTITY.get(operator or "")


def spans_to_tags(length: int, spans: List[Tuple[int, int, str]]) -> List[str]:
    """Build a BIOES tag sequence from (start, end, entity) spans.

    A later span wins where two spans overlap. Overlap is rare and always a
    sign of a bad label, so `find_overlaps` reports it rather than hiding it.
    """
    tags = [OUTSIDE] * length
    for start, end, entity in spans:
        if entity not in ENTITIES or end <= start:
            continue
        start = max(0, start)
        end = min(length, end)
        if end - start == 1:
            tags[start] = f"S-{entity}"
        else:
            tags[start] = f"B-{entity}"
            for i in range(start + 1, end - 1):
                tags[i] = f"I-{entity}"
            tags[end - 1] = f"E-{entity}"
    return tags


def tags_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """Read spans back out of a tag sequence.

    The decoding is forgiving. A model can emit an I tag with no B before it,
    and dropping such a span would hide a near miss during evaluation, so a
    stray continuation opens a span instead.
    """
    spans: List[Tuple[int, int, str]] = []
    start, entity = None, None

    for i, tag in enumerate(tags + [OUTSIDE]):
        if tag == OUTSIDE:
            prefix, ent = OUTSIDE, None
        else:
            prefix, ent = tag.split("-", 1)

        if prefix in ("B", "S") or (prefix in ("I", "E") and entity != ent):
            if start is not None:
                spans.append((start, i, entity))
            start, entity = i, ent
            if prefix == "S":
                spans.append((start, i + 1, entity))
                start, entity = None, None
        elif prefix == "E" and entity == ent:
            spans.append((start, i + 1, entity))
            start, entity = None, None
        elif prefix == OUTSIDE:
            if start is not None:
                spans.append((start, i, entity))
            start, entity = None, None

    return [s for s in spans if s[1] > s[0]]


def find_overlaps(spans: List[Tuple[int, int, str]]) -> List[Tuple]:
    """Return every pair of spans that overlap. An overlap is a bad label."""
    bad = []
    ordered = sorted(spans)
    for i in range(len(ordered) - 1):
        if ordered[i][1] > ordered[i + 1][0]:
            bad.append((ordered[i], ordered[i + 1]))
    return bad
