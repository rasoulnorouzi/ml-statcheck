"""Repair operators that the conversion from PDF destroyed.

A publisher sets mathematical symbols in a symbol font with its own private
encoding. When the file carries no valid ToUnicode map, the extractor writes
the raw glyph byte, so the operator is simply not in the text:

    F(1, 214) \x01 50.54, p \x05 .001
    F (6, 30) ¼ 2.66, p ¼ .04
    Q = 2.43, pN.05

No pattern can read these, because the character it needs is absent.

The repair works because the mapping is fixed inside one document. One font
produces the whole file, so a character means the same thing everywhere in it.

The mapping is inferred from an anchor that admits only one reading: a suspect
character between the degrees of freedom and a number is an equals sign. A test
statistic is reported as "t(23) = 2.45", never as "t(23) < 2.45". Once that
fixes the character, every other occurrence in the document is known, including
those after "p", where the reading would otherwise be ambiguous.

A character is repaired only when the document itself provides the evidence.
Nothing is repaired from a table of guesses, because the same byte means
different things in different documents.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Tuple

# Characters that can stand in for an operator. Letters are included because
# the corpus shows them, but a letter is only ever repaired in a position where
# a letter cannot belong.
CONTROL = r"\x00-\x08\x0b\x0c\x0e-\x1f"
SUSPECT = re.compile(f"[{CONTROL}¼\\\\!]")
SUSPECT_LETTER = re.compile(r"[bNp]")

# A suspect character between the closing bracket of the degrees of freedom and
# a number. Only an equals sign belongs here.
ANCHOR_STAT = re.compile(
    r"\)\s*([" + CONTROL + r"¼\\!bNp])\s*-?\d*\.?\d",
)
# A suspect character between "p" and a number. This is the ambiguous position.
ANCHOR_P = re.compile(
    r"\bp\s*([" + CONTROL + r"¼\\!bN])\s*(-?\d*\.?\d+)",
)


def infer_map(text: str) -> Dict[str, str]:
    """Work out what each suspect character means in this document."""
    mapping: Dict[str, str] = {}

    # 1. The unambiguous anchor. A character after the degrees of freedom and
    #    before a number is an equals sign.
    for ch in ANCHOR_STAT.findall(text):
        mapping.setdefault(ch, "=")

    # 2. The ambiguous position. A character between "p" and a number is one of
    #    three signs. Characters already fixed by step 1 keep their meaning.
    after_p: Dict[str, list] = defaultdict(list)
    for ch, value in ANCHOR_P.findall(text):
        if ch in mapping:
            continue
        try:
            after_p[ch].append(float(value if not value.startswith(".") else "0" + value))
        except ValueError:
            pass

    for ch, values in after_p.items():
        # A p-value written with a less-than sign is nearly always a round
        # threshold: .05, .01, .001. An equals sign carries an arbitrary value.
        thresholds = {0.05, 0.01, 0.001, 0.0001, 0.1}
        share_round = sum(v in thresholds for v in values) / max(len(values), 1)
        if share_round >= 0.8:
            mapping[ch] = "<"
        else:
            mapping[ch] = "="
    return mapping


def repair(text: str) -> Tuple[str, Dict[str, str], int]:
    """Return the repaired text, the mapping used, and how many were replaced.

    Only the positions that an operator can occupy are touched. A letter
    elsewhere in the document is left alone, so repairing "b" cannot damage an
    ordinary word.
    """
    mapping = infer_map(text)
    if not mapping:
        return text, {}, 0

    replaced = 0

    def fix_stat(m):
        nonlocal replaced
        whole = m.group(0)
        ch = m.group(1)
        if ch in mapping:
            replaced += 1
            return whole.replace(ch, mapping[ch], 1)
        return whole

    def fix_p(m):
        nonlocal replaced
        whole = m.group(0)
        ch = m.group(1)
        if ch in mapping:
            replaced += 1
            return whole.replace(ch, mapping[ch], 1)
        return whole

    out = ANCHOR_STAT.sub(fix_stat, text)
    out = ANCHOR_P.sub(fix_p, out)
    return out, mapping, replaced


def repair_document(text: str) -> str:
    """Repair a whole document, and return only the text."""
    return repair(text)[0]


def report(text: str) -> dict:
    """Describe what a repair would do, without hiding the detail."""
    fixed, mapping, n = repair(text)
    return {
        "mapping": {repr(k): v for k, v in mapping.items()},
        "replacements": n,
        "suspects_before": len(SUSPECT.findall(text)),
        "suspects_after": len(SUSPECT.findall(fixed)),
    }
