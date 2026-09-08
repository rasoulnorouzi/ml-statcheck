"""Make the text look the same whatever engine read the PDF.

Each port gets a different PDF engine. Python has PyMuPDF and PDFium, R has
poppler through pdftools, and a browser has PDF.js. They do not return the same
text, and two differences reach the rest of the system:

  Line breaks   Poppler returns a whole column as one line. PDF.js returns no
                lines at all, only positioned pieces. Every prefilter rule
                measures one line, so a joined line hides the result inside it.

  Operators     A publisher sets an operator in a font with no ToUnicode map,
                so the character is lost. PyMuPDF writes what is left as a
                control character. Poppler writes it as a letter in the Greek
                and Coptic block. The model reads characters, so a character it
                never saw becomes unknown.

This stage runs directly after the engine and before the prefilter. It adds and
removes no text. It only restores the line width and renames the damaged
operators.

Measured on 198 holdout documents with 323 gold results, the stage lifts
poppler prefilter recall from 0.793 to 0.879 and leaves PyMuPDF at 0.929. At
the model, what is found rises on poppler from 0.791 to 0.846 and no engine
loses anything.

The prefilter is not the only place to measure. An earlier version renamed a
character by its position alone, which looked free at the prefilter but cost
PyMuPDF 0.011 at the model. Only a character the model cannot read is renamed
now.

The rules live in `spec/normalize.json` so that the three ports read one
definition instead of three copies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

SPEC_PATH = Path(__file__).parent / "spec" / "normalize.json"
CHARMAP_PATH = Path(__file__).parent / "spec" / "charmap.json"

_SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

# Every character the model can read. Renaming one of these can only move the
# input away from the text the model was trained on, so the stage must not.
#
# This was learned the hard way. Renaming by position alone caught the footnote
# marker, the multiplication sign, the curly quotes and the significance star,
# because all of them can stand between a letter and a digit. On PyMuPDF text
# 91% of the renames were of characters the model already knew, and what the
# model found fell from 0.879 to 0.868.
KNOWN_CHARACTERS = frozenset(
    json.loads(CHARMAP_PATH.read_text(encoding="utf-8"))["chars"])

TARGET_WIDTH: int = _SPEC["target_line_width"]
REFLOW_TRIGGER: int = _SPEC["reflow_trigger_width"]
MAX_REFERENCE_LINE: int = _SPEC["max_reference_line"]
CANONICAL_SLOTS = list(_SPEC["canonical_operator_slots"])
_OPERATOR_SITE = re.compile(_SPEC["operator_site_pattern"])
_KEEP = set(_SPEC["text_characters"]) | set(_SPEC["protected_symbols"])


def reflow(text: str) -> str:
    """Cut joined columns back into lines of a normal width.

    A line shorter than the trigger is never touched, so text from an engine
    that already breaks lines per page line passes through almost unchanged.

    An earlier version skipped the whole document when its mean line was short,
    to protect PyMuPDF. That gate was added while chasing the wrong cause of a
    regression, and once the renaming rule was fixed it changed nothing:
    measured on 55 documents, PyMuPDF gave 0.879 and poppler 0.846 with the
    gate and without it. It was removed rather than kept for the sake of the
    work already done.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for line in text.split("\n"):
        while len(line) > REFLOW_TRIGGER:
            cut = line.rfind(" ", TARGET_WIDTH // 2, TARGET_WIDTH)
            if cut <= 0:
                cut = TARGET_WIDTH
            out.append(line[:cut])
            line = line[cut:].lstrip()
        out.append(line)
    return "\n".join(out)


def _is_canonical(ch: str) -> bool:
    """True when the character is already in the alphabet the model knows."""
    return ch in CANONICAL_SLOTS


def canonicalise(text: str) -> Tuple[str, Dict[str, str]]:
    """Rename damaged operator characters to the alphabet the model knows.

    A character counts as a damaged operator when the model cannot read it AND
    it stands where an operator belongs. Both halves are needed. The position
    alone is not enough, because the copyright sign, the multiplication sign
    and the significance star all stand between a letter and a digit, and the
    model reads all three well.

    The vocabulary decides the first half, so the rule does not depend on which
    engine read the PDF. The stage exists to bring a character the model cannot
    read into the alphabet it knows, and nothing else.

    Returns the text and the renaming that was applied.
    """
    damaged = set()
    for match in _OPERATOR_SITE.finditer(text):
        ch = match.group(1)
        if ch in KNOWN_CHARACTERS or ch in _KEEP or _is_canonical(ch):
            continue
        damaged.add(ch)
    if not damaged:
        return text, {}

    # A slot already used in this document keeps its meaning, so the rename
    # must not take it. Inside one document the correspondence between the
    # engines is one to one, so the remaining slots are enough.
    taken = {c for c in CANONICAL_SLOTS if c in text}
    free = [c for c in CANONICAL_SLOTS if c not in taken]
    if not free:
        return text, {}

    # The busiest damaged character takes the first free slot, so the mapping
    # is stable for the same document read by the same engine.
    order = sorted(damaged, key=lambda c: (-text.count(c), c))[:len(free)]
    mapping = {ch: free[i] for i, ch in enumerate(order)}
    return text.translate({ord(k): v for k, v in mapping.items()}), mapping


def normalize(text: str) -> Tuple[str, dict]:
    """Apply every rule that makes the text engine independent.

    Returns the text and a record of what the stage did, so a result can still
    be traced back to the document it came from.
    """
    before_lines = text.count("\n") + 1
    flowed = reflow(text)
    fixed, mapping = canonicalise(flowed)
    return fixed, {
        "lines_before": before_lines,
        "lines_after": fixed.count("\n") + 1,
        "operators_renamed": {k: ord(v) for k, v in mapping.items()},
    }
