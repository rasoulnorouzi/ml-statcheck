"""Adversarial and noise augmentation for the training windows.

The model must not only read the damage that happens to appear in these 3100
papers. A different PDF pipeline damages text differently, and a model that has
only seen one pipeline learns that pipeline rather than the task.

Each family is independently switchable so that an ablation can show which one
earned its place. The older families damage one character at random. The four
paper families (`font_map`, `negative_damage`, `spaced_digits`, `operator_drop`)
damage a whole example the same way, because a real font breaks a whole paper
the same way. They are off by default.

Every transform rewrites the text and the label offsets in one operation. A
transform that shifts text without shifting its labels silently poisons
training. The paper families describe their change as a list of edits, and
`_assert_aligned` checks each label against those edits and raises
`AlignmentError` when a label no longer covers its true value.

The damage tables are taken from the corpus, not invented. The corpus shows
control characters replacing operators, the fraction sign for the equals sign,
and the letters b and N for the comparison signs.

Usage:
    from statcheck_ml.augment import Augmenter
    aug = Augmenter(seed=0)
    extra = aug.expand(examples, per_example=2)
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple

from .labels import spans_to_tags, tags_to_spans

# --- family 1: characters an extractor confuses -----------------------------
OCR_CONFUSIONS = {
    "l": "1", "1": "l", "O": "0", "0": "O", "I": "1",
    "5": "S", "S": "5", "8": "B", "rn": "m", "cl": "d",
}

# --- family 2: symbols the same reader writes differently --------------------
UNICODE_VARIANTS = {
    "-": ["−", "–", "‐"],        # minus, en dash, hyphen
    " ": [" ", " ", " "],        # non-breaking and thin spaces
    "χ": ["X", "x", "chi", "c", "v"],      # the chi letter, as it degrades
    "²": ["2", "^2"],
}

# --- family 3: what the PDF pipeline does to an operator ---------------------
# Every replacement below appears in the corpus, as PyMuPDF reads it.
OPERATOR_DAMAGE = {
    "=": ["¼", "\x01", "\x02", "\x03", "\x04", "е", " "],
    "<": ["b", "\x03", "\x05", "\x07", "\\"],
    ">": ["N", "\x06", "\x08"],
}

# --- family 3b: the minus sign and the decimal point ------------------------
# The font that destroys an operator destroys these too, because all three are
# symbols and not digits. Every character below stands where a minus sign or a
# decimal point belongs somewhere in the corpus.
NUMBER_DAMAGE = ["\x01", "\x02", "\x03", "\x05", "\x06", "−", " ", ""]

# --- family 4: the same damage, as another PDF engine writes it --------------
# A destroyed operator has no Unicode value, so each engine invents one. PyMuPDF
# writes a control character and poppler writes a letter in the Greek and Coptic
# block. The characters below were measured by reading the same 198 documents
# with both engines and comparing the operator position.
#
# The model reads characters, so an engine it never saw produces unknown input.
# The normalisation stage renames these before the model, and training on them
# as well means the model degrades gently when a new engine invents a character
# that the stage does not yet know.
ENGINE_ALPHABETS = {
    "poppler": ["ϭ", "Ͻ", "Ϫ", "␤", "␩",
                "À", "Â"],
    # Private use is what an engine falls back to when it maps a glyph number
    # straight through. No engine in the corpus does this, and a port may meet
    # one that does.
    "private_use": ["", "", "", ""],
}


# --- the paper families: one font, one damage, the whole example -------------
# Rates measured on the bronze training windows (dataset/train.jsonl), never on
# gold. In 337 pairs of a document and an operator, 335 use one damaged
# character throughout, so a map is drawn once and applied everywhere.
#
# Of 527 damaged results that show both operators, the statistic operator is
# damaged in 439 and the p operator in 425: each operator joins a map with share
# 0.8. In quotes with a control character, 69 of 70 negative values lost the
# minus sign to a control character, and 20 of 608 decimal values lost the
# decimal point.
MAP_OPERATOR_SHARE = 0.8
MAP_MINUS_SHARE = 69 / 70
MAP_DECIMAL_SHARE = 20 / 608
#: Half the operator draws come from the font table, weighted by support. The
#: other half come from OPERATOR_DAMAGE, which also holds the letter forms (b, N)
#: that no table entry explains.
MAP_TABLE_SHARE = 0.5
CONTROL_SLOTS = [chr(c) for c in range(1, 9)]
#: The share of augmented copies that each switched-on paper family damages.
#: The originals stay clean, so this sets the clean-to-damaged ratio.
PAPER_FAMILY_SHARE = 0.5
#: Letter spacing, drawn once per example. Not measured: no bronze training
#: window holds spaced digits, and gold is not read for augmentation. The real
#: example "F1(1 , 7 3 ) = 3 .7 4" spaces three of its four digit gaps.
SPACING_RATE = (0.5, 1.0)

FONT_TABLE_PATH = Path(__file__).parent / "spec" / "font_table.json"

# A minus sign, not a hyphen in a range ("10-20") or a word ("pre-1990").
MINUS_SITE = re.compile(r"(?<![\w)\]])[-−–](?=\.?\d)")
# A decimal point, not the end of an abbreviation ("pp.12").
DECIMAL_SITE = re.compile(r"(?<![A-Za-z.])\.(?=\d)")
NUMBER = re.compile(r"\.?\d(?:[\d.]*\d)?")
# A window that a real font already damaged keeps its own map; a second map on
# top would mix two fonts in one paper.
ALREADY_DAMAGED = re.compile("[\x01-\x08¼]")


class AlignmentError(AssertionError):
    """A transform left a label that no longer covers its true value."""


@lru_cache(maxsize=1)
def font_glyphs() -> Dict[str, List[Tuple[str, int]]]:
    """The damaged character for each operator in the font table, with support.

    A font with no ToUnicode map passes the glyph code straight through, so the
    extractor returns chr(code): 188 is the fraction sign, 92 the backslash.
    """
    entries = json.loads(FONT_TABLE_PATH.read_text(encoding="utf-8"))["entries"]
    table: Dict[str, Counter] = {}
    for entry in entries.values():
        table.setdefault(entry["char"], Counter())[chr(entry["code"])] += entry["support"]
    return {op: sorted(counts.items()) for op, counts in table.items()}


Edit = Tuple[int, int, str]


def _assert_aligned(old_text: str, old_spans, edits: Sequence[Edit],
                    new_text: str, new_spans, family: str) -> None:
    """Check every label against the edits that made `new_text`.

    The expected content of a span is rebuilt from the old text and the edits
    alone, independently of the offset arithmetic in `_replace`. A span whose
    whole content was deleted must be gone; every other span must hold exactly
    its old characters with its own edits applied, under the same label.
    """
    expected = []
    for s, e, label in old_spans:
        inside = []
        for a, b, new in edits:
            if a < b and (a < s < b or a < e < b):
                raise AlignmentError(f"{family}: edit {a}-{b} cuts the {label} "
                                     f"span {s}-{e} in {old_text!r}")
            if (a < b and s <= a and b <= e) or (a == b and s < a < e):
                inside.append((a, b, new))
        pieces, pos = [], s
        for a, b, new in sorted(inside):
            pieces += [old_text[pos:a], new]
            pos = b
        pieces.append(old_text[pos:e])
        content = "".join(pieces)
        if content:
            expected.append((label, content))
    got = [(label, new_text[s:e]) for s, e, label in new_spans]
    if got != expected:
        raise AlignmentError(f"{family}: labels {got} where {expected} was expected; "
                             f"{old_text!r} became {new_text!r}")



@dataclass
class Augmenter:
    """Generate perturbed copies of annotated windows."""

    seed: int = 0
    ocr: bool = True
    unicode: bool = True
    operator: bool = True
    hard_negative: bool = True
    #: Destroy the minus sign or the decimal point inside a number. Measured
    #: and switched OFF, because it made every score worse. See
    #: `damage_number` for the numbers.
    number_damage: bool = False
    #: The hard negatives written from observed false positives. Measured
    #: together with `number_damage` only, so its own effect is not yet known.
    #: See `make_hard_negative`.
    extra_negatives: bool = False
    #: One damage map per example, applied to every operator, minus sign and
    #: decimal point in it. See `apply_font_map`.
    font_map: bool = False
    #: The font map damage on text that holds no result: windows with no label
    #: and hard negatives. Damage alone must never signal a result.
    negative_damage: bool = False
    #: Spaces between the digits of every number, as PDF letter spacing writes
    #: them. See `space_digits`.
    spaced_digits: bool = False
    #: Operators removed entirely, with their POP spans. See `drop_operators`.
    operator_drop: bool = False
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _replace(text: str, spans: List[Tuple[int, int, str]],
                 start: int, end: int, new: str):
        """Replace text[start:end] with `new`, and move every later span.

        A span that overlaps the replacement is resized rather than moved, so a
        one-character operator can become a two-character one without losing
        its label.
        """
        delta = len(new) - (end - start)
        out_text = text[:start] + new + text[end:]
        out_spans = []
        for s, e, label in spans:
            if e <= start:
                out_spans.append((s, e, label))
            elif s >= end:
                out_spans.append((s + delta, e + delta, label))
            else:
                # The span covers the edited region; keep it covering the new text.
                ns = s if s <= start else s + delta
                ne = e + delta if e >= end else e
                if ne > ns:
                    out_spans.append((ns, max(ne, ns + 1), label))
        return out_text, out_spans

    # -- families ------------------------------------------------------------

    def damage_operator(self, text, spans):
        """Replace an operator with something the corpus actually produces."""
        targets = [(s, e, lab) for s, e, lab in spans if lab.startswith("POP_")]
        if not targets:
            return text, spans
        s, e, lab = self.rng.choice(targets)
        real = {"POP_EQ": "=", "POP_LT": "<", "POP_GT": ">"}[lab]
        replacement = self.rng.choice(OPERATOR_DAMAGE[real])
        return self._replace(text, spans, s, e, replacement)

    def damage_number(self, text, spans):
        """Destroy the minus sign or the decimal point inside a number.

        The same font that destroys an operator also destroys the minus sign and
        the decimal point, because all three are symbols rather than digits.
        Error analysis on the holdout found this in the largest group of missed
        results:

            t(8) \x01 \x034.40, p \x04 .001      the statistic is -4.40
            t(454) \x02 \x031.950, p \x02 .052   the statistic is -1.950
            r = 0\x0592                          the statistic is 0.92

        Most replacements are one character for one character. Two are not: a
        space splits the number, and an empty string removes the decimal point
        altogether. Both happen in the corpus, and `_replace` moves the later
        spans, so the labels stay aligned either way.

        A number whose decimal point is gone cannot be read correctly by
        anything. The label still carries the true value, so the model learns
        to mark the span, and the arithmetic then reports what it can.
        """
        targets = [(s, e, lab) for s, e, lab in spans
                   if lab in ("STAT", "PVAL")]
        self.rng.shuffle(targets)
        for s, e, _ in targets:
            inner = text[s:e]
            sites = [i for i, ch in enumerate(inner) if ch in "-.−"]
            if not sites:
                continue
            at = s + self.rng.choice(sites)
            replacement = self.rng.choice(NUMBER_DAMAGE)
            return self._replace(text, spans, at, at + 1, replacement)
        return text, spans

    def corrupt_characters(self, text, spans, rate: float = 0.02):
        """Swap confusable characters outside the labelled spans.

        Labelled characters are left alone. Corrupting a digit inside a
        statistic would change the value while the label still claims the old
        one, which teaches the model a false pairing.
        """
        protected = set()
        for s, e, _ in spans:
            protected.update(range(s, e))
        chars = list(text)
        for i, ch in enumerate(chars):
            if i in protected or ch not in OCR_CONFUSIONS:
                continue
            if self.rng.random() < rate:
                chars[i] = OCR_CONFUSIONS[ch]
        return "".join(chars), spans

    def vary_unicode(self, text, spans):
        """Swap a symbol for another form of the same symbol."""
        options = [(i, ch) for i, ch in enumerate(text) if ch in UNICODE_VARIANTS]
        if not options:
            return text, spans
        i, ch = self.rng.choice(options)
        return self._replace(text, spans, i, i + 1,
                             self.rng.choice(UNICODE_VARIANTS[ch]))

    def break_line(self, text, spans):
        """Insert a line break inside a result.

        18.4% of real results are split this way, so the model must not depend
        on a result sitting on one line.
        """
        targets = [(s, e) for s, e, lab in spans if lab in ("STAT", "PVAL")]
        if not targets:
            return text, spans
        s, e = self.rng.choice(targets)
        cut = self.rng.choice([s, e])
        if cut <= 0 or cut >= len(text):
            return text, spans
        return self._replace(text, spans, cut, cut, "\n")

    def make_hard_negative(self) -> str:
        """Produce text that looks like a result and is not.

        These carry no labels. The regular expression fires on them, so they
        are the cheapest way to teach the model what a result is not.
        """
        rng = self.rng

        def dec(low=0, high=9):
            return f"{rng.randint(low, high)}.{rng.randint(10, 99)}"

        patterns = [
            f"see pp. {rng.randint(10, 99)}-{rng.randint(100, 300)} for the full table",
            f"as reported by Smith ({rng.randint(1990, 2020)}), n({rng.randint(10, 99)}) "
            f"= {dec(1)}",
            f"Journal of Testing, {rng.randint(10, 60)}({rng.randint(1, 4)}), "
            f"{rng.randint(100, 400)}-{rng.randint(400, 800)}",
            f"the model was M = {dec(1)} "
            f"(SD = {dec(0, 3)}) across conditions",
            f"* p < .05, ** p < .01, *** p < .001",
            f"the coefficient was b = -{dec(0, 2)}, p = .{rng.randint(10, 99)}",

        ]

        # Every pattern below was written from a false positive the model
        # produced on the holdout. Economics and finance use `t` for time, and
        # that alone accounted for four of the twenty-eight.
        #
        # They are OFF by default. Trained together with `number_damage` they
        # cut false positives from 28 to 6, and cost 0.06 of recall, which is
        # the wrong trade for a screening tool. Their effect on their own has
        # not been measured yet.
        extra = [
            f"and t = {rng.randint(1, 40)}, {rng.randint(41, 60)}, . . . , T "
            f"with T = {rng.randint(80, 200)}. For notational convenience",
            f"a share repurchase program at t = {dec(0, 3)} The riskfree rate "
            f"is set to {dec(0, 1)}",
            f"for years t = {rng.randint(1970, 1999)} and t = "
            f"{rng.randint(2000, 2020)}, respectively",
            f"of those excluded at t = {rng.randint(10, 60)}, "
            f"{dec(10, 80)}% are retired",

            # A convention or a rule of thumb, not a measurement.
            f"with r = .{rng.randint(1, 9):02d}, .{rng.randint(10, 30)}, and "
            f".{rng.randint(30, 60)} representing small, medium and large effects",
            f"to identify true small effects (f2 = .{rng.randint(1, 9):02d}, "
            f"r = .{rng.randint(10, 30)}) in a sample of {rng.randint(100, 900)} firms",
            f"falls below the Altman threshold value of z = {dec(1, 3)}, "
            f"denoting that the firm is in the bad state",

            # A percentage, a count, or a share of a sample.
            f"race and ethnicity were: White = {dec(60, 80)}%, "
            f"Black = {dec(5, 15)}%, Asian = {dec(1, 9)}%",
            f"{rng.randint(100, 400)} ({dec(40, 70)} per cent) were men, and "
            f"the average age was {dec(30, 60)} years",
            f"records of {rng.randint(300, 600)} of {rng.randint(600, 900)} "
            f"({dec(50, 90)} percent) employees",

            # A range, an interval bound, or a section number.
            f"mean age {dec(12, 16)} years, SD = .{rng.randint(40, 90)}; "
            f"range: [{dec(10, 13)}; {dec(15, 19)}]",
            f"95% confidence interval (CI) {dec(1, 9)}-{rng.randint(100, 700)}."
            f"{rng.randint(10, 99)}, controlling for the covariates",
            f"(from 0.{rng.randint(100, 500)} to 0.{rng.randint(500, 900)}, "
            f"t = {dec(1, 3)}). {rng.randint(2, 9)}.{rng.randint(1, 9)}. "
            f"Robustness In this subsection",

            # A mean beside another mean, which is not a test.
            f"(Msimultaneous = {dec(3, 5)}, Msequential = {dec(3, 5)}) across "
            f"the two presentation modes",
            f"Source: EB {dec(60, 62)}, {dec(62, 64)}, {dec(64, 66)}, "
            f"{dec(66, 68)}, {dec(68, 70)}, {dec(70, 72)}",
        ]
        if self.extra_negatives:
            patterns = patterns + extra
        return rng.choice(patterns)

    # -- the paper families ---------------------------------------------------

    def _edit(self, text, spans, edits: Sequence[Edit], family: str):
        """Apply non-overlapping edits from right to left, then check every label."""
        new_text, new_spans = text, list(spans)
        for start, end, new in sorted(edits, reverse=True):
            new_text, new_spans = self._replace(new_text, new_spans, start, end, new)
        _assert_aligned(text, spans, edits, new_text, new_spans, family)
        return new_text, new_spans

    def _draw_operators(self) -> List[str]:
        ops = [op for op in "=<>" if self.rng.random() < MAP_OPERATOR_SHARE]
        return ops or [self.rng.choice("=<>")]

    def draw_font_map(self) -> Dict[str, str]:
        """Draw what one symbol font does to each symbol it holds.

        Keys are "=", "<", ">", "-" (any minus sign) and "." (a decimal point).
        Two symbols never share a damaged character, because distinct glyphs in
        one font have distinct codes.
        """
        rng = self.rng
        glyphs = font_glyphs()
        fmap: Dict[str, str] = {}
        for op in self._draw_operators():
            table = glyphs.get(op)
            if table and rng.random() < MAP_TABLE_SHARE:
                chars, weights = zip(*table)
                pick = rng.choices(chars, weights)[0]
            else:
                pick = rng.choice(OPERATOR_DAMAGE[op])
            if pick in fmap.values():
                pick = rng.choice([c for c in CONTROL_SLOTS if c not in fmap.values()])
            fmap[op] = pick
        for symbol, share in (("-", MAP_MINUS_SHARE), (".", MAP_DECIMAL_SHARE)):
            if rng.random() < share:
                fmap[symbol] = rng.choice([c for c in CONTROL_SLOTS
                                           if c not in fmap.values()])
        return fmap

    def apply_font_map(self, text, spans, fmap: Dict[str, str] | None = None):
        """Damage every symbol of one font in the example, the same way.

        Returns the text, the spans and the map. A window that a real font
        already damaged is returned unchanged with an empty map.
        """
        if ALREADY_DAMAGED.search(text):
            return text, spans, {}
        if fmap is None:
            fmap = self.draw_font_map()
        edits = [(i, i + 1, fmap[ch]) for i, ch in enumerate(text)
                 if ch in "=<>" and ch in fmap]
        for symbol, site in (("-", MINUS_SITE), (".", DECIMAL_SITE)):
            if symbol in fmap:
                edits += [(m.start(), m.end(), fmap[symbol]) for m in site.finditer(text)]
        text, spans = self._edit(text, spans, edits, "font_map")
        return text, spans, fmap

    def space_digits(self, text, spans, rate: float | None = None):
        """Split the digits of every number with spaces, as letter spacing does.

        "F(1, 73) = 3.74" becomes "F(1 , 7 3 ) = 3 .7 4". A space falls only
        between two characters of a number, or between a number and the comma
        or parenthesis after it, so a label widens to cover its spaced value and
        the value itself does not change.
        """
        if rate is None:
            rate = self.rng.uniform(*SPACING_RATE)
        edits = []
        for m in NUMBER.finditer(text):
            gaps = list(range(m.start() + 1, m.end()))
            if m.end() < len(text) and text[m.end()] in ",)":
                gaps.append(m.end())
            edits += [(k, k, " ") for k in gaps if self.rng.random() < rate]
        return self._edit(text, spans, edits, "spaced_digits")

    def drop_operators(self, text, spans, operators: str | None = None,
                       collapse: bool | None = None):
        """Remove operators entirely, the way a glyph with no code vanishes.

        "z = 3.29, p = .01" becomes "z 3.29, p .01". The POP span of a removed
        operator is removed; every other label is kept. With `collapse`, one of
        the two spaces around a removed operator goes too.
        """
        if operators is None:
            operators = "".join(self._draw_operators())
        if collapse is None:
            collapse = self.rng.random() < 0.5
        edits = []
        for i, ch in enumerate(text):
            if ch not in operators:
                continue
            edits.append((i, i + 1, ""))
            if collapse and 0 < i < len(text) - 1 and text[i - 1] == " " == text[i + 1]:
                edits.append((i + 1, i + 2, ""))
        return self._edit(text, spans, edits, "operator_drop")

    def damage_paper(self, text, spans, always_map: bool = False):
        """Apply the switched-on paper families to one example.

        Each family fires with share PAPER_FAMILY_SHARE. With `always_map` the
        font map always fires, which is how text with no result is damaged.
        Returns the text, the spans and the names of the families that changed
        the text.
        """
        fired: Set[str] = set()
        if self.operator_drop and self.rng.random() < PAPER_FAMILY_SHARE:
            new_text, spans = self.drop_operators(text, spans)
            if new_text != text:
                fired.add("operator_drop")
            text = new_text
        if always_map or (self.font_map and self.rng.random() < PAPER_FAMILY_SHARE):
            new_text, spans, _ = self.apply_font_map(text, spans)
            if new_text != text:
                fired.add("font_map")
            text = new_text
        if self.spaced_digits and self.rng.random() < PAPER_FAMILY_SHARE:
            new_text, spans = self.space_digits(text, spans)
            if new_text != text:
                fired.add("spaced_digits")
            text = new_text
        return text, spans, fired

    def damage_negative(self, text: str) -> str:
        """Damage text that holds no result exactly as a result would be damaged."""
        damaged, spans, _ = self.damage_paper(text, [], always_map=True)
        if spans:
            raise AlignmentError(f"negative_damage: a label appeared in {damaged!r}")
        return damaged

    # -- driver --------------------------------------------------------------

    def perturb(self, example: dict) -> dict | None:
        """Apply a random subset of the families to one example."""
        text = example["text"]
        spans = tags_to_spans(example["tags"])
        if not spans:
            if not self.negative_damage:
                return None
            damaged = self.damage_negative(text)
            if damaged == text:
                return None
            return {
                **example,
                "window_id": example["window_id"] + "-aug",
                "text": damaged,
                "tags": ["O"] * len(damaged),
                "augmented": True,
            }

        fired: Set[str] = set()
        if self.font_map or self.spaced_digits or self.operator_drop:
            text, spans, fired = self.damage_paper(text, spans)
        # A second, random damage on an operator would break the one map the
        # paper families drew.
        one_font = bool(fired & {"font_map", "operator_drop"})

        steps: List[Callable] = []
        if self.operator and not one_font:
            steps.append(self.damage_operator)
        if self.number_damage and not one_font:
            steps.append(self.damage_number)
        if self.unicode:
            steps.append(self.vary_unicode)
        if self.ocr:
            steps.append(lambda t, s: self.corrupt_characters(t, s))
        steps.append(self.break_line)

        self.rng.shuffle(steps)
        for step in steps[: self.rng.randint(1, len(steps))]:
            text, spans = step(text, spans)

        if not spans:
            return None
        return {
            **example,
            "window_id": example["window_id"] + "-aug",
            "text": text,
            "tags": spans_to_tags(len(text), spans),
            "augmented": True,
        }

    def expand(self, examples: Sequence[dict], per_example: int = 1,
               negatives: int = 0) -> List[dict]:
        """Return new examples. The originals are never modified."""
        out: List[dict] = []
        for ex in examples:
            for _ in range(per_example):
                made = self.perturb(ex)
                if made:
                    out.append(made)
        if self.hard_negative:
            for i in range(negatives):
                text = self.make_hard_negative()
                out.append({
                    "window_id": f"hardneg-{self.seed}-{i}",
                    "text": text,
                    "tags": ["O"] * len(text),
                    "pool": "synthetic",
                    "journal": None,
                    "source_doc": None,
                    "augmented": True,
                })
                if self.negative_damage:
                    damaged = self.damage_negative(text)
                    if damaged != text:
                        out.append({
                            "window_id": f"hardneg-{self.seed}-{i}-aug",
                            "text": damaged,
                            "tags": ["O"] * len(damaged),
                            "pool": "synthetic",
                            "journal": None,
                            "source_doc": None,
                            "augmented": True,
                        })
        return out


def check_alignment(examples: Sequence[dict]) -> List[str]:
    """Report any example whose labels no longer fit its text."""
    bad = []
    for ex in examples:
        if len(ex["tags"]) != len(ex["text"]):
            bad.append(f"{ex['window_id']}: {len(ex['tags'])} tags for {len(ex['text'])} characters")
            continue
        for s, e, label in tags_to_spans(ex["tags"]):
            if s < 0 or e > len(ex["text"]) or e <= s:
                bad.append(f"{ex['window_id']}: span {s}-{e} ({label}) falls outside the text")
    return bad
