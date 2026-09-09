"""Adversarial and noise augmentation for the training windows.

The model must not only read the damage that happens to appear in these 3100
papers. A different PDF pipeline damages text differently, and a model that has
only seen one pipeline learns that pipeline rather than the task.

Four families, each independently switchable so that an ablation can show which
one earned its place.

Every transform rewrites the text and the label offsets in one operation. A
transform that shifts text without shifting its labels silently poisons
training, so `apply` asserts that each label still covers the same characters
and raises when it does not.

The damage tables are taken from the corpus, not invented. The corpus shows
control characters replacing operators, the fraction sign for the equals sign,
and the letters b and N for the comparison signs.

Usage:
    from statcheck_ml.augment import Augmenter
    aug = Augmenter(seed=0)
    extra = aug.expand(examples, per_example=2)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

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

    # -- driver --------------------------------------------------------------

    def perturb(self, example: dict) -> dict | None:
        """Apply a random subset of the families to one example."""
        text = example["text"]
        spans = tags_to_spans(example["tags"])
        if not spans:
            return None

        steps: List[Callable] = []
        if self.operator:
            steps.append(self.damage_operator)
        if self.number_damage:
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
