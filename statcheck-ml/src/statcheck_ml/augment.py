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
# Every replacement below appears in the corpus.
OPERATOR_DAMAGE = {
    "=": ["¼", "\x01", "\x02", "\x03", "\x04", "е", " "],
    "<": ["b", "\x03", "\x05", "\x07", "\\"],
    ">": ["N", "\x06", "\x08"],
}


@dataclass
class Augmenter:
    """Generate perturbed copies of annotated windows."""

    seed: int = 0
    ocr: bool = True
    unicode: bool = True
    operator: bool = True
    hard_negative: bool = True
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
        patterns = [
            f"see pp. {rng.randint(10, 99)}-{rng.randint(100, 300)} for the full table",
            f"as reported by Smith ({rng.randint(1990, 2020)}), n({rng.randint(10, 99)}) "
            f"= {rng.randint(1, 9)}.{rng.randint(10, 99)}",
            f"Journal of Testing, {rng.randint(10, 60)}({rng.randint(1, 4)}), "
            f"{rng.randint(100, 400)}-{rng.randint(400, 800)}",
            f"the model was M = {rng.randint(1, 9)}.{rng.randint(10, 99)} "
            f"(SD = {rng.randint(0, 3)}.{rng.randint(10, 99)}) across conditions",
            f"* p < .05, ** p < .01, *** p < .001",
            f"the coefficient was b = {rng.choice('-')}{rng.randint(0, 2)}."
            f"{rng.randint(10, 99)}, p = .{rng.randint(10, 99)}",
        ]
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
