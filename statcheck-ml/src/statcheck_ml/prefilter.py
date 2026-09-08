"""Select the passages of text that could hold a statistical result.

The corpus holds 3.66 million lines, and only 0.1% of them contain a statistic.
Running a model over all of them wastes almost all of the work, and in a
browser it is not possible at all.

The filter is tuned for recall alone. Text it discards can never be recovered
by the model, so the filter, and not the model, sets the recall ceiling of the
whole system. It therefore answers "could this hold a result", never "does this
look like a result".

Measured on the clean corpus at a density threshold of 0.20, the filter keeps
100% of lines that hold a statistic and removes 66% of the rest.

The rules live in `spec/prefilter.json` so that the Python, JavaScript and R
ports read one definition instead of three copies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

SPEC_PATH = Path(__file__).parent / "spec" / "prefilter.json"


@dataclass
class Window:
    """A passage offered to the model, with its place in the document."""

    text: str
    line: int
    start_line: int
    end_line: int


class Prefilter:
    """Apply the shared prefilter rules to a document."""

    def __init__(self, spec: dict | None = None):
        if spec is None:
            spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.spec = spec
        self.min_density = spec["min_non_letter_density"]
        self.min_length = spec["min_line_length"]
        self.context_lines = spec["context_lines"]
        self._digit = re.compile(spec["require_digit_pattern"])
        self._ref_head = re.compile(spec["reference_heading_pattern"], re.I)
        self._ref_line = [re.compile(p) for p in spec["reference_line_patterns"]]
        # The cap belongs to the normalisation spec, because it exists only to
        # survive the line breaks a PDF engine chooses.
        from .normalize import MAX_REFERENCE_LINE

        self.max_reference_line = spec.get("max_reference_line",
                                           MAX_REFERENCE_LINE)
        # A window grows until it holds this much text, so that a window means
        # the same amount of context whichever engine read the PDF. Zero turns
        # the rule off and restores a fixed count of lines.
        self.target_characters = spec.get("target_window_characters", 0)
        self.max_context_lines = spec.get("max_context_lines", 8)

    @staticmethod
    def density(line: str) -> float:
        """Return the share of characters in `line` that are not letters."""
        if not line:
            return 0.0
        return 1 - sum(c.isalpha() for c in line) / len(line)

    def is_reference(self, line: str) -> bool:
        """Say whether a line looks like an entry in a reference list.

        A reference entry is short. A line longer than the cap is a column that
        the PDF engine joined, and such a line can hold a result AND a citation.
        Judging it as one entry discards the result with the citation, which
        measured 26 of the 37 results poppler lost on the holdout. The cap keeps
        the rule away from joined lines and changes nothing for PyMuPDF text.
        """
        if len(line) > self.max_reference_line:
            return False
        return any(p.search(line) for p in self._ref_line)

    def strip_references(self, lines: List[str]) -> List[str]:
        """Blank the reference section and stray reference lines.

        A reference entry has the same density of digits and punctuation as a
        result, so the filter treats it as a candidate. Lines are blanked
        rather than deleted, which keeps every line number correct.
        """
        cut = len(lines)
        for i in range(int(len(lines) * 0.55), len(lines)):
            if self._ref_head.match(lines[i]):
                cut = i
                break
        out = [("" if self.is_reference(ln) else ln) for ln in lines[:cut]]
        out.extend("" for _ in lines[cut:])
        return out

    def keeps_line(self, line: str) -> bool:
        """Say whether one line survives the filter."""
        if len(line) < self.min_length:
            return False
        if not self._digit.search(line):
            return False
        return self.density(line) >= self.min_density

    def windows(self, text: str, drop_references: bool = True) -> Iterator[Window]:
        """Yield one window for each line that survives the filter.

        A window carries the lines around the candidate, because a result can
        be split by a line break. On the clean corpus 18.4% of results are
        separated from their p-value by at least one line break.
        """
        lines = text.split("\n")
        if drop_references:
            lines = self.strip_references(lines)
        for i, line in enumerate(lines):
            if not self.keeps_line(line):
                continue
            lo, hi = self._span(lines, i)
            yield Window("\n".join(lines[lo:hi]), i, lo, hi - 1)

    def _span(self, lines: List[str], i: int) -> tuple:
        """Choose how many lines of context this window needs.

        A fixed line count is not a fixed amount of context. The count was tuned
        on PyMuPDF text, whose lines run about 57 characters, and a window of
        two lines each side holds about 298 characters. PDF.js breaks the same
        page into shorter lines, so the same two lines hold only 175 characters
        and the result is cut off. That, and not the density rule, is what the
        other engines lose: removing the density rule recovers one result, and
        matching the amount of text recovers most of the gap.

        The window therefore grows until it holds about as much text as the
        window the model was trained on. It never shrinks below the line count,
        so text that already has long lines is untouched.
        """
        n = self.context_lines
        lo = max(0, i - n)
        hi = min(len(lines), i + n + 1)
        if not self.target_characters:
            return lo, hi
        while n < self.max_context_lines:
            if sum(len(x) + 1 for x in lines[lo:hi]) >= self.target_characters:
                break
            n += 1
            new_lo = max(0, i - n)
            new_hi = min(len(lines), i + n + 1)
            if (new_lo, new_hi) == (lo, hi):     # the document has no more text
                break
            lo, hi = new_lo, new_hi
        return lo, hi
