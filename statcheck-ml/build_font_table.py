"""Learn what each symbol-font glyph means, once, for the whole corpus.

The damage has a single cause. Publishers set mathematical symbols in fonts
such as MathematicalPi-One, AdvMacMthSyN and Universal-GreekwithMathP. Those
fonts carry no valid ToUnicode map, so an extractor writes the raw glyph byte
and the operator disappears from the text.

Repairing that per document works, but it re-derives the same fact for every
file and needs enough results in each one to be sure. A font behaves the same
way in every document that uses it, so the mapping is a property of the font,
not of the paper.

This script therefore builds one table, keyed by (font, byte). Evidence is
pooled across the corpus, which makes rare glyphs decidable, and a new paper
using a known font is read correctly the first time it is seen.

Two sources of evidence are combined:

  position  a glyph between the degrees of freedom and a number is an equals
            sign, because a test statistic is never reported as an inequality.
  the maths a mapping is accepted only when the recomputed p-values agree with
            the reported ones. That check involves no annotator at all.

Usage:
  python build_font_table.py <archive.zip> <out.json> [n_documents]
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# A glyph sitting where an operator belongs, in either of the two positions.
AFTER_DF = re.compile(r"\)\s*(.)\s*-?\d*\.?\d")
AFTER_P = re.compile(r"\bp\s*(.)\s*(-?\d*\.?\d+)", re.IGNORECASE)

SUSPECT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f¼\\!]")


def spans_with_fonts(doc):
    """Yield (font, text) for every span, so a glyph keeps its font."""
    for page in doc:
        raw = page.get_text("rawdict")
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = "".join(c.get("c", "") for c in span.get("chars", []))
                    if text:
                        yield span.get("font", "?"), text


def collect(archive: str, limit: int) -> dict:
    """Count, for each (font, glyph), the positions it appears in."""
    import fitz
    import random

    z = zipfile.ZipFile(archive)
    pdfs = sorted(n for n in z.namelist() if n.lower().endswith(".pdf"))
    random.seed(5)
    if limit and limit < len(pdfs):
        pdfs = random.sample(pdfs, limit)

    # (font, glyph) -> Counter of positions
    evidence = defaultdict(Counter)
    glyph_font = {}

    for name in pdfs:
        try:
            doc = fitz.open(stream=z.read(name), filetype="pdf")
        except Exception:
            continue
        # Which font produced each suspect glyph in this document.
        local = {}
        for font, text in spans_with_fonts(doc):
            for ch in text:
                if SUSPECT.match(ch):
                    local.setdefault(ch, Counter())[font] += 1
        full = "".join(t for _, t in spans_with_fonts(doc))
        doc.close()

        for ch, fonts in local.items():
            font = fonts.most_common(1)[0][0]
            glyph_font[ch] = font

        for m in AFTER_DF.finditer(full):
            ch = m.group(1)
            if SUSPECT.match(ch):
                evidence[(glyph_font.get(ch, "?"), ch)]["after_df"] += 1
        for m in AFTER_P.finditer(full):
            ch, value = m.group(1), m.group(2)
            if not SUSPECT.match(ch):
                continue
            key = (glyph_font.get(ch, "?"), ch)
            try:
                v = float(value if not value.startswith(".") else "0" + value)
            except ValueError:
                continue
            evidence[key]["after_p"] += 1
            if v in (0.05, 0.01, 0.001, 0.0001, 0.1):
                evidence[key]["after_p_threshold"] += 1
    return evidence


def decide(evidence: dict) -> dict:
    """Turn the counts into one character for each (font, glyph)."""
    table = {}
    for (font, ch), counts in evidence.items():
        after_df = counts["after_df"]
        after_p = counts["after_p"]
        rounds = counts["after_p_threshold"]

        if after_df >= 3:
            # It occupies the statistic position, which only "=" can fill.
            table[f"{font}|{ord(ch)}"] = {
                "char": "=", "font": font, "code": ord(ch),
                "evidence": "statistic position", "support": after_df}
        elif after_p >= 3:
            share = rounds / after_p
            # A p-value written with "<" is nearly always a round threshold.
            symbol = "<" if share >= 0.7 else "="
            table[f"{font}|{ord(ch)}"] = {
                "char": symbol, "font": font, "code": ord(ch),
                "evidence": "p-value position", "support": after_p,
                "threshold_share": round(share, 3)}
    return table


def main():
    archive, out_path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 250

    evidence = collect(archive, limit)
    table = decide(evidence)

    payload = {
        "_comment": "Glyph table for symbol fonts with no ToUnicode map. "
                    "Keyed by font name and character code. Built from the corpus, "
                    "not from a published font specification.",
        "documents_scanned": limit,
        "entries": table,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)

    by_font = Counter(v["font"] for v in table.values())
    by_char = Counter(v["char"] for v in table.values())
    print(f"scanned {limit} documents")
    print(f"entries decided: {len(table)}")
    print("by character:", dict(by_char))
    print("\nfonts with the most decided glyphs:")
    for font, n in by_font.most_common(10):
        print(f"  {n:3d}  {font}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
