"""Write the parity cases, and the answer the Python port gives for each.

The normalisation stage exists once, in `spec/normalize.json`, and three ports
apply it. A port that drifts silently is worse than no port, because the same
paper then gives a different answer in Python, in R and in a browser.

This writes real passages from every PDF engine, so the cases carry the damaged
operators and the joined lines the stage exists to handle, plus cases chosen by
hand for the edges a corpus may not reach.

Run this only when the rules change. Commit the result, so the JavaScript and R
checks can run without the corpus.

Usage: python tests/make_parity_cases.py [text_dir] [out.json]
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from statcheck_ml.normalize import canonicalise, normalize, reflow

#: Kept short on purpose. The file is committed, and a long case proves no more
#: than a short one.
EXCERPT = 800
PER_ENGINE = 2

HAND_CASES = [
    ("empty", ""),
    ("short", "F(1, 17) = 3.5, p = .05"),
    # PyMuPDF writes a destroyed operator as a control character. The stage must
    # leave these alone, or it would regress the engine the models were trained
    # on.
    ("pymupdf damage", "F(1, 17) \x03 35.72, p \x04 .0005"),
    # Poppler writes the same damage as letters in the Greek and Coptic block.
    ("poppler damage", "F(1, 17) ϭ 35.72, p Ͻ .0005"),
    ("mixed damage",
     "a ϭ 1.0 and b Ͻ 2.0 and c Ϫ 3.0 and d ␩ 4.0"),
    # These symbols carry meaning inside a statistic and must never be renamed.
    ("protected greek", "χ2(1) = 8.69, p = .003; ϕ = −.20"),
    ("long joined line",
     "word " * 60 + "F(2, 30) ϭ 4.11, p ϭ .03 " + "tail " * 30),
    ("crlf", "one\r\ntwo\rthree\n"),
    ("no spaces to cut", "x" * 400),
    ("every slot taken",
     "".join(chr(c) for c in range(1, 9)) + " t(9) ϭ 2.0"),
]


def main(text_dir: str | None = None, out_path: str | None = None):
    out_path = out_path or str(Path(__file__).parent / "parity_cases.json")
    rng = random.Random(7)
    cases = []

    if text_dir and os.path.isdir(text_dir):
        for folder in sorted(Path(text_dir).glob("txt_*")):
            engine = folder.name[len("txt_"):]
            names = sorted(p.name for p in folder.glob("*.txt"))[:40]
            for name in rng.sample(names, min(PER_ENGINE, len(names))):
                text = io.open(folder / name, encoding="utf-8",
                               errors="replace").read()
                if not text.strip():
                    continue
                start = rng.randrange(0, max(1, len(text) - EXCERPT))
                cases.append({"engine": engine, "name": name,
                              "text": text[start:start + EXCERPT]})

    cases += [{"engine": "hand", "name": n, "text": t} for n, t in HAND_CASES]

    rows = []
    for case in cases:
        text, _ = normalize(case["text"])
        renamed = canonicalise(reflow(case["text"]))[1]
        rows.append({**case, "expected": text,
                     "expected_reflow": reflow(case["text"]),
                     "renamed": {k: ord(v) for k, v in renamed.items()}})

    with io.open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=True, indent=1)
    print(f"wrote {len(rows)} parity cases to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None,
         sys.argv[2] if len(sys.argv) > 2 else None)
