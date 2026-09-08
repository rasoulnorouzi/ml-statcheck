"""Measure the system on every PDF engine, and fail when they disagree too much.

Each port gets a different PDF engine. Python can use PyMuPDF or PDFium, R gets
poppler through pdftools, and a browser gets PDF.js. They do not return the same
text, so the same paper can give a different answer in each port.

This is the gate for the R port and the browser port. It reports the recall of
each engine and the spread between the best and the worst. A port must not ship
while the spread is above the limit, because a user cannot choose the engine
their host provides.

The engines that need another runtime are read from folders of text prepared
earlier, so this script does not need Node or R to run.

Usage:
    python bench_engines.py <pdf_dir> <key.json> <labels.json> [--text-dir DIR]
                            [--max-spread 0.06]

`--text-dir` holds one folder for each engine, named `txt_<engine>`, each with
one text file per PDF. Use it for PDF.js and for R pdftools.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from statcheck_ml.normalize import normalize
from statcheck_ml.prefilter import Prefilter

#: How far the engines may disagree before a port must not ship. The spread was
#: 0.136 before the normalisation stage and 0.050 after it, so this limit holds
#: the gain without demanding more than the engines can give.
DEFAULT_MAX_SPREAD = 0.06


def read_pymupdf(path: str) -> str:
    import pymupdf

    doc = pymupdf.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def read_pdfium(path: str) -> str:
    import pypdfium2

    doc = pypdfium2.PdfDocument(path)
    parts = []
    for page in doc:
        page_text = page.get_textpage()
        parts.append(page_text.get_text_range())
        page_text.close()
        page.close()
    doc.close()
    return "".join(parts)


def read_poppler(path: str) -> str:
    done = subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"],
                          capture_output=True, timeout=180)
    return done.stdout.decode("utf-8", "replace")


NATIVE = {"pymupdf": read_pymupdf, "pdfium": read_pdfium,
          "poppler": read_poppler}


def numbers(result: dict):
    """The parts of a result that no engine can rename."""
    out = []
    for field in ("df1", "df2", "statistic", "p_value"):
        value = result.get(field)
        if value not in (None, ""):
            out.append(str(value).lstrip("<>=").strip())
    return out


def holds_result(flat: str, result: dict, span: int = 160) -> bool:
    """True when every number of the result sits inside one short stretch.

    White space is ignored, so a result split by a line break still counts.
    """
    nums = numbers(result)
    if not nums:
        return False
    for match in re.finditer(re.escape(nums[0]), flat):
        chunk = flat[match.start():match.start() + span]
        if all(n in chunk for n in nums[1:]):
            return True
    return False


def flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf_dir")
    ap.add_argument("key")
    ap.add_argument("labels")
    ap.add_argument("--text-dir", default=None)
    ap.add_argument("--max-spread", type=float, default=DEFAULT_MAX_SPREAD)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    key = {k["window_id"]: k for k in
           json.loads(io.open(args.key, encoding="utf-8").read(), strict=False)}
    labels = json.loads(io.open(args.labels, encoding="utf-8").read(),
                        strict=False)

    by_doc = {}
    for row in labels:
        if not row.get("results"):
            continue
        doc = key[row["window_id"]]["doc"]
        by_doc.setdefault(doc, []).extend(row["results"])

    # The PDF of a document is found by its stem, so a corpus can be renamed
    # without changing this script.
    stems = {Path(p).stem: p for p in
             (str(x) for x in Path(args.pdf_dir).glob("*.pdf"))}
    mapping_path = Path(args.pdf_dir).parent / "pdfmap.json"
    doc_to_stem = {}
    if mapping_path.exists():
        pdfmap = json.loads(mapping_path.read_text(encoding="utf-8"))
        doc_to_stem = {v: Path(k).stem for k, v in pdfmap.items()}
    else:
        for doc in by_doc:
            stem = Path(doc).stem
            if stem in stems:
                doc_to_stem[doc] = stem

    engines = list(NATIVE)
    if args.text_dir:
        for folder in sorted(Path(args.text_dir).glob("txt_*")):
            name = folder.name[len("txt_"):]
            if name not in engines:
                engines.append(name)

    prefilter = Prefilter()
    found = {e: 0 for e in engines}
    kept = {e: 0 for e in engines}
    total = 0
    missing = {e: 0 for e in engines}

    docs = sorted(d for d in by_doc if d in doc_to_stem)
    for n, doc in enumerate(docs):
        stem = doc_to_stem[doc]
        results = by_doc[doc]
        total += len(results)
        for engine in engines:
            if engine in NATIVE:
                pdf = stems.get(stem)
                if pdf is None:
                    missing[engine] += len(results)
                    continue
                try:
                    raw = NATIVE[engine](pdf)
                except Exception:
                    missing[engine] += len(results)
                    continue
            else:
                path = Path(args.text_dir) / f"txt_{engine}" / f"{stem}.txt"
                if not path.exists() or path.stat().st_size == 0:
                    missing[engine] += len(results)
                    continue
                raw = path.read_text(encoding="utf-8", errors="replace")

            flat = flatten(raw)
            for result in results:
                if holds_result(flat, result):
                    found[engine] += 1
            windows = [flatten(w.text)
                       for w in prefilter.windows(normalize(raw)[0])]
            for result in results:
                if any(holds_result(w, result) for w in windows):
                    kept[engine] += 1
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(docs)} documents", flush=True)

    print(f"\n{len(docs)} documents, {total} gold results\n")
    print(f"{'engine':<12}{'in text':>10}{'after prefilter':>18}"
          f"{'not read':>10}")
    print("-" * 50)
    rates = {}
    for engine in engines:
        seen = total - missing[engine]
        rate = kept[engine] / seen if seen else 0.0
        rates[engine] = rate
        print(f"{engine:<12}{found[engine] / max(seen, 1):>10.3f}"
              f"{rate:>18.3f}{missing[engine]:>10}")

    spread = max(rates.values()) - min(rates.values()) if rates else 0.0
    print(f"\nspread between the best and the worst engine: {spread:.3f}"
          f"   limit {args.max_spread:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"documents": len(docs), "gold": total, "recall": rates,
             "in_text": {e: found[e] / max(total - missing[e], 1)
                         for e in engines},
             "spread": spread, "limit": args.max_spread}, indent=2),
            encoding="utf-8")
        print(f"wrote {args.out}")

    if spread > args.max_spread:
        print("\nFAIL: the engines disagree by more than the limit. A port "
              "must not ship until the normalisation stage closes the gap.")
        return 1
    print("\nPASS: every engine stays inside the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
