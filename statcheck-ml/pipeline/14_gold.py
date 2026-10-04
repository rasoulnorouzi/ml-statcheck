"""Build the human gold test set from the owner's hand-coded sheet.

The sheet codes every reported result in 409 papers by hand, with the quote
copied from the PDF. This stage finds each quote in the PyMuPDF text, after the
same normalisation and repair the pipeline applies, and records where it is.
Gold is a test set only. No stage may train on it.

The sheet itself stays outside the repository. It quotes the articles at
length, like the corpus it was made from.

A quote is found by its numbers, not by its characters. The coders pasted
what the PDF gave them, control characters included, so the letters and
operators of a quote often differ from the text while its numbers do not.

Usage: python pipeline/14_gold.py <sheet.xlsx> <clean_text_dir> --out dataset/gold/gold.jsonl
"""
import argparse
import collections
import difflib
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

NUM = re.compile(r"\d*\.?\d+")
XESC = re.compile(r"_x([0-9A-Fa-f]{4})_")      # how Excel stores a control character
WS = re.compile(r"\s+")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")
NEIGHBOURS = 8      # numbers on each side of the anchor that may belong to one result


def numbers(s):
    return [(round(float(m.group()), 4), m.start(), m.end()) for m in NUM.finditer(s)]


def value(x):
    try:
        v = float(str(x).replace("−", "-").replace('"', "").strip())
    except (TypeError, ValueError):
        return None
    return None if v != v else round(abs(v), 4)


def test_type(raw_type):
    """Map the coders' spelling to the guideline's six names, or 'other'."""
    t = (raw_type or "").strip().lower().replace(" ", "")
    if not t:
        return None
    if "χ" in t or "x2" in t or "chi" in t or t == "x":
        return "chi2"
    if t in ("f", "δf"):
        return "F"
    if t.startswith("q"):
        return "Q"
    if t in ("t", "z", "r"):
        return t
    return "other"


def locate(raw, stat, text, doc_numbers):
    """The span in `text` that holds the numbers of `raw`, or None.

    The anchor is the statistic. Each other number of the quote takes its
    nearest occurrence around the anchor. Ties between anchors, such as a
    table cell that repeats the numbers of a sentence, go to the candidate
    whose characters look most like the quote.
    """
    want = [n for n, _, _ in numbers(raw)]
    if not want:
        return None
    anchor = stat if stat in want else max(want, key=lambda v: (len(str(v)), v))
    need = collections.Counter(want)
    best = None
    for i, (v, a_s, a_e) in enumerate(doc_numbers):
        if v != anchor:
            continue
        lo, hi = max(0, i - NEIGHBOURS), min(len(doc_numbers), i + NEIGHBOURS + 1)
        used, picked = set(), []
        for val, count in need.items():
            near = sorted((abs(j - i), j) for j in range(lo, hi)
                          if doc_numbers[j][0] == val and j not in used)
            for _, j in near[:count]:
                used.add(j)
                picked.append(doc_numbers[j])
        s, e = min(p[1] for p in picked), max(p[2] for p in picked)
        sim = difflib.SequenceMatcher(None, WS.sub("", text[max(0, s - 15):e + 15]),
                                      WS.sub("", raw)).ratio()
        cand = (len(picked), sim, s, e, a_s, a_e)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return None
    score, sim, s, e, a_s, a_e = best
    if score < max(2, len(want) - 1) and not (len(want) == 1 and score == 1):
        return None
    return {"span": [s, e], "anchor": [a_s, a_e], "numbers_matched": round(score / len(want), 3),
            "similarity": round(sim, 3)}


def build_paper(job):
    path, rows, clean_dir = job
    from statcheck_ml.normalize import normalize
    from statcheck_ml.repair_validated import repair_validated

    text, _ = normalize(open(os.path.join(clean_dir, path), encoding="utf-8").read())
    text, _ = repair_validated(text)
    doc_numbers = numbers(text)
    results = []
    for r in rows:
        if not r["raw"]:
            continue
        found = locate(r["raw"], r["stat"], text, doc_numbers)
        if found:
            found["line"] = text.count("\n", 0, found["anchor"][0])
        r = dict(r, located=found is not None, **(found or {}))
        results.append(r)
    return {"doc": path, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "results": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sheet")
    ap.add_argument("clean_dir")
    ap.add_argument("--key", default="dataset/windows/key.json")
    ap.add_argument("--out", default="dataset/gold/gold.jsonl")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    import pandas as pd

    sheet = pd.read_excel(args.sheet)
    sheet = sheet[sheet.pdf_filename.notna()]
    # The sheet spells some names without the hyphen or accent of the file,
    # so a paper is matched by the identifier at the end of its name.
    manifest = [json.loads(l) for l in open(os.path.join(args.clean_dir, "manifest.jsonl"))]
    by_id = {m["txt"][:-4].rsplit("_", 1)[-1]: m["txt"] for m in manifest if m.get("ok")}
    bronze = {k["doc"][:-4].rsplit("_", 1)[-1]: k["set"] for k in json.load(open(args.key))}

    papers = collections.defaultdict(list)
    for idx, r in sheet.iterrows():
        pid = str(r.pdf_filename).strip().rsplit("_", 1)[-1]
        raw = XESC.sub(lambda m: chr(int(m.group(1), 16)), r.raw_result) \
            if isinstance(r.raw_result, str) else ""
        name = None if pd.isna(r.test_type) else str(r.test_type)
        papers[by_id[pid]].append({
            "row": int(idx) + 2,          # the spreadsheet row, header included
            "raw": raw,
            "test_type_coded": name,
            "test_type": test_type(name),
            "stat": value(r.test_stat), "df1": value(r.df1), "df2": value(r.df2),
            "p_operator": None if pd.isna(r.p_comparison) else str(r.p_comparison).strip('"'),
            "p": value(r.p),
            "damaged": bool(CONTROL.search(raw)),
            "page": None if pd.isna(r.page) else str(r.page),
            "coder": r.coder_initials if isinstance(r.coder_initials, str) else None,
            "remarks": r.remarks if isinstance(r.remarks, str) else None,
        })

    jobs = [(path, rows, args.clean_dir) for path, rows in sorted(papers.items())]
    with ProcessPoolExecutor(args.workers) as ex:
        built = list(ex.map(build_paper, jobs, chunksize=4))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for paper in built:
            pid = paper["doc"][:-4].rsplit("_", 1)[-1]
            paper["bronze_set"] = bronze.get(pid, "none")
            paper["has_results"] = bool(paper["results"])
            fh.write(json.dumps(paper, ensure_ascii=False) + "\n")

    results = [r for p in built for r in p["results"]]
    print(f"papers {len(built)}  with results {sum(bool(p['results']) for p in built)}  "
          f"results {len(results)}  located {sum(r['located'] for r in results)}")


if __name__ == "__main__":
    main()
