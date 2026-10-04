"""Run the original R statcheck over the gold papers in four forms.

Gold is the measuring stick for the shipped system, and the original package
is the stick for the baseline, so the baseline must run on exactly the text
the gold spans live in. Four forms show where statcheck loses results:

  fulltext_raw       normalised PyMuPDF text, no repair
  fulltext_repaired  the text the gold spans are measured in (14_gold.py)
  passages           the repaired text cut into prefilter passages
  pdf                statcheck::checkPDF on the original PDF (pdftools)

Each CSV has the columns of dataset/baseline/statcheck_r.csv. The unit id is
the doc, or "<doc>#<first_line>-<last_line>" for a passage, with line numbers
in the repaired text. The PDF form runs one R process per paper, so a hanging
pdftools call loses one paper and not the run.

Usage: python pipeline/15_statcheck_gold.py --gold dataset/gold/gold.jsonl
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

RUN_TEXT = os.path.join(ROOT, "r", "run_statcheck.R")
RUN_PDF = os.path.join(ROOT, "r", "run_statcheck_pdf.R")
FORMS = ["fulltext_raw", "fulltext_repaired", "passages", "pdf"]
COLUMNS = "window_id,test_type,df1,df2,test_value,p_comp,reported_p,computed_p,error,decision_error,raw"
UNPARSED = re.compile(r"(?:could not parse|raised an error):\s*(\d+)")


def prepare(raw):
    """Return (normalised text, repaired text) exactly as 14_gold.py builds them."""
    from statcheck_ml.normalize import normalize
    from statcheck_ml.repair_validated import repair_validated

    text, _ = normalize(raw)
    fixed, _ = repair_validated(text)
    return text, fixed


def passage_id(doc, first, last):
    return f"{doc}#{first}-{last}"


def parse_passage_id(unit):
    doc, span = unit.rsplit("#", 1)
    first, last = span.split("-")
    return doc, int(first), int(last)


def passage_units(doc, fixed, prefilter):
    lines = fixed.split("\n")
    return [{"window_id": passage_id(doc, lo, hi), "text": "\n".join(lines[lo:hi + 1])}
            for lo, hi in prefilter.passages(prefilter.windows(fixed))]


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_r(rscript, script, units, out_csv, timeout=None):
    """Run an R script over units; return (returncode, stdout, unparsed count)."""
    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(units, f, ensure_ascii=False)
        p = subprocess.run([rscript, script, tmp, out_csv], capture_output=True,
                           text=True, encoding="utf-8", timeout=timeout)
    finally:
        os.remove(tmp)
    m = UNPARSED.search(p.stdout)
    return p.returncode, p.stdout + p.stderr, int(m.group(1)) if m else None


def read_rows(path):
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_pdf_one(job):
    rscript, doc, pdf, timeout = job
    fd, out = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        rc, log, unparsed = run_r(rscript, RUN_PDF, [{"window_id": doc, "path": pdf}],
                                  out, timeout)
        if rc not in (0, 3) or os.path.getsize(out) == 0:
            return doc, [], f"error: exit {rc}: {log.strip()[-300:]}"
        rows = read_rows(out)
        return doc, rows, ("checkPDF " + [l for l in log.splitlines() if l.startswith("error:")][-1] if rc == 3 else None)
    except subprocess.TimeoutExpired:
        return doc, [], f"timeout after {timeout}s"
    finally:
        if os.path.exists(out):
            os.remove(out)


def write_csv(path, rows):
    import csv
    cols = COLUMNS.split(",")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_NONNUMERIC)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def r_versions(rscript):
    code = ("cat(as.character(getRversion()), '|', as.character(packageVersion('statcheck')), '|', "
            "as.character(packageVersion('jsonlite')), '|', as.character(packageVersion('pdftools')), sep='')")
    out = subprocess.run([rscript, "-e", code], check=True, capture_output=True, text=True).stdout
    r, sc, jl, pt = out.strip().split("|")
    return {"r_version": r, "statcheck_version": sc, "jsonlite_version": jl, "pdftools_version": pt}


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def extract_pdfs(zip_path, clean_dir, docs, dest):
    manifest = {}
    with open(os.path.join(clean_dir, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            manifest[m["txt"]] = m["pdf"]
    os.makedirs(dest, exist_ok=True)
    paths = {}
    with zipfile.ZipFile(zip_path) as z:
        for doc in docs:
            member = manifest[doc]
            target = os.path.join(dest, member.split("/", 1)[1])
            if not os.path.exists(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            paths[doc] = target
    return paths


def rel(path):
    return os.path.relpath(os.path.abspath(path), ROOT).replace(os.sep, "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--clean-dir", default=os.path.join(ROOT, "data", "clean"))
    ap.add_argument("--pdf-zip", default=os.path.join(ROOT, "..", "data", "02_pdfs.zip"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "dataset", "baseline", "gold"))
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--seed", type=int, default=0,
                    help="unused; the stage is deterministic, the flag exists for the stage rule")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per PDF")
    ap.add_argument("--jobs", type=int, default=4, help="parallel R processes for the PDF form")
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()

    from statcheck_ml.prefilter import Prefilter

    with open(args.gold, encoding="utf-8") as f:
        docs = [json.loads(line)["doc"] for line in f if line.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    pf = Prefilter()

    raw_units, fixed_units, passage_list = [], [], []
    for doc in docs:
        with open(os.path.join(args.clean_dir, doc), encoding="utf-8") as f:
            text, fixed = prepare(f.read())
        raw_units.append({"window_id": doc, "text": text})
        fixed_units.append({"window_id": doc, "text": fixed})
        passage_list.extend(passage_units(doc, fixed, pf))
    print(f"{len(docs)} papers, {len(passage_list)} passages")

    versions = r_versions(args.rscript)
    versions["git_sha"] = git_sha()
    versions["command"] = "python pipeline/15_statcheck_gold.py " + " ".join(
        [f"--gold {rel(args.gold)}", f"--clean-dir {rel(args.clean_dir)}",
         f"--pdf-zip {rel(args.pdf_zip)}", f"--out-dir {rel(args.out_dir)}",
         "--rscript Rscript", f"--seed {args.seed}", f"--timeout {args.timeout}",
         f"--jobs {args.jobs}"] + (["--skip-pdf"] if args.skip_pdf else []))
    versions.update({"sha256": {}, "rows": {}, "units": {}, "units_with_results": {},
                     "unparsed_units": {}, "wall_seconds": {}, "pdf_failures": []})

    def record(name, path, units, unparsed, secs):
        rows = read_rows(path)
        versions["sha256"][name] = sha256_of(path)
        versions["rows"][name] = len(rows)
        versions["units"][name] = units
        versions["units_with_results"][name] = len({r["window_id"] for r in rows})
        versions["unparsed_units"][name] = unparsed
        versions["wall_seconds"][name] = round(secs, 1)
        print(name, versions["rows"][name], "rows", units, "units", unparsed, "unparsed",
              round(secs, 1), "s", flush=True)

    for name, units in (("fulltext_raw", raw_units), ("fulltext_repaired", fixed_units),
                        ("passages", passage_list)):
        path = os.path.join(args.out_dir, f"statcheck_{name}.csv")
        t0 = time.time()
        rc, log, unparsed = run_r(args.rscript, RUN_TEXT, units, path)
        if rc != 0:
            sys.exit(f"{name}: Rscript exit {rc}\n{log[-500:]}")
        record(name, path, len(units), unparsed, time.time() - t0)

    if not args.skip_pdf:
        pdf_dir = os.path.join(ROOT, "data", "gold_pdfs")
        paths = extract_pdfs(args.pdf_zip, args.clean_dir, docs, pdf_dir)
        t0 = time.time()
        jobs = [(args.rscript, d, paths[d], args.timeout) for d in docs]
        results = {}
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
            for i, (doc, rows, err) in enumerate(ex.map(run_pdf_one, jobs), 1):
                results[doc] = rows
                if err:
                    versions["pdf_failures"].append({"doc": doc, "reason": err})
                if i % 50 == 0:
                    print(f"pdf {i}/{len(jobs)}", flush=True)
        path = os.path.join(args.out_dir, "statcheck_pdf.csv")
        write_csv(path, [r for d in docs for r in results[d]])
        record("pdf", path, len(docs), len(versions["pdf_failures"]), time.time() - t0)

    with open(os.path.join(args.out_dir, "versions.json"), "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=1)
    print(json.dumps(versions, indent=1))


if __name__ == "__main__":
    main()
