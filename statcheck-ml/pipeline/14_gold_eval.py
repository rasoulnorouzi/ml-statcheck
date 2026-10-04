"""Score the systems on the human gold test set.

Gold codes every result in a paper, so this measures recall over whole
documents, the prefilter included. The bronze holdout measures windows only.
The two are reported apart and never pooled.

Systems:
  regex           the ported statcheck pattern, after operator repair
  model           the shipped model alone, on the prefilter windows
  cascade         the shipped pipeline: pattern first, the model adds
  model_passages  the model alone, on overlapping windows merged into one
                  passage, so each character is read once

The original R statcheck is scored in four forms, from the CSVs that
15_statcheck_gold.py wrote (--baseline-dir):
  statcheck_r_fulltext_raw       the normalised text, no repair
  statcheck_r_fulltext_repaired  the repaired text
  statcheck_r_passages           the repaired text cut into prefilter passages
  statcheck_r_pdf                checkPDF on the original PDF

Matching rule for these four. A gold result (located, in scope) is found by a
system when an unconsumed find of that system in the same paper has the same
absolute statistic rounded to four decimals. Each find serves one gold row, so
two gold rows with one statistic need two finds. statcheck reports values and
not positions, so for the full-text and PDF forms the paper is the unit. For
statcheck_r_passages the passage must also cover the gold line. A paper that
checkPDF failed on has no finds, so its gold results count as misses. Finds
that match no gold row are counted per paper by kind. A statcheck find always
has a test name, so the kind is "r" for a correlation and "named" otherwise.

The coders followed statcheck's scope, so a correlation with no degrees of
freedom is not in gold while the guideline marks it. A find with no gold
result is therefore not always wrong, and the report splits those finds by
kind instead of calling them false positives.

Results are reported by split. gold_test is the frozen test part and the
only part to quote as a result. gold_train may feed training in later
versions, so its score is a development figure. "all" is the sum, for scale.

Every run appends one line to results/gold_eval_log.jsonl. That log is the
record of each evaluation against the gold test. A model scored there, tuned,
and scored again is no longer a clean held-out score.

Usage: python pipeline/14_gold_eval.py --gold dataset/gold/gold.jsonl --out results/gold_eval.json
"""
import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

IN_SCOPE = {"t", "F", "r", "z", "chi2", "Q"}
LOCAL = ["regex", "model", "cascade", "model_passages"]
STATCHECK = {"statcheck_r_fulltext_raw": "statcheck_fulltext_raw.csv",
             "statcheck_r_fulltext_repaired": "statcheck_fulltext_repaired.csv",
             "statcheck_r_passages": "statcheck_passages.csv",
             "statcheck_r_pdf": "statcheck_pdf.csv"}
SYSTEMS = LOCAL + list(STATCHECK)
# statcheck writes Chi2 and Z; "Qw" and the like fall to other.
TEST_NAMES = {"t": "t", "f": "F", "r": "r", "z": "z", "chi2": "chi2", "q": "Q"}


def read_statcheck(path):
    """Map doc to its finds as [first line, last line, statistic, test type, used].

    Lines are None unless the unit id is a passage id, doc#first-last.
    """
    import csv

    out = collections.defaultdict(list)
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                stat = round(abs(float(row["test_value"])), 4)
            except ValueError:
                continue
            unit, lo, hi = row["window_id"], None, None
            if "#" in unit:
                unit, span = unit.rsplit("#", 1)
                lo, hi = map(int, span.split("-"))
            kind = TEST_NAMES.get(row["test_type"].lower(), "other")
            out[unit].append([lo, hi, stat, kind, False])
    return out


def match_statcheck(gold, finds):
    """Return (hit per gold row, finds left over by kind). Consumes finds in place."""
    hits = []
    for g in gold:
        match = next((f for f in finds if not f[4] and f[2] in (g["stat"], g["anchor_value"])
                      and (f[0] is None or f[0] <= g["line"] <= f[1])), None)
        if match:
            match[4] = True
        hits.append(match is not None)
    left = collections.Counter("r" if f[3] == "r" else "named" for f in finds if not f[4])
    return hits, left


def finds_for(system, pipe, text, windows, lines):
    """Every find as (first line, last line, statistic, test type)."""
    if system == "model_passages":
        out = []
        for lo, hi in pipe.prefilter.passages(windows):
            for f in pipe.find_with_model("\n".join(lines[lo:hi + 1]), lo):
                if f.statistic is not None:
                    out.append((lo, hi, round(abs(f.statistic), 4), f.test_type))
        return out
    span = {w.line: (w.start_line, w.end_line) for w in windows}
    out = []
    for r in pipe.run_text(text)["results"]:
        if r["statistic"] is None:
            continue
        lo, hi = span.get(r["line"], (r["line"], r["line"]))
        out.append((lo, hi, round(abs(r["statistic"]), 4), r["test_type"]))
    return out


def make_pipes(systems, model_path):
    from statcheck_ml.pipeline import Pipeline

    build = {"regex": lambda: Pipeline(model_path=None),
             "model": lambda: Pipeline(model_path=model_path, use_pattern=False),
             "cascade": lambda: Pipeline(model_path=model_path),
             "model_passages": lambda: Pipeline(model_path=model_path, use_pattern=False)}
    return {s: build[s]() for s in systems if s in build}


def score_paper(job):
    paper, clean_dir, systems, model_path = job[:4]
    sc_finds = job[4] if len(job) > 4 else {}
    from statcheck_ml.normalize import normalize

    raw = open(os.path.join(clean_dir, paper["doc"]), encoding="utf-8").read()
    pipes = make_pipes(systems, model_path)
    # Repair and prefilter do not depend on the model, so any pipe serves.
    base = next(iter(pipes.values())) if pipes else make_pipes(["regex"], None)["regex"]
    text, _ = normalize(raw)
    fixed, _ = base.repair(text)
    lines = fixed.split("\n")
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln) + 1)
    windows = list(base.prefilter.windows(fixed))
    ranges = [(starts[w.start_line], starts[w.end_line + 1]) for w in windows]

    gold = [g for g in paper["results"] if g["located"]]
    for g in gold:
        s, e = g["span"]
        g["kept"] = any(a <= s and e <= b for a, b in ranges)
        a_s, a_e = g["anchor"]
        g["anchor_value"] = round(abs(float(fixed[a_s:a_e])), 4) if fixed[a_s:a_e] else None

    out = {"doc": paper["doc"], "bronze_set": paper["bronze_set"],
           "has_results": paper["has_results"], "gold": [], "unmatched": {}}
    hits = {}
    for system in systems:
        if system in STATCHECK:
            hits[system], left = match_statcheck(gold, sc_finds.get(system, []))
            out["unmatched"][system] = left
            continue
        finds = [list(f) + [False] for f in finds_for(system, pipes[system], raw, windows, lines)]
        hits[system] = []
        for g in gold:
            match = next((f for f in finds if not f[4] and f[2] in (g["stat"], g["anchor_value"])
                          and f[0] <= g["line"] <= f[1]), None)
            if match:
                match[4] = True
            hits[system].append(match is not None)
        # One result is seen by several overlapping windows; count it once.
        left = {(f[2], f[3], f[0] // 5) for f in finds if not f[4]}
        out["unmatched"][system] = collections.Counter(
            "nameless" if not t else ("r" if t == "r" else "named") for _, t, _ in left)
    for i, g in enumerate(gold):
        out["gold"].append({"test_type": g["test_type"], "damaged": g["damaged"], "kept": g["kept"],
                            **{s: hits[s][i] for s in systems}})
    return out


def recall(papers, system, keep):
    k = n = 0
    for p in papers:
        for g in p["gold"]:
            if keep(g):
                n += 1
                k += g[system]
    return k, n


def bootstrap(papers, system, keep, resamples, seed):
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [papers[rng.randrange(len(papers))] for _ in papers]
        k, n = recall(sample, system, keep)
        if n:
            values.append(k / n)
    values.sort()
    return [round(values[int(0.025 * len(values))], 3), round(values[int(0.975 * len(values)) - 1], 3)]


def summarise(papers, resamples, seed, systems=SYSTEMS):
    in_scope = lambda g: g["test_type"] in IN_SCOPE
    out = {"papers": len(papers),
           "papers_with_results": sum(p["has_results"] for p in papers),
           "gold_in_scope": recall(papers, "kept", in_scope)[1]}
    k, n = recall(papers, "kept", in_scope)
    out["prefilter_keeps"] = round(k / n, 3) if n else None
    for system in systems:
        row = {}
        k, n = recall(papers, system, in_scope)
        row["recall"] = round(k / n, 3) if n else None
        row["recall_ci"] = bootstrap(papers, system, in_scope, resamples, seed) if n else None
        row["found"] = k
        k, n = recall(papers, system, lambda g: in_scope(g) and g["damaged"])
        row["recall_damaged"] = round(k / n, 3) if n else None
        row["damaged_n"] = n
        row["recall_by_test"] = {}
        for t in sorted(IN_SCOPE):
            k, n = recall(papers, system, lambda g, t=t: g["test_type"] == t)
            if n:
                row["recall_by_test"][t] = {"k": k, "n": n, "r": round(k / n, 3)}
        neg = [p for p in papers if not p["has_results"]]
        row["finds_in_papers_without_results"] = sum(sum(p["unmatched"][system].values()) for p in neg)
        pos = [p for p in papers if p["has_results"]]
        row["finds_not_in_gold"] = {kind: sum(p["unmatched"][system].get(kind, 0) for p in pos)
                                    for kind in ("named", "r", "nameless")}
        out[system] = row
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", default="dataset/gold/gold.jsonl")
    ap.add_argument("--split", default="dataset/gold/split.json")
    ap.add_argument("--model", default=None,
                    help="model directory; default is the packaged zoo that Pipeline() loads")
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--baseline-dir", default="dataset/baseline/gold",
                    help="the CSVs that 15_statcheck_gold.py wrote")
    ap.add_argument("--bronze", default="results/eval.json")
    ap.add_argument("--out", default="results/gold_eval.json")
    ap.add_argument("--log", default="results/gold_eval_log.jsonl")
    ap.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    from statcheck_ml.pipeline import bundled_model_path

    model = os.path.realpath(args.model) if args.model else os.path.realpath(bundled_model_path())
    model_files = sorted(os.path.join(model, f) for f in os.listdir(model)
                         if os.path.isfile(os.path.join(model, f)))
    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8")]
    split = json.load(open(args.split))
    test, train = set(split["gold_test"]), set(split["gold_train"])
    if test & train:
        sys.exit("split.json puts a paper in both parts")
    unknown = [p["doc"] for p in gold if p["doc"] not in test | train]
    if unknown:
        sys.exit(f"{len(unknown)} gold papers are in neither split, first: {unknown[0]}")

    baseline = {s: read_statcheck(os.path.join(args.baseline_dir, STATCHECK[s]))
                for s in args.systems if s in STATCHECK}
    jobs = [(p, args.clean_dir, args.systems, model,
             {s: b.get(p["doc"], []) for s, b in baseline.items()}) for p in gold]
    with ProcessPoolExecutor(args.workers) as ex:
        scored = list(ex.map(score_paper, jobs, chunksize=2))

    def part(papers):
        return summarise(papers, args.resamples, args.seed, args.systems)

    in_test = [p for p in scored if p["doc"] in test]
    in_train = [p for p in scored if p["doc"] in train]
    gold_test = part(in_test)
    gold_test["by_bronze_set"] = {
        b: part([p for p in in_test if p["bronze_set"] == b]) for b in ("holdout", "none")}

    bronze = json.load(open(args.bronze))
    baseline_files = [os.path.join(args.baseline_dir, STATCHECK[s])
                      for s in args.systems if s in STATCHECK]
    if baseline_files:
        baseline_files.append(os.path.join(args.baseline_dir, "versions.json"))
    inputs = [args.gold, args.split, args.bronze] + baseline_files + model_files
    report = {
        "note": "Gold is the owner's hand-coded sheet: whole papers, statcheck scope. "
                "gold_test is the frozen test part. Bronze is the three-rater consensus on "
                "sampled windows, a language-model annotation. They are never pooled.",
        "gold_test": gold_test,
        "gold_train": part(in_train),
        "all": part(scored),
        "bronze": {"source": args.bronze, **bronze["bronze"],
                   "systems": {name: {k: v for k, v in s.items()
                                      if k in ("overall", "checkable", "damaged")}
                               for name, s in bronze["systems"].items()}},
        "provenance": {
            "git_sha": git_sha(),
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "seed": args.seed,
            "resamples": args.resamples,
            "systems": args.systems,
            "model": model,
            "inputs": {os.path.relpath(f): sha256(f) for f in inputs},
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    line = {"date": report["provenance"]["date"], "git_sha": report["provenance"]["git_sha"],
            "model": model, "systems": args.systems,
            "gold_test_recall": {s: gold_test[s]["recall"] for s in args.systems}}
    with open(args.log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    print(json.dumps({k: report[k] for k in ("gold_test",)}, indent=1)[:3000])


if __name__ == "__main__":
    main()
