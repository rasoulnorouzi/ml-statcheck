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

The coders followed statcheck's scope, so a correlation with no degrees of
freedom is not in gold while the guideline marks it. A find with no gold
result is therefore not always wrong, and the report splits those finds by
kind instead of calling them false positives.

Usage: python pipeline/14_gold_eval.py --gold dataset/gold/gold.jsonl --out results/gold_eval.json
"""
import argparse
import collections
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

IN_SCOPE = {"t", "F", "r", "z", "chi2", "Q"}
SYSTEMS = ["regex", "model", "cascade", "model_passages"]


def passages(windows):
    """Merge windows that overlap or touch into (first line, last line) runs."""
    out = []
    for w in sorted(windows, key=lambda w: w.start_line):
        if out and w.start_line <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], w.end_line)
        else:
            out.append([w.start_line, w.end_line])
    return out


def finds_for(system, pipe, text, windows, lines):
    """Every find as (first line, last line, statistic, test type)."""
    if system == "model_passages":
        out = []
        for lo, hi in passages(windows):
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


def score_paper(job):
    paper, clean_dir = job
    from statcheck_ml.normalize import normalize
    from statcheck_ml.pipeline import Pipeline

    raw = open(os.path.join(clean_dir, paper["doc"]), encoding="utf-8").read()
    pipes = {"regex": Pipeline(model_path=None), "model": Pipeline(use_pattern=False),
             "cascade": Pipeline(), "model_passages": Pipeline(use_pattern=False)}
    text, _ = normalize(raw)
    fixed, _ = pipes["cascade"].repair(text)
    lines = fixed.split("\n")
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln) + 1)
    windows = list(pipes["cascade"].prefilter.windows(fixed))
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
    for system in SYSTEMS:
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
                            **{s: hits[s][i] for s in SYSTEMS}})
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


def summarise(papers, resamples, seed):
    in_scope = lambda g: g["test_type"] in IN_SCOPE
    out = {"papers": len(papers),
           "papers_with_results": sum(p["has_results"] for p in papers),
           "gold_in_scope": recall(papers, "model", in_scope)[1]}
    k, n = recall(papers, "kept", in_scope)
    out["prefilter_keeps"] = round(k / n, 3) if n else None
    for system in SYSTEMS:
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", default="dataset/gold/gold.jsonl")
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--bronze", default="results/eval.json")
    ap.add_argument("--out", default="results/gold_eval.json")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8")]
    with ProcessPoolExecutor(args.workers) as ex:
        scored = list(ex.map(score_paper, [(p, args.clean_dir) for p in gold], chunksize=2))

    report = {"note": "Gold is the owner's hand-coded sheet: whole papers, statcheck scope. "
                      "Bronze is the three-rater consensus on sampled windows. They are never pooled.",
              "gold": {"all": summarise(scored, args.resamples, args.seed)}}
    # The model trained on windows from the 'train' papers, so those are not
    # an unseen test. 'holdout' and 'none' are.
    for subset in ("holdout", "none", "train"):
        report["gold"]["bronze_" + subset + "_papers"] = summarise(
            [p for p in scored if p["bronze_set"] == subset], args.resamples, args.seed)
    unseen = [p for p in scored if p["bronze_set"] != "train"]
    report["gold"]["unseen_papers"] = summarise(unseen, args.resamples, args.seed)

    bronze = json.load(open(args.bronze))
    report["bronze_holdout"] = {
        "source": args.bronze,
        "systems": {name: {k: v for k, v in s.items() if k in ("overall", "checkable", "damaged")}
                    for name, s in bronze["systems"].items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(report["gold"]["unseen_papers"], indent=1)[:3000])


if __name__ == "__main__":
    main()
