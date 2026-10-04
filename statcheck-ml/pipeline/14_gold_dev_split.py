"""Create a development split from the gold_train papers.

The gold_train papers are split into dev and the remaining papers for use during
development. The split is deterministic and stratified by has_results.

Usage: python pipeline/14_gold_dev_split.py --gold dataset/gold/gold.jsonl \\
    --split dataset/gold/split.json --fraction 0.15 --seed 0 \\
    --out dataset/gold/dev.json
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_SCOPE = {"t", "F", "r", "z", "chi2", "Q"}

RULE = (
    "Papers in gold_train are grouped by has_results. Within each group, "
    "papers are sorted by doc name, shuffled with the given seed, and the "
    "first round(fraction * n) papers are selected as the development set."
)


def counts(papers, docs):
    c = {"papers": 0, "papers_with_results": 0, "papers_without_results": 0,
         "located_in_scope_results": 0}
    for p in papers:
        if p["doc"] not in docs:
            continue
        c["papers"] += 1
        c["papers_with_results" if p["has_results"] else "papers_without_results"] += 1
        c["located_in_scope_results"] += sum(
            1 for r in p["results"] if r.get("located") and r.get("test_type") in IN_SCOPE)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", default=str(ROOT / "dataset/gold/gold.jsonl"))
    ap.add_argument("--split", default=str(ROOT / "dataset/gold/split.json"))
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "dataset/gold/dev.json"))
    args = ap.parse_args()

    # Load the gold papers and existing split
    papers = [json.loads(l) for l in open(args.gold) if l.strip()]
    with open(args.split) as f:
        split_data = json.load(f)

    gold_train_set = set(split_data["gold_train"])
    gold_train_papers = [p for p in papers if p["doc"] in gold_train_set]

    # Group by has_results
    strata = defaultdict(list)
    for p in gold_train_papers:
        key = bool(p["has_results"])
        strata[key].append(p["doc"])

    # Sort, shuffle, and select dev within each stratum
    rng = random.Random(args.seed)
    dev = set()
    for key in sorted(strata.keys()):
        docs = sorted(strata[key])
        rng.shuffle(docs)
        dev_size = round(args.fraction * len(docs))
        dev.update(docs[:dev_size])

    # Verify and compute counts
    dev_papers = [p for p in papers if p["doc"] in dev]
    rest_papers = [p for p in papers if p["doc"] in gold_train_set and p["doc"] not in dev]

    out = {
        "seed": args.seed,
        "fraction": args.fraction,
        "rule": RULE,
        "dev": sorted(dev),
        "counts": {
            "dev": counts(papers, dev),
            "rest": counts(papers, {p["doc"] for p in rest_papers}),
        },
    }

    Path(args.out).write_text(json.dumps(out, sort_keys=True, indent=1) + "\n")
    print(json.dumps(out["counts"], sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
