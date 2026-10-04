"""Split the hand-coded gold papers into a frozen test part and a trainable part."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_SCOPE = {"t", "F", "r", "z", "chi2", "Q"}

RULE = (
    "Papers that gave windows to the bronze holdout go to gold_test. "
    "Papers that gave windows to the bronze training set go to gold_train, "
    "because the shipped model trained on them. "
    "The remaining papers are sorted by has_results, journal and doc, shuffled "
    "inside each has_results and journal stratum with the seed, and joined in "
    "sorted order. A paper at position i goes to gold_train when i % {k} == {r}, "
    "and to gold_test otherwise."
)


def journal(doc):
    return doc.split("/", 1)[0]


def split(papers, seed, every):
    test, train = set(), set()
    rest = []
    for p in papers:
        if p["bronze_set"] == "holdout":
            test.add(p["doc"])
        elif p["bronze_set"] == "train":
            train.add(p["doc"])
        else:
            rest.append(p)
    strata = defaultdict(list)
    for p in rest:
        strata[(bool(p["has_results"]), journal(p["doc"]))].append(p["doc"])
    rng = random.Random(seed)
    ordered = []
    for key in sorted(strata):
        docs = sorted(strata[key])
        rng.shuffle(docs)
        ordered.extend(docs)
    for i, doc in enumerate(ordered):
        (train if i % every == every - 1 else test).add(doc)
    return test, train


def counts(papers, docs):
    c = {"papers": 0, "papers_with_results": 0, "papers_without_results": 0,
         "located_in_scope_results": 0}
    by = Counter()
    for p in papers:
        if p["doc"] not in docs:
            continue
        c["papers"] += 1
        c["papers_with_results" if p["has_results"] else "papers_without_results"] += 1
        c["located_in_scope_results"] += sum(
            1 for r in p["results"] if r.get("located") and r.get("test_type") in IN_SCOPE)
        by[p["bronze_set"]] += 1
    c["by_bronze_set"] = dict(sorted(by.items()))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "dataset/gold/gold.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "dataset/gold/split.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-every", type=int, default=3)
    args = ap.parse_args()

    papers = [json.loads(l) for l in open(args.gold) if l.strip()]
    test, train = split(papers, args.seed, args.train_every)
    assert not test & train and len(test) + len(train) == len(papers)
    out = {
        "seed": args.seed,
        "rule": RULE.format(k=args.train_every, r=args.train_every - 1),
        "gold_test": sorted(test),
        "gold_train": sorted(train),
        "counts": {"gold_test": counts(papers, test), "gold_train": counts(papers, train)},
    }
    Path(args.out).write_text(json.dumps(out, sort_keys=True, indent=1) + "\n")
    print(json.dumps(out["counts"], sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
