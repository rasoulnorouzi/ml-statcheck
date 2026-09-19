"""Draw a test set from documents that no round has ever used.

The current test part shares a sampling run with the training data. The
documents do not overlap, but the draw did, so a reader can object that the
test windows were selected under the same conditions.

This script removes that objection. It reads every document already used, and
samples only from the papers that remain. About 1800 of the 3100 papers are
untouched.

The output is annotated by the full protocol, two blind passes and adjudication
of every dispute, and is then frozen.

Usage:
  python pipeline/01_sample_holdout.py <clean_dir> <used_key.json> <out_dir> [n_windows]
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

from sample_windows import (WEIGHTS, CONTEXT_LINES, classify, strip_references)


def used_documents(*key_paths) -> set:
    """Every document that has appeared in an earlier sample."""
    used = set()
    for path in key_paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for row in json.load(fh):
                if row.get("doc"):
                    used.add(row["doc"])
    return used


def windows_of(path):
    lines = strip_references(open(path, encoding="utf-8", errors="replace").read().split("\n"))
    for i, line in enumerate(lines):
        if len(line) < 20:
            continue
        lo = max(0, i - CONTEXT_LINES)
        hi = min(len(lines), i + CONTEXT_LINES + 1)
        text = "\n".join(lines[lo:hi])
        pool = classify(line, text)
        if pool:
            yield text, pool, i


def main():
    clean_dir, key_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    total = int(sys.argv[4]) if len(sys.argv) > 4 else 600

    used = used_documents(key_path)

    files = []
    for dp, _, fns in os.walk(clean_dir):
        for fn in fns:
            if fn.endswith(".txt"):
                rel = os.path.relpath(os.path.join(dp, fn), clean_dir).replace(os.sep, "/")
                if rel not in used:
                    files.append((os.path.join(dp, fn), rel))

    print(f"documents already used : {len(used)}")
    print(f"documents still unused : {len(files)}")
    if not files:
        raise SystemExit("no untouched documents remain")

    random.seed(101)                      # a different seed from every earlier draw
    random.shuffle(files)

    targets = {p: max(1, int(total * w)) for p, w in WEIGHTS.items()}
    pools = {p: [] for p in WEIGHTS}
    cap = {p: n * 30 for p, n in targets.items()}

    for path, rel in files:
        if all(len(pools[p]) >= cap[p] for p in pools):
            break
        taken = {p: 0 for p in WEIGHTS}
        try:
            for text, pool, ln in windows_of(path):
                if len(pools[pool]) >= cap[pool] or taken[pool] >= 2:
                    continue
                taken[pool] += 1
                pools[pool].append({"doc": rel, "journal": rel.split("/")[0],
                                    "line": ln, "text": text})
        except Exception:
            continue

    os.makedirs(out_dir, exist_ok=True)
    sample, key = [], []
    for pool in WEIGHTS:
        rows = pools[pool]
        random.shuffle(rows)
        seen, chosen = set(), []
        for r in rows:
            if r["doc"] in seen:          # one window per document, at most
                continue
            seen.add(r["doc"])
            chosen.append(r)
            if len(chosen) >= targets[pool]:
                break
        for r in chosen:
            wid = "ho" + hashlib.sha1((r["doc"] + str(r["line"])).encode()).hexdigest()[:8]
            sample.append({"window_id": wid, "text": r["text"]})
            key.append({"window_id": wid, "pool": pool, "doc": r["doc"],
                        "journal": r["journal"], "line": r["line"]})

    random.shuffle(sample)
    json.dump(sample, open(os.path.join(out_dir, "windows.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(key, open(os.path.join(out_dir, "key.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    overlap = {k["doc"] for k in key} & used
    print(f"drawn : " + ", ".join(f"{p}={sum(1 for k in key if k['pool']==p)}" for p in WEIGHTS))
    print(f"total : {len(sample)} windows from {len({k['doc'] for k in key})} documents")
    print(f"journals covered : {len({k['journal'] for k in key})}")
    print(f"overlap with used documents : {len(overlap)}  (must be 0)")
    if overlap:
        raise SystemExit("the holdout overlaps documents already used")


if __name__ == "__main__":
    main()
