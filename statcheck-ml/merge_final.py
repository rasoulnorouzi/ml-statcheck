"""Build the final label set for a round from two passes and an adjudication.

Each window gets the best label available, and the file records which:

  adjudicated  the two passes disagreed and a third pass decided
  agreed       both passes produced the same results
  single       only one pass covered this window

The tier is written into every row. A model may train on all three, but a
report must never present a `single` label as if two annotators had confirmed
it.

Usage: python merge_final.py <sample_dir>
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict


def read(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read(), strict=False)


def load_pass(sample_dir, prefix):
    out = {}
    chunk_dir = os.path.join(sample_dir, "chunks")
    for name in sorted(os.listdir(chunk_dir)):
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        try:
            rows = read(os.path.join(chunk_dir, name))
        except Exception:
            continue
        if isinstance(rows, list):
            for r in rows:
                if r.get("window_id"):
                    out[r["window_id"]] = r.get("results") or []
    return out


def as_number(text):
    if text is None:
        return None
    s = str(text).strip().replace("−", "-").replace("–", "-").lstrip("<>=").strip()
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]
    try:
        return round(float(s), 4)
    except ValueError:
        return None


def main(sample_dir):
    windows = [w["window_id"] for w in read(os.path.join(sample_dir, "windows.json"))]
    p1 = load_pass(sample_dir, "ann_")
    p2 = load_pass(sample_dir, "ann2_")

    adj = {}
    adj_path = os.path.join(sample_dir, "adjudicated.json")
    if os.path.exists(adj_path):
        for r in read(adj_path):
            if r.get("window_id"):
                adj[r["window_id"]] = r.get("results") or []

    final, tiers = [], Counter()
    for wid in windows:
        if wid in adj:
            results, tier = adj[wid], "adjudicated"
        elif wid in p1 and wid in p2:
            v1 = {as_number(r.get("statistic")) for r in p1[wid]} - {None}
            v2 = {as_number(r.get("statistic")) for r in p2[wid]} - {None}
            if v1 != v2:
                # The passes disagree and no decision was made. Leaving it in
                # would teach the model one annotator's mistake, so it is
                # dropped and counted.
                tiers["undecided_dropped"] += 1
                continue
            # Both agree. Prefer pass 2, which applied the fuller guide.
            results, tier = (p2[wid] or p1[wid]), "agreed"
        elif wid in p2:
            results, tier = p2[wid], "single"
        elif wid in p1:
            results, tier = p1[wid], "single"
        else:
            tiers["not_annotated"] += 1
            continue

        tiers[tier] += 1
        final.append({"window_id": wid, "contains_result": bool(results),
                      "results": results, "label_tier": tier})

    dest = os.path.join(sample_dir, "annotations_final.json")
    json.dump(final, open(dest, "w", encoding="utf-8"), ensure_ascii=False)

    n_res = sum(len(r["results"]) for r in final)
    n_dmg = sum(1 for r in final for x in r["results"] if x.get("damaged"))
    print(f"wrote {dest}")
    for k, v in tiers.most_common():
        print(f"  {k:20s} {v}")
    print(f"  windows kept         {len(final)}")
    print(f"  results              {n_res}")
    print(f"  marked damaged       {n_dmg} ({100*n_dmg/max(n_res,1):.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1])
