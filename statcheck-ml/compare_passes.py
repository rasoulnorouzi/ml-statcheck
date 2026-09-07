"""Compare two annotation passes, and prepare the disagreements for a decision.

Two passes over the same passages rarely agree. The rate of agreement is the
honest ceiling on what any model trained on these labels can reach, so it is
measured rather than assumed.

Results are matched on the window and the value of the test statistic. Offsets
are not used, because the two passes quote slightly different spans for the
same result and an offset match would call that a disagreement.

Outputs:
  agreement.json    the numbers
  disputed.json     the passages the two passes disagree about, for a third
                    pass to decide
  agreed.json       the results both passes found, which need no decision

Usage: python compare_passes.py <sample_dir>
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict


def read(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read(), strict=False)


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


def load_pass(sample_dir, prefix):
    """Merge the chunk files of one pass into {window_id: [results]}."""
    out = defaultdict(list)
    chunk_dir = os.path.join(sample_dir, "chunks")
    for name in sorted(os.listdir(chunk_dir)):
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        try:
            rows = read(os.path.join(chunk_dir, name))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            wid = r.get("window_id")
            if wid:
                out[wid] = r.get("results") or []
    return out


def main(sample_dir):
    windows = {w["window_id"]: w["text"] for w in read(os.path.join(sample_dir, "windows.json"))}
    p1 = load_pass(sample_dir, "ann_")
    p2 = load_pass(sample_dir, "ann2_")

    shared = [w for w in windows if w in p1 and w in p2]
    both = only1 = only2 = 0
    win_agree = win_total = 0
    disputed, agreed = [], []

    for wid in shared:
        v1 = {as_number(r.get("statistic")) for r in p1[wid]} - {None}
        v2 = {as_number(r.get("statistic")) for r in p2[wid]} - {None}
        both += len(v1 & v2)
        only1 += len(v1 - v2)
        only2 += len(v2 - v1)

        win_total += 1
        if v1 == v2:
            win_agree += 1
            if v1:
                agreed.append({"window_id": wid, "results": p1[wid]})
        else:
            disputed.append({
                "window_id": wid,
                "text": windows[wid],
                "pass_1": [{"quote": r.get("quote"), "statistic": r.get("statistic"),
                            "test_type": r.get("test_type")} for r in p1[wid]],
                "pass_2": [{"quote": r.get("quote"), "statistic": r.get("statistic"),
                            "test_type": r.get("test_type")} for r in p2[wid]],
            })

    total_pairs = both + only1 + only2
    jaccard = both / total_pairs if total_pairs else 0.0

    report = {
        "windows_compared": len(shared),
        "windows_in_full_agreement": win_agree,
        "window_agreement": win_agree / max(len(shared), 1),
        "results_found_by_both": both,
        "results_only_in_pass_1": only1,
        "results_only_in_pass_2": only2,
        "result_agreement_jaccard": jaccard,
        "windows_disputed": len(disputed),
    }

    json.dump(report, open(os.path.join(sample_dir, "agreement.json"), "w"), indent=1)
    json.dump(disputed, open(os.path.join(sample_dir, "disputed.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(agreed, open(os.path.join(sample_dir, "agreed.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    print(f"windows compared           : {len(shared)}")
    print(f"windows both passes agree  : {win_agree} ({100*report['window_agreement']:.1f}%)")
    print()
    print(f"results found by both      : {both}")
    print(f"results only in pass 1     : {only1}")
    print(f"results only in pass 2     : {only2}")
    print(f"agreement on results       : {100*jaccard:.1f}%")
    print()
    print(f"windows needing a decision : {len(disputed)}")
    print("\nThis agreement rate is the ceiling for any model trained on these labels.")


if __name__ == "__main__":
    main(sys.argv[1])
