"""Compare the real statcheck package against the annotations and the model.

The R package is the baseline. The Python port in `extract.py` is a
convenience and is never the reference, so this script reads the CSV that
`r/run_statcheck.R` produced.

A result is matched by its window and its test statistic value, not by its
character offsets. statcheck reports values, not positions, so an offset
comparison is not possible and a value comparison is what a user would judge.

Usage:
  python compare_baseline.py <windows.json> <annotations.json> <statcheck.csv>
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read(), strict=False)


def as_number(text):
    """Parse a reported number, tolerating a missing leading zero."""
    if text is None:
        return None
    s = str(text).strip().replace("−", "-").replace("–", "-")
    s = s.lstrip("<>=").strip()
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]
    try:
        return round(float(s), 4)
    except ValueError:
        return None


def main(windows_path, ann_path, csv_path):
    windows = {w["window_id"]: w["text"] for w in read_json(windows_path)}
    ann = read_json(ann_path)

    # statcheck results, keyed by window
    sc = defaultdict(set)
    sc_rows = defaultdict(list)
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            wid = row["window_id"]
            value = as_number(row["test_value"])
            sc[wid].add(value)
            sc_rows[wid].append(row)

    # annotator results, keyed by window
    an = defaultdict(set)
    an_rows = defaultdict(list)
    damaged_values = set()
    for r in ann:
        wid = r["window_id"]
        for res in r.get("results") or []:
            value = as_number(res.get("statistic"))
            an[wid].add(value)
            an_rows[wid].append(res)
            if res.get("damaged"):
                damaged_values.add((wid, value))

    all_ids = set(windows)
    both = missed_by_statcheck = only_statcheck = 0
    damaged_missed = 0
    examples = []

    for wid in all_ids:
        a = {v for v in an[wid] if v is not None}
        s = {v for v in sc[wid] if v is not None}
        both += len(a & s)
        gap = a - s
        missed_by_statcheck += len(gap)
        only_statcheck += len(s - a)
        for v in gap:
            if (wid, v) in damaged_values:
                damaged_missed += 1
            if len(examples) < 12 and (wid, v) in damaged_values:
                quote = next((x.get("quote") for x in an_rows[wid]
                              if as_number(x.get("statistic")) == v), "")
                examples.append((wid, repr(quote)[:80]))

    n_ann = sum(len(v) for v in an.values())
    n_sc = sum(len(v) for v in sc.values())

    print(f"passages compared            : {len(all_ids)}")
    print(f"passages with a statcheck hit: {len(sc)}")
    print(f"passages with an annotation  : {sum(1 for w in an if an[w])}")
    print()
    print(f"statcheck results            : {n_sc}")
    print(f"annotated results            : {n_ann}")
    print()
    print(f"found by both                : {both}")
    print(f"annotated, statcheck missed  : {missed_by_statcheck}")
    print(f"statcheck found, not annotated: {only_statcheck}")
    if n_ann:
        print(f"\nstatcheck recall against the annotations: {100*both/max(both+missed_by_statcheck,1):.1f}%")
    print(f"of what statcheck missed, marked damaged: {damaged_missed} "
          f"({100*damaged_missed/max(missed_by_statcheck,1):.1f}%)")
    print("\nexamples statcheck cannot read:")
    for wid, quote in examples[:8]:
        print(f"  {quote}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
