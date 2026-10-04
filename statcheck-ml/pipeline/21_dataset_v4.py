#!/usr/bin/env python
"""Build the mixed training set of version 4 (M5.3).

  python pipeline/21_dataset_v4.py \\
      --bronze dataset/train.jsonl \\
      --final dataset/annotations/passages/final.json \\
      --key dataset/passages/key.json \\
      --batches dataset/passages/batches \\
      --out dataset/train_v4.jsonl --splits-out dataset/splits_v4.json

Rows: the bronze training windows, written back byte for byte, then one row per
passage of final.json in the row format of train.jsonl, with `part_spans` found
by align.py, `label_tier` bronze, `round` v4.

A new stage and not an option of 06_dataset.py, because 06 builds one set from
one windows file and one final file, and reproduce.sh hashes its output. This
stage joins two sources, keeps one side of its output unchanged, inherits an
earlier split, and runs the gold guard. It reuses `align_result` from 06.

Gates, each of which stops the build before any file is written:
  - the gold test papers never enter, in either source;
  - the passage rows never come from a development paper;
  - the text of every passage matches the hash in key.json;
  - at most 1 % of the passage results fail to align.

The bronze windows of a development paper are left out. Their labels are
bronze, but the papers are the ones the gold evaluation scores on dev.json, and
a model that trained on the text of a dev paper scores too high there. The
count is in the summary. `--keep-dev-bronze` keeps them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statcheck_ml.gold import _pid, assert_not_gold_test, load_split  # noqa: E402
from statcheck_ml.splits import make_splits  # noqa: E402

_spec = importlib.util.spec_from_file_location("dataset_v2", ROOT / "pipeline" / "06_dataset.py")
dataset_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dataset_v2)

ROUND = "v4"
MAX_UNALIGNED = 0.01
RESULT_KEYS = ["test_type", "statistic", "df1", "df2", "n", "p_operator", "p_value",
               "quote", "damaged", "confidence"]


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"), strict=False)


def batch_texts(batches_dir: Path) -> dict:
    texts = {}
    for path in sorted(Path(batches_dir).glob("batch_*.json")):
        for passage in read_json(path):
            texts[passage["window_id"]] = passage["text"]
    return texts


def guard(docs, split: dict, dev: set) -> set:
    """Raise on a gold test paper. Return the docs that are development papers."""
    assert_not_gold_test(docs, split)
    dev_ids = {_pid(d) for d in dev}
    return {d for d in docs if d in dev or _pid(d) in dev_ids}


def passage_row(passage: dict, key: dict, text: str):
    """(row, failures): the row in train.jsonl format, and the results that did not align."""
    results, failures = [], []
    for res in passage["results"]:
        block_span, part_spans = dataset_v2.align_result(text, res)
        if not block_span:
            failures.append(res.get("quote"))
            continue
        out = {k: res.get(k) for k in RESULT_KEYS}
        out["damaged"] = bool(res.get("damaged"))
        out.update(
            block_span=block_span, part_spans=part_spans,
            checkable=bool(res.get("statistic")) and bool(res.get("df1")) and bool(res.get("p_value")),
            tier=res["tier"],
            raters=[v.removeprefix("annotator-") for v in res["voters"] if v != "coders"],
            voters=res["voters"], coder_vote=res.get("coder_vote"),
        )
        results.append(out)
    row = {
        "window_id": passage["window_id"], "round": ROUND,
        # The passage sample has strata, not the pools of the window sample; the
        # stratum is the nearest record of how the passage was drawn.
        "pool": key.get("stratum"), "journal": key.get("journal"),
        "source_doc": key["doc"], "source_line": key["start_line"], "text": text,
        "contains_result": bool(results), "label_tier": "bronze", "results": results,
        "set": "train", "tier_counts": dict(Counter(r["tier"] for r in results)),
    }
    return row, failures


def build(bronze_path, final_path, key_path, batches_dir, split_path, dev_path, old_splits_path,
          out_path, splits_out, seed=0, dev_share=0.15, keep_dev_bronze=False) -> dict:
    split = load_split(split_path)
    dev = set(read_json(dev_path)["dev"])
    old = read_json(old_splits_path)["documents"]

    bronze = []  # (source_doc, original line)
    for line in Path(bronze_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            bronze.append((json.loads(line)["source_doc"], line))
    bronze_docs = {d for d, _ in bronze}
    bronze_dev = guard(bronze_docs, split, dev)
    skipped = 0 if keep_dev_bronze else sum(1 for d, _ in bronze if d in bronze_dev)
    bronze = [(d, line) for d, line in bronze if keep_dev_bronze or d not in bronze_dev]
    unknown = sorted({d for d, _ in bronze} - set(old))
    if unknown:
        raise SystemExit(f"{len(unknown)} bronze documents are not in {old_splits_path}, "
                         f"e.g. {unknown[0]}")

    final = read_json(final_path)
    keys = {k["window_id"]: k for k in read_json(key_path)}
    texts = batch_texts(batches_dir)
    ids = [p["window_id"] for p in final["passages"]]
    if set(ids) != set(texts):
        diff = sorted(set(ids) ^ set(texts))
        raise SystemExit(f"final.json and the batches differ on {len(diff)} passages, e.g. {diff[0]}")
    missing = [i for i in ids if i not in keys]
    if missing:
        raise SystemExit(f"{len(missing)} passages are not in key.json, e.g. {missing[0]}")
    for wid in ids:
        if hashlib.sha256(texts[wid].encode("utf-8")).hexdigest() != keys[wid]["text_sha256"]:
            raise SystemExit(f"{wid}: batch text differs from the hash in key.json")
    passage_docs = {keys[i]["doc"] for i in ids}
    on_dev = guard(passage_docs, split, dev)
    if on_dev:
        raise SystemExit(f"development paper in the passage rows: {sorted(on_dev)[0]}")

    rows, unaligned, n_results, failed_passages = [], [], 0, []
    for passage in final["passages"]:
        wid = passage["window_id"]
        row, failures = passage_row(passage, keys[wid], texts[wid])
        n_results += len(passage["results"])
        if failures:
            unaligned += [(wid, q) for q in failures]
            failed_passages.append(wid)
            continue
        rows.append(row)
    share = len(unaligned) / n_results if n_results else 0.0
    if share > MAX_UNALIGNED:
        head = "\n  ".join(f"{w}: {q!r}" for w, q in unaligned[:20])
        raise SystemExit(f"alignment gate: {len(unaligned)} of {n_results} results "
                         f"({share:.2%}) do not align; the limit is {MAX_UNALIGNED:.0%}\n  {head}")

    # A passage with one result that does not align is left out whole: the
    # model would learn to skip the text of a true result it could not locate.
    kept_passage_docs = {r["source_doc"] for r in rows}
    new_docs = sorted(kept_passage_docs - set(old))
    fresh = make_splits(new_docs, dev_share=dev_share, seed=seed)["documents"] if new_docs else {}
    all_docs = sorted({d for d, _ in bronze} | kept_passage_docs)
    documents = {d: old[d] if d in old else fresh[d] for d in all_docs}
    splits = {
        "seed": seed, "dev_share": dev_share,
        "inherited_from": "dataset/splits.json",
        "inherited_documents": sum(1 for d in all_docs if d in old),
        "new_documents": len(new_docs),
        "excluded_gold_dev_documents": sorted(bronze_dev) if not keep_dev_bronze else [],
        "documents": documents,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for _, line in bronze] + [json.dumps(r, ensure_ascii=False) for r in rows]
    out_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    Path(splits_out).write_text(json.dumps(splits, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")

    side = Counter(documents.values())
    return {
        "rows": len(lines), "bronze_rows": len(bronze), "passage_rows": len(rows),
        "bronze_rows_left_out_dev_papers": skipped,
        "passage_results": n_results, "passage_results_unaligned": len(unaligned),
        "passage_rows_left_out_unaligned": len(failed_passages),
        "passage_results_aligned_share": round(1 - share, 4),
        "passage_results_no_part_spans": sum(1 for r in rows for x in r["results"]
                                             if not x["part_spans"]),
        "documents": len(documents), "documents_train": side["train"], "documents_dev": side["dev"],
        "new_documents": len(new_docs),
    }


def main() -> None:
    d = ROOT / "dataset"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bronze", default=str(d / "train.jsonl"))
    ap.add_argument("--final", default=str(d / "annotations/passages/final.json"))
    ap.add_argument("--key", default=str(d / "passages/key.json"))
    ap.add_argument("--batches", default=str(d / "passages/batches"))
    ap.add_argument("--split", default=str(d / "gold/split.json"))
    ap.add_argument("--dev", default=str(d / "gold/dev.json"))
    ap.add_argument("--old-splits", default=str(d / "splits.json"))
    ap.add_argument("--out", default=str(d / "train_v4.jsonl"))
    ap.add_argument("--splits-out", default=str(d / "splits_v4.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-share", type=float, default=0.15)
    ap.add_argument("--keep-dev-bronze", action="store_true")
    a = ap.parse_args()
    for path in (a.final, a.key):
        if not Path(path).exists():
            raise SystemExit(f"{path} does not exist; run the earlier stages first")
    summary = build(a.bronze, a.final, a.key, a.batches, a.split, a.dev, a.old_splits, a.out,
                    a.splits_out, a.seed, a.dev_share, a.keep_dev_bronze)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
