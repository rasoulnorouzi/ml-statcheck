#!/usr/bin/env python
"""Consensus over three raters, and adjudication of what they disagree on.

Three modes, run in order for a set (`train` or `holdout`):

  python pipeline/05_adjudicate.py --mode disputes --set train
      Reads the three raters and the windows. Writes one consensus record
      per window to data/annotation/train/consensus.json, and writes the
      disputes it could not settle as batch files under
      data/annotation/train/disputes/, for the adjudicator agent to read.

  python pipeline/05_adjudicate.py --mode collect --set train
      Reads the adjudicator's output from data/annotation/train/adjudicated/,
      validates it batch by batch, stamps it with provenance, and writes
      data/annotation/train/adjudicated.json.

  python pipeline/05_adjudicate.py --mode merge --set train
      Applies the adjudicator's decisions to consensus.json and writes the
      final labels to dataset/annotations/train/final.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statcheck_ml.consensus import consensus, NINE_FIELDS  # noqa: E402
from statcheck_ml.align import align_quote  # noqa: E402
from statcheck_ml.provenance import (  # noqa: E402
    sha256_file, stamp, normalise_operator, normalise_test_type,
)

RATERS = ["haiku", "sonnet", "opus"]
BATCH_SIZE = 20
CANDIDATE_KEYS = NINE_FIELDS + ["confidence"]
ADJUDICATOR_MODEL_ID = "claude-opus-5"

ROOT = Path(__file__).resolve().parents[1]              # statcheck-ml
REPO_ROOT = ROOT.parents[0]                              # holds .claude/
GUIDELINE_PATH = ROOT / "docs" / "GUIDELINE.md"
AGENT_PATH = REPO_ROOT / ".claude" / "agents" / "adjudicator.md"


def read_json(path: Path):
    """Load JSON the way rater and adjudicator output needs: quotes carry
    raw control characters, which the strict JSON reader rejects."""
    return json.loads(path.read_text(encoding="utf-8"), strict=False)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def trim_candidate(candidate: dict) -> dict:
    """The nine keys and confidence only: what the adjudicator agent reads."""
    return {k: candidate.get(k) for k in CANDIDATE_KEYS}


def dispute_id(set_name: str, window_id: str, k: int) -> str:
    return f"{set_name}-{window_id}-{k}"


# --------------------------------------------------------------- disputes --

def run_disputes(set_name: str) -> None:
    ann_dir = ROOT / "dataset" / "annotations" / set_name
    windows_path = ROOT / "dataset" / "windows" / f"{set_name}.json"
    out_dir = ROOT / "data" / "annotation" / set_name

    windows = read_json(windows_path)
    texts = {w["window_id"]: w["text"] for w in windows}
    rater_records = {name: {rec["window_id"]: rec for rec in read_json(ann_dir / f"{name}.json")}
                      for name in RATERS}

    consensus_out = []
    dispute_items = []
    rng = random.Random(0)
    tier_counts: dict = {}
    kind_counts: dict = {}

    for w in windows:
        wid = w["window_id"]
        per_rater = {name: rater_records[name].get(wid, {}).get("results", []) for name in RATERS}
        result = consensus(per_rater)
        for rec in result["kept"]:
            tier_counts[rec["tier"]] = tier_counts.get(rec["tier"], 0) + 1
        for k, d in enumerate(result["disputes"]):
            kind_counts[d["kind"]] = kind_counts.get(d["kind"], 0) + 1
            candidates = [trim_candidate(c) for c in d["candidates"]]
            rng.shuffle(candidates)
            dispute_items.append({
                "dispute_id": dispute_id(set_name, wid, k),
                "window_id": wid,
                "text": texts.get(wid, ""),
                "kind": d["kind"],
                "field": d["field"],
                "candidates": candidates,
            })
        consensus_out.append({"window_id": wid, "kept": result["kept"], "disputes": result["disputes"]})

    write_json(out_dir / "consensus.json", consensus_out)

    disputes_dir = out_dir / "disputes"
    disputes_dir.mkdir(parents=True, exist_ok=True)
    for stale in disputes_dir.glob("batch_*.json"):
        stale.unlink()

    n_batches = 0
    for i in range(0, len(dispute_items), BATCH_SIZE):
        write_json(disputes_dir / f"batch_{n_batches:03d}.json", dispute_items[i:i + BATCH_SIZE])
        n_batches += 1

    tier_str = ", ".join(f"{k}={v}" for k, v in sorted(tier_counts.items())) or "none"
    kind_str = ", ".join(f"{k}={v}" for k, v in sorted(kind_counts.items())) or "none"
    print(f"{set_name}: windows {len(windows)}")
    print(f"kept by tier:     {tier_str}")
    print(f"disputes by kind: {kind_str}")
    print(f"disputes total:  {len(dispute_items)}")
    print(f"batches:         {n_batches}")


# ---------------------------------------------------------------- collect --

def validate_adjudicated_batch(batch: list, output) -> list:
    """Validate one adjudicator batch output against its dispute batch."""
    errors = []
    if not isinstance(output, list):
        return ["output is not a list"]
    if len(output) != len(batch):
        errors.append(f"output length {len(output)} != batch length {len(batch)}")

    required = set(CANDIDATE_KEYS)
    for i, (d, decision) in enumerate(zip(batch, output)):
        want_id = d.get("dispute_id")
        if not isinstance(decision, dict):
            errors.append(f"record {i}: not a dict")
            continue
        got_id = decision.get("dispute_id")
        if got_id != want_id:
            errors.append(f"record {i}: dispute_id {got_id} != batch {want_id}")

        keep = decision.get("keep")
        if not isinstance(keep, bool):
            errors.append(f"record {i}: keep is not bool")

        result = decision.get("result")
        if result is None:
            if keep is True:
                errors.append(f"record {i}: result is null but keep is true")
        elif not isinstance(result, dict):
            errors.append(f"record {i}: result is neither null nor a dict")
        else:
            got_keys = set(result.keys())
            if got_keys != required:
                if required - got_keys:
                    errors.append(f"record {i}: result missing keys {required - got_keys}")
                if got_keys - required:
                    errors.append(f"record {i}: result extra keys {got_keys - required}")
            op = result.get("p_operator")
            if op is not None and normalise_operator(op) is None:
                errors.append(f"record {i}: p_operator {op!r} not recognised")
            tt = result.get("test_type")
            if tt is not None and normalise_test_type(tt) is None:
                errors.append(f"record {i}: test_type {tt!r} not recognised")
            conf = result.get("confidence")
            if conf not in ("high", "low"):
                errors.append(f"record {i}: confidence {conf!r} not in {{high, low}}")

        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"record {i}: reason is empty or not a string")
    return errors


def run_collect(set_name: str) -> None:
    out_dir = ROOT / "data" / "annotation" / set_name
    disputes_dir = out_dir / "disputes"
    adjudicated_dir = out_dir / "adjudicated"

    batch_files = sorted(disputes_dir.glob("batch_*.json"))
    guideline_sha = sha256_file(GUIDELINE_PATH)
    agent_sha = sha256_file(AGENT_PATH)
    collected_at = datetime.now(timezone.utc).isoformat()

    decisions = []
    failures = []

    for batch_file in batch_files:
        batch_id = batch_file.stem
        batch = read_json(batch_file)
        output_file = adjudicated_dir / f"{batch_id}.json"
        if not output_file.exists():
            failures.append({"batch_id": batch_id, "errors": [f"output file not found: {output_file}"]})
            continue
        try:
            output = read_json(output_file)
        except json.JSONDecodeError as e:
            failures.append({"batch_id": batch_id, "errors": [f"JSON parse error: {e}"]})
            continue

        errors = validate_adjudicated_batch(batch, output)
        if errors:
            failures.append({"batch_id": batch_id, "errors": errors})
            continue

        for decision in output:
            record = {"dispute_id": decision["dispute_id"], "keep": decision["keep"],
                      "result": decision["result"], "reason": decision["reason"]}
            stamp(record, rater="adjudicator", model_alias="opus", model_id=ADJUDICATOR_MODEL_ID,
                  guideline_sha=guideline_sha, agent_sha=agent_sha, batch_id=batch_id,
                  collected_at=collected_at)
            decisions.append(record)

    if failures:
        write_json(out_dir / "adjudicated.failed.json", failures)
        print(f"Failed batches: {len(failures)}")
        for f in failures:
            print(f"  {f['batch_id']}: {f['errors'][0]}")
        sys.exit(1)

    write_json(out_dir / "adjudicated.json", decisions)
    kept = sum(1 for d in decisions if d["keep"])
    print(f"{set_name}: decisions {len(decisions)}, kept {kept}, rejected {len(decisions) - kept}")


# ------------------------------------------------------------------ merge --

def _apply_singleton(results: list, decision: Optional[dict], raters: list, text: str) -> bool:
    """Returns True when a result was added."""
    if decision is None or not decision.get("keep"):
        return False
    result = decision["result"]
    quote = result.get("quote")
    span = align_quote(text, quote) if quote else None
    record = {
        "test_type": normalise_test_type(result.get("test_type")),
        "statistic": result.get("statistic"),
        "df1": result.get("df1"),
        "df2": result.get("df2"),
        "n": result.get("n"),
        "p_operator": normalise_operator(result.get("p_operator")),
        "p_value": result.get("p_value"),
        "quote": quote,
        "damaged": result.get("damaged"),
        "confidence": result.get("confidence"),
        "span": list(span) if span else None,
        "tier": "adjudicated",
        "raters": raters,
    }
    results.append(record)
    return True


def run_merge(set_name: str) -> None:
    out_dir = ROOT / "data" / "annotation" / set_name
    consensus_path = out_dir / "consensus.json"
    adjudicated_path = out_dir / "adjudicated.json"
    windows_path = ROOT / "dataset" / "windows" / f"{set_name}.json"
    final_path = ROOT / "dataset" / "annotations" / set_name / "final.json"

    consensus_records = read_json(consensus_path)
    decisions_by_id = {d["dispute_id"]: d for d in read_json(adjudicated_path)}
    texts = {w["window_id"]: w["text"] for w in read_json(windows_path)}

    guideline_sha = sha256_file(GUIDELINE_PATH)
    built_at = datetime.now(timezone.utc).isoformat()

    final_records = []
    tier_counts: dict = {}
    n_disputes = 0
    n_adjudicated = 0
    unresolved = []
    n_none_in_disputed_field = 0

    for entry in consensus_records:
        wid = entry["window_id"]
        text = texts.get(wid, "")
        results = [dict(rec) for rec in entry["kept"]]
        pending = iter([rec for rec in results if rec.get("pending_field")])

        for k, d in enumerate(entry["disputes"]):
            n_disputes += 1
            did = dispute_id(set_name, wid, k)
            decision = decisions_by_id.get(did)

            if d["kind"] == "singleton":
                if _apply_singleton(results, decision, d["raters"], text):
                    n_adjudicated += 1
            else:  # field_conflict
                rec = next(pending)
                field = d["field"]
                if decision is None or not decision.get("keep"):
                    unresolved.append(did)
                    n_none_in_disputed_field += 1  # the pending field stays None
                    continue
                value = decision["result"].get(field)
                if field == "test_type":
                    value = normalise_test_type(value)
                elif field == "p_operator":
                    value = normalise_operator(value)
                rec[field] = value
                if field == "quote" and value:
                    span = align_quote(text, value)
                    rec["span"] = list(span) if span else None
                rec.pop("pending_field", None)
                rec["tier"] = "adjudicated"
                n_adjudicated += 1
                if value is None:
                    n_none_in_disputed_field += 1

        for rec in results:
            rec.pop("pending_field", None)  # a field left unresolved stays None
            tier_counts[rec["tier"]] = tier_counts.get(rec["tier"], 0) + 1

        final_records.append({
            "window_id": wid,
            "contains_result": len(results) > 0,
            "results": [{key: rec.get(key) for key in
                         (NINE_FIELDS + ["confidence", "span", "tier", "raters"])} for rec in results],
            "provenance": {},
        })

    provenance = {
        "guideline_sha": guideline_sha,
        "consensus_rule": "2-of-3 lenient",
        "adjudicator_model_id": ADJUDICATOR_MODEL_ID,
        "built_at": built_at,
        "n_disputes": n_disputes,
        "n_adjudicated": n_adjudicated,
    }
    for rec in final_records:
        rec["provenance"] = provenance

    final_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(final_path, final_records)

    print(f"{set_name}: windows {len(final_records)}")
    print("tiers: " + ", ".join(f"{k}={v}" for k, v in sorted(tier_counts.items())))
    print(f"disputes {n_disputes}, adjudicated {n_adjudicated}, unresolved {len(unresolved)}")
    if unresolved:
        print("unresolved dispute ids: " + ", ".join(unresolved))
    print(f"results with a None field on a disputed field/adjudicated result: {n_none_in_disputed_field}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["disputes", "collect", "merge"])
    ap.add_argument("--set", required=True, dest="set_name", choices=["train", "holdout"])
    args = ap.parse_args()

    if args.mode == "disputes":
        run_disputes(args.set_name)
    elif args.mode == "collect":
        run_collect(args.set_name)
    else:
        run_merge(args.set_name)


if __name__ == "__main__":
    main()
