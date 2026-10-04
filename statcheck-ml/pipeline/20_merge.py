#!/usr/bin/env python
"""Merge the blind adjudicator decisions into the passage labels (M5.2).

  python pipeline/20_merge.py \\
      --consensus dataset/passages/consensus.json \\
      --disputes-dir dataset/passages/disputes \\
      --decisions-dir data/annotation/passages/adjudicator \\
      --out dataset/annotations/passages/final.json

A singleton that the adjudicator keeps joins the passage with tier `adjudicated`.
A singleton it drops is recorded under `dropped` with its reason. For a field
conflict the result takes the adjudicator's value for that field, and its tier
becomes `adjudicated`. Every other result keeps the tier that 19_consensus.py gave.
Every label is bronze.

The merge stops, and writes nothing, when a decision file is missing, does not
validate, or names a dispute that the consensus does not hold. Nothing in the
output depends on the clock, so two runs give the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statcheck_ml.align import align_quote  # noqa: E402
from statcheck_ml.provenance import (  # noqa: E402
    normalise_operator, normalise_test_type, sha256_file,
)


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipeline" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


consensus_stage = _load("consensus_v4", "19_consensus.py")
adjudicate_v2 = _load("adjudicate_v2", "05_adjudicate.py")

CANDIDATE_KEYS = consensus_stage.CANDIDATE_KEYS
MAJORITY_FIELDS = consensus_stage.MAJORITY_FIELDS
CODERS = consensus_stage.CODERS
read_json = consensus_stage.read_json
dumps = consensus_stage.dumps
batch_files = consensus_stage.batch_files

GUIDELINE_PATH = ROOT / "docs" / "GUIDELINE.md"
PROVENANCE_FILE = "provenance.json"


def _digest(paths: List[Path]) -> str:
    return hashlib.sha256("".join(hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in paths).encode()).hexdigest()


# ------------------------------------------------------------- decisions --

def load_decisions(batches: List[tuple], decisions_dir: Path, items: Dict[str, dict]):
    """Every decision, validated against its dispute batch. Stops on any fault."""
    decisions: Dict[str, dict] = {}
    errors: List[str] = []
    used = set()
    for _, bpath in batches:
        dpath = decisions_dir / bpath.name
        used.add(dpath.name)
        if not dpath.exists():
            errors.append(f"{bpath.name}: decision file missing")
            continue
        try:
            output = read_json(dpath)
        except json.JSONDecodeError as e:
            errors.append(f"{bpath.name}: JSON parse error: {e}")
            continue
        batch = read_json(bpath)
        problems = []
        if isinstance(output, list):
            # Checked before the order test, so a stray id is named as unknown
            # and not as a mismatch of position.
            for i, d in enumerate(output):
                did = d.get("dispute_id") if isinstance(d, dict) else None
                if did not in items:
                    problems.append(f"record {i}: unknown dispute_id {did!r}")
        problems += adjudicate_v2.validate_adjudicated_batch(batch, output)
        if not problems:
            for item, d in zip(batch, output):
                if item["kind"] == "field_conflict" and d["keep"] is not True:
                    problems.append(f"{d['dispute_id']}: a field conflict is never dropped")
                if d["keep"] is False and d["result"] is not None:
                    problems.append(f"{d['dispute_id']}: keep is false but result is not null")
        if problems:
            errors.append(f"{bpath.name}: {problems[0]}"
                          + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""))
            continue
        for d in output:
            decisions[d["dispute_id"]] = d
    stray = sorted(p.name for _, p in batch_files(decisions_dir) if p.name not in used)
    errors += [f"{name}: decision file without a dispute batch" for name in stray]
    if errors:
        head = "\n  ".join(errors[:20])
        raise SystemExit(f"{len(errors)} decision files missing or invalid; "
                         f"the merge needs every one:\n  {head}")
    return decisions


def adjudicator_guideline_hashes(decisions_dir: Path, decisions: Dict[str, dict]):
    """The guideline hash the adjudicator ran under, when a provenance record says."""
    found = set()
    sidecar = decisions_dir / PROVENANCE_FILE
    if sidecar.exists():
        sha = read_json(sidecar).get("guideline_sha")
        if sha:
            found.add(sha)
    for d in decisions.values():
        sha = (d.get("provenance") or {}).get("guideline_sha")
        if sha:
            found.add(sha)
    return sorted(found) or None


# ----------------------------------------------------------------- merge --

def final_result(res: dict, adjudication) -> dict:
    out = {k: res.get(k) for k in CANDIDATE_KEYS}
    out["span"] = res.get("span")
    out["voters"] = res["voters"]
    out["coder_vote"] = res.get("coder_vote")
    out["coder_row"] = res.get("coder_row")
    out["tier"] = res["tier"]
    out["adjudication"] = adjudication
    return out


def normalised(field: str, value):
    if field == "test_type":
        return normalise_test_type(value)
    if field == "p_operator":
        return normalise_operator(value)
    return value


def position(res: dict) -> tuple:
    span = res.get("span")
    return (0, span[0], span[1]) if span else (1, 0, 0)


def merge(consensus: dict, items: Dict[str, dict], decisions: Dict[str, dict]):
    passages, tiers, counts = [], Counter(), Counter()
    for entry in consensus["passages"]:
        results = [final_result(r, None) for r in entry["results"]]
        pendings = [r["pending"] for r in entry["results"]]
        dropped = []
        added = []
        for disp in entry["disputes"]:
            did = disp["dispute_id"]
            item, decision = items[did], decisions[did]
            if disp["kind"] == "field_conflict":
                field, idx = disp["field"], disp["result"]
                if pendings[idx].get(field) != did:
                    raise SystemExit(f"{did}: the pending map of result {idx} does not name this dispute")
                target = results[idx]
                target[field] = normalised(field, decision["result"][field])
                target["tier"] = "adjudicated"
                adj = target["adjudication"] or {"kind": "field_conflict", "fields": [],
                                                 "dispute_ids": []}
                adj["fields"].append(field)
                adj["dispute_ids"].append(did)
                target["adjudication"] = adj
                counts["field_conflicts_resolved"] += 1
            elif decision["keep"]:
                r = decision["result"]
                quote = r.get("quote")
                span = align_quote(item["text"], quote) if quote else None
                voter = disp["voters"][0]
                new = {k: normalised(k, r.get(k)) for k in CANDIDATE_KEYS}
                new.update(span=list(span) if span else None, voters=[voter],
                           coder_vote="span" if voter == CODERS else None,
                           coder_row=disp.get("coder_row"), tier="adjudicated")
                added.append(final_result(new, {"kind": "singleton", "fields": None,
                                                "dispute_ids": [did]}))
                counts["singletons_kept"] += 1
            else:
                dropped.append({"dispute_id": did, "voter": disp["voters"][0],
                                "reason": decision["reason"],
                                "candidate": item["candidates"][0]})
                counts["singletons_dropped"] += 1
        seen = {d["dispute_id"] for d in entry["disputes"]}
        if any(did not in seen for p in pendings for did in p.values()):
            raise SystemExit(f"{entry['window_id']}: a pending dispute is not in the disputes list")
        results = sorted(results + added, key=position)
        for r in results:
            tiers[r["tier"]] += 1
        counts["results"] += len(results)
        passages.append({"window_id": entry["window_id"],
                         "contains_result": bool(results),
                         "results": results, "dropped": dropped})
    return passages, tiers, counts


def run(consensus_path: Path, disputes_dir: Path, decisions_dir: Path, out_path: Path,
        guideline_path: Path = GUIDELINE_PATH) -> dict:
    consensus_path, decisions_dir = Path(consensus_path), Path(decisions_dir)
    if not consensus_path.exists():
        raise SystemExit(f"{consensus_path} does not exist; run 19_consensus.py first")
    consensus = read_json(consensus_path)
    batches = batch_files(disputes_dir)
    if not batches:
        raise SystemExit(f"no dispute batches in {disputes_dir}")

    items: Dict[str, dict] = {}
    for _, bpath in batches:
        for item in read_json(bpath):
            if item["dispute_id"] in items:
                raise SystemExit(f"{item['dispute_id']}: in two dispute batches")
            items[item["dispute_id"]] = item
    known = {d["dispute_id"] for p in consensus["passages"] for d in p["disputes"]}
    if known != set(items):
        diff = sorted(known ^ set(items))
        raise SystemExit(f"consensus and dispute batches differ on {len(diff)} disputes, "
                         f"e.g. {diff[0]}; rerun 19_consensus.py")
    if not decisions_dir.is_dir():
        raise SystemExit(f"{decisions_dir} does not exist; the adjudicator has not run")

    decisions = load_decisions(batches, decisions_dir, items)
    passages, tiers, counts = merge(consensus, items, decisions)

    summary = {
        "passages": len(passages),
        "results": counts["results"],
        "results_by_tier": dict(sorted(tiers.items())),
        "disputes": len(items),
        "singletons_kept": counts["singletons_kept"],
        "singletons_dropped": counts["singletons_dropped"],
        "field_conflicts_resolved": counts["field_conflicts_resolved"],
    }
    out = {
        "protocol": "docs/PROTOCOL.md version 3, four-vote consensus, adjudicated",
        "label_tier": "bronze",
        "inputs": {
            "consensus_sha256": sha256_file(consensus_path),
            "dispute_batches_sha256": _digest([p for _, p in batches]),
            "decisions_sha256": _digest([decisions_dir / p.name for _, p in batches]),
            "guideline_sha256": sha256_file(guideline_path),
            "adjudicator_guideline_sha256": adjudicator_guideline_hashes(decisions_dir, decisions),
        },
        "summary": summary,
        "passages": passages,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps(out), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consensus", default=str(ROOT / "dataset/passages/consensus.json"))
    ap.add_argument("--disputes-dir", default=str(ROOT / "dataset/passages/disputes"))
    ap.add_argument("--decisions-dir", default=str(ROOT / "data/annotation/passages/adjudicator"))
    ap.add_argument("--out", default=str(ROOT / "dataset/annotations/passages/final.json"))
    args = ap.parse_args()
    summary = run(Path(args.consensus), Path(args.disputes_dir), Path(args.decisions_dir),
                  Path(args.out))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
