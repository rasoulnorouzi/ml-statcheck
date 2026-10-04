#!/usr/bin/env python
"""Four-vote consensus over the passages (docs/PROTOCOL.md version 3).

Voters: annotator-haiku, annotator-sonnet, annotator-opus, and the coders,
whose rows come from the inventory. A result stays with two votes of four.
A singleton and every field conflict go to the adjudicator in blind batches
of 20. Every label written here is bronze.

  python pipeline/19_consensus.py \\
      --batches dataset/passages/batches \\
      --raters data/annotation/passages \\
      --inventory dataset/passages/inventory.jsonl \\
      --out dataset/passages/consensus.json \\
      --disputes-dir dataset/passages/disputes \\
      --seed 0

`--out` holds the labels before adjudication: the kept results, with a
`pending` map from each field in conflict to its dispute, and every dispute
with the voter of each candidate in the order the adjudicator sees them.
The adjudicator never reads `--out`. The merge of the decisions is M5.2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statcheck_ml.agreement import NUMERIC_FIELDS  # noqa: E402
from statcheck_ml.align import align_quote, as_number  # noqa: E402
from statcheck_ml.consensus import NINE_FIELDS, _cluster  # noqa: E402
from statcheck_ml.provenance import (  # noqa: E402
    normalise_operator, normalise_test_type, sha256_file, validate_batch,
)

RATERS = ("haiku", "sonnet", "opus")
CODERS = "coders"
VOTERS = tuple(f"annotator-{r}" for r in RATERS) + (CODERS,)
RATER_VOTERS = VOTERS[:3]
SIX_TESTS = frozenset({"t", "F", "r", "z", "chi2", "Q"})
MAJORITY_FIELDS = ["test_type", "statistic", "df1", "df2", "n",
                   "p_operator", "p_value", "damaged"]
CODER_NEVER_VOTES = frozenset({"n", "quote", "confidence"})
CANDIDATE_KEYS = NINE_FIELDS + ["confidence"]
BATCH_SIZE = 20
GUIDELINE_PATH = ROOT / "docs" / "GUIDELINE.md"

_NUMBER_RE = re.compile(r"[-−–]?(?:\d+(?:\.\d+)?|\.\d+)")
_BATCH_RE = re.compile(r"batch_(\d+)\.json$")


def read_json(path: Path):
    # Quotes carry raw control characters, which the strict reader rejects.
    return json.loads(Path(path).read_text(encoding="utf-8"), strict=False)


def dumps(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


def field_key(field: str, value):
    if field in NUMERIC_FIELDS:
        return as_number(value)
    if field == "test_type":
        return normalise_test_type(value)
    if field == "p_operator":
        return normalise_operator(value)
    return value


# ---------------------------------------------------------------- voters --

def rater_result(res: dict, text: str) -> dict:
    out = {k: res.get(k) for k in CANDIDATE_KEYS}
    out["test_type"] = normalise_test_type(out["test_type"])
    out["p_operator"] = normalise_operator(out["p_operator"])
    span = align_quote(text, out["quote"]) if out["quote"] else None
    out["span"] = list(span) if span else None
    return out


def coder_votes(rows: List[dict], text: str) -> Tuple[List[dict], Counter]:
    """The coder rows of one passage that vote, as results that can be matched."""
    counts: Counter = Counter()
    votes = []
    for row in rows:
        counts["rows"] += 1
        if normalise_test_type(row.get("test_type")) not in SIX_TESTS:
            counts["out_of_scope"] += 1
            continue
        span = row.get("span_in_passage")
        if span is not None:
            start, end = max(0, span[0]), min(len(text), span[1])
            if (start, end) != (span[0], span[1]):
                counts["span_clamped"] += 1
            # A span wholly outside the piece points nowhere in this text, so
            # the row can only vote the way an unlocated row does.
            span = [start, end] if end > start else None
        # The gold locator spans a result from the first to the last of the
        # numbers it picked, and it sometimes picks one far away (38 rows of
        # 1644 span past their passage or over 200 characters). The position
        # of the statistic is exact in every row, and a rater result that holds
        # the statistic is the same result, so the statistic is what matches.
        stat = row.get("statistic_span_in_passage")
        if stat is not None and 0 <= stat[0] < stat[1] <= len(text):
            if span is None or not (span[0] <= stat[0] and stat[1] <= span[1]) or span[1] - span[0] > 300:
                counts["quote_span_from_line"] += 1
                line_start = text.rfind("\n", 0, stat[0]) + 1
                line_end = text.find("\n", stat[1])
                span = [line_start, len(text) if line_end < 0 else line_end]
            quote_span, span = span, list(stat)
        else:
            quote_span = span
        vote = {f: row.get(f) for f in MAJORITY_FIELDS if f not in CODER_NEVER_VOTES}
        vote["test_type"] = normalise_test_type(vote["test_type"])
        vote["p_operator"] = normalise_operator(vote["p_operator"])
        vote["span"] = span
        vote["quote_span"] = quote_span
        vote["row"] = row.get("row")
        counts["in_scope_located" if span is not None else "in_scope_unlocated"] += 1
        votes.append(vote)
    return votes, counts


# ------------------------------------------------------------------ vote --

def vote_field(field: str, members: List[Tuple[str, dict]]) -> Tuple[object, int, bool]:
    """(value, votes for it, conflict) for one field among the voters of a cluster."""
    voting = [(v, r) for v, r in members if not (v == CODERS and field in CODER_NEVER_VOTES)]
    keys = [field_key(field, r.get(field)) for _, r in voting]
    if len(voting) == 1:
        return voting[0][1].get(field), 1, False
    ranked = Counter(keys).most_common()
    best_key, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if best >= 2 and best > runner_up:
        # Members are in voter order, so a rater's written form wins over
        # the coders' typed form whenever a rater holds the value.
        rep = next(r for (_, r), k in zip(voting, keys) if k == best_key)
        return rep.get(field), best, False
    return None, best, True


def choose_span(raters: List[dict]) -> Tuple[Optional[list], Optional[str]]:
    spans = [tuple(r["span"]) if r.get("span") else None for r in raters]
    shared = Counter(s for s in spans if s is not None).most_common(1)
    if shared and shared[0][1] >= 2:
        chosen = shared[0][0]
    else:
        # The first rater whose quote aligned; a rater with no span has no
        # span to give, and the protocol names no other source for one.
        chosen = next((s for s in spans if s is not None), None)
    owner = next((r for r, s in zip(raters, spans) if s == chosen), raters[0])
    return (list(chosen) if chosen else None), owner.get("quote")


def choose_confidence(raters: List[dict]) -> str:
    value, count = Counter(r.get("confidence") for r in raters).most_common(1)[0]
    return value if count * 2 > len(raters) else "low"


def coder_vote_kind(coder: Optional[dict]) -> Optional[str]:
    if coder is None:
        return None
    return "span" if coder.get("span") is not None else "value"


def _cluster_position(members: List[Tuple[str, dict]]) -> tuple:
    spans = [r["span"] for _, r in members if r.get("span")]
    if not spans:
        return (1, 0, 0)
    first = min(spans)
    return (0, first[0], first[1])


def four_vote(voters: Dict[str, List[dict]]) -> dict:
    """The four-vote rule for one passage.

    `voters` maps each name in VOTERS to its results; a coder result is the
    output of `coder_votes`. Returns kept results, disputes with their
    voters (not yet blind), and the dropped coder rows.
    """
    named = {v: voters.get(v, []) for v in VOTERS}
    clusters = [[(v, named[v][i]) for v, i in group] for group in _cluster(named)]
    # Reading order, so that the dispute numbering says nothing about voters.
    clusters.sort(key=_cluster_position)

    kept, disputes, dropped = [], [], []
    for members in clusters:
        names = [v for v, _ in members]
        coder = next((r for v, r in members if v == CODERS), None)

        if len(members) == 1:
            if coder is not None and coder.get("span") is None:
                dropped.append(coder.get("row"))
                continue
            disputes.append({"kind": "singleton", "members": members})
            continue

        raters = [r for v, r in members if v != CODERS]
        record = {}
        votes = {}
        conflicts = []
        for field in MAJORITY_FIELDS:
            value, count, conflict = vote_field(field, members)
            record[field] = value
            votes[field] = count
            if conflict:
                conflicts.append(field)
        span, quote = choose_span(raters)
        record["quote"] = quote
        record["confidence"] = choose_confidence(raters)
        record = {k: record[k] for k in CANDIDATE_KEYS}
        record["span"] = span

        n_field_voters = {f: sum(1 for v in names if not (v == CODERS and f in CODER_NEVER_VOTES))
                          for f in MAJORITY_FIELDS}
        unanimous = (not conflicts and set(RATER_VOTERS) <= set(names)
                     and all(votes[f] == n_field_voters[f] for f in MAJORITY_FIELDS))
        record["voters"] = names
        record["coder_vote"] = coder_vote_kind(coder)
        record["coder_row"] = coder.get("row") if coder is not None else None
        record["votes"] = votes
        # A result with a field in conflict is settled by the merge, which
        # sets the tier to adjudicated.
        record["tier"] = "unanimous" if unanimous else "majority"
        record["pending"] = {}
        kept.append(record)
        for field in conflicts:
            disputes.append({"kind": "field_conflict", "field": field,
                             "members": members, "result": len(kept) - 1})
    return {"kept": kept, "disputes": disputes, "dropped": dropped}


# ------------------------------------------------------------ candidates --

def written_form(value, quote: Optional[str]):
    target = as_number(value)
    if target is None or not quote:
        return value
    for m in _NUMBER_RE.finditer(quote):
        if as_number(m.group()) == target:
            return m.group()
    return value


def coder_candidate(coder: dict, text: str, cluster_span: Optional[list],
                    cluster_quote: Optional[str]) -> dict:
    if coder.get("quote_span") is not None:
        quote = text[coder["quote_span"][0]:coder["quote_span"][1]]
    elif cluster_span is not None:
        # An unlocated row has no place of its own; it voted for the result
        # of the cluster, so it shows the text of that result.
        quote = text[cluster_span[0]:cluster_span[1]]
    else:
        quote = cluster_quote
    cand = {k: None for k in CANDIDATE_KEYS}
    cand["quote"] = quote
    for f in ("statistic", "df1", "df2", "p_value"):
        cand[f] = written_form(coder.get(f), quote)
    for f in ("test_type", "p_operator", "damaged"):
        cand[f] = coder.get(f)
    cand["n"] = None
    cand["confidence"] = "high"
    return cand


def candidate(voter: str, res: dict, text: str, kept: Optional[dict]) -> dict:
    if voter == CODERS:
        return coder_candidate(res, text, kept and kept.get("span"), kept and kept.get("quote"))
    return {k: res.get(k) for k in CANDIDATE_KEYS}


# ------------------------------------------------------------------- run --

def batch_files(directory: Path) -> List[Tuple[int, Path]]:
    found = []
    for p in Path(directory).glob("batch_*.json"):
        m = _BATCH_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def load_inventory(path: Path) -> Dict[str, dict]:
    inventory = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                inventory[rec["passage_id"]] = rec
    return inventory


def load_raters(batches: List[Tuple[int, Path]], raters_dir: Path) -> Tuple[dict, dict, list]:
    """Every rater output, validated as 03_collect.py validates it."""
    results: Dict[str, Dict[str, list]] = {v: {} for v in RATER_VOTERS}
    digests: Dict[str, List[str]] = {v: [] for v in RATER_VOTERS}
    errors = []
    for _, bpath in batches:
        batch = read_json(bpath)
        texts = {b["window_id"]: b["text"] for b in batch}
        for rater, voter in zip(RATERS, RATER_VOTERS):
            opath = Path(raters_dir) / rater / bpath.name
            if not opath.exists():
                errors.append(f"{rater}/{bpath.name}: missing")
                continue
            try:
                output = read_json(opath)
            except json.JSONDecodeError as e:
                errors.append(f"{rater}/{bpath.name}: JSON parse error: {e}")
                continue
            problems = validate_batch(batch, output)
            if problems:
                errors.append(f"{rater}/{bpath.name}: {problems[0]}")
                continue
            digests[voter].append(hashlib.sha256(opath.read_bytes()).hexdigest())
            for rec in output:
                wid = rec["window_id"]
                results[voter][wid] = [rater_result(r, texts[wid]) for r in rec["results"]]
    combined = {v: hashlib.sha256("".join(d).encode()).hexdigest() for v, d in digests.items()}
    return results, combined, errors


def run(batches_dir: Path, raters_dir: Path, inventory_path: Path, out_path: Path,
        disputes_dir: Path, seed: int) -> dict:
    batches = batch_files(batches_dir)
    if not batches:
        raise SystemExit(f"no batch files in {batches_dir}")
    rater_results, rater_digests, errors = load_raters(batches, raters_dir)
    if errors:
        head = "\n  ".join(errors[:20])
        raise SystemExit(f"{len(errors)} rater files missing or invalid; consensus needs all "
                         f"{3 * len(batches)}:\n  {head}")
    inventory = load_inventory(inventory_path)

    rng = random.Random(seed)
    passages_out = []
    blind = []
    coder_counts: Counter = Counter()
    counts: Counter = Counter()
    rater_counts: Counter = Counter()
    conflict_fields: Counter = Counter()
    singleton_voters: Counter = Counter()
    tier_counts: Counter = Counter()
    coder_by_tier: Dict[str, Counter] = {}

    for _, bpath in batches:
        for passage in read_json(bpath):
            wid, text = passage["window_id"], passage["text"]
            rec = inventory.get(wid)
            if rec is None:
                raise SystemExit(f"{wid}: not in the inventory")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != rec["text_sha256"]:
                raise SystemExit(f"{wid}: batch text differs from the inventory text; "
                                 "the coder spans would point at the wrong characters")
            coders, c = coder_votes(rec.get("coder_results", []), text)
            coder_counts.update(c)
            voters = {v: rater_results[v].get(wid, []) for v in RATER_VOTERS}
            voters[CODERS] = coders
            for v in RATER_VOTERS:
                rater_counts[f"{v}:results"] += len(voters[v])
                rater_counts[f"{v}:unaligned"] += sum(1 for r in voters[v] if r["span"] is None)

            outcome = four_vote(voters)
            kept = outcome["kept"]
            coder_counts["dropped_unlocated"] += len(outcome["dropped"])
            disputes_out = []
            for k, d in enumerate(outcome["disputes"]):
                did = f"{wid}-{k}"
                target = kept[d["result"]] if d["kind"] == "field_conflict" else None
                pairs = [(v, candidate(v, r, text, target)) for v, r in d["members"]]
                rng.shuffle(pairs)
                item = {"dispute_id": did, "window_id": wid, "text": text, "kind": d["kind"]}
                entry = {"dispute_id": did, "kind": d["kind"]}
                if d["kind"] == "field_conflict":
                    item["field"] = d["field"]
                    entry["field"] = d["field"]
                    entry["result"] = d["result"]
                    target["pending"][d["field"]] = did
                    conflict_fields[d["field"]] += 1
                else:
                    singleton_voters[pairs[0][0]] += 1
                    if pairs[0][0] == CODERS:
                        entry["coder_row"] = d["members"][0][1].get("row")
                item["candidates"] = [c for _, c in pairs]
                entry["voters"] = [v for v, _ in pairs]
                blind.append(item)
                disputes_out.append(entry)

            for r in kept:
                tier_counts[r["tier"]] += 1
                by = coder_by_tier.setdefault(r["tier"], Counter())
                by[r["coder_vote"] or "none"] += 1
            counts["passages"] += 1
            counts["kept"] += len(kept)
            counts["kept_with_field_conflict"] += sum(1 for r in kept if r["pending"])
            passages_out.append({"window_id": wid, "results": kept, "disputes": disputes_out,
                                 "dropped_coder_rows": outcome["dropped"]})

    n_batches = (len(blind) + BATCH_SIZE - 1) // BATCH_SIZE
    disputes_dir = Path(disputes_dir)
    stale = [p.name for i, p in batch_files(disputes_dir) if i >= n_batches]
    if stale:
        raise SystemExit(f"{disputes_dir} holds {len(stale)} dispute batches from an earlier "
                         f"run that this run would not overwrite, e.g. {stale[0]}; clear it first")

    def share(by: Counter) -> Optional[float]:
        total = sum(by.values())
        return round((by["span"] + by["value"]) / total, 4) if total else None

    all_by = sum(coder_by_tier.values(), Counter())
    summary = {
        "label_tier": "bronze",
        "passages": counts["passages"],
        "rater_results": dict(sorted(rater_counts.items())),
        "coder_rows": {k: coder_counts[k] for k in
                       ("rows", "out_of_scope", "in_scope_located", "in_scope_unlocated",
                        "span_clamped", "quote_span_from_line", "dropped_unlocated")},
        "kept": counts["kept"],
        "kept_by_tier": dict(sorted(tier_counts.items())),
        "kept_with_field_conflict": counts["kept_with_field_conflict"],
        "coder_vote": {
            "total": {"span": all_by["span"], "value": all_by["value"], "none": all_by["none"],
                      "share_with_coder_vote": share(all_by)},
            "by_tier": {t: {"span": by["span"], "value": by["value"], "none": by["none"],
                            "share_with_coder_vote": share(by)}
                        for t, by in sorted(coder_by_tier.items())},
        },
        "singletons_by_voter": {v: singleton_voters[v] for v in VOTERS},
        "field_conflicts_by_field": {f: conflict_fields[f] for f in MAJORITY_FIELDS},
        "disputes": len(blind),
        "dispute_batches": n_batches,
    }
    out = {
        "protocol": "docs/PROTOCOL.md version 3, four-vote consensus",
        "label_tier": "bronze",
        "seed": seed,
        "inputs": {
            "guideline_sha256": sha256_file(GUIDELINE_PATH),
            "inventory_sha256": sha256_file(inventory_path),
            "batches": len(batches),
            "rater_outputs_sha256": rater_digests,
        },
        "summary": summary,
        "passages": passages_out,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps(out), encoding="utf-8")
    disputes_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_batches):
        (disputes_dir / f"batch_{i}.json").write_text(
            dumps(blind[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", default=str(ROOT / "dataset/passages/batches"))
    ap.add_argument("--raters", default=str(ROOT / "data/annotation/passages"))
    ap.add_argument("--inventory", default=str(ROOT / "dataset/passages/inventory.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "dataset/passages/consensus.json"))
    ap.add_argument("--disputes-dir", default=str(ROOT / "dataset/passages/disputes"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    summary = run(Path(args.batches), Path(args.raters), Path(args.inventory),
                  Path(args.out), Path(args.disputes_dir), args.seed)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
