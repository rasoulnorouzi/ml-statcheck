"""Draw a stratified sample of passages from the gold-train inventory.

The sampler reads the passage inventory and draws from four strata without overlap:
  1. gold: passages with in-scope coder rows
  2. hard: passages of papers without results but cascade fires
  3. damaged: passages of damaged papers with no gold, cascade_fires first
  4. clean: passages with no gold and cascade does not fire, round-robin by journal

Output: a batch ready for annotation (`{"window_id", "text"}`), a key with metadata
no rater should see, and a summary of the draw.

Usage: python pipeline/18_sample_passages.py <clean_text_dir> \\
    --inventory dataset/passages/inventory.jsonl \\
    --inventory-summary dataset/passages/inventory_summary.json \\
    --split dataset/gold/split.json \\
    --dev dataset/gold/dev.json \\
    --seed 0 \\
    --cap-gold 900 --cap-hard 400 --cap-damaged 200 --cap-clean 300 \\
    --max-chars 1500 \\
    --out dataset/passages/sample.json \\
    --key dataset/passages/key.json \\
    --summary dataset/passages/sample_summary.json
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statcheck_ml.gold import assert_not_gold_test, load_split


def load_script(name, file):
    """Load a pipeline script as a module by path."""
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "pipeline" / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


passage_inventory_module = load_script("passage_inventory", "17_passage_inventory.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clean_text_dir")
    ap.add_argument("--inventory", default=str(ROOT / "dataset/passages/inventory.jsonl"))
    ap.add_argument("--inventory-summary", default=str(ROOT / "dataset/passages/inventory_summary.json"))
    ap.add_argument("--split", default=str(ROOT / "dataset/gold/split.json"))
    ap.add_argument("--dev", default=str(ROOT / "dataset/gold/dev.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-gold", type=int, default=900)
    ap.add_argument("--cap-hard", type=int, default=400)
    ap.add_argument("--cap-damaged", type=int, default=200)
    ap.add_argument("--cap-clean", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--out", default=str(ROOT / "dataset/passages/sample.json"))
    ap.add_argument("--key", default=str(ROOT / "dataset/passages/key.json"))
    ap.add_argument("--summary", default=str(ROOT / "dataset/passages/sample_summary.json"))
    args = ap.parse_args()

    # Gate: check inventory summary
    with open(args.inventory_summary) as f:
        inv_summary = json.load(f)

    cascade_fires_count = inv_summary.get("passages_by_flag", {}).get("cascade_fires", 0)
    if cascade_fires_count == 0:
        raise SystemExit("Inventory shows cascade_fires = 0; cannot sample.")

    span_check_failures = inv_summary.get("coder_rows", {}).get("span_check_failures", 0)
    if span_check_failures > 5:
        raise SystemExit(f"Inventory shows span_check_failures = {span_check_failures} > 5; inventory is broken.")

    # Load inventory records
    inventory = []
    with open(args.inventory) as f:
        for line in f:
            if line.strip():
                inventory.append(json.loads(line))

    # Load split and dev
    split_data = load_split(args.split)
    with open(args.dev) as f:
        dev_data = json.load(f)

    dev_set = set(dev_data["dev"])
    gold_train_set = set(split_data["gold_train"])

    # Assign each passage to a stratum
    gold_passages = []
    hard_passages = []
    damaged_cascade_passages = []
    damaged_no_cascade_passages = []
    clean_passages = []

    for rec in inventory:
        doc = rec["doc"]

        # Only process gold-train papers not in dev
        if doc not in gold_train_set or doc in dev_set:
            continue

        flags = rec["flags"]

        # Stratum assignment in order
        if flags["has_gold_in_scope"]:
            gold_passages.append(rec)
        elif not rec["paper_has_results"] and flags["cascade_fires"]:
            hard_passages.append(rec)
        elif rec["paper_damaged"] and not flags["has_gold"]:
            if flags["cascade_fires"]:
                damaged_cascade_passages.append(rec)
            else:
                damaged_no_cascade_passages.append(rec)
        elif not flags["has_gold"] and not flags["cascade_fires"]:
            clean_passages.append(rec)

    # Get docs for assert_not_gold_test
    docs_to_check = [rec["doc"] for rec in
                     gold_passages + hard_passages + damaged_cascade_passages +
                     damaged_no_cascade_passages + clean_passages]
    # Create minimal dicts for assert_not_gold_test
    docs_for_check = [{"doc": d} for d in docs_to_check]
    assert_not_gold_test(docs_for_check, split_data)

    # Assert no passages from dev papers
    for rec in inventory:
        if rec["doc"] in dev_set:
            raise AssertionError(f"Passage from dev paper found: {rec['doc']}")

    # Sample from each stratum
    rng = random.Random(args.seed)

    # Gold: sort by passage_id, shuffle, take up to cap
    gold_passages.sort(key=lambda x: x["passage_id"])
    rng_gold = random.Random(args.seed)
    rng_gold.shuffle(gold_passages)
    drawn_gold = gold_passages[:args.cap_gold]

    # Hard: sort by passage_id, shuffle, take up to cap
    hard_passages.sort(key=lambda x: x["passage_id"])
    rng_hard = random.Random(args.seed)
    rng_hard.shuffle(hard_passages)
    drawn_hard = hard_passages[:args.cap_hard]

    # Damaged: cascade_fires first, then rest; within each, sort and shuffle, take up to cap
    damaged_cascade_passages.sort(key=lambda x: x["passage_id"])
    damaged_no_cascade_passages.sort(key=lambda x: x["passage_id"])
    rng_dam_c = random.Random(args.seed)
    rng_dam_c.shuffle(damaged_cascade_passages)
    rng_dam_n = random.Random(args.seed)
    rng_dam_n.shuffle(damaged_no_cascade_passages)
    drawn_damaged = (damaged_cascade_passages + damaged_no_cascade_passages)[:args.cap_damaged]

    # Clean: round-robin by journal
    # Group by journal
    clean_by_journal = collections.defaultdict(list)
    for rec in clean_passages:
        clean_by_journal[rec["journal"]].append(rec)

    # Sort journals, and within each journal sort passages by passage_id and shuffle
    journals_sorted = sorted(clean_by_journal.keys())
    shuffled_by_journal = {}
    for journal in journals_sorted:
        passages_in_journal = clean_by_journal[journal]
        passages_in_journal.sort(key=lambda x: x["passage_id"])
        # Shuffle with seed
        rng_clean = random.Random(args.seed)
        rng_clean.shuffle(passages_in_journal)
        shuffled_by_journal[journal] = passages_in_journal

    # Round-robin: iterate in rounds, taking from each journal in order
    drawn_clean = []
    round_idx = 0
    while len(drawn_clean) < args.cap_clean:
        added_any = False
        for journal in journals_sorted:
            if len(drawn_clean) >= args.cap_clean:
                break
            if round_idx < len(shuffled_by_journal[journal]):
                drawn_clean.append(shuffled_by_journal[journal][round_idx])
                added_any = True
        if not added_any:
            break
        round_idx += 1

    # Combine all drawn passages
    all_drawn = drawn_gold + drawn_hard + drawn_damaged + drawn_clean

    # Map to stratum name
    stratum_map = {}
    for rec in drawn_gold:
        stratum_map[rec["passage_id"]] = "gold"
    for rec in drawn_hard:
        stratum_map[rec["passage_id"]] = "hard"
    for rec in drawn_damaged:
        stratum_map[rec["passage_id"]] = "damaged"
    for rec in drawn_clean:
        stratum_map[rec["passage_id"]] = "clean"

    # Rebuild passage texts and verify sha256
    for rec in all_drawn:
        doc = rec["doc"]
        text_path = os.path.join(args.clean_text_dir, doc)

        # Load the paper's clean text
        with open(text_path, encoding="utf-8") as f:
            raw_text = f.read()

        # Normalize and repair (same as inventory script)
        from statcheck_ml.normalize import normalize
        from statcheck_ml.repair_validated import repair_validated

        text, _ = normalize(raw_text)
        text, _ = repair_validated(text)

        # Rebuild pieces with paper_pieces
        pieces = passage_inventory_module.paper_pieces(text, args.max_chars)

        # Find the matching piece
        matching = [p for p in pieces
                    if p["start_line"] == rec["start_line"]
                    and p["end_line"] == rec["end_line"]
                    and p["piece"] == rec["piece"]]

        if len(matching) != 1:
            raise ValueError(f"Could not find piece for {rec['passage_id']}: "
                           f"found {len(matching)} matches")

        piece = matching[0]
        piece_text = piece["text"]

        # Verify sha256
        piece_hash = hashlib.sha256(piece_text.encode("utf-8")).hexdigest()
        if piece_hash != rec["text_sha256"]:
            raise ValueError(f"SHA256 mismatch for {rec['passage_id']}: "
                           f"expected {rec['text_sha256']}, got {piece_hash}")

        # Store the text for later
        rec["_piece_text"] = piece_text

    # Shuffle all drawn passages with seed (so batch mixes strata)
    rng_final = random.Random(args.seed)
    rng_final.shuffle(all_drawn)

    # Build sample and key
    sample = []
    key = []
    for rec in all_drawn:
        window_id = rec["passage_id"]
        piece_text = rec.pop("_piece_text")

        sample.append({
            "window_id": window_id,
            "text": piece_text,
        })

        key.append({
            "window_id": window_id,
            "stratum": stratum_map[window_id],
            "doc": rec["doc"],
            "journal": rec["journal"],
            "start_line": rec["start_line"],
            "end_line": rec["end_line"],
            "piece": rec["piece"],
            "pieces": rec["pieces"],
            "n_chars": rec["n_chars"],
            "text_sha256": rec["text_sha256"],
            "flags": rec["flags"],
            "n_coder_results": len(rec.get("coder_results", [])),
            "cascade": rec["cascade"],
        })

    # Write outputs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.key).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(sample, f, indent=1, sort_keys=True)
        f.write("\n")

    with open(args.key, "w") as f:
        json.dump(key, f, indent=1, sort_keys=True)
        f.write("\n")

    # Build summary
    summary = {
        "caps": {
            "gold": args.cap_gold,
            "hard": args.cap_hard,
            "damaged": args.cap_damaged,
            "clean": args.cap_clean,
        },
        "seed": args.seed,
        "drawn": {
            "gold": len(drawn_gold),
            "hard": len(drawn_hard),
            "damaged": len(drawn_damaged),
            "clean": len(drawn_clean),
            "total": len(all_drawn),
        },
        "pools": {
            "gold": len(gold_passages),
            "hard": len(hard_passages),
            "damaged": len(damaged_cascade_passages) + len(damaged_no_cascade_passages),
            "clean": len(clean_passages),
        },
        "caps_hit": {
            "gold": len(drawn_gold) == args.cap_gold and len(drawn_gold) < len(gold_passages),
            "hard": len(drawn_hard) == args.cap_hard and len(drawn_hard) < len(hard_passages),
            "damaged": len(drawn_damaged) == args.cap_damaged and len(drawn_damaged) < len(damaged_cascade_passages) + len(damaged_no_cascade_passages),
            "clean": len(drawn_clean) == args.cap_clean and len(drawn_clean) < len(clean_passages),
        },
        "papers": len(set(rec["doc"] for rec in all_drawn)),
        "journals": len(set(rec["journal"] for rec in all_drawn)),
        "passages_per_journal": dict(sorted(collections.Counter(rec["journal"] for rec in all_drawn).items())),
    }

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
        f.write("\n")

    # Print summary to stdout
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
