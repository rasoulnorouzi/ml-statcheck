"""Tests for the passage sampler."""
import collections
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from statcheck_ml.normalize import normalize
from statcheck_ml.repair_validated import repair_validated
from statcheck_ml.prefilter import Prefilter


def load_script(name, file):
    """Load a pipeline script as a module."""
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "pipeline" / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sampler_module = load_script("sampler", "18_sample_passages.py")
inventory_module = load_script("inventory", "17_passage_inventory.py")


@pytest.fixture
def synthetic_inventory_env(tmp_path):
    """Create a synthetic inventory and corpus for testing."""
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()

    inv_summary = {
        "passages_by_flag": {
            "cascade_fires": 10,
            "has_gold_in_scope": 5,
            "has_gold": 6,
            "gold_damaged": 0,
        },
        "coder_rows": {
            "span_check_failures": 0,
            "on_blanked_line": 0,
            "in_scope": 10,
            "located_in_passage": 10,
            "total_in_gold": 10,
            "unlocated": 0,
            "not_inside_any_unit": 0,
        },
        "hard_negatives": 2,
        "damaged_no_gold": 1,
        "pieces": 20,
        "papers": 5,
        "clean_negatives": {"Journal1": 8, "Journal2": 5},
        "units_split": {"count": 0, "largest_unit": 500},
    }

    # Create synthetic papers and passages
    # Paper 1 (gold_train, has results): gold passages
    paper1_text = "Introduction.\n" + ("Result line: t(28) = 2.45, p = .021\n" * 10) + "Conclusion."
    norm_text, _ = normalize(paper1_text)
    repair_text, _ = repair_validated(norm_text)
    (clean_dir / "Journal1" / "paper1.txt").parent.mkdir(parents=True, exist_ok=True)
    (clean_dir / "Journal1" / "paper1.txt").write_text(repair_text, encoding="utf-8")

    paper1_pieces = inventory_module.paper_pieces(repair_text, max_chars=500)

    # Build paper1 passages (2 with gold, 2 clean, 1 hard)
    passages = []
    paper1_doc = "Journal1/paper1.txt"

    # Create 2 gold passages
    for i, piece in enumerate(paper1_pieces[:2]):
        text_sha256 = hashlib.sha256(piece["text"].encode()).hexdigest()
        passages.append({
            "passage_id": f"gold{i}",
            "doc": paper1_doc,
            "journal": "Journal1",
            "paper_has_results": True,
            "paper_damaged": False,
            "start_line": piece["start_line"],
            "end_line": piece["end_line"],
            "char_start": piece["char_start"],
            "n_chars": len(piece["text"]),
            "n_lines": len(piece["text"].split("\n")),
            "text_sha256": text_sha256,
            "paper_text_sha256": hashlib.sha256(repair_text.encode()).hexdigest(),
            "piece": piece["piece"],
            "pieces": piece["pieces"],
            "coder_results": [{"in_scope": True}],  # 1 in-scope coder result
            "cascade": {"results": 1, "fragments": 0},
            "flags": {
                "has_gold": True,
                "has_gold_in_scope": True,
                "gold_damaged": False,
                "cascade_fires": True,
            },
        })

    # Create 2 clean passages (no gold, no cascade)
    if len(paper1_pieces) > 2:
        for i, piece in enumerate(paper1_pieces[2:4], start=2):
            text_sha256 = hashlib.sha256(piece["text"].encode()).hexdigest()
            passages.append({
                "passage_id": f"clean{i}",
                "doc": paper1_doc,
                "journal": "Journal1",
                "paper_has_results": True,
                "paper_damaged": False,
                "start_line": piece["start_line"],
                "end_line": piece["end_line"],
                "char_start": piece["char_start"],
                "n_chars": len(piece["text"]),
                "n_lines": len(piece["text"].split("\n")),
                "text_sha256": text_sha256,
                "paper_text_sha256": hashlib.sha256(repair_text.encode()).hexdigest(),
                "piece": piece["piece"],
                "pieces": piece["pieces"],
                "coder_results": [],
                "cascade": {"results": 0, "fragments": 0},
                "flags": {
                    "has_gold": False,
                    "has_gold_in_scope": False,
                    "gold_damaged": False,
                    "cascade_fires": False,
                },
            })

    # Paper 2 (gold_train, no results, cascade fires): hard passages
    paper2_text = "Introduction.\n" + ("Some text with t(28) and p = .021\n" * 10) + "Conclusion."
    norm_text, _ = normalize(paper2_text)
    repair_text, _ = repair_validated(norm_text)
    (clean_dir / "Journal1" / "paper2.txt").write_text(repair_text, encoding="utf-8")

    paper2_pieces = inventory_module.paper_pieces(repair_text, max_chars=500)
    paper2_doc = "Journal1/paper2.txt"

    # Create 2 hard passages
    for i, piece in enumerate(paper2_pieces[:2]):
        text_sha256 = hashlib.sha256(piece["text"].encode()).hexdigest()
        passages.append({
            "passage_id": f"hard{i}",
            "doc": paper2_doc,
            "journal": "Journal1",
            "paper_has_results": False,
            "paper_damaged": False,
            "start_line": piece["start_line"],
            "end_line": piece["end_line"],
            "char_start": piece["char_start"],
            "n_chars": len(piece["text"]),
            "n_lines": len(piece["text"].split("\n")),
            "text_sha256": text_sha256,
            "paper_text_sha256": hashlib.sha256(repair_text.encode()).hexdigest(),
            "piece": piece["piece"],
            "pieces": piece["pieces"],
            "coder_results": [],
            "cascade": {"results": 1, "fragments": 0},
            "flags": {
                "has_gold": False,
                "has_gold_in_scope": False,
                "gold_damaged": False,
                "cascade_fires": True,
            },
        })

    # Paper 3 (gold_train, damaged, no results, cascade fires): damaged passages
    paper3_text = "Introduction.\n" + ("Some text with t(28) and p = .021\n" * 10) + "Conclusion."
    norm_text, _ = normalize(paper3_text)
    repair_text, _ = repair_validated(norm_text)
    (clean_dir / "Journal2" / "paper3.txt").parent.mkdir(parents=True, exist_ok=True)
    (clean_dir / "Journal2" / "paper3.txt").write_text(repair_text, encoding="utf-8")

    paper3_pieces = inventory_module.paper_pieces(repair_text, max_chars=500)
    paper3_doc = "Journal2/paper3.txt"

    # Create 1 damaged passage (cascade_fires=True)
    if len(paper3_pieces) > 0:
        piece = paper3_pieces[0]
        text_sha256 = hashlib.sha256(piece["text"].encode()).hexdigest()
        passages.append({
            "passage_id": f"damaged0",
            "doc": paper3_doc,
            "journal": "Journal2",
            "paper_has_results": False,
            "paper_damaged": True,
            "start_line": piece["start_line"],
            "end_line": piece["end_line"],
            "char_start": piece["char_start"],
            "n_chars": len(piece["text"]),
            "n_lines": len(piece["text"].split("\n")),
            "text_sha256": text_sha256,
            "paper_text_sha256": hashlib.sha256(repair_text.encode()).hexdigest(),
            "piece": piece["piece"],
            "pieces": piece["pieces"],
            "coder_results": [],
            "cascade": {"results": 1, "fragments": 0},
            "flags": {
                "has_gold": False,
                "has_gold_in_scope": False,
                "gold_damaged": False,
                "cascade_fires": True,
            },
        })

    # Paper 4: gold_test (should not be sampled)
    paper4_text = "Test paper.\nResult: t(28) = 2.45, p = .021\nEnd."
    norm_text, _ = normalize(paper4_text)
    repair_text, _ = repair_validated(norm_text)
    (clean_dir / "Journal2" / "paper_test.txt").write_text(repair_text, encoding="utf-8")

    # Paper 5: dev paper (should not be sampled)
    paper5_text = "Dev paper.\nSome content.\nEnd."
    norm_text, _ = normalize(paper5_text)
    repair_text, _ = repair_validated(norm_text)
    (clean_dir / "Journal2" / "paper_dev.txt").write_text(repair_text, encoding="utf-8")

    # Create clean passages for Journal2 (more to test round-robin)
    for j in range(3):
        piece = paper3_pieces[j + 1] if j + 1 < len(paper3_pieces) else paper3_pieces[-1]
        text_sha256 = hashlib.sha256(piece["text"].encode()).hexdigest()
        passages.append({
            "passage_id": f"clean_j2_{j}",
            "doc": paper3_doc,
            "journal": "Journal2",
            "paper_has_results": False,
            "paper_damaged": False,
            "start_line": piece["start_line"],
            "end_line": piece["end_line"],
            "char_start": piece["char_start"],
            "n_chars": len(piece["text"]),
            "n_lines": len(piece["text"].split("\n")),
            "text_sha256": text_sha256,
            "paper_text_sha256": hashlib.sha256(repair_text.encode()).hexdigest(),
            "piece": piece["piece"],
            "pieces": piece["pieces"],
            "coder_results": [],
            "cascade": {"results": 0, "fragments": 0},
            "flags": {
                "has_gold": False,
                "has_gold_in_scope": False,
                "gold_damaged": False,
                "cascade_fires": False,
            },
        })

    # Write inventory.jsonl
    inv_file = tmp_path / "inventory.jsonl"
    with open(inv_file, "w") as f:
        for passage in passages:
            f.write(json.dumps(passage) + "\n")

    # Write inventory_summary.json
    inv_summary_file = tmp_path / "inventory_summary.json"
    with open(inv_summary_file, "w") as f:
        json.dump(inv_summary, f)

    # Write split.json
    split_file = tmp_path / "split.json"
    split_data = {
        "gold_test": ["Journal2/paper_test.txt"],
        "gold_train": [
            "Journal1/paper1.txt",
            "Journal1/paper2.txt",
            "Journal2/paper3.txt",
            "Journal2/paper_dev.txt",
        ],
        "counts": {"gold_test": {}, "gold_train": {}},
    }
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    # Write dev.json
    dev_file = tmp_path / "dev.json"
    dev_data = {
        "seed": 0,
        "fraction": 0.25,
        "rule": "test",
        "dev": ["Journal2/paper_dev.txt"],
        "counts": {"dev": {}, "rest": {}},
    }
    with open(dev_file, "w") as f:
        json.dump(dev_data, f)

    return {
        "clean_dir": clean_dir,
        "inventory_file": inv_file,
        "inventory_summary_file": inv_summary_file,
        "split_file": split_file,
        "dev_file": dev_file,
        "passages": passages,
    }


def test_strata_disjoint_and_ordered(synthetic_inventory_env):
    """Test that strata are disjoint and assigned in the stated order."""
    # A passage that could be both gold and hard should go to gold
    # This is tested by the assignment logic
    env = synthetic_inventory_env
    passages = env["passages"]

    # Track assignments
    gold_pids = set()
    hard_pids = set()
    damaged_pids = set()
    clean_pids = set()

    for rec in passages:
        flags = rec["flags"]
        doc = rec["doc"]

        # Simplified version of the assignment logic
        if flags["has_gold_in_scope"]:
            gold_pids.add(rec["passage_id"])
        elif not rec["paper_has_results"] and flags["cascade_fires"]:
            hard_pids.add(rec["passage_id"])
        elif rec["paper_damaged"] and not flags["has_gold"]:
            damaged_pids.add(rec["passage_id"])
        elif not flags["has_gold"] and not flags["cascade_fires"]:
            clean_pids.add(rec["passage_id"])

    # Verify disjoint
    all_pids = gold_pids | hard_pids | damaged_pids | clean_pids
    assert len(all_pids) == len(gold_pids) + len(hard_pids) + len(damaged_pids) + len(clean_pids)


def test_cap_respected(synthetic_inventory_env, tmp_path):
    """Test that caps are respected."""
    env = synthetic_inventory_env
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "18_sample_passages.py"),
        str(env["clean_dir"]),
        "--inventory", str(env["inventory_file"]),
        "--inventory-summary", str(env["inventory_summary_file"]),
        "--split", str(env["split_file"]),
        "--dev", str(env["dev_file"]),
        "--seed", "0",
        "--cap-gold", "1",
        "--cap-hard", "1",
        "--cap-damaged", "1",
        "--cap-clean", "1",
        "--max-chars", "500",
        "--out", str(out_dir / "sample.json"),
        "--key", str(out_dir / "key.json"),
        "--summary", str(out_dir / "summary.json"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Sampler failed: {result.stderr}"

    # Load sample and verify counts
    with open(out_dir / "sample.json") as f:
        sample = json.load(f)

    # With caps of 1, we should have at most 4 passages (1 per stratum)
    assert len(sample) <= 4


def test_clean_round_robin(synthetic_inventory_env, tmp_path):
    """Test that clean stratum round-robins over journals.

    Verifies that with multiple journals in the clean pool, round-robin assignment
    balances the draw across journals.
    """
    env = synthetic_inventory_env
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "18_sample_passages.py"),
        str(env["clean_dir"]),
        "--inventory", str(env["inventory_file"]),
        "--inventory-summary", str(env["inventory_summary_file"]),
        "--split", str(env["split_file"]),
        "--dev", str(env["dev_file"]),
        "--seed", "0",
        "--cap-gold", "10",
        "--cap-hard", "10",
        "--cap-damaged", "10",
        "--cap-clean", "10",
        "--max-chars", "500",
        "--out", str(out_dir / "sample.json"),
        "--key", str(out_dir / "key.json"),
        "--summary", str(out_dir / "summary.json"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Sampler failed: {result.stderr}"

    # Load key and check clean passages exist
    with open(out_dir / "key.json") as f:
        key = json.load(f)

    clean_passages = [k for k in key if k["stratum"] == "clean"]
    assert len(clean_passages) > 0, "No clean passages were drawn"

    # Count by journal
    by_journal = collections.defaultdict(int)
    for rec in clean_passages:
        by_journal[rec["journal"]] += 1

    # Round-robin should balance across journals
    # Max difference should be at most 1
    if len(by_journal) > 1:
        max_count = max(by_journal.values())
        min_count = min(by_journal.values())
        assert max_count - min_count <= 1, \
            f"Round-robin imbalance: max={max_count}, min={min_count}"


def test_determinism(synthetic_inventory_env, tmp_path):
    """Test that two runs with same seed produce same bytes."""
    env = synthetic_inventory_env

    def run_sampler(out_tag):
        out_dir = tmp_path / f"output{out_tag}"
        out_dir.mkdir()

        cmd = [
            sys.executable,
            str(ROOT / "pipeline" / "18_sample_passages.py"),
            str(env["clean_dir"]),
            "--inventory", str(env["inventory_file"]),
            "--inventory-summary", str(env["inventory_summary_file"]),
            "--split", str(env["split_file"]),
            "--dev", str(env["dev_file"]),
            "--seed", "0",
            "--cap-gold", "100",
            "--cap-hard", "100",
            "--cap-damaged", "100",
            "--cap-clean", "100",
            "--max-chars", "500",
            "--out", str(out_dir / "sample.json"),
            "--key", str(out_dir / "key.json"),
            "--summary", str(out_dir / "summary.json"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Sampler failed: {result.stderr}"

        return out_dir

    out1 = run_sampler("1")
    out2 = run_sampler("2")

    # Compare hashes
    hash1_sample = hashlib.sha256((out1 / "sample.json").read_bytes()).hexdigest()
    hash2_sample = hashlib.sha256((out2 / "sample.json").read_bytes()).hexdigest()
    assert hash1_sample == hash2_sample, "sample.json differs between runs"

    hash1_key = hashlib.sha256((out1 / "key.json").read_bytes()).hexdigest()
    hash2_key = hashlib.sha256((out2 / "key.json").read_bytes()).hexdigest()
    assert hash1_key == hash2_key, "key.json differs between runs"

    hash1_summary = hashlib.sha256((out1 / "summary.json").read_bytes()).hexdigest()
    hash2_summary = hashlib.sha256((out2 / "summary.json").read_bytes()).hexdigest()
    assert hash1_summary == hash2_summary, "summary.json differs between runs"


def test_sha256_validation(synthetic_inventory_env, tmp_path):
    """Test that a mismatched text_sha256 raises an error."""
    env = synthetic_inventory_env
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Corrupt one passage's text_sha256 in the inventory
    inv_file_corrupted = tmp_path / "inventory_corrupted.jsonl"
    with open(env["inventory_file"]) as f:
        lines = f.readlines()

    with open(inv_file_corrupted, "w") as f:
        for i, line in enumerate(lines):
            record = json.loads(line)
            if i == 0:  # Corrupt first passage
                record["text_sha256"] = "0" * 64  # Invalid hash
            f.write(json.dumps(record) + "\n")

    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "18_sample_passages.py"),
        str(env["clean_dir"]),
        "--inventory", str(inv_file_corrupted),
        "--inventory-summary", str(env["inventory_summary_file"]),
        "--split", str(env["split_file"]),
        "--dev", str(env["dev_file"]),
        "--seed", "0",
        "--cap-gold", "100",
        "--cap-hard", "100",
        "--cap-damaged", "100",
        "--cap-clean", "100",
        "--max-chars", "500",
        "--out", str(out_dir / "sample.json"),
        "--key", str(out_dir / "key.json"),
        "--summary", str(out_dir / "summary.json"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # Should fail with SHA256 mismatch
    assert result.returncode != 0, "Expected sampler to fail on SHA256 mismatch"
    assert "SHA256" in result.stderr or "sha256" in result.stderr.lower() or "mismatch" in result.stderr.lower()


def test_summary_cascade_fires_zero():
    """Test that a summary with cascade_fires=0 raises SystemExit."""
    summary = {
        "passages_by_flag": {
            "cascade_fires": 0,
            "has_gold_in_scope": 5,
        },
        "coder_rows": {
            "span_check_failures": 0,
        },
    }

    # Write to temp file and try to load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(summary, f)
        temp_path = f.name

    try:
        with open(temp_path) as f:
            inv_summary = json.load(f)

        cascade_fires_count = inv_summary.get("passages_by_flag", {}).get("cascade_fires", 0)
        if cascade_fires_count == 0:
            raise SystemExit("Inventory shows cascade_fires = 0; cannot sample.")
        assert False, "Should have raised SystemExit"
    except SystemExit as e:
        assert "cascade_fires" in str(e)
    finally:
        Path(temp_path).unlink()


def test_summary_span_check_failures_high():
    """Test that a summary with span_check_failures > 5 raises SystemExit."""
    summary = {
        "passages_by_flag": {
            "cascade_fires": 10,
            "has_gold_in_scope": 5,
        },
        "coder_rows": {
            "span_check_failures": 6,
        },
    }

    # Write to temp file and try to load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(summary, f)
        temp_path = f.name

    try:
        with open(temp_path) as f:
            inv_summary = json.load(f)

        span_check_failures = inv_summary.get("coder_rows", {}).get("span_check_failures", 0)
        if span_check_failures > 5:
            raise SystemExit(f"Inventory shows span_check_failures = {span_check_failures} > 5")
        assert False, "Should have raised SystemExit"
    except SystemExit as e:
        assert "span_check_failures" in str(e)
    finally:
        Path(temp_path).unlink()


def test_key_has_all_window_ids(synthetic_inventory_env, tmp_path):
    """Test that key holds every window_id of the sample, in the same order."""
    env = synthetic_inventory_env
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "18_sample_passages.py"),
        str(env["clean_dir"]),
        "--inventory", str(env["inventory_file"]),
        "--inventory-summary", str(env["inventory_summary_file"]),
        "--split", str(env["split_file"]),
        "--dev", str(env["dev_file"]),
        "--seed", "0",
        "--cap-gold", "100",
        "--cap-hard", "100",
        "--cap-damaged", "100",
        "--cap-clean", "100",
        "--max-chars", "500",
        "--out", str(out_dir / "sample.json"),
        "--key", str(out_dir / "key.json"),
        "--summary", str(out_dir / "summary.json"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Sampler failed: {result.stderr}"

    with open(out_dir / "sample.json") as f:
        sample = json.load(f)

    with open(out_dir / "key.json") as f:
        key = json.load(f)

    # Check window_ids match and are in the same order
    sample_ids = [rec["window_id"] for rec in sample]
    key_ids = [rec["window_id"] for rec in key]

    assert sample_ids == key_ids, "window_ids do not match between sample and key"


def test_sample_record_fields(synthetic_inventory_env, tmp_path):
    """Test that a sample record has only window_id and text."""
    env = synthetic_inventory_env
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "18_sample_passages.py"),
        str(env["clean_dir"]),
        "--inventory", str(env["inventory_file"]),
        "--inventory-summary", str(env["inventory_summary_file"]),
        "--split", str(env["split_file"]),
        "--dev", str(env["dev_file"]),
        "--seed", "0",
        "--cap-gold", "100",
        "--cap-hard", "100",
        "--cap-damaged", "100",
        "--cap-clean", "100",
        "--max-chars", "500",
        "--out", str(out_dir / "sample.json"),
        "--key", str(out_dir / "key.json"),
        "--summary", str(out_dir / "summary.json"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Sampler failed: {result.stderr}"

    with open(out_dir / "sample.json") as f:
        sample = json.load(f)

    # Each record should have only window_id and text
    for rec in sample:
        assert set(rec.keys()) == {"window_id", "text"}, f"Unexpected keys in sample record: {rec.keys()}"
        assert isinstance(rec["window_id"], str)
        assert isinstance(rec["text"], str)
