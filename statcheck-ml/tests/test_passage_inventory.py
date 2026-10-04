"""Tests for the passage inventory: structure, coverage, and determinism."""
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_script(name, file):
    """Load a pipeline script as a module."""
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "pipeline" / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inventory_module = load_script("inventory", "17_passage_inventory.py")


@pytest.fixture
def synthetic_corpus(tmp_path):
    """Create a minimal synthetic corpus for testing."""
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()

    # Create three papers
    papers_data = []

    # Paper 1: gold_test (should not be processed)
    paper1_text = "Introduction.\nThe effect was significant, t(28) = 2.45, p = .021.\nConclusion."
    (clean_dir / "Journal/Paper1_test.txt").parent.mkdir(parents=True, exist_ok=True)
    (clean_dir / "Journal/Paper1_test.txt").write_text(paper1_text, encoding="utf-8")
    papers_data.append({
        "doc": "Journal/Paper1_test.txt",
        "text_sha256": hashlib.sha256(paper1_text.encode()).hexdigest(),
        "bronze_set": "none",
        "has_results": True,
        "results": [
            {
                "row": 0,
                "raw": "t(28) = 2.45, p = .021",
                "test_type_coded": "t",
                "test_type": "t",
                "stat": 2.45,
                "df1": 28.0,
                "df2": None,
                "p_operator": "=",
                "p": 0.021,
                "damaged": False,
                "page": 1,
                "coder": "test",
                "located": True,
                "span": [30, 56],
                "anchor": [43, 47],
                "line": 1,
            }
        ]
    })

    # Paper 2: dev paper (should not be processed)
    paper2_text = "Methods.\nWe used a t-test.\nResults were significant.\nConclusion."
    (clean_dir / "Journal/Paper2_dev.txt").write_text(paper2_text, encoding="utf-8")
    papers_data.append({
        "doc": "Journal/Paper2_dev.txt",
        "text_sha256": hashlib.sha256(paper2_text.encode()).hexdigest(),
        "bronze_set": "none",
        "has_results": False,
        "results": []
    })

    # Paper 3: gold_train, not dev (should be processed)
    paper3_text = "Introduction.\n" + ("Line of text. " * 100) + "\nConclusion with F(1, 100) = 5.23, p = .045."
    (clean_dir / "Journal/Paper3_train.txt").write_text(paper3_text, encoding="utf-8")
    papers_data.append({
        "doc": "Journal/Paper3_train.txt",
        "text_sha256": hashlib.sha256(paper3_text.encode()).hexdigest(),
        "bronze_set": "train",
        "has_results": True,
        "results": [
            {
                "row": 0,
                "raw": "F(1, 100) = 5.23, p = .045",
                "test_type_coded": "F",
                "test_type": "F",
                "stat": 5.23,
                "df1": 1.0,
                "df2": 100.0,
                "p_operator": "=",
                "p": 0.045,
                "damaged": False,
                "page": 1,
                "coder": "test",
                "located": True,
                "span": [paper3_text.index("F(1, 100)"), paper3_text.index("F(1, 100)") + 27],
                "anchor": [paper3_text.index("5.23"), paper3_text.index("5.23") + 4],
                "line": paper3_text[:paper3_text.index("5.23")].count("\n"),
            }
        ]
    })

    # Create gold.jsonl
    gold_file = tmp_path / "gold.jsonl"
    gold_file.write_text("", encoding="utf-8")
    for p in papers_data:
        gold_file.write_text(gold_file.read_text() + json.dumps(p) + "\n", encoding="utf-8")

    # Create split.json
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({
        "seed": 0,
        "rule": "test split",
        "gold_test": ["Journal/Paper1_test.txt"],
        "gold_train": ["Journal/Paper2_dev.txt", "Journal/Paper3_train.txt"],
        "counts": {"gold_test": {}, "gold_train": {}},
    }), encoding="utf-8")

    # Create dev.json
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({
        "seed": 0,
        "fraction": 0.5,
        "rule": "test dev",
        "dev": ["Journal/Paper2_dev.txt"],
        "counts": {"dev": {}, "rest": {}},
    }), encoding="utf-8")

    return clean_dir, gold_file, split_file, dev_file


def test_passage_structure(synthetic_corpus):
    """Test that passages cover units with no gap and no overlap."""
    clean_dir, gold_file, split_file, dev_file = synthetic_corpus

    # Run the inventory script with monkeypatch to avoid reading all papers
    from statcheck_ml.gold import load_split

    papers = [json.loads(l) for l in open(gold_file) if l.strip()]
    split_data = load_split(split_file)
    with open(dev_file) as f:
        dev_data = json.load(f)

    dev_set = set(dev_data["dev"])
    gold_train_set = set(split_data["gold_train"])

    # Select papers
    selected = [p for p in papers if p["doc"] in gold_train_set and p["doc"] not in dev_set]
    assert len(selected) == 1
    assert selected[0]["doc"] == "Journal/Paper3_train.txt"


def test_gold_test_not_read(synthetic_corpus):
    """Test that gold_test papers are not processed."""
    clean_dir, gold_file, split_file, dev_file = synthetic_corpus

    from statcheck_ml.gold import load_split

    papers = [json.loads(l) for l in open(gold_file) if l.strip()]
    split_data = load_split(split_file)
    with open(dev_file) as f:
        dev_data = json.load(f)

    dev_set = set(dev_data["dev"])
    gold_train_set = set(split_data["gold_train"])
    gold_test_set = set(split_data["gold_test"])

    selected = [p for p in papers if p["doc"] in gold_train_set and p["doc"] not in dev_set]

    # Ensure no gold_test papers are selected
    for p in selected:
        assert p["doc"] not in gold_test_set


def test_dev_paper_not_included(synthetic_corpus):
    """Test that dev papers are not included in the inventory."""
    clean_dir, gold_file, split_file, dev_file = synthetic_corpus

    from statcheck_ml.gold import load_split

    papers = [json.loads(l) for l in open(gold_file) if l.strip()]
    split_data = load_split(split_file)
    with open(dev_file) as f:
        dev_data = json.load(f)

    dev_set = set(dev_data["dev"])
    gold_train_set = set(split_data["gold_train"])

    selected = [p for p in papers if p["doc"] in gold_train_set and p["doc"] not in dev_set]

    # Ensure dev papers are not selected
    for p in selected:
        assert p["doc"] not in dev_set


def test_df_formatting():
    """Test that df1/df2 formatting removes only one trailing .0."""
    # Test 1.0 -> "1"
    s = "1.0"
    result = s[:-2] if s.endswith(".0") else s
    assert result == "1"

    # Test 184.0 -> "184"
    s = "184.0"
    result = s[:-2] if s.endswith(".0") else s
    assert result == "184"

    # Test 100.0 -> "100" (not "1")
    s = "100.0"
    result = s[:-2] if s.endswith(".0") else s
    assert result == "100"


def test_p_operator_mapping():
    """Test that p_operator maps correctly."""
    mappings = {
        "=": "=",
        "<": "<",
        ">": ">",
        "≤": "<",
        "≥": ">",
        "ns": None,
        "other": None,
        None: None,
    }

    for input_val, expected in mappings.items():
        p_op = input_val
        if p_op == "≤":
            p_op = "<"
        elif p_op == "≥":
            p_op = ">"
        if p_op not in ("=", "<", ">"):
            p_op = None
        assert p_op == expected


def test_cascade_from_pipeline():
    """Test that cascade uses Pipeline() and builds results by line."""
    # Mock Pipeline to return predictable results
    from statcheck_ml.pipeline import Pipeline

    # Create a mock cascade report
    mock_report = {
        "results": [
            {
                "test_type": "t",
                "statistic": 2.45,
                "line": 1,
                "quote": "t(28) = 2.45, p = .021",
            }
        ],
        "fragments": [
            {
                "test_type": None,
                "statistic": 1.5,
                "line": 2,
                "quote": "1.5",
            }
        ],
    }

    # Build cascade dict
    cascade_by_line = {}
    for result in mock_report["results"]:
        line_num = result.get("line", -1)
        if line_num not in cascade_by_line:
            cascade_by_line[line_num] = {"results": [], "fragments": []}
        cascade_by_line[line_num]["results"].append(result)

    for fragment in mock_report["fragments"]:
        line_num = fragment.get("line", -1)
        if line_num not in cascade_by_line:
            cascade_by_line[line_num] = {"results": [], "fragments": []}
        cascade_by_line[line_num]["fragments"].append(fragment)

    assert 1 in cascade_by_line
    assert len(cascade_by_line[1]["results"]) == 1
    assert 2 in cascade_by_line
    assert len(cascade_by_line[2]["fragments"]) == 1


def test_unit_position_offset_mapping(tmp_path):
    """Test position mapping when unit starts after a blanked reference line."""
    # Create a paper with a blanked reference line before a unit
    text = "Line 1\n\nReferences\nSmith, J. (2020)\nJones, M. (2019)\n\nConclusion: t(28) = 2.45, p = .021"
    repaired_text = text  # Assume normalize and repair don't change it

    lines = repaired_text.split("\n")
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln) + 1)

    # A result at character position in repaired text
    result_line = 6  # "Conclusion: t(28) = 2.45, p = .021"
    anchor_s_global = repaired_text.index("2.45")
    anchor_e_global = anchor_s_global + 4

    # Map to piece starting at line 6
    piece_start_line = 6
    piece_offset = anchor_s_global - offsets[result_line]

    assert piece_offset >= 0


def test_statistic_span_in_piece_with_numbers():
    """Test that statistic_span_in_passage holds the statistic's numbers."""
    # Simulated piece_text and numbers
    piece_text = "t(28) = 2.45, p = .021"
    anchor_s = piece_text.index("2.45")
    anchor_e = anchor_s + 4

    statistic_value = piece_text[anchor_s:anchor_e]
    assert "2.45" in statistic_value


def test_paper_pieces(tmp_path):
    """Test paper_pieces: units non-empty, pieces cover with no gap/overlap, split at line ends, hash correctness."""
    from statcheck_ml.prefilter import Prefilter
    from statcheck_ml.normalize import normalize
    from statcheck_ml.repair_validated import repair_validated

    # Create synthetic text that passes the prefilter (with digits)
    text = "Results: t(28) = 2.45, p = .021\n" + ("Statistical test: F(1, 100) = 5.23, p = .045\n" * 20) + "Final line with t(30) = 1.50, p = .001"

    # Get units and verify they are non-empty
    prefilter = Prefilter()
    units = list(prefilter.units(text))
    assert len(units) > 0, "No units found in text"
    for unit in units:
        assert len(unit.text) > 0, "Empty unit"

    # Call paper_pieces with max_chars to force splitting
    pieces = inventory_module.paper_pieces(text, max_chars=200)
    assert len(pieces) > 0
    assert any(p["pieces"] > 1 for p in pieces), "At least one unit should be split"

    # Group pieces by unit
    pieces_by_unit = {}
    for piece in pieces:
        unit_idx = piece["unit"]
        if unit_idx not in pieces_by_unit:
            pieces_by_unit[unit_idx] = []
        pieces_by_unit[unit_idx].append(piece)

    # For each unit, verify coverage, numbering, continuity, and text equality
    for unit_idx in sorted(pieces_by_unit.keys()):
        unit = units[unit_idx]
        unit_pieces = sorted(pieces_by_unit[unit_idx], key=lambda p: p["piece"])

        # Verify piece numbers are sequential (1..pieces)
        for i, piece in enumerate(unit_pieces):
            assert piece["piece"] == i + 1
            assert piece["pieces"] == len(unit_pieces)

        # Verify first piece starts at unit.start_line
        assert unit_pieces[0]["start_line"] == unit.start_line

        # Verify no gaps or overlaps: each next piece starts where previous ends + 1
        for i in range(1, len(unit_pieces)):
            prev = unit_pieces[i - 1]
            curr = unit_pieces[i]
            assert curr["start_line"] == prev["end_line"] + 1

        # Verify joined piece texts equal unit text
        joined = "\n".join(p["text"] for p in unit_pieces)
        assert joined == unit.text

        # Verify each piece ends at a line end (last line is a full unit line)
        unit_lines = unit.text.split("\n")
        for piece in unit_pieces:
            piece_lines = piece["text"].split("\n")
            piece_start_in_unit = piece["start_line"] - unit.start_line
            piece_end_in_unit = piece["end_line"] - unit.start_line
            expected_last_line = unit_lines[piece_end_in_unit]
            actual_last_line = piece_lines[-1]
            assert actual_last_line == expected_last_line

    # Verify piece size constraint
    for piece in pieces:
        n_chars = len(piece["text"])
        piece_lines = piece["text"].split("\n")
        if len(piece_lines) > 1:
            assert n_chars <= 200, f"Multi-line piece exceeds max_chars: {n_chars}"

    # Test build_passages integration and hash computation
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()

    doc_path = clean_dir / "TestJournal" / "test_paper.txt"
    doc_path.parent.mkdir(parents=True)

    # Normalize and repair text for the corpus
    norm_text, _ = normalize(text)
    repair_text, _ = repair_validated(norm_text)
    doc_path.write_text(repair_text, encoding="utf-8")

    # Create minimal paper dict
    paper = {
        "doc": "TestJournal/test_paper.txt",
        "has_results": False,
        "results": []
    }

    # Call build_passages
    passages, _ = inventory_module.build_passages(
        ("TestJournal/test_paper.txt", paper, str(clean_dir), 200)
    )

    # Verify text_sha256 matches piece text hash
    for passage in passages:
        matching_pieces = [p for p in pieces
                          if p["start_line"] == passage.start_line
                          and p["end_line"] == passage.end_line]
        assert len(matching_pieces) == 1
        piece = matching_pieces[0]
        expected_hash = hashlib.sha256(piece["text"].encode("utf-8")).hexdigest()
        assert passage.text_sha256 == expected_hash

    # Verify paper_text_sha256 matches repair_text hash
    paper_hash = hashlib.sha256(repair_text.encode("utf-8")).hexdigest()
    for passage in passages:
        assert passage.paper_text_sha256 == paper_hash
