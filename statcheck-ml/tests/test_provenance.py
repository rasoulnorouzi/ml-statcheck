"""Tests for provenance module."""
import json
import tempfile
from pathlib import Path
import pytest

from statcheck_ml.provenance import (
    sha256_file, sha256_text, write_manifest, verify_manifest,
    stamp, validate_batch, normalise_test_type, normalise_operator
)


def test_sha256_text():
    """Test SHA256 hash of text."""
    text = "hello world"
    h = sha256_text(text)
    # Known hash value
    assert len(h) == 64
    assert h == sha256_text(text)  # Deterministic


def test_sha256_file():
    """Test SHA256 hash of file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write("test content")
        f.flush()
        path = f.name

    try:
        h1 = sha256_file(path)
        h2 = sha256_file(path)
        assert h1 == h2
        assert len(h1) == 64
    finally:
        Path(path).unlink()


def test_write_and_verify_manifest():
    """Test manifest write and verify."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        file1 = tmpdir / "file1.txt"
        file1.write_text("content1", encoding='utf-8')
        file2 = tmpdir / "file2.txt"
        file2.write_text("content2", encoding='utf-8')

        manifest_path = tmpdir / "manifest.json"

        # Write manifest
        manifest = write_manifest([file1, file2], manifest_path, "test command", seed=42)

        # Verify manifest uses POSIX paths (no backslashes)
        for key in manifest['files'].keys():
            assert '\\' not in key, f"Manifest key contains backslash: {key}"

        # Verify returns empty list (all files match)
        errors = verify_manifest(manifest_path)
        assert errors == []

        # Modify a file
        file1.write_text("modified", encoding='utf-8')

        # Verify now returns the modified file
        errors = verify_manifest(manifest_path)
        assert "file1.txt" in errors[0]


def test_stamp():
    """Test stamping record with provenance."""
    record = {"window_id": "test123", "text": "hello"}
    stamped = stamp(record, rater="haiku", model_id="claude-haiku")

    assert stamped["window_id"] == "test123"
    assert stamped["provenance"]["rater"] == "haiku"
    assert stamped["provenance"]["model_id"] == "claude-haiku"


def test_normalise_test_type():
    """Test test type normalisation."""
    assert normalise_test_type("t") == "t"
    assert normalise_test_type("F") == "F"
    assert normalise_test_type("f") == "F"
    assert normalise_test_type("r") == "r"
    assert normalise_test_type("z") == "z"
    assert normalise_test_type("q") == "Q"
    assert normalise_test_type("Q") == "Q"

    # Chi-square variants
    assert normalise_test_type("chi2") == "chi2"
    assert normalise_test_type("chi-square") == "chi2"
    assert normalise_test_type("x2") == "chi2"
    assert normalise_test_type("χ2") == "chi2"
    assert normalise_test_type("chi²") == "chi2"

    # Invalid
    assert normalise_test_type("invalid") is None
    assert normalise_test_type(None) is None


def test_normalise_operator():
    """Test operator normalisation."""
    assert normalise_operator("<") == "<"
    assert normalise_operator(">") == ">"
    assert normalise_operator("=") == "="
    assert normalise_operator(None) is None

    # Variants that map to <
    assert normalise_operator("≤") == "<"
    assert normalise_operator("le") == "<"

    # Variants that map to >
    assert normalise_operator("≥") == ">"
    assert normalise_operator("ge") == ">"

    # Invalid
    assert normalise_operator("invalid") is None


def test_validate_batch_wrong_length():
    """Test validation fails on wrong output length."""
    batch = [{"window_id": "w1", "text": "text1"}]
    output = [{"window_id": "w1", "contains_result": False, "results": []}] * 2

    errors = validate_batch(batch, output)
    assert len(errors) > 0
    assert "length" in errors[0]


def test_validate_batch_wrong_window_id():
    """Test validation fails on mismatched window_id."""
    batch = [{"window_id": "w1", "text": "text1"}]
    output = [{"window_id": "w2", "contains_result": False, "results": []}]

    errors = validate_batch(batch, output)
    assert len(errors) > 0


def test_validate_batch_missing_key():
    """Test validation fails on missing required key."""
    batch = [{"window_id": "w1", "text": "text1"}]
    output = [{"window_id": "w1", "results": []}]  # Missing contains_result

    errors = validate_batch(batch, output)
    assert len(errors) > 0


def test_validate_batch_bad_operator():
    """Test validation fails on bad p_operator."""
    batch = [{"window_id": "w1", "text": "text1"}]
    output = [{
        "window_id": "w1",
        "contains_result": True,
        "results": [{
            "test_type": "t", "statistic": "1.23", "df1": "10", "df2": None,
            "n": None, "p_operator": "invalid_op", "p_value": "0.05",
            "quote": "t(10) = 1.23", "damaged": False, "confidence": "high"
        }]
    }]

    errors = validate_batch(batch, output)
    assert any("p_operator" in e for e in errors)


def test_validate_batch_valid_v2():
    """Test validation passes on valid v2 output."""
    # Build a small batch from chunk_00.json and create valid v2 output
    chunk_path = Path(__file__).parent.parent / "data" / "sample_round2" / "chunks" / "chunk_00.json"

    if not chunk_path.exists():
        pytest.skip("Test data not found")

    with open(chunk_path, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Use first two windows
    batch = all_chunks[:2]

    # Create valid v2 output: first window has no result, second has one result
    output = [
        {
            "window_id": batch[0]["window_id"],
            "contains_result": False,
            "results": []
        },
        {
            "window_id": batch[1]["window_id"],
            "contains_result": True,
            "results": [{
                "test_type": "F",
                "statistic": "6.14",
                "df1": "2",
                "df2": "89",
                "n": None,
                "p_operator": "<",
                "p_value": ".01",
                "quote": "F(2, 89) = 6.14, p < .01",
                "damaged": False,
                "confidence": "high"
            }]
        }
    ]

    errors = validate_batch(batch, output)
    assert errors == [], f"Validation errors: {errors}"


def test_sha256_of_text_ignores_line_endings(tmp_path):
    from statcheck_ml.provenance import sha256_file
    lf, crlf = tmp_path / "a.json", tmp_path / "b.json"
    lf.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    crlf.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
    assert sha256_file(lf) == sha256_file(crlf)
    # a binary file is hashed as it is
    x, y = tmp_path / "a.onnx", tmp_path / "b.onnx"
    x.write_bytes(b'\x00\n'); y.write_bytes(b'\x00\r\n')
    assert sha256_file(x) != sha256_file(y)
