"""Tests for pipeline/08_port_kit.py, run as a subprocess of the venv python.

A subprocess, not importlib, because the script's own argument parsing and
its `--verify` mode are what a port actually calls.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "pipeline" / "08_port_kit.py"

EXPECTED_FILES_GRU_CRF = {
    "spec/charmap.json", "spec/font_table.json", "spec/normalize.json",
    "spec/prefilter.json", "spec/repair.json",
    "model/gru-crf/tagger.onnx", "model/gru-crf/decoder.json",
    "model/gru-crf/charmap.json", "model/gru-crf/weights.json",
    "parity/cases.json", "manifest.json", "README.md",
}


def run(args):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8", env=env)


def files_in(kit: Path) -> set:
    return {p.relative_to(kit).as_posix() for p in kit.rglob("*") if p.is_file()}


def test_export_default_config_exact_tree_and_verify(tmp_path):
    kit = tmp_path / "kit"
    result = run([str(kit)])
    assert result.returncode == 0, result.stderr

    assert files_in(kit) == EXPECTED_FILES_GRU_CRF

    manifest = json.loads((kit / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"].keys()) == EXPECTED_FILES_GRU_CRF - {"manifest.json"}
    assert manifest["model_default"] == "gru-crf"
    assert manifest["models"] == ["gru-crf"]
    assert manifest["mother_repository"] == "https://github.com/rasoulnorouzi/ml-statcheck"
    assert len(manifest["mother_commit"]) == 40

    verify = run(["--verify", str(kit)])
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_verify_catches_a_modified_file(tmp_path):
    kit = tmp_path / "kit"
    assert run([str(kit)]).returncode == 0

    path = kit / "spec" / "normalize.json"
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))

    verify = run(["--verify", str(kit)])
    assert verify.returncode == 1
    assert "spec/normalize.json" in verify.stdout


def test_manifest_hashes_do_not_depend_on_target_path(tmp_path):
    kit1 = tmp_path / "a" / "kit"
    kit2 = tmp_path / "b" / "kit"
    assert run([str(kit1)]).returncode == 0
    assert run([str(kit2)]).returncode == 0

    files1 = json.loads((kit1 / "manifest.json").read_text(encoding="utf-8"))["files"]
    files2 = json.loads((kit2 / "manifest.json").read_text(encoding="utf-8"))["files"]
    assert files1 == files2


def test_config_not_matching_parity_model_is_refused(tmp_path):
    kit = tmp_path / "kit"
    result = run([str(kit), "--config", "lstm-softmax"])
    assert result.returncode != 0
    assert "gru-crf" in (result.stdout + result.stderr)
    assert not kit.exists() or not (kit / "manifest.json").exists()


def test_two_configs_ship_two_models(tmp_path):
    kit = tmp_path / "kit"
    result = run([str(kit), "--config", "gru-crf", "lstm-softmax"])
    assert result.returncode == 0, result.stderr

    assert (kit / "model" / "gru-crf" / "tagger.onnx").exists()
    assert (kit / "model" / "lstm-softmax" / "tagger.onnx").exists()
    manifest = json.loads((kit / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["models"] == ["gru-crf", "lstm-softmax"]
    assert manifest["model_default"] == "gru-crf"

    verify = run(["--verify", str(kit)])
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_refuses_a_config_not_in_the_zoo(tmp_path):
    kit = tmp_path / "kit"
    result = run([str(kit), "--config", "does-not-exist"])
    assert result.returncode != 0
    assert "does-not-exist" in (result.stdout + result.stderr)


def test_kit_replaces_stale_contents(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    stale = kit / "leftover.txt"
    stale.write_text("from a previous export", encoding="utf-8")

    result = run([str(kit)])
    assert result.returncode == 0, result.stderr
    assert not stale.exists()
    assert files_in(kit) == EXPECTED_FILES_GRU_CRF
