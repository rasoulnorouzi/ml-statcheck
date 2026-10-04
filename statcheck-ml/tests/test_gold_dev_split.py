"""Tests for the gold dev split: determinism, stratification, and counts."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "pipeline/14_gold_dev_split.py"
SPLIT = ROOT / "dataset/gold/split.json"
GOLD = ROOT / "dataset/gold/gold.jsonl"


def run(out, seed="0"):
    subprocess.run([sys.executable, str(STAGE), "--split", str(SPLIT),
                    "--gold", str(GOLD), "--out", str(out), "--seed", seed],
                   check=True, capture_output=True)
    return Path(out).read_bytes()


@pytest.fixture(scope="module")
def papers():
    return [json.loads(l) for l in open(GOLD) if l.strip()]


@pytest.fixture(scope="module")
def split_data():
    with open(SPLIT) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dev(tmp_path_factory):
    return json.loads(run(tmp_path_factory.mktemp("d") / "dev.json"))


def test_deterministic(tmp_path):
    assert run(tmp_path / "a.json") == run(tmp_path / "b.json")


def test_all_dev_in_gold_train(dev, split_data):
    gold_train = set(split_data["gold_train"])
    assert all(doc in gold_train for doc in dev["dev"])


def test_no_dev_in_gold_test(dev, split_data):
    gold_test = set(split_data["gold_test"])
    assert not any(doc in gold_test for doc in dev["dev"])


def test_counts_match_list(dev, papers):
    dev_set = set(dev["dev"])
    dev_counts = dev["counts"]["dev"]

    papers_count = 0
    with_results = 0
    without_results = 0
    located_results = 0

    IN_SCOPE = {"t", "F", "r", "z", "chi2", "Q"}
    for p in papers:
        if p["doc"] not in dev_set:
            continue
        papers_count += 1
        if p["has_results"]:
            with_results += 1
        else:
            without_results += 1
        located_results += sum(
            1 for r in p["results"] if r.get("located") and r.get("test_type") in IN_SCOPE)

    assert dev_counts["papers"] == papers_count
    assert dev_counts["papers_with_results"] == with_results
    assert dev_counts["papers_without_results"] == without_results
    assert dev_counts["located_in_scope_results"] == located_results
