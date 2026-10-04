import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from statcheck_ml.gold import assert_not_gold_test  # noqa: E402

GOLD = ROOT / "dataset/gold/gold.jsonl"
STAGE = ROOT / "pipeline/14_gold_split.py"


@pytest.fixture(scope="module")
def papers():
    return [json.loads(l) for l in open(GOLD) if l.strip()]


def run(out, seed="0"):
    subprocess.run([sys.executable, str(STAGE), "--out", str(out), "--seed", seed],
                   check=True, capture_output=True)
    return Path(out).read_bytes()


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    return json.loads(run(tmp_path_factory.mktemp("s") / "split.json"))


def test_holdout_in_test_and_train_not(papers, split):
    test = set(split["gold_test"])
    for p in papers:
        if p["bronze_set"] == "holdout":
            assert p["doc"] in test
        if p["bronze_set"] == "train":
            assert p["doc"] not in test


def test_each_paper_in_one_set(papers, split):
    all_docs = split["gold_test"] + split["gold_train"]
    assert sorted(all_docs) == sorted(p["doc"] for p in papers)
    assert len(set(all_docs)) == len(all_docs)


def test_none_share_and_strata(papers, split):
    test = set(split["gold_test"])
    none = [p for p in papers if p["bronze_set"] == "none"]
    share = sum(p["doc"] in test for p in none) / len(none)
    assert 0.64 <= share <= 0.70
    for side in (True, False):
        c = Counter(bool(p["has_results"]) for p in none if (p["doc"] in test) == side)
        assert c[True] > 0 and c[False] > 0


def test_deterministic(tmp_path):
    assert run(tmp_path / "a.json") == run(tmp_path / "b.json")


def test_guard(split):
    t, tr = split["gold_test"][0], split["gold_train"][0]
    with pytest.raises(ValueError, match=re.escape(t)):
        assert_not_gold_test(["x/other_1.txt", t], split)
    assert_not_gold_test([tr], split)
