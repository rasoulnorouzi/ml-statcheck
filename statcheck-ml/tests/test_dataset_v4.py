"""The mixed training set (pipeline/21_dataset_v4.py), on synthetic inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dataset4", ROOT / "pipeline" / "21_dataset_v4.py")
d4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d4)

TEXT = "As expected, t(23) = 2.45, p = .02, and F(1, 184) = 7.64, p < .01."
T_QUOTE = "t(23) = 2.45, p = .02"
F_QUOTE = "F(1, 184) = 7.64, p < .01"


def doc(name, pid):
    return f"J/{name}_J_2011_{pid}.txt"


OLD_TRAIN, OLD_DEV, GOLD_DEV, GOLD_TEST = (doc("A", "aaaa"), doc("B", "bbbb"),
                                           doc("D", "dddd"), doc("T", "tttt"))
NEW_DOCS = [doc(f"N{i}", f"n{i:03d}") for i in range(10)]


def result(quote, **fields):
    base = {"test_type": "t", "statistic": "2.45", "df1": "23", "df2": None, "n": None,
            "p_operator": "=", "p_value": ".02", "quote": quote, "damaged": False,
            "confidence": "high", "span": None, "voters": ["annotator-haiku", "annotator-sonnet",
                                                           "coders"],
            "coder_vote": "span", "tier": "majority"}
    if quote == F_QUOTE:
        base.update(test_type="F", statistic="7.64", df1="1", df2="184",
                    p_operator="<", p_value=".01")
    base.update(fields)
    return base


def bronze_row(wid, source_doc):
    return {"window_id": wid, "round": "v2", "pool": "A", "journal": "J", "source_doc": source_doc,
            "source_line": 3, "text": "no result here — ünïcode", "contains_result": False,
            "label_tier": "bronze", "results": [], "set": "train", "tier_counts": {}}


class World:
    def __init__(self, tmp, passages, bronze=None):
        """`passages` is a list of (window_id, doc, [results])."""
        self.tmp = tmp
        self.bronze = bronze or [bronze_row("b1", OLD_TRAIN), bronze_row("b2", OLD_DEV),
                                 bronze_row("b3", GOLD_DEV)]
        self.bronze_path = tmp / "train.jsonl"
        # Hand-written spacing, to prove that the stage copies the line and does not re-dump it.
        self.bronze_path.write_text("".join(json.dumps(r, ensure_ascii=False, indent=None,
                                                       separators=(",  ", ": ")) + "\n"
                                            for r in self.bronze), encoding="utf-8")
        self.final = tmp / "final.json"
        self.final.write_text(json.dumps({"passages": [
            {"window_id": w, "results": rs} for w, _, rs in passages]}))
        self.key = tmp / "key.json"
        self.key.write_text(json.dumps([
            {"window_id": w, "doc": dc, "journal": "J", "start_line": 10 + i, "stratum": "damaged",
             "text_sha256": hashlib.sha256(TEXT.encode()).hexdigest()}
            for i, (w, dc, _) in enumerate(passages)]))
        self.batches = tmp / "batches"
        self.batches.mkdir()
        (self.batches / "batch_0.json").write_text(json.dumps(
            [{"window_id": w, "text": TEXT} for w, _, _ in passages]))
        self.split = tmp / "split.json"
        self.split.write_text(json.dumps({"gold_test": [GOLD_TEST],
                                          "gold_train": [OLD_TRAIN, OLD_DEV, GOLD_DEV]}))
        self.dev = tmp / "dev.json"
        self.dev.write_text(json.dumps({"dev": [GOLD_DEV]}))
        self.old = tmp / "splits.json"
        self.old.write_text(json.dumps({"seed": 0, "dev_share": 0.15, "documents": {
            OLD_TRAIN: "train", OLD_DEV: "dev", GOLD_DEV: "train"}}))
        self.out = tmp / "out" / "train_v4.jsonl"
        self.splits_out = tmp / "out" / "splits_v4.json"

    def build(self, **kw):
        return d4.build(self.bronze_path, self.final, self.key, self.batches, self.split,
                        self.dev, self.old, self.out, self.splits_out, **kw)

    def rows(self):
        return [json.loads(l) for l in self.out.read_text(encoding="utf-8").splitlines()]


def good_passages():
    both = [result(T_QUOTE), result(F_QUOTE, tier="adjudicated")]
    return [("p0", OLD_TRAIN, both)] + [(f"q{i}", d, [result(T_QUOTE)]) for i, d in enumerate(NEW_DOCS)]


def test_bronze_rows_come_first_and_unchanged(tmp_path):
    w = World(tmp_path, good_passages())
    summary = w.build()
    out_lines = w.out.read_text(encoding="utf-8").splitlines()
    in_lines = w.bronze_path.read_text(encoding="utf-8").splitlines()
    # b3 is a gold dev paper and is left out; the two others are copied as written.
    assert out_lines[:2] == in_lines[:2]
    assert summary["bronze_rows"] == 2 and summary["bronze_rows_left_out_dev_papers"] == 1
    assert summary["passage_rows"] == 11 and summary["rows"] == 13


def test_keep_dev_bronze_keeps_the_rows(tmp_path):
    w = World(tmp_path, good_passages())
    summary = w.build(keep_dev_bronze=True)
    assert summary["bronze_rows"] == 3 and summary["bronze_rows_left_out_dev_papers"] == 0


def test_a_passage_row_has_the_keys_of_a_bronze_row(tmp_path):
    w = World(tmp_path, good_passages())
    w.build()
    rows = w.rows()
    bronze, passage = rows[0], rows[2]
    assert list(passage) == list(bronze)
    assert passage["round"] == "v4" and passage["set"] == "train"
    assert passage["label_tier"] == "bronze"
    assert passage["source_doc"] == OLD_TRAIN and passage["source_line"] == 10
    assert passage["tier_counts"] == {"majority": 1, "adjudicated": 1}
    t, f = passage["results"]
    assert list(t)[:10] == list(bronze_result_keys())
    assert TEXT[slice(*t["part_spans"]["STAT"])] == "2.45"
    assert TEXT[slice(*f["part_spans"]["DF2"])] == "184"
    assert TEXT[slice(*f["part_spans"]["POP"])] == "<"
    assert t["checkable"] is True and t["raters"] == ["haiku", "sonnet"]
    assert t["voters"][-1] == "coders" and t["coder_vote"] == "span"


def bronze_result_keys():
    return ["test_type", "statistic", "df1", "df2", "n", "p_operator", "p_value", "quote",
            "damaged", "confidence"]


def test_the_split_keeps_old_sides_and_splits_new_documents(tmp_path):
    w = World(tmp_path, good_passages())
    w.build()
    splits = json.loads(w.splits_out.read_text())
    docs = splits["documents"]
    assert docs[OLD_TRAIN] == "train" and docs[OLD_DEV] == "dev"
    assert GOLD_DEV not in docs and splits["excluded_gold_dev_documents"] == [GOLD_DEV]
    assert splits["new_documents"] == 10 and splits["inherited_documents"] == 2
    # round(10 * 0.15) = 2 of the new documents go to dev.
    assert sorted(docs[d] for d in NEW_DOCS).count("dev") == 2
    # Every document has one side, and every row's document is in the map.
    assert all(r["source_doc"] in docs for r in w.rows())


def test_the_split_is_seeded(tmp_path):
    sides = []
    for seed in (0, 1):
        (tmp_path / str(seed)).mkdir()
        w = World(tmp_path / str(seed), good_passages())
        w.build(seed=seed)
        docs = json.loads(w.splits_out.read_text())["documents"]
        sides.append({d for d in NEW_DOCS if docs[d] == "dev"})
    assert sides[0] != sides[1]


def test_two_builds_give_the_same_bytes(tmp_path):
    w = World(tmp_path, good_passages())
    w.build()
    first = (w.out.read_bytes(), w.splits_out.read_bytes())
    w.build()
    assert (w.out.read_bytes(), w.splits_out.read_bytes()) == first


def test_a_gold_test_paper_in_the_passages_stops(tmp_path):
    w = World(tmp_path, good_passages() + [("g", GOLD_TEST, [result(T_QUOTE)])])
    with pytest.raises(ValueError, match="gold test paper"):
        w.build()
    assert not w.out.exists()


def test_a_gold_test_paper_in_the_bronze_rows_stops(tmp_path):
    w = World(tmp_path, good_passages(), bronze=[bronze_row("b1", GOLD_TEST)])
    with pytest.raises(ValueError, match="gold test paper"):
        w.build()


def test_the_pid_alone_names_a_gold_test_paper(tmp_path):
    other_path = "other/Renamed_X_2012_tttt.txt"
    w = World(tmp_path, good_passages() + [("g", other_path, [result(T_QUOTE)])])
    with pytest.raises(ValueError, match="gold test paper"):
        w.build()


def test_a_dev_paper_in_the_passages_stops(tmp_path):
    w = World(tmp_path, good_passages() + [("g", GOLD_DEV, [result(T_QUOTE)])])
    with pytest.raises(SystemExit, match="development paper"):
        w.build()
    assert not w.out.exists()


def test_a_result_that_does_not_align_stops_above_one_percent(tmp_path):
    bad = result("t(99) = 9.99, p = .99")
    w = World(tmp_path, [("p0", OLD_TRAIN, [result(T_QUOTE), bad])])
    with pytest.raises(SystemExit, match="alignment gate: 1 of 2"):
        w.build()
    assert not w.out.exists()


def test_one_unaligned_result_in_two_hundred_and_one_is_counted_and_its_passage_left_out(tmp_path):
    bad = result("t(99) = 9.99, p = .99")
    passages = [("p0", OLD_TRAIN, [result(T_QUOTE)] * 199 + [bad]),
                ("p1", OLD_TRAIN, [result(T_QUOTE)])]
    w = World(tmp_path, passages)
    summary = w.build()
    assert summary["passage_results"] == 201 and summary["passage_results_unaligned"] == 1
    assert summary["passage_rows"] == 1 and summary["passage_rows_left_out_unaligned"] == 1
    assert [r["window_id"] for r in w.rows() if r["round"] == "v4"] == ["p1"]


def test_a_result_with_no_quote_does_not_align(tmp_path):
    w = World(tmp_path, [("p0", OLD_TRAIN, [result(None)])])
    with pytest.raises(SystemExit, match="alignment gate"):
        w.build()


def test_a_changed_passage_text_stops(tmp_path):
    w = World(tmp_path, good_passages())
    (w.batches / "batch_0.json").write_text(json.dumps(
        [{"window_id": p, "text": TEXT + " "} for p, _, _ in good_passages()]))
    with pytest.raises(SystemExit, match="differs from the hash"):
        w.build()


def test_a_passage_missing_from_final_stops(tmp_path):
    w = World(tmp_path, good_passages())
    final = json.loads(w.final.read_text())
    final["passages"].pop()
    w.final.write_text(json.dumps(final))
    with pytest.raises(SystemExit, match="differ on 1 passages"):
        w.build()
