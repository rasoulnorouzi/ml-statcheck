import math

from statcheck_ml.evalutil import (build_result, damage_family, extract_gold, family_recall,
                                   group_spans, mcnemar_pair, prf, score_span_strict,
                                   score_system, verdict_agreement)


def test_extract_gold_reports_unparseable_statistics():
    windows = [
        {"window_id": "w1", "source_doc": "d1", "results": [
            {"statistic": "1.48", "block_span": [0, 5]},
            {"statistic": "0\x0592", "block_span": [10, 15]},
        ]},
    ]
    gold_by_window, gold_meta, _, doc_to_windows, unparseable = extract_gold(windows)
    assert gold_by_window["w1"] == {1.48}
    assert unparseable == [{"window_id": "w1", "statistic": "0\x0592"}]
    assert doc_to_windows["d1"] == ["w1"]


def test_group_spans_and_build_result():
    parts_text = {"TEST": "t", "DF1": "67", "STAT": "1.48", "POP_EQ": "=", "PVAL": ".143"}
    text = "".join(parts_text.values())
    spans, pos = [], 0
    for label, s in parts_text.items():
        spans.append((pos, pos + len(s), label))
        pos += len(s)

    grouped = group_spans(text, spans)
    assert len(grouped) == 1
    parts, part_spans = grouped[0]
    res, ptext = build_result(parts)

    assert res.test_type == "t"
    assert res.statistic == 1.48
    assert res.df1 == 67.0
    assert res.p_operator == "="
    assert res.p_value == 0.143
    assert ptext == ".143"
    starts = [s for s, _ in part_spans]
    ends = [e for _, e in part_spans]
    assert (min(starts), max(ends)) == (0, len(text))


def test_group_spans_splits_on_second_test():
    # Two results in one window, back to back, each opened by a TEST tag.
    spans = [(0, 1, "TEST"), (2, 5, "STAT"), (6, 7, "TEST"), (8, 11, "STAT")]
    grouped = group_spans("x" * 12, spans)
    assert len(grouped) == 2


def test_build_result_needs_a_statistic():
    res, ptext = build_result({"TEST": "t", "DF1": "67"})
    assert res is None and ptext is None


def test_damage_family_control_character():
    assert damage_family("F(1,313) \x02 .03, p \x02 .86") == "control character"


def test_damage_family_default():
    assert damage_family("r = .71") == "other damage"


def test_prf_known_values():
    out = prf(tp=8, fp=2, fn=2)
    assert math.isclose(out["p"], 0.8)
    assert math.isclose(out["r"], 0.8)
    assert math.isclose(out["f1"], 0.8)


def test_score_system_counts_and_buckets():
    gold_by_window = {"w1": {1.0, 2.0}, "w2": {3.0}}
    gold_meta = {
        ("w1", 1.0): {"damaged": True, "test_type": "t", "checkable": True, "family": "control character"},
        ("w1", 2.0): {"damaged": False, "test_type": "f", "checkable": False, "family": None},
        ("w2", 3.0): {"damaged": False, "test_type": "t", "checkable": True, "family": None},
    }
    found_by_window = {"w1": {1.0, 9.0}, "w2": set()}
    out = score_system(found_by_window, gold_by_window, gold_meta, ["w1", "w2"])
    assert out["overall"]["tp"] == 1
    assert out["overall"]["fp"] == 1
    assert out["overall"]["fn"] == 2
    assert out["damaged"]["tp"] == 1 and out["damaged"]["fn"] == 0
    assert out["undamaged"]["fn"] == 2
    assert out["damage:control character"]["tp"] == 1
    assert out["checkable"]["tp"] == 1 and out["checkable"]["fn"] == 1


def test_score_span_strict():
    gold_spans = {"w1": {(0, 5)}}
    found_spans = {"w1": {(0, 5), (10, 15)}}
    out = score_span_strict(found_spans, gold_spans, ["w1"])
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 0


def test_mcnemar_pair_counts():
    gold_by_window = {"w1": {1.0}, "w2": {2.0}, "w3": {3.0}}
    found_a = {"w1": {1.0}, "w2": set(), "w3": {3.0}}
    found_b = {"w1": set(), "w2": {2.0}, "w3": {3.0}}
    b, c = mcnemar_pair(found_a, found_b, gold_by_window)
    assert b == 1 and c == 1


def test_family_recall_basic():
    gold_by_window = {"w1": {1.0}, "w2": {2.0}}
    gold_meta = {
        ("w1", 1.0): {"family": "control character"},
        ("w2", 2.0): {"family": "control character"},
    }
    found_by_window = {"w1": {1.0}, "w2": set()}
    out = family_recall(found_by_window, gold_by_window, gold_meta)
    fam = out["control character"]
    assert fam["k"] == 1 and fam["n"] == 2
    assert math.isclose(fam["r"], 0.5)
    lo, hi = fam["wilson"]
    assert lo <= 0.5 <= hi


def test_verdict_agreement_basic():
    # Real statcheck CSV rows: single-df tests carry their df in whichever
    # column applies (df1 for chi-square, df2 for t) and leave the other NA.
    rows = [
        {"test_type": "t", "df1": "NA", "df2": "67", "test_value": "1.48",
         "p_comp": "=", "reported_p": "0.143", "error": "FALSE"},
        {"test_type": "t", "df1": "NA", "df2": "99", "test_value": "5.59",
         "p_comp": "<", "reported_p": "0.01", "error": "FALSE"},
        {"test_type": "Chi2", "df1": "10", "df2": "NA", "test_value": "13.76",
         "p_comp": "=", "reported_p": "0.18", "error": "FALSE"},
        {"test_type": "t", "df1": "NA", "df2": "178", "test_value": "0.54",
         "p_comp": "ns", "reported_p": "NA", "error": "FALSE"},
    ]
    out = verdict_agreement(rows)
    # Every row agrees with statcheck, including the t(67) row that the old
    # rule called an error: the statistic's own rounding leaves room for a
    # reported .143. `ns` reads as p > alpha, as statcheck reads it.
    assert out == {"agree": 4, "disagree": 0, "undecidable": 0}
