"""Tests for the gold stages: quote location, scoring, and summary figures."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "pipeline" / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gold = load("gold_build", "14_gold.py")
gold_eval = load("gold_eval", "14_gold_eval.py")


def locate_in(text, quote, stat):
    return gold.locate(quote, stat, text, gold.numbers(text))


def test_locate_with_control_character_operators():
    result = "F(1,184) \x02 7.64, p \x03 .01"
    text = "x" * 150 + result + "y" * (300 - 150 - len(result))
    assert len(text) == 300
    found = locate_in(text, "F(1,184) 7.64, p .01", 7.64)
    assert found is not None
    s, e = found["span"]
    assert text[found["anchor"][0]:found["anchor"][1]] == "7.64"
    assert s >= 150 and e <= 150 + len(result)


def test_locate_prefers_the_candidate_that_looks_like_the_quote():
    table = "Table 2   1  184  7.64  .01   other cells   "
    sentence = "The effect was reliable, F(1, 184) = 7.64, p = .01, as expected."
    text = table + sentence
    found = locate_in(text, "F(1, 184) = 7.64, p = .01", 7.64)
    assert found["span"][0] >= len(table)
    assert found["anchor"][0] >= len(table)


def test_locate_returns_none_when_numbers_are_absent():
    assert locate_in("nothing numeric here at all", "F(1, 184) = 7.64", 7.64) is None


def test_test_type():
    t = gold.test_type
    assert t("χ2") == "chi2"
    assert t("X2") == "chi2"
    assert t("chi") == "chi2"
    assert t("F") == "F"
    assert t("ΔF") == "F"
    assert t("Qb") == "Q"
    assert t("t") == "t"
    assert t("Wald") == "other"
    assert t("") is None


def test_score_paper_regex_matches_a_synthetic_paper(tmp_path):
    body = "Introduction. We ran the study.\n\nThe effect was reliable, t(28) = 2.45, p = .021, as expected.\n\nEnd."
    (tmp_path / "paper.txt").write_text(body, encoding="utf-8")
    s = body.index("2.45")
    paper = {"doc": "paper.txt", "bronze_set": "none", "has_results": True,
             "results": [{"located": True, "span": [s - 8, s + 20], "anchor": [s, s + 4],
                          "line": body[:s].count("\n"), "stat": 2.45,
                          "test_type": "t", "damaged": False}]}
    out = gold_eval.score_paper((paper, str(tmp_path), ["regex"], None))
    assert out["gold"][0]["regex"] is True
    assert out["gold"][0]["kept"] is True
    assert set(out["unmatched"]) == {"regex"}


def paper(has, *hits):
    return {"has_results": has, "bronze_set": "none",
            "gold": [{"test_type": "t", "damaged": d, "kept": True, "regex": h,
                        "regex_with_fragments": h} for h, d in hits],
            "unmatched": {"regex": {"named": 0 if has else 2, "nameless": 0 if has else 1}}}


def test_summarise_recall_and_interval():
    papers = [paper(True, (True, False), (True, False)),
              paper(True, (False, True), (True, True)),
              paper(False)]
    out = gold_eval.summarise(papers, 500, 0, ["regex"])
    r = out["regex"]
    assert out["papers"] == 3 and out["papers_with_results"] == 2
    assert out["gold_in_scope"] == 4
    assert r["found"] == 3 and r["recall"] == 0.75
    assert r["recall_damaged"] == 0.5 and r["damaged_n"] == 2
    lo, hi = r["recall_ci"]
    assert lo <= 0.75 <= hi and 0 <= lo and hi <= 1
    assert r["finds_in_papers_without_results"] == 2
    assert out["prefilter_keeps"] == 1.0


def test_statcheck_reader_maps_case(tmp_path):
    f = tmp_path / "sc.csv"
    f.write_text('"window_id","test_type","test_value"\n'
                 '"a.txt","Chi2","6.93"\n"a.txt","Z","-1.96"\n"a.txt","Qw","3"\n'
                 '"b.txt#4-9","t","NA"\n"b.txt#4-9","F","2.5"\n', encoding="utf-8")
    got = gold_eval.read_statcheck(str(f))
    assert [(x[2], x[3]) for x in got["a.txt"]] == [(6.93, "chi2"), (1.96, "z"), (3.0, "other")]
    assert got["b.txt"] == [[4, 9, 2.5, "F", False]]


def test_statcheck_find_serves_one_gold_row():
    rows = [{"stat": 2.45, "anchor_value": 2.45, "line": 3},
            {"stat": 2.45, "anchor_value": 2.45, "line": 40}]
    finds = [[None, None, 2.45, "t", False]]
    hits, left = gold_eval.match_statcheck(rows, finds)
    assert hits == [True, False] and not left
    finds = [[None, None, 2.45, "t", False], [None, None, 2.45, "t", False],
             [None, None, 0.3, "r", False]]
    hits, left = gold_eval.match_statcheck(rows, finds)
    assert hits == [True, True] and left == {"r": 1}


def test_statcheck_passage_must_cover_the_gold_line():
    row = [{"stat": 2.45, "anchor_value": None, "line": 40}]
    hits, _ = gold_eval.match_statcheck(row, [[0, 10, 2.45, "t", False]])
    assert hits == [False]
    hits, _ = gold_eval.match_statcheck(row, [[38, 42, 2.45, "t", False]])
    assert hits == [True]


def test_recall_with_fragments_on_a_synthetic_paper(tmp_path, monkeypatch):
    body = ("Introduction. We ran the study.\n\n"
            "The difference was 2.31, p = .02, in the first sample.\n\nEnd.")
    (tmp_path / "paper.txt").write_text(body, encoding="utf-8")
    s = body.index("2.31")
    paper = {"doc": "paper.txt", "bronze_set": "none", "has_results": True,
             "results": [{"located": True, "span": [s - 4, s + 12], "anchor": [s, s + 4],
                          "line": body[:s].count("\n"), "stat": 2.31,
                          "test_type": "t", "damaged": False}]}
    pipe = gold_eval.make_pipes(["regex"], None)["regex"]
    pipe.model = object()
    pipe.find_with_model = lambda text, line: [
        gold_eval_found(text, line)] if "2.31" in text else []
    monkeypatch.setattr(gold_eval, "make_pipes", lambda systems, model_path: {"regex": pipe})
    out = gold_eval.score_paper((paper, str(tmp_path), ["regex"], None))
    row = out["gold"][0]
    assert row["regex"] is False
    assert row["regex_with_fragments"] is True
    assert out["unmatched"]["regex"].get("nameless", 0) == 0
    summary = gold_eval.summarise([out], 50, 0, ["regex"])
    assert summary["regex"]["recall"] == 0.0
    assert summary["regex"]["recall_with_fragments"] == 1.0


def gold_eval_found(text, line):
    from statcheck_ml.pipeline import Found

    i = text.index("2.31")
    return Found(test_type=None, statistic=2.31, p_operator="=", p_value=0.02,
                 source="model", line=line, statistic_span=(i, i + 4))
