"""pipeline/25_verdicts_v4.py: the matching wrapper against 14_gold_eval.py, the APA
strings, the R reference, and the metrics on hand-made rows."""
import importlib.util
import random
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verdicts_v4", ROOT / "pipeline/25_verdicts_v4.py")
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)
ge = vd.ge

C, I, D, U = vd.CONSISTENT, vd.INCONSISTENT, vd.DECISION_ERROR, vd.UNDECIDABLE


# ---------------- matching: the wrapper equals 14_gold_eval.py ----------------

def random_case(rng, lines_optional):
    values = [1.5, 2.25, 3.0, 0.42]
    gold = [{"stat": rng.choice(values + [None]), "anchor_value": rng.choice(values + [None]),
             "line": rng.randrange(0, 12)} for _ in range(rng.randrange(0, 8))]
    finds = []
    for _ in range(rng.randrange(0, 9)):
        lo = rng.randrange(0, 12)
        hi = lo + rng.randrange(0, 4)
        if lines_optional and rng.random() < 0.4:
            lo = hi = None
        finds.append([lo, hi, rng.choice(values), rng.choice(["t", "F", "r"]), False,
                      rng.choice([C, I, D, U])])
    return gold, finds


def test_match_served_equals_match_local():
    rng = random.Random(0)
    for _ in range(1000):
        gold, finds = random_case(rng, lines_optional=False)
        mine, theirs = [list(f) for f in finds], [list(f) for f in finds]
        served = vd.match_served(gold, mine, "local")
        assert [i is not None for i in served] == ge.match_local(gold, theirs)
        assert [f[4] for f in mine] == [f[4] for f in theirs]
        used = [i for i in served if i is not None]
        assert len(used) == len(set(used))


def test_match_served_equals_match_statcheck():
    rng = random.Random(1)
    for _ in range(1000):
        gold, finds = random_case(rng, lines_optional=True)
        mine, theirs = [list(f) for f in finds], [list(f) for f in finds]
        served = vd.match_served(gold, mine, "statcheck")
        hits, _ = ge.match_statcheck(gold, theirs)
        assert [i is not None for i in served] == hits
        assert [f[4] for f in mine] == [f[4] for f in theirs]


def test_match_served_names_the_find():
    gold = [{"stat": 2.0, "anchor_value": None, "line": 5},
            {"stat": 2.0, "anchor_value": None, "line": 5}]
    finds = [[0, 3, 2.0, "t", False, C], [4, 6, 2.0, "t", False, I], [4, 6, 2.0, "t", False, D]]
    assert vd.match_served(gold, finds, "local") == [1, 2]


def test_read_statcheck_verdicts_equals_read_statcheck():
    for name in ge.STATCHECK.values():
        path = ROOT / "dataset/baseline/gold" / name
        if not path.exists():
            pytest.skip("the baseline CSVs are not in this checkout")
        theirs = ge.read_statcheck(str(path))
        mine = vd.read_statcheck_verdicts(str(path))
        assert set(mine) == set(theirs)
        for doc in theirs:
            assert [f[:5] for f in mine[doc]] == theirs[doc]
            assert all(f[5] in (C, I, D, U) for f in mine[doc])


def test_r_verdict_from_columns():
    assert vd.r_verdict(False, False) == C
    assert vd.r_verdict(True, False) == I
    assert vd.r_verdict(True, True) == D
    assert vd.r_verdict(None, None) is None


TEXT = ("Some prose with no result.\n"
        "Participants were faster, t(30) = 2.10, p = .04, than before.\n"
        "The effect was clear, F(1, 60) = 9.02, p = .50.\n"
        "More prose.\n" * 3)


def test_local_finds_equals_finds_for():
    from statcheck_ml.normalize import normalize
    from statcheck_ml.pipeline import Pipeline

    pipe = Pipeline(model_path=None)
    text, _ = normalize(TEXT)
    fixed, _ = pipe.repair(text)
    units = list(pipe.prefilter.units(fixed))
    named, _ = ge.finds_for(pipe, TEXT, units)
    report = pipe.run_text(TEXT)
    mine = vd.local_finds(report, units)
    assert named and [tuple(f[:4]) for f in mine] == named
    assert [f[5] for f in mine] == [report["results"][f[6]]["verdict"] for f in mine]
    assert {f[5] for f in mine} == {C, D}


# ---------------- the APA strings ----------------

def gold_row(**kw):
    row = {"test_type": "t", "stat": 2.1, "df1": None, "df2": 30.0, "p_operator": "=", "p": 0.04,
           "raw": "t(30) = 2.10, p = .040"}
    row.update(kw)
    return row


def test_build_keeps_printed_decimals():
    b = vd.build(gold_row())
    assert b["string"] == " t(30) = 2.10, p = .040"
    assert b["stat_from_raw"] and b["p_from_raw"] and b["df1"] == 30.0


def test_build_falls_back_to_the_coded_value():
    b = vd.build(gold_row(raw="damaged", p=1e-05, p_operator="<"))
    assert b["string"] == " t(30) = 2.1, p < 0.00001"
    assert not b["stat_from_raw"] and not b["p_from_raw"]


def test_statistic_prefers_more_decimals_over_an_equal_df():
    text, at, ambiguous = vd.stat_as_written("F(2, 61) = 2.00, p = .14", 2.0)
    assert (text, ambiguous) == ("2.00", True)


def test_statistic_drops_the_sign_and_reads_thousands():
    assert vd.stat_as_written("t(109) = −29.86, p < .001", -29.86)[0] == "29.86"
    assert vd.stat_as_written("χ^2 (20)=2,995.73, p=0.00", 2995.73)[0] == "2995.73"


def test_p_is_read_after_the_statistic():
    raw = "r = .05, t(30) = 2.10, p = .050"
    text, at, _ = vd.stat_as_written(raw, 2.1)
    assert vd.p_as_written(raw, 0.05, at) == (".050", True)


def test_build_tests_and_operators():
    assert vd.build(gold_row(test_type="F", df1=1.0, df2=60.0, stat=0.02, p_operator="ns", p=None,
                             raw="F(1, 60) .02, ns"))["string"] == " F(1, 60) = .02, ns"
    b = vd.build(gold_row(p_operator="≤", raw="t(30) = 2.10, p ≤ .040"))
    assert b["string"] == " t(30) = 2.10, p < .040" and b["operator_mapped"]
    assert vd.build(gold_row(p_operator='<" ', p=0.05, raw="t(30) = 2.10, p < .05"))["op"] == "<"
    assert vd.build(gold_row(test_type="z", df2=None, raw="z = 2.10, p = .04"))["string"] \
        == " z = 2.10, p = .04"
    assert vd.build(gold_row(test_type="chi2", df1=3.0, df2=None))["string"].startswith(" χ2(3) = ")
    assert vd.build(gold_row(test_type="F", df1=1.5, df2=91.67))["string"].startswith(" F(1.5, 91.67) = ")


@pytest.mark.parametrize("kw,cause", [
    ({"stat": None}, "no statistic coded"),
    ({"df2": None}, "no degrees of freedom coded"),
    ({"test_type": "F", "df1": 1.0, "df2": None}, "no degrees of freedom coded"),
    ({"p": None}, "no p-value coded"),
    ({"p_operator": None}, "no p operator coded"),
    ({"p_operator": "~"}, "no p operator coded"),
])
def test_build_names_the_missing_part(kw, cause):
    b = vd.build(gold_row(**kw))
    assert b["string"] is None and b["cause"] == cause


# ---------------- the R reference ----------------

def r_outcome(**kw):
    out = {"status": "ok", "test_type": "t", "test_value": 2.1, "p_comp": "=", "reported_p": 0.04,
           "computed_p": 0.0442, "error": False, "decision_error": False}
    out.update(kw)
    return out


def test_reference_from_outcome():
    b = vd.build(gold_row())
    assert vd.reference(b, r_outcome()) == (C, None, 0.0442)
    assert vd.reference(b, r_outcome(error=True, decision_error=True))[0] == D
    assert vd.reference(b, r_outcome(error=None))[1] == "statcheck gave no verdict"
    assert vd.reference(b, r_outcome(test_value=21.0))[1] == "statcheck read another value"
    assert vd.reference(b, r_outcome(test_type="F"))[1] == "statcheck read another value"
    assert vd.reference(b, {"status": "none"})[1] == "statcheck found no result"
    assert vd.reference(b, {"status": "error"})[1] == "statcheck raised an error"


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R is not installed")
def test_r_and_pvalue_agree_on_known_strings():
    rows = [gold_row(raw="t(30) = 2.10, p = .044", p=0.044),                               # consistent
            gold_row(raw="t(30) = 2.10, p = .03", p=0.03),                          # inconsistent
            gold_row(raw="t(30) = 2.10, p = .07", p=0.07),                          # decision error
            gold_row(test_type="F", df1=1.0, df2=60.0, stat=9.02, p_operator="ns", p=None,
                     raw="F(1, 60) = 9.02, ns"),                                    # decision error
            gold_row(test_type="z", df2=None, stat=2.5, p=0.01, raw="z = 2.50, p = .01"),
            gold_row(test_type="chi2", df1=2.0, df2=None, stat=4.5, p=0.11, raw="χ2(2) = 4.50, p = .11"),
            gold_row(test_type="Q", df1=5.0, df2=None, stat=12.3, p=0.03, raw="Q(5) = 12.3, p = .03"),
            gold_row(test_type="r", df2=40.0, stat=0.35, p=0.02, raw="r(40) = .35, p = .02"),
            gold_row(test_type="r", df2=40.0, stat=1.35, p=0.02, raw="r(40) = 1.35, p = .02")]
    built = [vd.build(g) for g in rows]
    out = vd.run_r([{"id": str(k), "text": b["string"]} for k, b in enumerate(built)])
    got = [vd.reference(b, out[str(k)]) for k, b in enumerate(built)]
    assert [g[0] for g in got] == [C, I, D, D, C, C, C, C, None]
    assert got[-1][1] in ("statcheck found no result", "statcheck raised an error")
    for b, (verdict, _, r_p) in zip(built, got):
        if verdict:
            ours, p = vd.ours(b)
            assert ours == verdict
            assert p == pytest.approx(r_p, rel=vd.P_REL_TOL)


# ---------------- metrics ----------------

def rows_and_papers():
    s = "cascade"
    rows = [
        {"doc": "a", "reference": I, "systems": {s: I}},
        {"doc": "a", "reference": D, "systems": {s: D}},
        {"doc": "a", "reference": D, "systems": {s: I}},
        {"doc": "b", "reference": I, "systems": {s: None}},
        {"doc": "b", "reference": I, "systems": {s: C}},
        {"doc": "b", "reference": I, "systems": {s: U}},
        {"doc": "b", "reference": C, "systems": {s: D}},
        {"doc": "b", "reference": C, "systems": {s: C}},
        {"doc": "b", "reference": C, "systems": {s: None}},
        {"doc": "b", "reference": None, "systems": {s: I}},
    ]
    papers = [{"doc": "a", "has_results": True,
               "outside": {s: {"unmatched": {I: 2, C: 1}, "out_of_scope": {D: 1}}}},
              {"doc": "b", "has_results": True, "outside": {s: {"unmatched": {}, "out_of_scope": {}}}},
              {"doc": "c", "has_results": False,
               "outside": {s: {"unmatched": {D: 1, U: 3}, "out_of_scope": {}}}}]
    return rows, papers


def test_system_metrics_counts():
    rows, papers = rows_and_papers()
    m = vd.system_metrics(rows, papers, "cascade", 200, 0)
    i = m["inconsistencies"]
    assert (i["n"], i["flagged"], i["not_found"], i["judged_consistent"], i["not_checked"]) == (6, 3, 1, 1, 1)
    assert i["n"] == i["flagged"] + i["not_found"] + i["judged_consistent"] + i["not_checked"]
    assert i["share_flagged"] == 0.5 and len(i["share_flagged_ci"]) == 2
    d = m["decision_errors"]
    assert (d["n"], d["flagged"], d["flagged_inconsistent_only"]) == (2, 1, 1)
    assert m["false_flags"] == {"n_consistent": 3, "found": 2, "flagged": 1, "flagged_decision_error": 1}
    assert m["flags_outside_gold"] == {"papers_with_results": {"flagged": 2, "decision_errors": 0},
                                       "papers_without_results": {"flagged": 1, "decision_errors": 1}}
    assert m["flags_on_gold_rows_without_reference"] == {"in_scope": 1, "out_of_scope": 1}
    assert m["papers"] == {"with_inconsistency": 2, "flagged_correctly": 1}
    assert m["recall"] == 0.8


def test_bootstrap_matches_14_on_the_same_papers():
    rows, papers = rows_and_papers()
    m = vd.system_metrics(rows, papers, "cascade", 500, 0)
    boot = [{"gold": [{"k": True, "cascade": r["systems"]["cascade"] in vd.FLAGS}
                      for r in rows if r["doc"] == p["doc"] and r["reference"] in vd.FLAGS]}
            for p in papers]
    assert m["inconsistencies"]["share_flagged_ci"] == ge.bootstrap(boot, "cascade", lambda g: g["k"], 500, 0)
