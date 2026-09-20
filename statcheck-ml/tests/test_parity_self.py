"""The Python reference reproduces every value in `parity_cases.json`.

One test per section. Every check calls the same wrapper
`tests/make_parity_cases.py` uses to build the file, imported from
`tests/parity_lib.py`, so the generator and this test cannot drift apart: a
wrapper that changes changes what both of them see.

This is a self-test, not a parity test: it proves the committed file still
matches the code that wrote it, not that any other port matches. JavaScript
and R have their own runners (`tests/parity.mjs`, `tests/parity.R`) for that,
today only for the `normalise` section.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import parity_lib as pl

CASES_PATH = Path(__file__).parent / "parity_cases.json"


def _load():
    with open(CASES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


DATA = _load()
SECTIONS = DATA["sections"]


def test_top_level_shape():
    assert DATA["version"] == 2
    assert DATA["generated_by"] == "tests/make_parity_cases.py"
    assert DATA["model"] == "gru-crf"
    assert set(SECTIONS) == {
        "normalise", "repair", "prefilter", "extract", "model",
        "model_logits", "group", "pvalue", "pipeline",
    }
    for name, rows in SECTIONS.items():
        names = [r["name"] for r in rows]
        assert len(names) == len(set(names)), f"{name}: duplicate case name"
        assert names == sorted(names), f"{name}: not sorted by name"


# --------------------------------------------------------------- normalise

@pytest.mark.parametrize("case", SECTIONS["normalise"], ids=lambda c: c["name"])
def test_normalise(case):
    assert pl.normalise_case(case["text"]) == case["expected"]


# ------------------------------------------------------------------ repair

@pytest.mark.parametrize("case", SECTIONS["repair"], ids=lambda c: c["name"])
def test_repair(case):
    assert pl.repair_case(case["text"]) == case["expected"]


# --------------------------------------------------------------- prefilter

@pytest.mark.parametrize("case", SECTIONS["prefilter"], ids=lambda c: c["name"])
def test_prefilter(case):
    assert pl.prefilter_case(case["text"]) == case["expected"]


# ----------------------------------------------------------------- extract

@pytest.mark.parametrize("case", SECTIONS["extract"], ids=lambda c: c["name"])
def test_extract(case):
    assert pl.extract_case(case["text"]) == case["expected"]


# ------------------------------------------------------------------- model

@pytest.mark.parametrize("case", SECTIONS["model"], ids=lambda c: c["name"])
def test_model(case):
    tagger = pl.get_tagger()
    tags = tagger.tag_text(case["text"])
    assert tags == case["expected"]


# ------------------------------------------------------------ model_logits

@pytest.mark.parametrize("case", SECTIONS["model_logits"], ids=lambda c: c["name"])
def test_model_logits(case):
    tagger = pl.get_tagger()
    ids = tagger.encode(case["text"])
    logits = tagger.session.run(["logits"], {"ids": ids})[0][0]
    got = [[round(float(x), 5) for x in row] for row in logits]
    assert len(got) == len(case["expected"])
    for got_row, want_row in zip(got, case["expected"]):
        assert len(got_row) == len(want_row) == 37
        for g, w in zip(got_row, want_row):
            assert abs(g - w) <= pl.LOGIT_TOLERANCE, (g, w)
    # Every argmax tag exactly, per `logit_note`.
    got_tags = [tagger.tags[int(max(range(len(row)), key=row.__getitem__))]
               for row in got]
    want_tags = [tagger.tags[int(max(range(len(row)), key=row.__getitem__))]
                for row in case["expected"]]
    assert got_tags == want_tags


# ------------------------------------------------------------------- group

@pytest.mark.parametrize("case", SECTIONS["group"], ids=lambda c: c["name"])
def test_group(case):
    assert pl.group_case(case["text"], case["tags"]) == case["expected"]


# ------------------------------------------------------------------ pvalue

@pytest.mark.parametrize("case", SECTIONS["pvalue"], ids=lambda c: c["name"])
def test_pvalue(case):
    outcome = pl.pvalue_case(case["test_type"], case["statistic"], case["df1"],
                             case["df2"], case["p_operator"], case["p_text"],
                             case.get("statistic_text"))
    got = pl.pvalue_expected(outcome)
    assert got is not None, f"{case['name']}: reference now returns NaN"
    assert got["verdict"] == case["expected"]["verdict"]
    want_p = case["expected"]["computed_p"]
    got_p = got["computed_p"]
    if want_p is None:
        assert got_p is None
    else:
        assert got_p is not None
        assert not math.isnan(got_p)
        assert got_p == pytest.approx(want_p, rel=1e-9, abs=1e-12)


def test_pvalue_rounding_rule_present():
    assert DATA["pvalue_rounding_rule"] == pl.PVALUE_ROUNDING_RULE


# ----------------------------------------------------------------- pipeline

@pytest.mark.parametrize("case", SECTIONS["pipeline"], ids=lambda c: c["name"])
def test_pipeline(case):
    got = pl.pipeline_case(case["text"])
    assert len(got) == len(case["expected"])
    for g, w in zip(got, case["expected"]):
        assert g.keys() == w.keys()
        for k in g:
            if k == "computed_p" and g[k] is not None and w[k] is not None:
                assert g[k] == pytest.approx(w[k], rel=1e-9, abs=1e-12)
            else:
                assert g[k] == w[k], (case["name"], k, g[k], w[k])


def test_logit_metadata_present():
    assert DATA["logit_tolerance"] == pl.LOGIT_TOLERANCE
    assert DATA["logit_note"] == pl.LOGIT_NOTE
    assert len(SECTIONS["model_logits"]) == 3
    for case in SECTIONS["model_logits"]:
        assert len(case["text"]) <= 120
