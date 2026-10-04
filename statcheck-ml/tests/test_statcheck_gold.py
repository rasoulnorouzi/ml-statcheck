import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

PIPE = os.path.join(os.path.dirname(__file__), "..", "pipeline")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PIPE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage = load("stage15", "15_statcheck_gold.py")
gold = load("stage14", "14_gold.py")


def test_passage_id_round_trips():
    for doc in ("Journal/Name_2014_1990.txt", "J #2/odd#name.txt"):
        assert stage.parse_passage_id(stage.passage_id(doc, 12, 40)) == (doc, 12, 40)


def test_form_b_text_equals_what_14_gold_reads(tmp_path):
    from statcheck_ml.normalize import normalize
    from statcheck_ml.repair_validated import repair_validated

    raw = ("Intro words here.\n"
           "The effect was reliable, t\x01(24) \x03 2.31, p \x01 .029, and F\x01(1, 40) \x03 5.33, p \x01 .026.\n"
           "More words.\n")
    _, fixed = stage.prepare(raw)
    expect, _ = normalize(raw)
    expect, _ = repair_validated(expect)
    assert fixed == expect
    (tmp_path / "p.txt").write_text(raw, encoding="utf-8")
    built = gold.build_paper(("p.txt", [], str(tmp_path)))
    assert built["text_sha256"] == hashlib.sha256(fixed.encode("utf-8")).hexdigest()
    assert fixed != stage.prepare(raw)[0]      # the damaged operator really was repaired


def test_passages_cover_the_candidate_lines():
    from statcheck_ml.prefilter import Prefilter

    raw = "\n".join(["plain words"] * 4 + ["t(24) = 2.31, p = .029"] + ["plain words"] * 20)
    _, fixed = stage.prepare(raw)
    units = stage.passage_units("d.txt", fixed, Prefilter())
    assert len(units) == 1
    doc, lo, hi = stage.parse_passage_id(units[0]["window_id"])
    assert doc == "d.txt" and lo <= 4 <= hi
    assert units[0]["text"] == "\n".join(fixed.split("\n")[lo:hi + 1])


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")
def test_r_runs_on_one_unit(tmp_path):
    out = str(tmp_path / "o.csv")
    rc, log, unparsed = stage.run_r("Rscript", stage.RUN_TEXT,
                                    [{"window_id": "u", "text": "t(24) = 2.31, p = .029"}], out)
    assert rc == 0 and unparsed == 0
    assert [r["window_id"] for r in stage.read_rows(out)] == ["u"]
