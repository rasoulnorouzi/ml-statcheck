"""pipeline/26_paired_v4.py: counts, McNemar, bootstrap and the stored-file cross-checks."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_v4", ROOT / "pipeline/26_paired_v4.py")
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)

FINAL = ROOT / "results/v4/final"


def row(doc, a, b, ref="consistent"):
    return {"doc": doc, "reference": ref, "systems": {"A": a, "B": b}}


def test_counts_and_mcnemar():
    rows = [row("p1", "consistent", None), row("p1", "consistent", "consistent"),
            row("p2", None, "consistent"), row("p2", None, None), row("p3", "inconsistent", None)]
    r = pv.paired(rows, "A", "B", pv.found, 0, 200)
    assert (r["a_only"], r["b_only"], r["both"], r["neither"]) == (2, 1, 1, 1)
    assert r["recall_a"] == 0.6 and r["recall_b"] == 0.4
    assert r["recall_diff"] == pytest.approx(0.2)
    assert r["mcnemar_p"] == pytest.approx(1.0)
    assert r["papers"] == 3


def test_flagged_counts_only_flag_verdicts():
    rows = [row("p", "inconsistent", "consistent"), row("p", "decision_error", "undecidable"),
            row("p", "consistent", None)]
    r = pv.paired(rows, "A", "B", pv.flagged, 0, 100)
    assert (r["a_only"], r["b_only"], r["both"]) == (2, 0, 0)


def test_identical_systems_give_zero_difference():
    rows = [row(f"p{i % 4}", "consistent" if i % 3 else None, "consistent" if i % 3 else None)
            for i in range(20)]
    r = pv.paired(rows, "A", "B", pv.found, 0, 200)
    assert r["recall_diff"] == 0 and r["recall_diff_ci95"] == [0, 0] and r["mcnemar_p"] == 1.0


def test_bootstrap_resamples_papers_and_is_seeded():
    rows = [row(f"p{i}", "consistent", None) for i in range(10)] + \
           [row(f"q{i}", None, "consistent") for i in range(2)]
    one = pv.paired(rows, "A", "B", pv.found, 3, 300)
    assert one == pv.paired(rows, "A", "B", pv.found, 3, 300)
    assert one["papers"] == 12
    assert one["recall_diff_ci95"][0] < one["recall_diff"] < one["recall_diff_ci95"][1]


def test_empty_population():
    r = pv.paired([], "A", "B", pv.found, 0, 10)
    assert r["n"] == 0 and r["recall_a"] is None and r["mcnemar_p"] == 1.0


@pytest.mark.skipif(not (FINAL / "paired.json").exists(), reason="stage not run")
def test_stored_output_matches_inputs():
    out = json.load(open(FINAL / "paired.json"))
    assert out["n_rows"] == 476 and out["seed"] == 0 and out["resamples"] == 2000
    for p in out["found"] + out["reference_flagged"]:
        assert p["a_only"] + p["b_only"] + p["both"] + p["neither"] == p["n"]
    for r in out["table"]:
        g = r["gold"]
        if g:
            # recall rebuilt from the verdict rows equals the recall in summary.json
            assert g["recall_from_rows"] == pytest.approx(g["recall_named"], abs=0.0011)
    assert out["provenance"]["inputs"]["results/v4/final/verdicts.json"] == \
        pv.sha256(FINAL / "verdicts.json")


@pytest.mark.skipif(not (FINAL / "verdicts.json").exists(), reason="stage not run")
def test_rerun_is_byte_stable(tmp_path):
    for n in ("verdicts.json", "summary.json", "bronze.json"):
        (tmp_path / n).write_bytes((FINAL / n).read_bytes())
    a = pv.render(pv.build(str(tmp_path), 0, 100))
    b = pv.render(pv.build(str(tmp_path), 0, 100))
    assert a == b
