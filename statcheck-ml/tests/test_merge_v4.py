"""The merge of the adjudicator decisions (pipeline/20_merge.py), on hand-built inputs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("merge4", ROOT / "pipeline" / "20_merge.py")
m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4)

TEXT = "As expected, t(23) = 2.45, p = .02, and F(1, 184) = 7.64, p < .01, then r = .73."
T_QUOTE = "t(23) = 2.45, p = .02"
F_QUOTE = "F(1, 184) = 7.64, p < .01"
R_QUOTE = "r = .73"
H, S, O, C = "annotator-haiku", "annotator-sonnet", "annotator-opus", "coders"


def span_of(quote):
    i = TEXT.index(quote)
    return [i, i + len(quote)]


def result(quote, **fields):
    base = {"test_type": "t", "statistic": "2.45", "df1": "23", "df2": None, "n": None,
            "p_operator": "=", "p_value": ".02", "quote": quote, "damaged": False,
            "confidence": "high"}
    if quote == F_QUOTE:
        base.update(test_type="F", statistic="7.64", df1="1", df2="184",
                    p_operator="<", p_value=".01")
    if quote == R_QUOTE:
        base.update(test_type="r", statistic=".73", df1=None, p_operator=None, p_value=None)
    base.update(fields)
    return base


def kept(quote, pending=None, tier="majority", **fields):
    rec = result(quote, **fields)
    rec.update(span=span_of(quote), voters=[H, S, O], coder_vote="span", coder_row=3,
               votes={}, tier=tier, pending=pending or {})
    return rec


def decision(did, keep=True, res=None, reason="The passage reports the result."):
    return {"dispute_id": did, "keep": keep, "result": res if keep else None, "reason": reason}


@pytest.fixture
def world(tmp_path):
    """One passage: a kept t result whose p_value is in conflict, a singleton F
    that a rater alone found, and a singleton r that the coders alone found."""
    consensus = {"passages": [{
        "window_id": "p1",
        "results": [kept(T_QUOTE, pending={"p_value": "p1-1"}, p_value=None)],
        "disputes": [
            {"dispute_id": "p1-0", "kind": "singleton", "voters": [S]},
            {"dispute_id": "p1-1", "kind": "field_conflict", "field": "p_value",
             "result": 0, "voters": [O, C, H]},
            {"dispute_id": "p1-2", "kind": "singleton", "voters": [C], "coder_row": 9},
        ],
        "dropped_coder_rows": [],
    }]}
    items = [
        {"dispute_id": "p1-0", "window_id": "p1", "text": TEXT, "kind": "singleton",
         "candidates": [result(F_QUOTE, statistic="7.46")]},
        {"dispute_id": "p1-1", "window_id": "p1", "text": TEXT, "kind": "field_conflict",
         "field": "p_value", "candidates": [result(T_QUOTE, p_value=".02"),
                                           result(T_QUOTE, p_value=".03")]},
        {"dispute_id": "p1-2", "window_id": "p1", "text": TEXT, "kind": "singleton",
         "candidates": [result(R_QUOTE)]},
    ]
    paths = {
        "consensus": tmp_path / "consensus.json",
        "disputes": tmp_path / "disputes",
        "decisions": tmp_path / "decisions",
        "out": tmp_path / "out" / "final.json",
    }
    paths["consensus"].write_text(json.dumps(consensus))
    paths["disputes"].mkdir()
    paths["decisions"].mkdir()
    (paths["disputes"] / "batch_0.json").write_text(json.dumps(items))
    good = [
        decision("p1-0", res=result(F_QUOTE)),
        decision("p1-1", res=result(T_QUOTE, p_value=".02")),
        decision("p1-2", keep=False, reason="A correlation matrix row, not a reported test."),
    ]
    (paths["decisions"] / "batch_0.json").write_text(json.dumps(good))
    paths["good"] = good
    return paths


def merge(w):
    return m4.run(w["consensus"], w["disputes"], w["decisions"], w["out"])


def final(w):
    return json.loads(w["out"].read_text())


def test_keep_a_singleton(world):
    merge(world)
    rec = final(world)["passages"][0]
    f = next(r for r in rec["results"] if r["test_type"] == "F")
    assert f["tier"] == "adjudicated"
    # The adjudicator's corrected value replaces the candidate's 7.46.
    assert f["statistic"] == "7.64"
    assert f["voters"] == [S]
    assert f["coder_vote"] is None
    assert f["span"] == span_of(F_QUOTE)
    assert f["adjudication"] == {"kind": "singleton", "fields": None, "dispute_ids": ["p1-0"]}


def test_drop_a_singleton(world):
    summary = merge(world)
    rec = final(world)["passages"][0]
    assert [r["test_type"] for r in rec["results"]] == ["t", "F"]
    assert len(rec["dropped"]) == 1
    dropped = rec["dropped"][0]
    assert dropped["dispute_id"] == "p1-2"
    assert dropped["voter"] == C
    assert dropped["reason"].startswith("A correlation matrix")
    assert dropped["candidate"]["quote"] == R_QUOTE
    assert summary["singletons_dropped"] == 1


def test_resolve_a_field_conflict(world):
    merge(world)
    t = final(world)["passages"][0]["results"][0]
    assert t["p_value"] == ".02"
    assert t["tier"] == "adjudicated"
    # The other fields and the voters stay as the consensus wrote them.
    assert t["voters"] == [H, S, O]
    assert t["coder_vote"] == "span"
    assert t["statistic"] == "2.45"
    assert t["adjudication"]["fields"] == ["p_value"]
    assert "pending" not in t and "votes" not in t


def test_a_kept_result_without_a_dispute_keeps_its_tier(world):
    consensus = json.loads(world["consensus"].read_text())
    consensus["passages"][0]["results"].append(kept(F_QUOTE, tier="unanimous"))
    world["consensus"].write_text(json.dumps(consensus))
    # F is both kept and a singleton in this hand-built case; drop the singleton.
    bad = world["good"]
    bad[0] = decision("p1-0", keep=False, reason="Already kept.")
    (world["decisions"] / "batch_0.json").write_text(json.dumps(bad))
    merge(world)
    unanimous = [r for r in final(world)["passages"][0]["results"] if r["tier"] == "unanimous"]
    assert len(unanimous) == 1 and unanimous[0]["adjudication"] is None


def test_results_come_in_reading_order(world):
    merge(world)
    spans = [r["span"][0] for r in final(world)["passages"][0]["results"]]
    assert spans == sorted(spans)


def test_a_missing_decision_file_stops(world):
    (world["decisions"] / "batch_0.json").unlink()
    with pytest.raises(SystemExit, match="batch_0.json: decision file missing"):
        merge(world)
    assert not world["out"].exists()


def test_an_unknown_dispute_id_stops(world):
    bad = world["good"]
    bad[1]["dispute_id"] = "p1-77"
    (world["decisions"] / "batch_0.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="unknown dispute_id 'p1-77'"):
        merge(world)
    assert not world["out"].exists()


def test_a_decision_file_without_a_batch_stops(world):
    (world["decisions"] / "batch_5.json").write_text("[]")
    with pytest.raises(SystemExit, match="batch_5.json: decision file without a dispute batch"):
        merge(world)


def test_an_invalid_decision_stops(world):
    bad = world["good"]
    bad[0]["reason"] = ""
    (world["decisions"] / "batch_0.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="reason is empty"):
        merge(world)


def test_a_dropped_field_conflict_stops(world):
    bad = world["good"]
    bad[1] = decision("p1-1", keep=False, reason="Not a result.")
    (world["decisions"] / "batch_0.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="never dropped"):
        merge(world)


def test_a_short_decision_file_stops(world):
    (world["decisions"] / "batch_0.json").write_text(json.dumps(world["good"][:2]))
    with pytest.raises(SystemExit, match="output length 2 != batch length 3"):
        merge(world)


def test_a_missing_consensus_stops(world):
    world["consensus"].unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        merge(world)


def test_two_runs_give_the_same_bytes(world):
    merge(world)
    first = world["out"].read_bytes()
    merge(world)
    assert world["out"].read_bytes() == first


def test_the_adjudicator_guideline_hash_is_recorded_when_present(world):
    merge(world)
    assert final(world)["inputs"]["adjudicator_guideline_sha256"] is None
    (world["decisions"] / "provenance.json").write_text(json.dumps({"guideline_sha": "ab" * 32}))
    merge(world)
    inputs = final(world)["inputs"]
    assert inputs["adjudicator_guideline_sha256"] == ["ab" * 32]
    assert len(inputs["guideline_sha256"]) == 64


def test_summary_counts(world):
    summary = merge(world)
    assert summary == {"passages": 1, "results": 2,
                       "results_by_tier": {"adjudicated": 2}, "disputes": 3,
                       "singletons_kept": 1, "singletons_dropped": 1,
                       "field_conflicts_resolved": 1}
