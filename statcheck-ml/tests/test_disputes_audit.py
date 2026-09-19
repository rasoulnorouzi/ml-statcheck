"""The disputes.json audit trail round-trips: write, then read, same ids."""
import importlib.util
import json
from pathlib import Path

# Import from pipeline/05_adjudicate.py using importlib (filename starts with a digit).
spec = importlib.util.spec_from_file_location(
    "adjudicate_module", str(Path(__file__).parent.parent / "pipeline" / "05_adjudicate.py")
)
adjudicate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adjudicate_module)


def test_disputes_audit_round_trips(tmp_path):
    out_dir = tmp_path / "annotation"
    disputes_dir = out_dir / "disputes"
    disputes_dir.mkdir(parents=True)

    batch = [
        {"dispute_id": "train-abc123-0", "window_id": "abc123", "text": "t(23) = 2.45",
         "kind": "singleton", "field": None,
         "candidates": [{"test_type": "t", "statistic": "2.45", "df1": "23", "df2": None,
                          "n": None, "p_operator": "=", "p_value": ".02",
                          "quote": "t(23) = 2.45, p = .02", "damaged": False, "confidence": "high"}]},
        {"dispute_id": "train-abc123-1", "window_id": "abc123", "text": "F(2, 89) = 6.14",
         "kind": "field_conflict", "field": "p_operator",
         "candidates": [{"test_type": "F", "statistic": "6.14", "df1": "2", "df2": "89",
                          "n": None, "p_operator": "=", "p_value": ".01",
                          "quote": "F(2, 89) = 6.14, p = .01", "damaged": False, "confidence": "high"},
                         {"test_type": "F", "statistic": "6.14", "df1": "2", "df2": "89",
                          "n": None, "p_operator": "<", "p_value": ".01",
                          "quote": "F(2, 89) = 6.14, p < .01", "damaged": False, "confidence": "high"}]},
    ]
    (disputes_dir / "batch_000.json").write_text(json.dumps(batch), encoding="utf-8")

    decisions_by_id = {
        "train-abc123-0": {"dispute_id": "train-abc123-0", "keep": True,
                            "result": batch[0]["candidates"][0], "reason": "It is a result.",
                            "provenance": {"rater": "adjudicator"}},
        # train-abc123-1 deliberately has no decision, to exercise the "missing" path.
    }

    audit_path = tmp_path / "disputes.json"
    kind_counts = adjudicate_module._write_disputes_audit(
        "train", out_dir, decisions_by_id, audit_path=audit_path
    )

    assert kind_counts == {"singleton": 1, "field_conflict": 1}

    written = json.loads(audit_path.read_text(encoding="utf-8"))
    assert [d["dispute_id"] for d in written] == [d["dispute_id"] for d in batch]
    assert written[0]["decision"]["keep"] is True
    assert written[1]["decision"] is None
