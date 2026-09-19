"""End-to-end smoke test of one training run.

A run must end with a report.json next to model.pt. The version 2 grid
found a NameError at the very end of train(), after 30 epochs of work,
because no test ran train() to completion. This one does, on 40 windows
for one epoch, in a few seconds.
"""
import json
from pathlib import Path

from statcheck_ml.data import load_jsonl
from statcheck_ml.splits import make_splits
from statcheck_ml.train import train

ROWS = Path(__file__).resolve().parents[1] / "dataset" / "train.jsonl"


def test_train_writes_report(tmp_path):
    rows = [r for r in load_jsonl(str(ROWS)) if r["results"]][:40]
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(make_splits([r["source_doc"] for r in rows],
                                             dev_share=0.25, seed=0)), encoding="utf-8")
    out = tmp_path / "run"
    report = train([str(data)], str(out), epochs=1, batch_size=8, seed=0,
                   use_crf=False, augment=0, hard_negatives=0, unit="cnn",
                   splits_path=str(splits))
    assert (out / "model.pt").exists()
    assert (out / "report.json").exists()
    assert (out / "charmap.json").exists()
    assert report["best_epoch"] == 1
    assert "f1" in report["dev"]["overall"]
