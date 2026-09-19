import json

from statcheck_ml.figutil import COLORS, load_history

def test_color_map_has_every_required_system():
    for name in ("statcheck_raw", "statcheck_repaired", "cascade", "lstm", "gru", "cnn"):
        assert name in COLORS
        assert isinstance(COLORS[name], str) and COLORS[name].startswith("#")


def test_load_history_reads_report_json(tmp_path):
    run = tmp_path / "gru-crf-s0"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"history": [
        {"epoch": 1, "loss": 0.4, "precision": 0.68, "recall": 0.86, "f1": 0.76},
        {"epoch": 2, "loss": 0.07, "precision": 0.72, "recall": 0.92, "f1": 0.80}]}),
        encoding="utf-8")
    epochs = load_history(run)
    assert [e["epoch"] for e in epochs] == [1, 2]
    assert epochs[1] == {"epoch": 2, "loss": 0.07, "p": 0.72, "r": 0.92, "f1": 0.80}
    assert load_history(tmp_path / "missing") == []
