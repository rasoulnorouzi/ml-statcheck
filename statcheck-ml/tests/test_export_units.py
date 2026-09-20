"""Every unit exports through statcheck_ml.export.export, end to end.

The version 2 grid export died on the first recurrent run: torch 2.14 selects
the dynamo exporter by default, which freezes the time axis of a GRU or LSTM.
No test had run export() on a recurrent checkpoint under the pinned torch, so
this one runs all three units, with and without a CRF, at a tiny size.
"""
import pytest

torch = pytest.importorskip("torch")  # only in the `train` extra, not `test`

from statcheck_ml.export import export
from statcheck_ml.labels import TAG_TO_ID
from statcheck_ml.model import CharTagger


def _checkpoint(tmp_path, unit, crf):
    vocab = {chr(i): i for i in range(40)}
    tags = list(TAG_TO_ID) if crf else None
    model = CharTagger(len(vocab), len(TAG_TO_ID), tags=tags, unit=unit)
    path = tmp_path / f"{unit}-{'crf' if crf else 'softmax'}.pt"
    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "tags": list(TAG_TO_ID), "unit": unit, "crf": crf}, path)
    return path


@pytest.mark.parametrize("unit", ["lstm", "gru", "cnn"])
@pytest.mark.parametrize("crf", [False, True])
def test_export_keeps_the_time_axis_free(tmp_path, unit, crf):
    ckpt = _checkpoint(tmp_path, unit, crf)
    report = export(str(ckpt), str(tmp_path / "t.onnx"), check_lengths=(16, 283, 400))
    assert report["unit"] == unit
    assert report["has_crf"] is crf
    assert (tmp_path / "t.onnx").exists()
    assert report["max_difference"] < 1e-4
    assert all(c["same_tag_share"] == 1.0 for c in report["checks"])
