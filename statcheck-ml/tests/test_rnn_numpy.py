"""`rnn_numpy.forward` must match the checkpoint it was exported from.

Two things are checked. First, a tiny random GRU and a tiny random LSTM built
directly through `CharTagger`, so the test runs without the zoo and catches a
wrong gate order or a wrong weight transpose immediately. Second, every real
zoo checkpoint (skipped where its `model.pt` is missing), on real dev windows
batched the way `pipeline/08_export.py` batches them: mixed lengths, batch
size 8, so padding is actually exercised.

The comparison is against the checkpoint run *without* padding, one window at
a time — exactly how every deployed port ever calls it. As
`rnn_numpy`'s module docstring explains, the checkpoint's own padded forward
pass (`model(ids)` on the padded batch directly) does not match that, because
`CharTagger` never packs its sequences and the backward direction picks up
whatever the padding steps leave behind. `forward()` is written to match the
no-padding answer even when fed a padded batch, so that is what it is tested
against.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from statcheck_ml.data import apply_splits, load_jsonl, load_splits, row_to_example
from statcheck_ml.export import load as load_checkpoint
from statcheck_ml.labels import TAG_TO_ID
from statcheck_ml.model import CharTagger
from statcheck_ml.rnn_numpy import forward
from statcheck_ml.train import make_batches
from statcheck_ml.weights import export_weights, weights_from_model

REPO = Path(__file__).resolve().parents[1]           # statcheck-ml
DIFF_TOLERANCE = 1e-3   # float32 torch vs float64 numpy: the GRU drifts ~2e-4 over a long window (same as the ONNX parity bound); every argmax tag must still agree

# zoo config name -> the run its model.pt lives under
ZOO_CONFIGS = [
    ("gru-crf", "gru-crf-s0"),
    ("gru-softmax", "gru-softmax-s0"),
    ("lstm-softmax", "lstm-softmax-s0"),
]


def _check_batch(model, weights, ids: torch.Tensor, rows) -> tuple:
    """Return (max abs diff, all-argmax-agree) on the real positions of one
    padded batch, comparing `forward()` against the checkpoint run unpadded,
    one row at a time."""
    numpy_logits = forward(weights, ids.numpy())
    worst = 0.0
    all_agree = True
    for j, row in enumerate(rows):
        n = len(row["text"])
        with torch.no_grad():
            torch_row = model(ids[j:j + 1, :n]).numpy()[0]     # (n, tags), unpadded
        numpy_row = numpy_logits[j, :n]
        worst = max(worst, float(np.abs(torch_row - numpy_row).max()))
        if not np.array_equal(torch_row.argmax(-1), numpy_row.argmax(-1)):
            all_agree = False
    return worst, all_agree


@pytest.mark.parametrize("unit", ["gru", "lstm"])
def test_synthetic_matches_torch(tmp_path, unit):
    """A tiny random model, so the gate math is checked without the zoo."""
    torch.manual_seed(0)
    vocab_size, hidden, layers = 10, 4, 2
    vocab = {chr(i): i for i in range(vocab_size)}
    model = CharTagger(vocab_size, len(TAG_TO_ID), hidden=hidden, layers=layers,
                       unit=unit, tags=None)
    model.eval()

    # export.load rebuilds a model at the default size, which a hidden of 4
    # cannot fill, so the tiny model is exported from the object directly.
    weights = weights_from_model(model, vocab, unit, has_crf=False)
    assert weights["unit"] == unit
    assert weights["hidden"] == hidden
    assert weights["layers"] == layers
    assert weights["crf"] is None

    # Three rows of different lengths, padded to the longest, so the backward
    # mask is actually exercised.
    lengths = [3, 7, 5]
    width = max(lengths)
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    for j, n in enumerate(lengths):
        ids[j, :n] = torch.randint(2, vocab_size, (n,), generator=g)

    worst, agree = _check_batch(model, weights, ids,
                                [{"text": "x" * n} for n in lengths])
    assert worst < DIFF_TOLERANCE, f"{unit}: max abs diff {worst:.2e}"
    assert agree, f"{unit}: argmax tags disagree on a real position"


@pytest.mark.parametrize("zoo_config,run_name", ZOO_CONFIGS)
def test_zoo_checkpoint_matches_torch(zoo_config, run_name):
    model_path = REPO / "models" / run_name / "model.pt"
    if not model_path.exists():
        pytest.skip(f"{model_path} missing")

    model, vocab, unit, has_crf = load_checkpoint(str(model_path))
    weights = export_weights(str(model_path))
    assert weights["unit"] == unit
    assert (weights["crf"] is not None) == has_crf

    rows_all = load_jsonl(str(REPO / "dataset" / "train.jsonl"))
    examples = [row_to_example(r) for r in rows_all]
    splits = load_splits(str(REPO / "dataset" / "splits.json"))
    _, dev_examples = apply_splits(examples, splits)
    dev_examples = dev_examples[:20]
    assert len(dev_examples) > 0, "no dev windows to test against"

    worst = 0.0
    all_agree = True
    saw_padding = False
    for ids, _, rows in make_batches(dev_examples, vocab, batch_size=8, shuffle=False):
        lengths = [len(r["text"]) for r in rows]
        if max(lengths) > min(lengths):
            saw_padding = True
        diff, agree = _check_batch(model, weights, ids, rows)
        worst = max(worst, diff)
        all_agree = all_agree and agree

    assert saw_padding, "the 20-window sample never exercised padding"
    assert worst < DIFF_TOLERANCE, f"{zoo_config}: max abs diff {worst:.2e}"
    assert all_agree, f"{zoo_config}: argmax tags disagree on a real position"
