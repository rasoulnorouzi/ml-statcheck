"""Export a checkpoint's raw numbers to `weights.json`, for a pure-R forward pass.

`export.py` ships the graph (`tagger.onnx`) and the CRF (`decoder.json`), and a
port runs the graph through an ONNX runtime. The R port has no ONNX runtime of
its own for this project, so it needs the numbers themselves: the embedding
table, every recurrent gate's weight and bias, the output projection, and the
CRF if the run has one. `weights.json` is that file. It is self-describing on
purpose, because the only spec for the R forward pass is this file plus
`rnn_numpy.py`'s docstring: no separate design document exists for it.

Only the `gru` and `lstm` units are supported. The CNN unit has no recurrence
and is not part of the model zoo, so it is out of scope here.

Usage:
    python -m statcheck_ml.weights models/gru-crf-s0/model.pt \
        models/zoo/gru-crf/weights.json
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from .data import PAD, UNK
from .export import load as load_checkpoint
from .labels import ID_TO_TAG


def _formulas(unit: str) -> str:
    """One-line PyTorch reference for the unit, plus the two facts a port
    cannot see from the weight shapes alone: how layer 1's input is formed,
    and that dropout does nothing at inference."""
    shared = (
        "layer 1 (the second BiGRU/BiLSTM layer) reads, at each position, the "
        "concatenation of the forward and backward hidden states that layer 0 "
        "produced at that same position; dropout is the identity function at "
        "inference, so it is skipped entirely."
    )
    if unit == "gru":
        return (
            "GRU, gate order r,z,n: r = sigmoid(W_ir x + b_ir + W_hr h + b_hr); "
            "z = sigmoid(W_iz x + b_iz + W_hz h + b_hz); "
            "n = tanh(W_in x + b_in + r*(W_hn h + b_hn)); "
            "h' = (1-z)*n + z*h. " + shared
        )
    return (
        "LSTM, gate order i,f,g,o: i = sigmoid(W_ii x + b_ii + W_hi h + b_hi); "
        "f = sigmoid(W_if x + b_if + W_hf h + b_hf); "
        "g = tanh(W_ig x + b_ig + W_hg h + b_hg); "
        "o = sigmoid(W_io x + b_io + W_ho h + b_ho); "
        "c' = f*c + i*g; h' = o*tanh(c'). " + shared
    )


def _arr(t: torch.Tensor) -> dict:
    """A tensor as JSON: its shape, and its float32 values as base64.

    Decimal text would be seven times larger than the checkpoint. Base64 of
    the little-endian float32 bytes, row-major, is exact and one call to
    decode in every language: numpy `frombuffer`, R `readBin`, JavaScript
    `Float32Array` over the decoded bytes.
    """
    a = np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4")
    return {"shape": list(a.shape), "dtype": "float32", "order": "C",
            "encoding": "base64", "byte_order": "little",
            "data": base64.b64encode(a.tobytes()).decode("ascii")}



def export_weights(model_path: str) -> dict:
    """Read a checkpoint and return every number a pure-R forward pass needs.

    Reuses `export.load`, so a `weights.json` and a `tagger.onnx` built from
    the same checkpoint always agree on the unit, the CRF, and the vocabulary
    size: there is exactly one place a checkpoint is loaded from disk.
    """
    model, vocab, unit, has_crf = load_checkpoint(model_path)
    return weights_from_model(model, vocab, unit, has_crf)


def weights_from_model(model, vocab: dict, unit: str, has_crf: bool) -> dict:
    """The same numbers from a model object already in memory. Tests use this
    for a tiny synthetic model that `export.load` cannot rebuild, because a
    checkpoint does not record its hidden size."""
    if unit not in ("gru", "lstm"):
        raise ValueError(
            f"weights.json supports the gru and lstm units only, got {unit!r}; "
            "the cnn unit has no recurrence and is not part of the model zoo")

    rnn = model.lstm                    # named .lstm on CharTagger for both units
    hidden = rnn.hidden_size
    layers = rnn.num_layers
    embed_dim = model.embed.embedding_dim
    gate_order = "r,z,n" if unit == "gru" else "i,f,g,o"

    rnn_state = rnn.state_dict()
    rnn_layers = []
    for layer in range(layers):
        for direction, suffix in (("forward", ""), ("backward", "_reverse")):
            rnn_layers.append({
                "layer": layer,
                "direction": direction,
                "W_ih": _arr(rnn_state[f"weight_ih_l{layer}{suffix}"]),
                "W_hh": _arr(rnn_state[f"weight_hh_l{layer}{suffix}"]),
                "b_ih": _arr(rnn_state[f"bias_ih_l{layer}{suffix}"]),
                "b_hh": _arr(rnn_state[f"bias_hh_l{layer}{suffix}"]),
            })

    crf = None
    if has_crf and model.crf is not None:
        with torch.no_grad():
            crf = {
                "transitions": model.crf._transitions().tolist(),
                "start": model.crf.start.tolist(),
                "end": model.crf.end.tolist(),
            }

    return {
        "unit": unit,
        "layers": layers,
        "hidden": hidden,
        "embed_dim": embed_dim,
        "bidirectional": bool(rnn.bidirectional),
        "gate_order": gate_order,
        "formulas": _formulas(unit),
        "embedding": _arr(model.embed.weight),
        "rnn": rnn_layers,
        "out": {"W": _arr(model.out.weight), "b": _arr(model.out.bias)},
        "crf": crf,
        "tags": [ID_TO_TAG[i] for i in range(len(ID_TO_TAG))],
        "pad_id": vocab.get(PAD, 0),
        "unk_id": vocab.get(UNK, 1),
    }


def write_weights(model_path: str, out_path: str) -> dict:
    weights = export_weights(model_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(weights, indent=1), encoding="utf-8")
    return weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_path", help="checkpoint, e.g. models/gru-crf-s0/model.pt")
    ap.add_argument("out_path", help="where to write weights.json")
    args = ap.parse_args()
    write_weights(args.model_path, args.out_path)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
