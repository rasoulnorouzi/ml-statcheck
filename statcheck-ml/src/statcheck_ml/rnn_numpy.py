"""A pure-numpy forward pass over `weights.json`, and the spec the R port mirrors.

This module exists because the R port has no `nn.GRU`/`nn.LSTM` and no ONNX
runtime. `weights.json` (written by `weights.py`) holds the embedding table,
every recurrent gate's weight and bias, and the output projection, in
PyTorch's own layout. `forward()` below turns a batch of character ids into
tag logits using only those numbers and matrix arithmetic, and it is written
to be reimplemented line for line in R: every step below is one R port
mirrors, in the same order, with the same shapes.

Architecture (fixed, from `CharTagger` in `model.py`):
    embedding lookup -> 2 stacked bidirectional layers -> linear(256, 37)
Layer 1 reads, at each position, the concatenation of the forward and
backward hidden states layer 0 produced at that position. Dropout is the
identity function at inference and is not implemented here at all.

Gate formulas (also stated in `weights.json["formulas"]`), in PyTorch's own
gate order:
    GRU  (order r, z, n):
        r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
        z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
        n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
        h' = (1 - z) * n + z * h
    LSTM (order i, f, g, o):
        i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
        f = sigmoid(W_if x + b_if + W_hf h + b_hf)
        g = tanh(W_ig x + b_ig + W_hg h + b_hg)
        o = sigmoid(W_io x + b_io + W_ho h + b_ho)
        c' = f * c + i * g
        h' = o * tanh(c')
`W_ih` has shape (gate*hidden, input_size); `W_hh` has shape (gate*hidden,
hidden). A step is `x @ W_ih.T + b_ih` plus `h @ W_hh.T + b_hh`, computed for
the whole batch at once, one matrix product per timestep, never one sample at
a time.

Padding, and why the backward direction needs a mask
------------------------------------------------------
`CharTagger.forward` never packs the sequence: it runs `nn.GRU`/`nn.LSTM`
directly over a batch padded with id 0 at the tail of the shorter rows, the
same way `train.make_batches` builds a batch. The forward direction is causal,
so padding after a real position never reaches it and needs no special
handling. The backward direction is not causal: it starts at the last column
of the batch and walks left, so at the last real position it arrives carrying
whatever hidden state came from processing every padding step beyond that
position. PyTorch does not reset that state — a single padding column moves
the checkpoint's own logits by double digits at the last real position — so a
naive backward pass over a padded batch does not match what a length-aware
model computes, and it does not match what this project ever actually runs:
every deployed inference (the ONNX graph, `pipeline.Pipeline`, the R port) is
called with one window at its exact length, batch size 1, no padding at all.

`forward()` reproduces that exact, no-padding computation even when it is
given a padded, mixed-length batch, so that batching windows together for
speed never changes a single window's answer. It does this with a mask, not a
per-sample Python loop: at every backward timestep t, compute the candidate
hidden state as usual from the previous step, then multiply it by a (batch,)
mask that is 1 where `ids[:, t] != pad_id` and 0 where it is padding. A
padding column's hidden state (and, for the LSTM, its cell state) is thereby
forced to exactly zero regardless of the candidate. Walking one more step
left, from a padding column to the last real column, the incoming hidden
state is therefore exactly zero — precisely the initial state a fresh,
unpadded backward pass would start from at that position. The same mask,
built once from the original input ids, is reused unchanged at every layer:
padding never moves and never needs recomputing layer to layer.

The forward direction is never masked. It needs none: it depends only on
positions at or before it, and padding always sits after the real text.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

import base64


def tensor(entry: dict) -> np.ndarray:
    """Decode one tensor entry of weights.json: base64 of little-endian
    float32 bytes in row-major order, to a float64 array of `shape`. This
    module needs numpy only; torch is not imported."""
    raw = base64.b64decode(entry["data"])
    return np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(entry["shape"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _split_gates(g: np.ndarray, n_gates: int) -> list:
    """Split the last axis of `g` (..., n_gates * hidden) into `n_gates` equal
    chunks, in PyTorch's own packed order."""
    return np.split(g, n_gates, axis=-1)


def _rnn_layer(x: np.ndarray, mask: np.ndarray, unit: str, hidden: int,
              fwd: dict, bwd: dict) -> np.ndarray:
    """One bidirectional layer. `x` is (batch, time, in). `mask` is (batch,
    time), 1.0 for a real position and 0.0 for padding. Returns (batch, time,
    2*hidden): the forward output concatenated with the backward output.
    """
    batch, time, _ = x.shape
    n_gates = 3 if unit == "gru" else 4

    def step(x_t, h, c, weights):
        gi = x_t @ weights["W_ih"].T + weights["b_ih"]
        gh = h @ weights["W_hh"].T + weights["b_hh"]
        if unit == "gru":
            i_r, i_z, i_n = _split_gates(gi, 3)
            h_r, h_z, h_n = _split_gates(gh, 3)
            r = _sigmoid(i_r + h_r)
            z = _sigmoid(i_z + h_z)
            n = np.tanh(i_n + r * h_n)
            h_new = (1.0 - z) * n + z * h
            return h_new, c
        i_i, i_f, i_g, i_o = _split_gates(gi, 4)
        h_i, h_f, h_g, h_o = _split_gates(gh, 4)
        i = _sigmoid(i_i + h_i)
        f = _sigmoid(i_f + h_f)
        g = np.tanh(i_g + h_g)
        o = _sigmoid(i_o + h_o)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    # Forward direction: causal, so it runs over the raw padded input with no
    # masking. Padding, which always sits after the real text, never reaches
    # a real position's forward state.
    h = np.zeros((batch, hidden), dtype=x.dtype)
    c = np.zeros((batch, hidden), dtype=x.dtype)
    fwd_out = np.empty((batch, time, hidden), dtype=x.dtype)
    for t in range(time):
        h, c = step(x[:, t, :], h, c, fwd)
        fwd_out[:, t, :] = h

    # Backward direction: walks right to left, so it is not causal and needs
    # the mask described in the module docstring.
    h = np.zeros((batch, hidden), dtype=x.dtype)
    c = np.zeros((batch, hidden), dtype=x.dtype)
    bwd_out = np.empty((batch, time, hidden), dtype=x.dtype)
    for t in range(time - 1, -1, -1):
        h, c = step(x[:, t, :], h, c, bwd)
        m = mask[:, t][:, None]
        h = h * m
        c = c * m
        bwd_out[:, t, :] = h

    return np.concatenate([fwd_out, bwd_out], axis=-1)


def forward(weights: Dict, ids: np.ndarray) -> np.ndarray:
    """Map a batch of character ids to tag logits, matching `CharTagger`.

    `ids` is `(batch, time)` of int64. Padding, where present, must be
    `weights["pad_id"]` at the tail of a row (as `train.make_batches` and the
    R port both build it). Returns `(batch, time, len(weights["tags"]))`.

    Every position is computed, padded ones included, the same way
    `CharTagger.forward` computes them — but the backward direction is masked
    so that a real position's answer never depends on how much padding
    follows it. See the module docstring for why.
    """
    ids = np.asarray(ids, dtype=np.int64)
    unit = weights["unit"]
    hidden = weights["hidden"]
    layers = weights["layers"]
    pad_id = weights["pad_id"]

    embedding = tensor(weights["embedding"])
    x = embedding[ids]                              # (batch, time, embed_dim)
    mask = (ids != pad_id).astype(np.float64)        # (batch, time)

    by_layer: Dict[int, Dict[str, dict]] = {}
    for entry in weights["rnn"]:
        layer_weights = by_layer.setdefault(entry["layer"], {})
        layer_weights[entry["direction"]] = {
            "W_ih": tensor(entry["W_ih"]),
            "W_hh": tensor(entry["W_hh"]),
            "b_ih": tensor(entry["b_ih"]),
            "b_hh": tensor(entry["b_hh"]),
        }

    for layer in range(layers):
        fwd = by_layer[layer]["forward"]
        bwd = by_layer[layer]["backward"]
        x = _rnn_layer(x, mask, unit, hidden, fwd, bwd)

    w_out = tensor(weights["out"]["W"])
    b_out = tensor(weights["out"]["b"])
    logits = x @ w_out.T + b_out                      # (batch, time, n_tags)
    return logits
