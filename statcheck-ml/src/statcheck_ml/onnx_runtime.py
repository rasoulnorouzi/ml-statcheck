"""Run an exported tagger the way a port does: ONNX graph plus a numpy Viterbi.

`pipeline/08_export.py` already runs the exported graph, but it checks parity
against the PyTorch model and so decodes with the PyTorch CRF
(`statcheck_ml.crf.CRF.decode`). That is fine for a parity check: it needs the
checkpoint anyway. It is not what a deployed port does. A port has only
`tagger.onnx`, `decoder.json` and `charmap.json`, and no PyTorch. This module
is that path: it loads exactly those three files and decodes with plain numpy,
so evaluating a model here measures what a user of the exported artefact gets.

The refactor that would let `pipeline/08_export.py` share this code is not
small: its `dev_parity` check exists to compare the ONNX graph against the
loaded checkpoint, which needs the PyTorch CRF as the reference to compare
against, not this decoder. So `08_export.py` is left as it is, and this module
is the one copy of the decode-from-`decoder.json` path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def viterbi_decode(emissions: np.ndarray, transitions: np.ndarray,
                    start: np.ndarray, end: np.ndarray) -> List[int]:
    """Best tag path for one sequence, mirroring `statcheck_ml.crf.CRF.decode`.

    `emissions` is (time, n_tags). `transitions` already carries the mask, so
    entry [i, j] is the score of tag j following tag i, or a large negative
    number when that step is forbidden.
    """
    time, n = emissions.shape
    if time == 0:
        return []
    score = start + emissions[0]
    if time == 1:
        return [int(np.argmax(score + end))]

    history = np.zeros((time - 1, n), dtype=np.int64)
    for t in range(1, time):
        nxt = score[:, None] + transitions          # (prev, next)
        best_prev = np.argmax(nxt, axis=0)           # (next,)
        best_score = nxt[best_prev, np.arange(n)]
        score = best_score + emissions[t]
        history[t - 1] = best_prev
    score = score + end

    tag = int(np.argmax(score))
    path = [tag]
    for t in range(time - 2, -1, -1):
        tag = int(history[t, tag])
        path.append(tag)
    path.reverse()
    return path


class OnnxTagger:
    """Loads `tagger.onnx` + `decoder.json` + `charmap.json` from a run directory.

    Single-threaded by default: the evaluator scores many windows through many
    runs, and one thread per run keeps a multi-run evaluation predictable on a
    CPU that a training grid may also be using.
    """

    def __init__(self, run_dir: str | Path, intra_threads: int = 1):
        import onnxruntime as ort

        run_dir = Path(run_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        self.session = ort.InferenceSession(
            str(run_dir / "tagger.onnx"), sess_options=opts,
            providers=["CPUExecutionProvider"])

        charmap = json.loads((run_dir / "charmap.json").read_text(encoding="utf-8"))
        self.vocab: dict = charmap["chars"]
        self.unk_id = charmap.get("unk_id", 1)

        decoder_path = run_dir / "decoder.json"
        decoder = json.loads(decoder_path.read_text(encoding="utf-8"))
        self.tags: List[str] = decoder["tags"]
        self.has_crf = bool(decoder.get("has_crf"))
        if self.has_crf:
            self.transitions = np.array(decoder["transitions"], dtype=np.float64)
            self.start = np.array(decoder["start"], dtype=np.float64)
            self.end = np.array(decoder["end"], dtype=np.float64)
        else:
            self.transitions = self.start = self.end = None

    def encode(self, text: str) -> np.ndarray:
        return np.array([[self.vocab.get(ch, self.unk_id) for ch in text]], dtype=np.int64)

    def tag_text(self, text: str) -> List[str]:
        if not text:
            return []
        ids = self.encode(text)
        logits = self.session.run(["logits"], {"ids": ids})[0][0]   # (time, n_tags)
        if self.has_crf:
            path = viterbi_decode(logits.astype(np.float64), self.transitions,
                                  self.start, self.end)
        else:
            path = logits.argmax(-1).tolist()
        return [self.tags[i] for i in path[:len(text)]]
