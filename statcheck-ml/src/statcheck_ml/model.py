"""The character tagger.

A two-layer bidirectional LSTM over characters. It is small on purpose: the
`lite` tier must run in a browser and natively in R, and it must export to ONNX
without a custom operator.

The model has one output. It labels every character, and the spans follow from
the labels. A window-level class is not needed, because a window holds a result
when the model tags one. The operator is named inside the tag set, so no second
head is needed for that either.

A conditional random field is deliberately not part of the exported graph. It
does not export cleanly, and it would have to be reimplemented in JavaScript
and in R. Where the tag sequence needs tidying, `labels.tags_to_spans` already
accepts an imperfect sequence.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CharTagger(nn.Module):
    """Label each character of a window."""

    def __init__(self, vocab_size: int, n_tags: int,
                 embed_dim: int = 64, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.25,
                 tags: list | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_tags = n_tags
        # When `tags` is given, a CRF sits above the emissions. It is trained
        # and decoded in Python and never exported, so the ONNX graph still
        # returns plain emission scores.
        self.crf = None
        if tags is not None:
            from .crf import CRF
            self.crf = CRF(tags)
        # Index 0 is PAD. Its embedding stays at zero and never trains.
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, n_tags)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Map a batch of character ids to tag scores.

        `ids` has shape (batch, time). The output has shape (batch, time, tags).

        Padded positions are still computed. Masking them costs nothing at
        training time, because the loss ignores them, and leaving the graph
        free of packed sequences keeps the ONNX export simple.
        """
        x = self.embed(ids)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return self.out(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_report(self) -> str:
        n = self.n_parameters()
        return (f"{n:,} parameters, "
                f"{n * 4 / 1e6:.1f} MB as float32, "
                f"about {n / 1e6:.1f} MB as int8")


def make_model(vocab_size: int, n_tags: int, **kwargs) -> CharTagger:
    return CharTagger(vocab_size, n_tags, **kwargs)
