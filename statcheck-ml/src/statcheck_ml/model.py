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
                 tags: list | None = None, unit: str = "lstm"):
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
        self.unit = unit.lower()
        self.dropout = nn.Dropout(dropout)
        if self.unit == "cnn":
            # A dilated causal-free stack: four layers with dilation 1, 2, 4, 8
            # and a fixed kernel of 5. Each layer's padding (2 * dilation) keeps
            # the sequence length exact, so no cropping is needed around the
            # convolutions. This unit has no recurrence, which makes it the
            # cheapest of the three to run and the easiest to export.
            self.lstm = None
            self.convs = nn.ModuleList()
            in_channels = embed_dim
            for dilation in (1, 2, 4, 8):
                self.convs.append(nn.Conv1d(
                    in_channels, hidden, kernel_size=5,
                    dilation=dilation, padding=2 * dilation))
                in_channels = hidden
            self.out = nn.Linear(hidden, n_tags)
        else:
            # A GRU has three gates where an LSTM has four, so it carries about
            # a quarter fewer recurrent parameters. Whether that costs accuracy
            # on this task is a question for the measurement, not for an
            # assumption.
            rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[self.unit]
            self.convs = None
            self.lstm = rnn(
                embed_dim, hidden, num_layers=layers, batch_first=True,
                bidirectional=True, dropout=dropout if layers > 1 else 0.0,
            )
            self.out = nn.Linear(hidden * 2, n_tags)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Map a batch of character ids to tag scores.

        `ids` has shape (batch, time). The output has shape (batch, time, tags).

        Padded positions are still computed. Masking them costs nothing at
        training time, because the loss ignores them, and leaving the graph
        free of packed sequences keeps the ONNX export simple.
        """
        x = self.embed(ids)
        if self.unit == "cnn":
            x = x.transpose(1, 2)                 # (batch, channels, time)
            for conv in self.convs:
                x = self.dropout(torch.relu(conv(x)))
            x = x.transpose(1, 2)                 # (batch, time, channels)
        else:
            x, _ = self.lstm(x)
            x = self.dropout(x)
        return self.out(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_report(self) -> str:
        n = self.n_parameters()
        return (f"{self.unit} model, {n:,} parameters, "
                f"{n * 4 / 1e6:.1f} MB as float32, "
                f"about {n / 1e6:.1f} MB as int8")


def make_model(vocab_size: int, n_tags: int, **kwargs) -> CharTagger:
    return CharTagger(vocab_size, n_tags, **kwargs)
