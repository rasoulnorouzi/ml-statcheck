"""A linear-chain conditional random field for the tag layer.

The research on sequence labelling is consistent that a CRF above a BiLSTM
beats a softmax, because it models which tag may follow which. On this data the
gain is expected to be small: an error analysis of the softmax model found only
23 invalid transitions among 363 errors. It is measured rather than assumed.

The CRF stays outside the exported ONNX graph. Only the emission scores are
exported, and the Viterbi decode runs in the host language. That decode is
about forty lines in JavaScript and in R, and it avoids a custom ONNX operator
that neither runtime supports.

Impossible transitions are masked to a large negative score, so the model
cannot spend capacity learning what the tag scheme already forbids. In BIOES,
`I-X` may only follow `B-X` or `I-X`, and `B-X` may only be followed by `I-X`
or `E-X`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NEG = -1e4


def transition_mask(tags: list[str]) -> torch.Tensor:
    """Build a (n_tags, n_tags) mask. Entry [i, j] allows tag i then tag j."""
    n = len(tags)
    mask = torch.zeros(n, n)

    def parts(tag):
        return ("O", None) if tag == "O" else tuple(tag.split("-", 1))

    for i, a in enumerate(tags):
        pa, ea = parts(a)
        for j, b in enumerate(tags):
            pb, eb = parts(b)
            if pb in ("B", "S", "O"):
                ok = pa in ("E", "S", "O")          # a span must have closed
            elif pb in ("I", "E"):
                ok = pa in ("B", "I") and ea == eb  # continue the same entity
            else:
                ok = True
            if not ok:
                mask[i, j] = NEG
    return mask


class CRF(nn.Module):
    """Scores a tag sequence, and finds the best one."""

    def __init__(self, tags: list[str]):
        super().__init__()
        self.n = len(tags)
        self.transitions = nn.Parameter(torch.randn(self.n, self.n) * 0.01)
        self.start = nn.Parameter(torch.randn(self.n) * 0.01)
        self.end = nn.Parameter(torch.randn(self.n) * 0.01)
        self.register_buffer("mask_matrix", transition_mask(tags))

    def _transitions(self) -> torch.Tensor:
        return self.transitions + self.mask_matrix

    def _score(self, emissions, tags, mask):
        """Score of the true path."""
        batch, time, _ = emissions.shape
        trans = self._transitions()
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, time):
            step = trans[tags[:, t - 1], tags[:, t]] + \
                emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * mask[:, t]
        last = mask.sum(1).long() - 1
        score = score + self.end[tags.gather(1, last.unsqueeze(1)).squeeze(1)]
        return score

    def _partition(self, emissions, mask):
        """Log of the sum over every path, by the forward algorithm."""
        batch, time, n = emissions.shape
        trans = self._transitions()
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, time):
            nxt = alpha.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            nxt = torch.logsumexp(nxt, dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = nxt * m + alpha * (1 - m)
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def forward(self, emissions, tags, mask):
        """Return the mean negative log likelihood over the batch."""
        mask = mask.float()
        safe = tags.clamp(min=0)
        return (self._partition(emissions, mask) - self._score(emissions, safe, mask)).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi. Returns the best tag sequence for each row."""
        batch, time, n = emissions.shape
        trans = self._transitions()
        score = self.start.unsqueeze(0) + emissions[:, 0]
        history = []
        for t in range(1, time):
            nxt = score.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            best, idx = nxt.max(dim=1)
            m = mask[:, t].unsqueeze(1).float()
            score = best * m + score * (1 - m)
            history.append(idx)
        score = score + self.end.unsqueeze(0)

        lengths = mask.sum(1).long()
        best_last = score.argmax(dim=1)
        paths = []
        for b in range(batch):
            n_steps = int(lengths[b])
            tag = int(best_last[b])
            path = [tag]
            for t in range(n_steps - 2, -1, -1):
                tag = int(history[t][b, tag])
                path.append(tag)
            paths.append(path[::-1])
        return paths
