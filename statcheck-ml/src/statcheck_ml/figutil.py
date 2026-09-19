"""Shared pieces for `pipeline/10_figures.py`: the system colour map, the
training-log parser, and small file/skip helpers that do not touch
matplotlib (so importing this module never forces a plotting backend).

One colour per system lives here, once, so every figure the pipeline draws
uses the same colour for the same system instead of letting matplotlib's
default cycle assign a different one per chart.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, TypedDict

# --------------------------------------------------------------- colours --

# Fixed, not matplotlib's default cycle: every figure must reuse the same
# colour for the same system.
COLORS = {
    "statcheck_raw": "#7f7f7f",
    "statcheck_repaired": "#1f77b4",
    "cascade": "#2ca02c",
    "lstm": "#d62728",
    "gru": "#9467bd",
    "cnn": "#ff7f0e",
    "other": "#8c564b",
}

_FAMILIES = ("lstm", "gru", "cnn")


def color_for(name: str) -> str:
    """The colour for a system or run name.

    Exact keys (`statcheck_raw`, `cascade`, ...) match first; a run name such
    as `lstm-crf-s0` or `cascade_gru-crf-s0` falls back to the family it
    names. Anything else gets the shared `other` colour rather than a colour
    invented per call.
    """
    if name in COLORS:
        return COLORS[name]
    if name.startswith("cascade"):
        return COLORS["cascade"]
    if name.startswith("statcheck_raw"):
        return COLORS["statcheck_raw"]
    if name.startswith("statcheck_repaired"):
        return COLORS["statcheck_repaired"]
    for family in _FAMILIES:
        if family in name:
            return COLORS[family]
    return COLORS["other"]


def family_of(name: str) -> str:
    """The architecture family named inside a run or config name, or ''."""
    for family in _FAMILIES:
        if family in name:
            return family
    return ""


# ---------------------------------------------------------------- logs ----

class Epoch(TypedDict):
    epoch: int
    loss: float
    p: float
    r: float
    f1: float
    seconds: float


_EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s+loss\s+([0-9.]+)\s+dev\s+P\s+([0-9.]+)\s+R\s+([0-9.]+)\s+"
    r"F1\s+([0-9.]+)\s+\(([0-9.]+)s\)"
)


def parse_log(text: str) -> List[Epoch]:
    """Every `epoch N loss L dev P p R r F1 f (Ns)` line in a training log.

    Lines that do not match (banner lines such as `splits: ...` or
    `vocabulary: ...`) are skipped rather than raising.
    """
    out: List[Epoch] = []
    for line in text.splitlines():
        m = _EPOCH_RE.search(line)
        if not m:
            continue
        epoch, loss, p, r, f1, seconds = m.groups()
        out.append({"epoch": int(epoch), "loss": float(loss), "p": float(p),
                    "r": float(r), "f1": float(f1), "seconds": float(seconds)})
    return out


# ------------------------------------------------------------ figure I/O --

GRID_LOG = re.compile(r"^(lstm|gru|cnn)-(softmax|crf)-s0\.log$")


class Skip(Exception):
    """A figure could not be drawn from what is on disk."""


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise Skip(msg)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def clean_axes(ax) -> None:
    """Drop the top and right spines: one less line of chart junk."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def seed0_configs(systems: dict) -> list:
    """(config, name, holdout F1) for every `<config>-s0` run, best first."""
    out = []
    for name, s in systems.items():
        m = re.fullmatch(r"(?!cascade_)([a-z0-9]+-[a-z0-9]+(?:-noaug)?)-s0", name)
        if m and "overall" in s:
            out.append((m.group(1), name, s["overall"]["f1"]))
    return sorted(out, key=lambda t: t[2], reverse=True)


def curve_configs(eval_data, logs_dir: Path) -> list:
    """Seed-0 configs to plot: top-3 holdout F1, else lexical order over logs."""
    if eval_data is not None:
        top3 = [cfg for cfg, _, _ in seed0_configs(eval_data.get("systems", {}))[:3]]
        if top3:
            return top3
    names = sorted(p.name for p in logs_dir.glob("*-s0.log") if GRID_LOG.match(p.name))
    return [n[: -len("-s0.log")] for n in names[:3]]
