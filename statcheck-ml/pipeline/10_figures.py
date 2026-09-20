"""Draw the version-2 report figures from files other stages already wrote.

Every number plotted comes from `results/eval.json`, `dataset/agreement`,
`models/runs.json`, `models/export.json`, or a training log, never typed
here. A figure whose input is missing is skipped; the rest still run.

Usage:
  python pipeline/10_figures.py --eval results/eval.json \\
      --agreement dataset/agreement --runs models/runs.json \\
      --export models/export.json --logs models --out results/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from statcheck_ml.figutil import (COLORS, Skip, clean_axes, color_for, curve_configs, load_history,
                                  load_json, need, seed0_configs)

DPI = 150
META = {"Software": None, "Creation Time": None}


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, metadata=META)
    plt.close(fig)
    return path


def fig_agreement(agreement_dir: Path, out: Path) -> Path:
    """1: rater-pair strict-F1 heatmap (train) and F1-vs-final bars (both sets)."""
    train = load_json(agreement_dir / "train.json")
    holdout = load_json(agreement_dir / "holdout.json")
    need(train is not None, f"{agreement_dir / 'train.json'} not found")
    need(holdout is not None, f"{agreement_dir / 'holdout.json'} not found")

    raters = train["raters"]
    pairwise = train["result"]["strict"]["pairwise"]
    mat = np.eye(len(raters))
    for i, a in enumerate(raters):
        for j, b in enumerate(raters):
            key = f"{a}-{b}" if f"{a}-{b}" in pairwise else f"{b}-{a}"
            if i != j and key in pairwise:
                mat[i, j] = pairwise[key]["value"]

    vmin = min(v["value"] for v in pairwise.values()) - 0.02
    vmax = 1.0
    threshold = vmin + 0.6 * (vmax - vmin)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
    im = left.imshow(mat, vmin=vmin, vmax=vmax, cmap="Blues")
    left.set_xticks(range(len(raters)), raters)
    left.set_yticks(range(len(raters)), raters)
    for i in range(len(raters)):
        for j in range(len(raters)):
            left.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                      color="white" if mat[i, j] > threshold else "black")
    left.set_title("Pairwise strict span F1 (train)")
    fig.colorbar(im, ax=left, fraction=0.046, pad=0.04, label="F1")

    x = np.arange(len(raters))
    width = 0.35
    train_bars = right.bar(x - width / 2, [train["result"]["vs_final"][r]["f1"] for r in raters],
                           width, label="train", color="#1f77b4")
    holdout_bars = right.bar(x + width / 2,
                             [holdout["result"]["vs_final"][r]["f1"] for r in raters],
                             width, label="holdout", color="#ff7f0e")
    right.bar_label(train_bars, fmt="%.3f", padding=2, fontsize=8)
    right.bar_label(holdout_bars, fmt="%.3f", padding=2, fontsize=8)
    right.set_xticks(x, raters)
    right.set_ylabel("F1 vs. final label (lenient span)")
    right.set_title("Per-rater agreement with the adjudicated label")
    right.set_ylim(0.90, 1.02)
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    clean_axes(right)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return save(fig, out / "agreement_heatmap.png")


def fig_benchmark(eval_data: dict, out: Path) -> Path:
    """2: holdout F1 per model config, seed 0, with CI, seed 1-2 dots where they exist."""
    need(eval_data is not None, "eval.json not found")
    systems = eval_data.get("systems", {})
    configs = seed0_configs(systems)
    need(bool(configs), "no seed-0 model runs in eval.json systems")
    need("statcheck_repaired" in systems, "statcheck_repaired missing from eval.json")

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.1), 5))
    x = np.arange(len(configs))
    f1s = [f for _, _, f in configs]
    lo, hi = [], []
    for cfg, name, f1 in configs:
        ci = systems[name].get("bootstrap", {}).get("overall", {}).get("f1", (f1, f1))
        lo.append(f1 - ci[0])
        hi.append(ci[1] - f1)
    ax.bar(x, f1s, yerr=[lo, hi], capsize=4, color=[color_for(c) for c, _, _ in configs])

    # Seeds 1 and 2 exist only for the grid's top three by dev F1, which is
    # not the top three by holdout F1 that orders this chart.
    for i, (cfg, _, _) in enumerate(configs):
        for seed in (1, 2):
            other = f"{cfg}-s{seed}"
            if other in systems:
                ax.scatter([i], [systems[other]["overall"]["f1"]], color="black",
                          zorder=3, marker="o", s=30)

    ref = systems["statcheck_repaired"]["overall"]["f1"]
    ax.axhline(ref, color=COLORS["statcheck_repaired"], linestyle="--",
              label=f"statcheck_repaired ({ref:.3f})")
    ax.set_xticks(x, [c for c, _, _ in configs], rotation=30, ha="right")
    ax.set_ylabel("Holdout overall F1")
    ax.set_title("Holdout F1 by model configuration, seed 0 (dots: seeds 1 and 2)")
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    clean_axes(ax)
    fig.tight_layout()
    return save(fig, out / "benchmark_f1.png")


def fig_family_recall(eval_data: dict, out: Path) -> Path:
    """3: recall by damage family for statcheck_repaired, best model, cascade."""
    need(eval_data is not None, "eval.json not found")
    fam_recall = eval_data.get("family_recall", {})
    need(bool(fam_recall), "eval.json family_recall is empty (no models scored)")

    groups = [g for g in ("statcheck_repaired", "best_model", "cascade") if g in fam_recall]
    families = sorted({fam for g in groups for fam in fam_recall[g]})
    n_of = {fam: next(fam_recall[g][fam]["n"] for g in groups if fam in fam_recall[g])
           for fam in families}
    group_color = {"statcheck_repaired": COLORS["statcheck_repaired"],
                  "cascade": COLORS["cascade"],
                  "best_model": color_for(eval_data.get("best_model") or "best_model")}

    fig, ax = plt.subplots(figsize=(max(8, len(families) * 1.3), 5.5))
    x = np.arange(len(families))
    width = 0.8 / len(groups)
    for i, g in enumerate(groups):
        r = [fam_recall[g].get(fam, {"r": 0.0}).get("r", 0.0) for fam in families]
        wl = [fam_recall[g].get(fam, {"wilson": [0.0, 0.0]})["wilson"] for fam in families]
        lo = [ri - w[0] for ri, w in zip(r, wl)]
        hi = [w[1] - ri for ri, w in zip(r, wl)]
        ax.bar(x + i * width - 0.4 + width / 2, r, width, yerr=[lo, hi], capsize=3,
              label=g, color=group_color[g])

    ax.set_xticks(x, [f"{f}\n(n={n_of[f]})" for f in families], rotation=30, ha="right",
                 fontsize=8)
    ax.set_ylabel("Recall")
    ax.set_title("Recall by damage family (Wilson 95% CI)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    return save(fig, out / "family_recall.png")


def fig_precision_recall(eval_data: dict, out: Path) -> Path:
    """4: precision vs. recall per system, holdout overall, with iso-F1 curves."""
    need(eval_data is not None, "eval.json not found")
    systems = eval_data.get("systems", {})
    need(bool(systems), "eval.json has no systems")

    fig, ax = plt.subplots(figsize=(7, 7))
    rr = np.linspace(0.01, 1, 200)
    for target in (0.5, 0.7, 0.9):
        pp = target * rr / (2 * rr - target)
        ax.plot(rr, np.where((pp > 0) & (pp <= 1), pp, np.nan), color="#cccccc",
               linewidth=1, zorder=0)
        ax.annotate(f"F1={target}", (0.98, target * 0.98 / (2 * 0.98 - target)),
                   color="#999999", fontsize=8)

    # The main axes show the whole range, where the two statcheck systems sit
    # far from the models. The models crowd one corner, so an inset in the
    # empty lower left magnifies it. Seeds 1 and 2 are drawn without labels.
    def is_seed0(name):
        return not (name[-3:-1] == "-s" and name[-1] in "12")

    def draw(axis, label):
        for name, s in systems.items():
            o = s.get("overall", {})
            if "p" in o and "r" in o:
                axis.scatter([o["r"]], [o["p"]], color=color_for(name),
                             s=60 if is_seed0(name) else 25, zorder=3,
                             alpha=1.0 if is_seed0(name) else 0.6)
                if label and is_seed0(name):
                    # The cascade sits on top of its model; its label goes below.
                    offset = (4, -9) if name.startswith("cascade") else (4, 3)
                    axis.annotate(name, (o["r"], o["p"]), fontsize=7, xytext=offset,
                                  textcoords="offset points")

    draw(ax, label=False)
    for name in ("statcheck_raw", "statcheck_repaired"):
        o = systems.get(name, {}).get("overall", {})
        if "p" in o and "r" in o:
            ax.annotate(name, (o["r"], o["p"]), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")

    inset = ax.inset_axes([0.08, 0.08, 0.55, 0.55])
    draw(inset, label=True)
    inset.set_xlim(0.70, 0.92)
    inset.set_ylim(0.80, 1.005)
    inset.set_title("models, magnified", fontsize=8)
    inset.tick_params(labelsize=7)
    ax.indicate_inset_zoom(inset, edgecolor="#999999")

    ax.set_xlabel("Recall (holdout, overall)")
    ax.set_ylabel("Precision (holdout, overall)")
    ax.set_title("Precision vs. recall by system, with iso-F1 curves")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    clean_axes(ax)
    fig.tight_layout()
    return save(fig, out / "precision_recall.png")


def fig_zoo(eval_data: dict, export: list, out: Path) -> Path:
    """5: ONNX size vs. holdout F1, marker area by latency, in_zoo highlighted."""
    need(eval_data is not None, "eval.json not found")
    need(export is not None, "export.json not found")
    systems = eval_data.get("systems", {})
    points = [(r, systems[r["name"]]["overall"]["f1"]) for r in export if r["name"] in systems]
    need(bool(points), "no export.json run names match eval.json systems")

    fig, ax = plt.subplots(figsize=(8, 6))
    for record, f1 in points:
        size_mb = record["onnx_bytes"] / 1e6
        latency = record.get("latency_ms_median") or 1.0
        in_zoo = bool(record.get("in_zoo"))
        ax.scatter([size_mb], [f1], s=max(20, latency * 20), color=color_for(record["name"]),
                  edgecolors="black" if in_zoo else "none", linewidths=1.5 if in_zoo else 0,
                  alpha=0.85)
        ax.annotate(record["name"], (size_mb, f1), fontsize=8, xytext=(5, 5),
                   textcoords="offset points")

    ax.set_xlabel("ONNX size (MB)")
    ax.set_ylabel("Holdout overall F1")
    ax.set_title("Accuracy vs. size (marker area scales with latency in ms; "
                "black edge = in zoo)")
    clean_axes(ax)
    fig.tight_layout()
    return save(fig, out / "zoo_size_latency.png")


def fig_learning_curves(eval_data: dict, logs_dir: Path, out: Path) -> Path:
    """6: dev F1 per epoch, top-3 configs seed 0, from each run's report.json."""
    need(logs_dir.exists(), f"{logs_dir} not found")
    configs = curve_configs(eval_data, logs_dir)
    need(bool(configs), "no seed-0 run directory with a report.json found")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    drawn = 0
    for cfg in configs:
        epochs = load_history(logs_dir / f"{cfg}-s0")
        if epochs:
            # Colour is the family; the head is the line style, so two heads of
            # one family stay apart.
            ax.plot([e["epoch"] for e in epochs], [e["f1"] for e in epochs], label=cfg,
                   color=color_for(cfg), marker=".",
                   linestyle="-" if cfg.endswith("-crf") else "--")
            drawn += 1
    need(drawn > 0, "no report.json had a history")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev F1")
    ax.set_title("Learning curves, seed 0, top-3 configs")
    ax.legend()
    clean_axes(ax)
    fig.tight_layout()
    return save(fig, out / "learning_curves.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default="results/eval.json")
    ap.add_argument("--agreement", default="dataset/agreement")
    ap.add_argument("--runs", default="models/runs.json")
    ap.add_argument("--export", default="models/export.json")
    ap.add_argument("--logs", default="models")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else REPO / path

    out = resolve(args.out)
    eval_data = load_json(resolve(args.eval))
    export = load_json(resolve(args.export))
    logs_dir = resolve(args.logs)

    jobs = [
        ("agreement_heatmap", lambda: fig_agreement(resolve(args.agreement), out)),
        ("benchmark_f1", lambda: fig_benchmark(eval_data, out)),
        ("family_recall", lambda: fig_family_recall(eval_data, out)),
        ("precision_recall", lambda: fig_precision_recall(eval_data, out)),
        ("zoo_size_latency", lambda: fig_zoo(eval_data, export, out)),
        ("learning_curves", lambda: fig_learning_curves(eval_data, logs_dir, out)),
    ]

    produced = []
    for label, job in jobs:
        try:
            path = job()
        except Skip as exc:
            print(f"skipped {label}: {exc}")
            continue
        produced.append(label)
        print(f"wrote {path}")

    if "agreement_heatmap" not in produced:
        print("agreement_heatmap could not be produced; nothing else guarantees a figure exists")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
