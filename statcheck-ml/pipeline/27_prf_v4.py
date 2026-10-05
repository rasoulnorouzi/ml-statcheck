"""Precision, recall and F1 of every version 4 model on the gold test, the bronze holdout,
and the two pooled.

Reads only stored results (results/v4/final/gold/*.json, bronze.json); it opens no paper
and no model, so it is not a new read of the gold test. All figures are the model alone,
because the bronze holdout scores the model alone.

Gold precision needs a count of wrong finds, and the gold set gives only a lower bound:
the coders followed statcheck's scope, so a find outside gold is not always wrong.
  strict  every find outside gold is wrong: TP / (TP + finds outside gold in papers with
          results + finds in papers with no result)
  scope   the same, without the correlations outside gold in papers with results, which
          statcheck's scope leaves out when they carry no degrees of freedom
The "mix" pools the counts of the gold test (scope precision) and the bronze holdout. Its
two halves have different labels (human, machine) and different units (whole papers,
windows); it is given because the owner asked for it, beside the two, never instead.

Usage: python pipeline/27_prf_v4.py [--final results/v4/final]
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "old-gru-crf-s0"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def run_scores(gold_path, bronze):
    g = json.loads(gold_path.read_text())["gold_test"]
    m = g["model"]
    tp, fn = m["found"], g["gold_in_scope"] - m["found"]
    out_named, out_r = m["finds_not_in_gold"]["named"], m["finds_not_in_gold"]["r"]
    none = m["finds_in_papers_without_results"]
    b = bronze["overall"]
    gold_scope = prf(tp, out_named + none, fn)
    return {"gold_strict": prf(tp, out_named + out_r + none, fn),
            "gold": gold_scope,
            "bronze": prf(b["tp"], b["fp"], b["fn"]),
            "mix": prf(tp + b["tp"], gold_scope["fp"] + b["fp"], fn + b["fn"])}


def group(name):
    if name == "gate-none":
        return "gru-crf", "none"
    if name == "gate-old+all":
        return "gru-crf", "all"
    if name.startswith("gate-") or name == OLD:
        return None
    cfg, noise, _ = name.rsplit("-", 2)
    return cfg, noise


def mean_scores(rows):
    return {part: {k: statistics.fmean(r[part][k] for r in rows) for k in ("p", "r", "f1")}
            for part in rows[0]}


def write_md(path, out):
    f = lambda x: f"{x:.3f}"
    cols = ("gold", "bronze", "mix")
    head = ("| {} | human gold P | R | F1 | AI bronze P | R | F1 | mix P | R | F1 | gold P strict |",
            "|---|" + "---|" * 10)

    def line(label, s):
        return (f"| {label} | " + " | ".join(f(s[c][k]) for c in cols for k in ("p", "r", "f1"))
                + f" | {f(s['gold_strict']['p'])} |")

    md = ["# Version 4: precision, recall and F1 by test set", "",
          "Model alone. Gold test: 169 papers, 476 results, human labels. Bronze holdout: "
          "576 windows, machine labels. Mix: the two pooled by counts. Read after the choice "
          "from stored results; no new read of the gold test.", "",
          "Gold P is the scope precision: finds outside gold count as wrong, except "
          "correlations in papers with results. Gold P strict counts every find outside gold "
          "as wrong. Both are lower bounds, because the coders followed statcheck's scope.", "",
          "## Means over seeds 0 to 4", "", head[0].format("model, noise"), head[1]]
    for key, s in out["means"].items():
        md.append(line(key.replace("|", ", "), s))
    md.append(line("gru-crf-s0, shipped (old)", out["runs"][OLD]))
    md += ["", "## Noise families, gru-crf seed 0", "", head[0].format("run"), head[1]]
    for name, s in out["runs"].items():
        if name.startswith("gate-"):
            md.append(line(name, s))
    md += ["", "## Every run", "", head[0].format("run"), head[1]]
    for name, s in out["runs"].items():
        md.append(line(name, s))
    Path(path).write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", default="results/v4/final")
    args = ap.parse_args()
    final = ROOT / args.final
    bronze = json.loads((final / "bronze.json").read_text())["systems"]
    runs = {}
    for path in sorted((final / "gold").glob("*.json")):
        runs[path.stem] = run_scores(path, bronze[path.stem])
    groups = {}
    for name, s in runs.items():
        g = group(name)
        if g:
            groups.setdefault(f"{g[0]}|{g[1]}", []).append(s)
    out = {"note": "model alone; gold precision is a lower bound; mix pools counts of two label sources",
           "runs": runs, "means": {k: mean_scores(v) for k, v in sorted(groups.items())},
           "seeds_per_mean": {k: len(v) for k, v in sorted(groups.items())}}
    (final / "prf.json").write_text(json.dumps(out, indent=1) + "\n")
    write_md(final / "prf.md", out)
    print(f"wrote {final / 'prf.md'}")


if __name__ == "__main__":
    main()
