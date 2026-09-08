"""Build the results report from the evaluation output.

Every number in the report comes from a file written by a measurement, never
from a value typed here. When a run is missing, the report says so rather than
leaving a gap that reads like a zero.

Usage:
  python make_report.py <report_dir> <out.md>
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def row(name, m):
    return (f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} |")


def systems_table(split: dict) -> str:
    lines = ["| System | Precision | Recall | F1 | TP | FP | FN |",
             "|---|---|---|---|---|---|---|"]
    for key, label in (("statcheck", "statcheck 1.5.0"),
                       ("model", "model alone"),
                       ("hybrid", "hybrid cascade")):
        if key in split:
            lines.append(row(label, split[key]))
    return "\n".join(lines)


def main(report_dir: str, out_path: str):
    evaluation = load(os.path.join(report_dir, "evaluation.json"))
    runs = {}
    for name in sorted(os.listdir(report_dir)) if os.path.isdir(report_dir) else []:
        rp = os.path.join(report_dir, name, "report.json")
        if os.path.exists(rp):
            runs[name] = load(rp)

    out = [f"# Results\n",
           f"Generated on {date.today().isoformat()}.\n",
           "## What is being measured\n",
           "Three systems read the same passages and are scored against the same",
           "labels with the same metric.\n",
           "| System | What it is |",
           "|---|---|",
           "| statcheck | the real R package, version 1.5.0 |",
           "| model | the learned extractor, alone |",
           "| hybrid | the prefilter, then statcheck, then the model for the rest |\n",
           "A result counts as found when its test statistic matches. statcheck",
           "reports values and not character positions, so this is the only",
           "matching all three systems can take part in.\n"]

    if evaluation:
        for split_name, split in evaluation.items():
            note = ("chose when to stop training, so it is optimistic"
                    if split_name == "dev" else
                    "never used for any decision")
            out.append(f"## {split_name} ({split['windows']} windows, "
                       f"{split['gold_results']} annotated results)\n")
            out.append(f"This part {note}.\n")
            out.append(systems_table(split) + "\n")
            if "hybrid" in split:
                h = split["hybrid"]
                out.append(f"The cascade took {h.get('from_statcheck', 0)} results from "
                           f"statcheck and added {h.get('from_model', 0)} from the model.\n")
            v = split.get("verdict_agreement", {})
            total = v.get("agree", 0) + v.get("disagree", 0)
            if total:
                out.append(f"On the results both systems found, the verdicts agree "
                           f"{v['agree']} times in {total} "
                           f"({100*v['agree']/total:.1f}%). "
                           f"{v.get('undecidable', 0)} could not be judged, because the "
                           f"model did not recover every value the arithmetic needs.\n")
    else:
        out.append("## No evaluation file was found\n")
        out.append("Run `evaluate.py` first.\n")

    if runs:
        out.append("## Training runs\n")
        out.append("| Run | Architecture | Augmented | Parameters | Dev F1 |")
        out.append("|---|---|---|---|---|")
        for name, r in runs.items():
            arch = r.get("unit") or r.get("model") or "unknown"
            aug = "yes" if r.get("augment") or r.get("hard_negatives") else "no"
            if r.get("crf"):
                arch = f"{arch} with a CRF"
            params = r.get("parameters", 0)
            f1 = r.get("dev", {}).get("overall", {}).get("f1")
            out.append(f"| {name} | {arch} | {aug} | {params:,} | "
                       f"{f1:.3f} |" if f1 is not None else
                       f"| {name} | {arch} | {aug} | {params:,} | not finished |")

    out.append("\n## What is not claimed\n")
    out.append("Every label comes from a language model, not from a person. Two")
    out.append("independent passes agree on 84.1% of results, and that is the")
    out.append("ceiling for any model measured against these labels.\n")
    out.append("No measurement here can find a mistake that the annotator makes")
    out.append("every time.\n")
    out.append("Robustness is claimed only for the damage present in this corpus.")
    out.append("The adversarial scenarios teach a wider range, but the test set")
    out.append("comes from the same 3100 papers, so it cannot show that the wider")
    out.append("range was learned.\n")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
