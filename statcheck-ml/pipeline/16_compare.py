"""Compare the version 4 "before" results with the current results.

Gold and bronze are reported apart and never pooled. The before files under
results/v4/before are frozen copies of the M1 outputs. The steps table reads
the gold evaluation log, so the path from the M1 state to the final state is
visible, one row per labelled evaluation.

Usage: python pipeline/16_compare.py
"""
import argparse
import json
import os

GOLD_FIELDS = ("recall", "recall_ci", "recall_damaged", "finds_in_papers_without_results")
BRONZE_KEYS = ("p", "r", "f1")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def delta(a, b):
    return None if a is None or b is None else round(b - a, 3)


def gold_rows(before, after):
    rows, only_before, only_after = [], [], []
    names = [k for k, v in before.items() if isinstance(v, dict) and "recall" in v]
    after_names = [k for k, v in after.items() if isinstance(v, dict) and "recall" in v]
    for name in names:
        if name not in after_names:
            only_before.append(name)
            continue
        b, a = before[name], after[name]
        row = {"system": name, "recall_delta": delta(b["recall"], a["recall"])}
        for f in GOLD_FIELDS:
            row[f + "_before"], row[f + "_after"] = b[f], a[f]
        rows.append(row)
    only_after = [n for n in after_names if n not in names]
    return {"rows": rows, "only_before": only_before, "only_after": only_after}


def bronze_rows(before, after):
    b_sys, a_sys = before["systems"], after["systems"]
    rows = []
    for name in b_sys:
        if name not in a_sys:
            continue
        row = {"system": name}
        for k in BRONZE_KEYS:
            x, y = b_sys[name]["overall"][k], a_sys[name]["overall"][k]
            row[k + "_before"], row[k + "_after"], row[k + "_delta"] = x, y, delta(x, y)
        rows.append(row)
    return {"rows": rows,
            "only_before": [n for n in b_sys if n not in a_sys],
            "only_after": [n for n in a_sys if n not in b_sys]}


def steps(log_path):
    out = []
    with open(log_path, encoding="utf-8") as fh:
        for text in fh:
            if not text.strip():
                continue
            line = json.loads(text)
            if not line.get("label"):
                continue
            per = line.get("gold_test", {})
            out.append({"label": line["label"], "date": line.get("date"),
                        "git_sha": line.get("git_sha"),
                        "systems": {s: {"recall": per.get(s, {}).get("recall",
                                                                    line.get("gold_test_recall", {}).get(s)),
                                        "finds_in_papers_without_results":
                                        per.get(s, {}).get("finds_in_papers_without_results")}
                                    for s in line.get("systems", [])}})
    return out


def f3(x):
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def ci(x):
    return "n/a" if x is None else f"[{x[0]:.3f}, {x[1]:.3f}]"


def render(report):
    md = ["# Version 4: before and after", "",
          "Gold and bronze are never pooled. Each table names its tier.", ""]
    for tier in ("gold_test", "gold_train"):
        t = report["gold"][tier]
        md += [f"## Gold, {tier} (human-coded, recall)", "",
               "| system | recall before | recall after | delta | CI before | CI after | damaged before "
               "| damaged after | finds without results before | after |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in t["rows"]:
            md.append(f"| {r['system']} | {f3(r['recall_before'])} | {f3(r['recall_after'])} "
                      f"| {f3(r['recall_delta'])} | {ci(r['recall_ci_before'])} | {ci(r['recall_ci_after'])} "
                      f"| {f3(r['recall_damaged_before'])} | {f3(r['recall_damaged_after'])} "
                      f"| {r['finds_in_papers_without_results_before']} "
                      f"| {r['finds_in_papers_without_results_after']} |")
        md += ["", f"Only before: {', '.join(t['only_before']) or 'none'}. "
                   f"Only after: {', '.join(t['only_after']) or 'none'}.", ""]
    b = report["bronze_holdout"]
    md += ["## Bronze holdout (language-model annotation, windows only)", "",
           "| system | P before | P after | dP | R before | R after | dR | F1 before | F1 after | dF1 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in b["rows"]:
        md.append(f"| {r['system']} | " + " | ".join(
            f3(r[f"{k}_{s}"]) for k in BRONZE_KEYS for s in ("before", "after", "delta")) + " |")
    md += ["", f"Only before: {', '.join(b['only_before']) or 'none'}. "
               f"Only after: {', '.join(b['only_after']) or 'none'}.", "",
           "## Steps (gold_test, one row per labelled evaluation in the log)", "",
           "| label | date | git_sha | system | recall | finds without results |",
           "|---|---|---|---|---|---|"]
    for s in report["steps"]:
        for name, v in s["systems"].items():
            md.append(f"| {s['label']} | {s['date']} | {(s['git_sha'] or '')[:7]} | {name} "
                      f"| {f3(v['recall'])} | {v['finds_in_papers_without_results'] if v['finds_in_papers_without_results'] is not None else 'n/a'} |")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before-gold", default="results/v4/before/gold_eval.json")
    ap.add_argument("--after-gold", default="results/gold_eval.json")
    ap.add_argument("--before-bronze", default="results/v4/before/eval.json")
    ap.add_argument("--after-bronze", default="results/eval.json")
    ap.add_argument("--log", default="results/gold_eval_log.jsonl")
    ap.add_argument("--out", default="results/v4/compare.json")
    ap.add_argument("--md", default="results/v4/compare.md")
    ap.add_argument("--seed", type=int, default=0, help="unused; the stage is deterministic")
    args = ap.parse_args()

    bg, ag = load(args.before_gold), load(args.after_gold)
    report = {
        "inputs": {"before_gold": args.before_gold, "after_gold": args.after_gold,
                   "before_bronze": args.before_bronze, "after_bronze": args.after_bronze,
                   "log": args.log},
        "gold": {t: gold_rows(bg[t], ag[t]) for t in ("gold_test", "gold_train")},
        "bronze_holdout": bronze_rows(load(args.before_bronze), load(args.after_bronze)),
        "steps": steps(args.log),
    }
    for path in (args.out, args.md):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write(render(report))


if __name__ == "__main__":
    main()
