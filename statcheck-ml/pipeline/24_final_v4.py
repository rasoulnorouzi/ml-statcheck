"""The one read of the gold test for version 4, after the choice (docs/v4/preregistration.md).

Runs after 23_select_v4.py and refuses to run before it: results/v4/selection.json
must exist, so the choice is fixed before any gold test number is seen. Then, for
the 35 runs and the old gru-crf-s0:

  gold test      14_gold_eval.py --papers gold_test, model and cascade, one log line
                 per model with the label v4-final:<run>. The old model's call also
                 scores the regex and the four R statcheck forms, once.
  bronze holdout 09_evaluate.py on a view of the 36 ONNX models, precision, recall
                 and F1 on the 576 windows.

Nothing here feeds back into the choice. The two sets are reported apart and never
pooled; summary.md puts them side by side.

Usage: python pipeline/24_final_v4.py [--models models/v4] [--out-dir results/v4/final]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "old-gru-crf-s0"
R_SYSTEMS = ["regex", "statcheck_r_fulltext_raw", "statcheck_r_fulltext_repaired",
             "statcheck_r_passages", "statcheck_r_pdf"]


def group(name):
    """(config, noise, seed) of a run; the five single-family gate runs have no group."""
    if name == "gate-none":
        return "gru-crf", "none", 0
    if name == "gate-old+all":
        return "gru-crf", "all", 0
    if name.startswith("gate-"):
        return None
    cfg, noise, seed = name.rsplit("-", 2)
    return cfg, noise, int(seed[1:])


def view(models, names, old_zoo, out):
    """A runs.json and one link per model, the layout 09_evaluate.py reads."""
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names + [OLD]:
        target = old_zoo if name == OLD else (models / name / "zoo").resolve()
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
        g = group(name) if name != OLD else ("old-gru-crf", "old", 0)
        report = {} if name == OLD else json.loads((models / name / "report.json").read_text())
        records.append({"name": name, "status": "done",
                        "config": f"{g[0]}-{g[1]}" if g else name, "seed": g[2] if g else 0,
                        "dev_f1": report.get("dev", {}).get("overall", {}).get("f1")})
    (out / "runs.json").write_text(json.dumps(records, indent=1) + "\n")
    return out / "runs.json"


def gold_row(path, system):
    g = json.loads(Path(path).read_text())["gold_test"]
    row = g[system]
    return {"R_named": row["recall"], "R_named_ci": row["recall_ci"],
            "R_frag": row.get("recall_with_fragments"),
            "FP": row["finds_in_papers_without_results"],
            "R_damaged": row["recall_damaged"], "by_test": row["recall_by_test"],
            "finds_not_in_gold": row["finds_not_in_gold"], "n": g["gold_in_scope"]}


def mean(values):
    return round(statistics.fmean(values), 3) if values else None


def write_md(path, summary, selection):
    f = lambda x: "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    lines = ["# Version 4 final evaluation", "",
             f"Chosen by the rules on the dev papers: `{summary['shipped']}` "
             f"(rule 3: {selection['selection']['rule3']['reason']}).",
             "The gold test was read once, after the choice. The bronze holdout and the gold "
             "test are different labels and are never pooled.", "",
             "## Every model", "",
             "| run | dev cascade R_named | dev cascade FP | gold R_named | 95% CI | gold R_frag | gold FP (120 no-result papers) "
             "| gold R damaged | gold cascade R_named | gold cascade FP | bronze P | bronze R | bronze F1 |",
             "|" + "---|" * 13]
    for name, r in summary["runs"].items():
        g, c, b = r["gold_model"], r["gold_cascade"], r["bronze"]
        lines.append(f"| {name} | {f(r['dev'].get('R_named'))} | {f(r['dev'].get('FP'))} | "
                     f"{f(g['R_named'])} | {g['R_named_ci'][0]:.3f}–{g['R_named_ci'][1]:.3f} | "
                     f"{f(g['R_frag'])} | {g['FP']} | {f(g['R_damaged'])} | {f(c['R_named'])} | {c['FP']} | "
                     f"{f(b.get('precision'))} | {f(b.get('recall'))} | {f(b.get('f1'))} |")
    lines += ["", "## Means over seeds 0 to 4", "",
              "| configuration | noise | gold R_named | gold FP | cascade R_named | cascade FP | bronze F1 |",
              "|---|---|---|---|---|---|---|"]
    for key, m in summary["means"].items():
        cfg, noise = key.split("|")
        lines.append(f"| {cfg} | {noise} | {f(m['gold_R_named'])} | {f(m['gold_FP'])} | "
                     f"{f(m['cascade_R_named'])} | {f(m['cascade_FP'])} | {f(m['bronze_f1'])} |")
    lines += ["", "## Baselines on the gold test (one read, with the old model's call)", "",
              "| system | R_named | 95% CI | FP (120 no-result papers) | R damaged |", "|---|---|---|---|---|"]
    for s, g in summary["baselines"].items():
        lines.append(f"| {s} | {f(g['R_named'])} | {g['R_named_ci'][0]:.3f}–{g['R_named_ci'][1]:.3f} | "
                     f"{g['FP']} | {f(g['R_damaged'])} |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="models/v4")
    ap.add_argument("--selection", default="results/v4/selection.json")
    ap.add_argument("--out-dir", default="results/v4/final")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    selection_path = ROOT / args.selection
    if not selection_path.exists():
        sys.exit(f"{args.selection} is missing: run 23_select_v4.py first, so the choice is fixed "
                 "before the gold test is read")
    selection = json.loads(selection_path.read_text())
    if selection.get("partial"):
        sys.exit("the selection was partial; the final read needs the full selection")
    models, out = ROOT / args.models, ROOT / args.out_dir
    # The same zoo the selection scored as the old model: the packaged one.
    old_zoo = Path(selection["old"]["zoo"]).resolve()
    names = sorted(selection["runs"])
    (out / "gold").mkdir(parents=True, exist_ok=True)

    py = sys.executable
    # The bronze holdout first: it reads no gold paper, so a fault in it costs no read.
    bronze_path = out / "bronze.json"
    if not bronze_path.exists():
        runs_json = view(models, names, old_zoo, ROOT / args.models / "_eval_view")
        started = time.time()
        subprocess.run([py, "pipeline/09_evaluate.py", "--runs", str(runs_json),
                        "--export", "/nonexistent", "--out", str(bronze_path),
                        "--resamples", "2000", "--seed", "0"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        print(f"bronze holdout in {time.time() - started:.0f} s", flush=True)
    for name in [OLD] + names:
        dest = out / "gold" / f"{name}.json"
        if dest.exists():
            # The read happened; a second call would add a second log line for one model.
            print(f"{name}: gold test read before, kept")
            continue
        zoo = old_zoo if name == OLD else models / name / "zoo"
        systems = ["model", "cascade"] + (R_SYSTEMS if name == OLD else [])
        started = time.time()
        subprocess.run([py, "pipeline/14_gold_eval.py", "--papers", "gold_test", "--model", str(zoo),
                        "--systems", *systems, "--label", f"v4-final:{name}", "--out", str(dest),
                        "--workers", str(args.workers)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        print(f"{name}: gold test in {time.time() - started:.0f} s", flush=True)

    bronze = json.loads(bronze_path.read_text())["systems"]

    dev = {n: r for n, r in selection["runs"].items()}
    dev[OLD] = selection["old"]
    summary = {"shipped": selection["selection"]["chosen_run"] if selection["selection"]["replaces_gru_crf_s0"] else "gru-crf-s0", "runs": {}, "means": {},
               "baselines": {s: gold_row(out / "gold" / f"{OLD}.json", s) for s in R_SYSTEMS}}
    for name in names + [OLD]:
        path = out / "gold" / f"{name}.json"
        summary["runs"][name] = {
            "dev": {k: dev[name].get(k) for k in ("R_named", "R_frag", "FP")},
            "gold_model": gold_row(path, "model"), "gold_cascade": gold_row(path, "cascade"),
            "bronze": {k: bronze[name]["overall"][s] for k, s in (("precision", "p"), ("recall", "r"), ("f1", "f1"))}}
    for name in names:
        g = group(name)
        if g is None:
            continue
        key = f"{g[0]}|{g[1]}"
        r = summary["runs"][name]
        m = summary["means"].setdefault(key, {"runs": [], "gold_R_named": [], "gold_FP": [],
                                              "cascade_R_named": [], "cascade_FP": [], "bronze_f1": []})
        m["runs"].append(name)
        m["gold_R_named"].append(r["gold_model"]["R_named"])
        m["gold_FP"].append(r["gold_model"]["FP"])
        m["cascade_R_named"].append(r["gold_cascade"]["R_named"])
        m["cascade_FP"].append(r["gold_cascade"]["FP"])
        m["bronze_f1"].append(r["bronze"]["f1"])
    for m in summary["means"].values():
        for k in list(m):
            if k != "runs":
                m[k] = mean(m[k])
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_md(out / "summary.md", summary, selection)
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
