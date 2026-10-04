"""Apply the version 4 pre-registered selection rule (docs/v4/preregistration.md).

For every run in models/v4/ this exports the run to ONNX (<run>/zoo), proves the
export on real dev windows, and scores the 36 gold-train dev papers with
14_gold_eval.py --papers dev. The old shipped model is scored the same way. Then
rules 1, 2 and 3 run on the three dev metrics and nothing else.

Reads no gold_test paper, no bronze file, and writes nothing to
results/gold_eval_log.jsonl: the scoring call uses --papers dev, which opens only
the dev papers. The metrics come from the system named by --rule-system. The
default is "model", the tagger alone, because the cascade's regex half does not
change from run to run and would hide the difference between runs. The cascade
figures are stored beside it.

Usage: python pipeline/23_select_v4.py [--models models/v4] [--out-dir results/v4]
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIGS = ["gru-crf", "gru-softmax", "lstm-softmax"]
SEEDS = (0, 1, 2)
TIE = 0.01          # R_named gap below which two candidates count as tied
FP_SLACK = 2        # rule 1: extra FP a setting may have over "none"
EPS = 1e-9          # float guard so that 0.01 exactly is not "below 0.01"


def load_stage(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3].replace("-", "_"), ROOT / "pipeline" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_name(config, noise, seed):
    """The directory name 22_train_v4.py gives a run. gru-crf seed 0 is a gate run."""
    if config == "gru-crf" and seed == 0:
        return "gate-none" if noise == "none" else "gate-old+all"
    return f"{config}-{noise}-s{seed}"


def dev_metrics(report, system):
    row = report["dev"][system]
    n = report["dev"]["gold_in_scope"]
    # Recall is rebuilt from counts so a tie at 0.01 is not decided by 3-digit rounding.
    k_frag = round(row["recall_with_fragments"] * n) if row.get("recall_with_fragments") is not None else None
    return {"R_named": row["found"] / n, "R_frag": None if k_frag is None else k_frag / n,
            "FP": row["finds_in_papers_without_results"], "found": row["found"], "n": n}


def rule1(m):
    """Pick the noise setting from the gru-crf seed-0 runs; m maps run name to metrics."""
    none, new = m[run_name("gru-crf", "none", 0)], m[run_name("gru-crf", "all", 0)]
    eligible = {"none": True, "all": new["FP"] <= none["FP"] + FP_SLACK}
    out = {"inputs": {"none": none, "all": new}, "fp_limit": none["FP"] + FP_SLACK, "eligible": eligible}
    if not eligible["all"]:
        return {**out, "chosen": "none", "reason": "setting 'all' has FP above the FP of 'none' plus 2"}
    gap = new["R_named"] - none["R_named"]
    if abs(gap) + EPS >= TIE:
        chosen = "all" if gap > 0 else "none"
        return {**out, "chosen": chosen, "reason": "R_named differs by 0.01 or more; the higher wins"}
    if new["FP"] != none["FP"]:
        chosen = "all" if new["FP"] < none["FP"] else "none"
        return {**out, "chosen": chosen, "reason": "R_named differs by less than 0.01; the lower FP wins"}
    return {**out, "chosen": "none", "reason": "R_named and FP tie; the setting 'none' wins"}


def rule2(m, params, noise):
    """Pick the configuration by its mean R_named over seeds 0 to 2.

    The 0.01 tie is read against the best mean, so the rule is not circular when
    three means lie close together: every configuration within 0.01 of the top
    mean is tied, and the tied ones go by mean FP, then by parameters.
    """
    table = {}
    for cfg in CONFIGS:
        runs = [m[run_name(cfg, noise, s)] for s in SEEDS]
        table[cfg] = {"mean_R_named": sum(r["R_named"] for r in runs) / len(runs),
                      "mean_FP": sum(r["FP"] for r in runs) / len(runs),
                      "parameters": params[run_name(cfg, noise, 0)],
                      "R_named_by_seed": [r["R_named"] for r in runs]}
    top = max(t["mean_R_named"] for t in table.values())
    tied = [c for c in CONFIGS if top - table[c]["mean_R_named"] + EPS < TIE]
    best = min(tied, key=lambda c: (table[c]["mean_FP"], table[c]["parameters"]))
    if len(tied) == 1:
        reason = "highest mean R_named"
    else:
        reason = f"means within 0.01 of the top for {tied}; lowest mean FP, then fewest parameters"
    return {"noise": noise, "table": table, "tied": tied, "chosen": best, "reason": reason}


def rule3(m, old, config, noise):
    """The seed-0 run of the choice ships only when it matches the old model on R_named."""
    run = run_name(config, noise, 0)
    cand = m[run]
    replaces = cand["R_named"] + EPS >= old["R_named"]
    return {"candidate_run": run, "candidate_R_named": cand["R_named"], "old_R_named": old["R_named"],
            "replaces_gru_crf_s0": replaces,
            "reason": "candidate R_named is at least the old R_named" if replaces
            else "candidate R_named is below the old R_named; the old model stays"}


TOP3 = ["gate-old+all", "gru-softmax-all-s0", "lstm-softmax-all-s0", "gate-none"]


def rule2_top3(m, params, gru_crf_run):
    """The amended rule 2: three seed-0 candidates, one per top configuration."""
    cands = [gru_crf_run, "gru-softmax-all-s0", "lstm-softmax-all-s0"]
    table = {c: {"R_named": m[c]["R_named"], "FP": m[c]["FP"], "parameters": params[c]} for c in cands}
    top = max(t["R_named"] for t in table.values())
    tied = [c for c in cands if top - table[c]["R_named"] + EPS < TIE]
    best = min(tied, key=lambda c: (table[c]["FP"], table[c]["parameters"]))
    reason = ("highest R_named" if len(tied) == 1
              else f"R_named within 0.01 of the top for {tied}; lowest FP, then fewest parameters")
    return {"table": table, "tied": tied, "chosen": best, "reason": reason}


def select_top3(m, params, old):
    r1 = rule1(m)
    r2 = rule2_top3(m, params, run_name("gru-crf", r1["chosen"], 0))
    cand = m[r2["chosen"]]
    replaces = cand["R_named"] + EPS >= old["R_named"]
    r3 = {"candidate_run": r2["chosen"], "candidate_R_named": cand["R_named"], "old_R_named": old["R_named"],
          "replaces_gru_crf_s0": replaces,
          "reason": "candidate R_named is at least the old R_named" if replaces
          else "candidate R_named is below the old R_named; the old model stays"}
    return {"plan": "top3", "rule1": r1, "rule2": r2, "rule3": r3,
            "chosen_run": r2["chosen"], "replaces_gru_crf_s0": replaces}


def select(m, params, old):
    r1 = rule1(m)
    r2 = rule2(m, params, r1["chosen"])
    r3 = rule3(m, old, r2["chosen"], r1["chosen"])
    return {"rule1": r1, "rule2": r2, "rule3": r3,
            "chosen_run": r3["candidate_run"], "replaces_gru_crf_s0": r3["replaces_gru_crf_s0"]}


def expected_runs():
    return list(load_stage("22_train_v4.py").runs())


def check_runs(models, names, allow_partial):
    need = ("model.pt", "charmap.json", "report.json", "train.log")
    missing = [n for n in names if not all((models / n / f).exists() for f in need)]
    if missing and not allow_partial:
        sys.exit(f"{len(missing)} of {len(names)} runs are missing or unfinished in {models}: "
                 f"{', '.join(missing)}. Nothing was exported or scored.")
    if len(missing) == len(names):
        sys.exit(f"no run of the plan found in {models}")
    return [n for n in names if n not in missing], missing


def export_and_check(run_dir, dev_examples):
    """Export to <run>/zoo and run 08_export's real-window parity on dev windows."""
    from statcheck_ml.export import ensure_zoo, load

    ex = load_stage("08_export.py")
    zoo = ensure_zoo(run_dir)
    model, vocab, _, has_crf = load(str(run_dir / "model.pt"))
    diff, agree = ex.dev_parity(model, vocab, ex.open_session(zoo / "tagger.onnx"), has_crf, dev_examples)
    ok = bool(diff < ex.LOGIT_TOLERANCE and agree == 1.0)
    params = sum(p.numel() for p in model.parameters())
    return zoo, {"parity": ok, "max_abs_diff": diff, "tag_agreement": agree, "parameters": params}


def onnx_parameters(path):
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    return int(sum(numpy_helper.to_array(t).size for t in graph.initializer))


def score_dev(zoo, out, args):
    cmd = [sys.executable, str(ROOT / "pipeline/14_gold_eval.py"), "--papers", "dev",
           "--model", str(zoo), "--out", str(out), "--systems", "model", "cascade",
           "--resamples", str(args.resamples), "--label", "v4-dev-selection",
           "--gold", str(args.gold), "--split", str(args.split), "--dev", str(args.dev),
           "--clean-dir", str(args.clean_dir)]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads(Path(out).read_text())


def write_md(path, result, order):
    def f(x):
        return "-" if x is None else f"{x:.3f}"

    lines = ["| run | R_named | R_frag | FP | cascade R_named | cascade FP | parameters | export+score s |",
             "|---|---|---|---|---|---|---|---|"]
    for name in order:
        r = result["runs"].get(name)
        if r is None:
            lines.append(f"| {name} | not run | | | | | | |")
            continue
        lines.append(f"| {name} | {f(r['R_named'])} | {f(r['R_frag'])} | {r['FP']} | "
                     f"{f(r['cascade']['R_named'])} | {r['cascade']['FP']} | {r['parameters']} | {r['seconds']:.0f} |")
    o = result["old"]
    lines.append(f"| gru-crf-s0 (shipped) | {f(o['R_named'])} | {f(o['R_frag'])} | {o['FP']} | "
                 f"{f(o['cascade']['R_named'])} | {o['cascade']['FP']} | {o['parameters']} | {o['seconds']:.0f} |")
    text = ["# Version 4 selection on the 36 dev papers", "",
            f"Rule system: `{result['rule_system']}`. Gold test and bronze holdout were not read.", ""]
    if result["partial"]:
        text += [f"PARTIAL RUN, for testing only. Missing: {', '.join(result['missing'])}", ""]
    text += lines + [""]
    sel = result["selection"]
    if sel is None:
        text.append(f"Rules not applied: {result['selection_error']}")
    else:
        r1, r2, r3 = sel["rule1"], sel["rule2"], sel["rule3"]
        text += [f"Rule 1: noise setting `{r1['chosen']}`. {r1['reason']}. FP limit {r1['fp_limit']}.",
                 f"Rule 2: configuration `{r2['chosen']}`. {r2['reason']}.", ""]
        text += ["| configuration | mean R_named | mean FP | parameters |", "|---|---|---|---|"]
        text += [f"| {c} | {t['mean_R_named']:.3f} | {t['mean_FP']:.1f} | {t['parameters']} |"
                 for c, t in r2["table"].items()]
        text += ["", f"Rule 3: run `{sel['chosen_run']}` R_named {r3['candidate_R_named']:.3f} against "
                     f"{r3['old_R_named']:.3f} for gru-crf-s0. "
                     f"{'It replaces gru-crf-s0' if sel['replaces_gru_crf_s0'] else 'The old model stays'}. "
                     f"{r3['reason']}."]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="models/v4")
    ap.add_argument("--out-dir", default="results/v4")
    ap.add_argument("--old-model", default=None, help="default: the packaged zoo model")
    ap.add_argument("--rule-system", default="cascade", choices=["model", "cascade"])
    ap.add_argument("--plan", default="top3", choices=["top3", "full"],
                    help="top3: the four runs of the 2026-10-05 amendment; full: the 23 runs")
    ap.add_argument("--allow-partial", action="store_true",
                    help="testing only: skip missing runs and skip the rules that need them")
    ap.add_argument("--data", default="dataset/train_v4.jsonl")
    ap.add_argument("--splits", default="dataset/splits_v4.json")
    ap.add_argument("--gold", default="dataset/gold/gold.jsonl")
    ap.add_argument("--split", default="dataset/gold/split.json")
    ap.add_argument("--dev", default="dataset/gold/dev.json")
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--parity-windows", type=int, default=150,
                    help="dev windows for the torch against ONNX check; 0 means all (473)")
    ap.add_argument("--resamples", type=int, default=20,
                    help="bootstrap resamples; the selection reads no interval")
    args = ap.parse_args()

    def path(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    models, out_dir = path(args.models), path(args.out_dir)
    for a in ("data", "splits", "gold", "split", "dev", "clean_dir"):
        setattr(args, a, path(getattr(args, a)))
    names = TOP3 if args.plan == "top3" else expected_runs()
    present, missing = check_runs(models, names, args.allow_partial)

    from statcheck_ml.data import apply_splits, load_jsonl, load_splits, row_to_example
    from statcheck_ml.pipeline import bundled_model_path

    examples = [row_to_example(r) for r in load_jsonl(str(args.data))]
    _, dev_examples = apply_splits(examples, load_splits(str(args.splits)))
    (out_dir / "dev").mkdir(parents=True, exist_ok=True)

    def one(label, zoo, info):
        started = time.time()
        report = score_dev(zoo, out_dir / "dev" / f"{label}.json", args)
        m = dev_metrics(report, args.rule_system)
        m["cascade"] = dev_metrics(report, "cascade")
        m.update(info, zoo=str(zoo))
        m["seconds"] = time.time() - started + info.get("export_seconds", 0)
        print(f"{label}: R_named {m['R_named']:.3f} R_frag {m['R_frag']:.3f} FP {m['FP']} "
              f"in {m['seconds']:.0f} s", flush=True)
        return m

    runs = {}
    for n in present:
        t = time.time()
        zoo, info = export_and_check(models / n, dev_examples[:args.parity_windows or None])
        if not info["parity"]:
            sys.exit(f"{n}: ONNX parity failed (max diff {info['max_abs_diff']:.2e}, "
                     f"tag agreement {info['tag_agreement']:.4f})")
        print(f"{n}: exported and checked in {time.time() - t:.0f} s", flush=True)
        runs[n] = one(n, zoo, {**info, "export_seconds": time.time() - t})
    old_zoo = path(args.old_model) if args.old_model else bundled_model_path()
    old = one("old-gru-crf-s0", old_zoo, {"parameters": onnx_parameters(old_zoo / "tagger.onnx")})

    selection = error = None
    try:
        chooser = select_top3 if args.plan == "top3" else select
        selection = chooser({n: runs[n] for n in runs}, {n: runs[n]["parameters"] for n in runs}, old)
    except KeyError as exc:
        error = f"the rules need run {exc} and it is missing (partial run)"
    result = {"rule_system": args.rule_system, "partial": bool(missing), "missing": missing,
              "runs": runs, "old": old, "selection": selection, "selection_error": error,
              "tie": TIE, "fp_slack": FP_SLACK,
              "scored_papers": "dev (36 gold-train papers); no gold_test paper, no bronze file"}
    (out_dir / "selection.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_md(out_dir / "selection.md", result, names)
    if selection:
        print(f"chosen {selection['chosen_run']}; replaces gru-crf-s0: {selection['replaces_gru_crf_s0']}")
    else:
        print(error)


if __name__ == "__main__":
    main()
