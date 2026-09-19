"""Score every system against the holdout: statcheck, each trained model, and
the cascade that follows statcheck with a model.

This is the measurement the project exists to produce, and it runs once
against a gold-discipline holdout: log the run, do not tune against it and
run again pretending the first run did not happen.

Systems scored:

  statcheck_raw          the R package, unmodified text
  statcheck_repaired      the same package, operator-repaired text
  <config>-s<seed>        every finished model run named in runs.json
  cascade_<config>-s0     statcheck_repaired, plus what the best model adds

"Best" is chosen from `runs.json`'s own `dev_f1`, the development score
recorded at training time, never from a score computed in this file. Picking
"best" from the holdout score computed here would be the leak the project's
own rules warn against: the holdout is evaluated, not tuned against.

Usage:
  python pipeline/09_evaluate.py --windows dataset/holdout.jsonl \\
      --statcheck dataset/baseline/statcheck_r.csv \\
      --repaired dataset/baseline/statcheck_repaired.csv \\
      --runs models/runs.json --zoo models/zoo --export models/export.json \\
      --out results/eval.json --resamples 2000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]      # statcheck-ml
sys.path.insert(0, str(REPO / "src"))

from statcheck_ml import evalutil as ev
from statcheck_ml.onnx_runtime import OnnxTagger
from statcheck_ml.stats import mcnemar_exact, paired_bootstrap


def resolve(p) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


# --------------------------------------------------------------- models ---

def load_run_records(runs_path: Path, only) -> list:
    if not runs_path.exists():
        print(f"note: {runs_path} not found, no model runs to score")
        return []
    records = [r for r in ev.read_json(runs_path) if r.get("status") == "done"]
    if only:
        records = [r for r in records if r["name"] in only]
    return records


def load_tagger(models_dir: Path, name: str):
    run_dir = models_dir / name
    if not (run_dir / "tagger.onnx").exists():
        print(f"note: {run_dir}/tagger.onnx missing, skipping {name} "
              f"(run pipeline/08_export.py first)")
        return None
    return OnnxTagger(run_dir)


def pick_best_seed0(records: list):
    """The seed-0 run with the highest recorded dev F1, excluding the
    augmentation ablation. Chosen from training-time dev_f1, not from any
    score in this file.
    """
    seed0 = [r for r in records if r.get("seed") == 0 and "noaug" not in r.get("config", "")]
    seed0 = [r for r in seed0 if r.get("dev_f1") is not None]
    return max(seed0, key=lambda r: r["dev_f1"]) if seed0 else None


def top3_seed0(records: list) -> list:
    seed0 = [r for r in records if r.get("seed") == 0 and "noaug" not in r.get("config", "")]
    seed0 = [r for r in seed0 if r.get("dev_f1") is not None]
    return sorted(seed0, key=lambda r: r["dev_f1"], reverse=True)[:3]


# --------------------------------------------------------------- scoring --

def build_system_report(found_by_window, window_ids, doc_ids, doc_to_windows,
                        gold_by_window, gold_meta, resamples, seed,
                        block_spans_by_window=None, gold_spans_by_window=None) -> dict:
    report = ev.score_system(found_by_window, gold_by_window, gold_meta, window_ids)
    report["bootstrap"] = ev.bootstrap_prf(doc_ids, doc_to_windows, found_by_window,
                                           gold_by_window, gold_meta, n=resamples, seed=seed)
    if block_spans_by_window is not None:
        report["span_strict"] = ev.score_span_strict(block_spans_by_window,
                                                      gold_spans_by_window, window_ids)
    return report


def f1_stat(found_by_window, gold_by_window, gold_meta, doc_to_windows, bucket="overall"):
    def stat(doc_sample):
        window_ids = [wid for d in doc_sample for wid in doc_to_windows.get(d, [])]
        scored = ev.score_system(found_by_window, gold_by_window, gold_meta, window_ids)
        return scored.get(bucket, {}).get("f1", 0.0)
    return stat


def paired_entry(name_a, found_a, name_b, found_b, gold_by_window, gold_meta,
                 doc_ids, doc_to_windows, resamples, seed) -> dict:
    diff, (lo, hi), p = paired_bootstrap(
        doc_ids,
        f1_stat(found_a, gold_by_window, gold_meta, doc_to_windows),
        f1_stat(found_b, gold_by_window, gold_meta, doc_to_windows),
        n=resamples, seed=seed)
    return {"a": name_a, "b": name_b, "diff": diff, "ci": [lo, hi], "p": p}


def mcnemar_entry(name_a, found_a, name_b, found_b, gold_by_window) -> dict:
    b, c = ev.mcnemar_pair(found_a, found_b, gold_by_window)
    return {"a": name_a, "b_only_a": b, "b_only_b": c, "p": mcnemar_exact(b, c)}


# ------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default="dataset/holdout.jsonl")
    ap.add_argument("--statcheck", default="dataset/baseline/statcheck_r.csv")
    ap.add_argument("--repaired", default="dataset/baseline/statcheck_repaired.csv")
    ap.add_argument("--runs", default="models/runs.json")
    ap.add_argument("--zoo", default="models/zoo")
    ap.add_argument("--export", default="models/export.json")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict model runs to these names from runs.json")
    ap.add_argument("--no-models", action="store_true",
                    help="score statcheck alone; skip every model and the cascade")
    args = ap.parse_args()

    windows_path = resolve(args.windows)
    windows = ev.read_jsonl(windows_path)
    window_ids = [w["window_id"] for w in windows]
    gold_by_window, gold_meta, gold_spans_by_window, doc_to_windows, gold_unparseable = \
        ev.extract_gold(windows)
    doc_ids = sorted(doc_to_windows)
    n_gold = sum(len(v) for v in gold_by_window.values())
    n_damaged = sum(1 for m in gold_meta.values() if m["damaged"])
    n_checkable = sum(1 for m in gold_meta.values() if m["checkable"])

    notes: list = []
    systems: dict = {}
    found_by_system: dict = {}          # name -> {wid: {value, ...}}
    input_files = [windows_path]

    def add_system(name, found, block_spans=None):
        systems[name] = build_system_report(
            found, window_ids, doc_ids, doc_to_windows, gold_by_window, gold_meta,
            args.resamples, args.seed, block_spans, gold_spans_by_window)
        found_by_system[name] = found

    # --- statcheck, raw and repaired ---
    statcheck_path = resolve(args.statcheck)
    if statcheck_path.exists():
        raw_found, _ = ev.load_statcheck(statcheck_path)
        add_system("statcheck_raw", raw_found)
        input_files.append(statcheck_path)
    else:
        notes.append(f"{statcheck_path} not found; statcheck_raw skipped")

    repaired_path = resolve(args.repaired)
    repaired_rows = []
    if repaired_path.exists():
        repaired_found, repaired_rows = ev.load_statcheck(repaired_path)
        add_system("statcheck_repaired", repaired_found)
        input_files.append(repaired_path)
    else:
        notes.append(f"{repaired_path} not found; statcheck_repaired skipped")

    verdicts = ev.verdict_agreement(repaired_rows) if repaired_rows else None

    # --- models ---
    best_record = top3 = None
    export_by_name = {}
    if not args.no_models:
        runs_path = resolve(args.runs)
        records = load_run_records(runs_path, set(args.only) if args.only else None)
        if records:
            input_files.append(runs_path)
        export_path = resolve(args.export)
        if export_path.exists():
            export_by_name = {r["name"]: r for r in ev.read_json(export_path)}

        models_dir = runs_path.parent        # each run lives in <runs.json's dir>/<name>/
        block_spans_by_system = {}
        for record in records:
            tagger = load_tagger(models_dir, record["name"])
            if tagger is None:
                continue
            preds = ev.model_predictions(tagger, windows)
            found = {wid: {round(r.statistic, 4) for r, _, _ in v} for wid, v in preds.items()}
            spans = {wid: {span for _, _, span in v} for wid, v in preds.items()}
            add_system(record["name"], found, spans)
            block_spans_by_system[record["name"]] = spans
            if record["name"] in export_by_name:
                a = export_by_name[record["name"]]
                systems[record["name"]]["artifact"] = {
                    "onnx_bytes": a.get("onnx_bytes"), "quant_bytes": a.get("quant_bytes"),
                    "latency_ms_median": a.get("latency_ms_median"), "dev_f1": a.get("dev_f1")}

        loaded = {r["name"] for r in records if r["name"] in found_by_system}
        best_record = pick_best_seed0([r for r in records if r["name"] in loaded])
        top3 = [r for r in top3_seed0(records) if r["name"] in loaded]
        if best_record is None:
            notes.append("no seed-0 model loaded; cascade and best_model comparisons skipped")
    else:
        notes.append("--no-models: model runs, cascade, and model comparisons skipped")

    # --- cascade: statcheck_repaired, plus what the best model adds ---
    cascade_name = None
    if best_record is not None and "statcheck_repaired" in found_by_system:
        cascade_name = f"cascade_{best_record['config']}-s0"
        base = found_by_system["statcheck_repaired"]
        model_found = found_by_system[best_record["name"]]
        cascade_found = {wid: base.get(wid, set()) | model_found.get(wid, set())
                         for wid in set(base) | set(model_found)}
        add_system(cascade_name, cascade_found)
    elif best_record is not None:
        notes.append("statcheck_repaired unavailable; cascade skipped")

    # --- seeds: mean/sd of holdout F1 per config ---
    seeds_report = {}
    if not args.no_models:
        by_config: dict = {}
        for record in [r for r in load_run_records(resolve(args.runs),
                                                    set(args.only) if args.only else None)
                       if r["name"] in found_by_system]:
            by_config.setdefault(record["config"], []).append(
                systems[record["name"]]["overall"]["f1"])
        for config, values in by_config.items():
            seeds_report[config] = {
                "mean_f1": statistics.fmean(values),
                "sd_f1": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }

    # --- paired tests and McNemar ---
    paired = {}
    mcnemar = {}
    if top3 and len(top3) >= 2:
        ranked = sorted(top3, key=lambda r: systems[r["name"]]["overall"]["f1"], reverse=True)
        names = [r["name"] for r in ranked]
        paired["top1_vs_top2"] = paired_entry(
            names[0], found_by_system[names[0]], names[1], found_by_system[names[1]],
            gold_by_window, gold_meta, doc_ids, doc_to_windows, args.resamples, args.seed)
        if len(names) >= 3:
            paired["top1_vs_top3"] = paired_entry(
                names[0], found_by_system[names[0]], names[2], found_by_system[names[2]],
                gold_by_window, gold_meta, doc_ids, doc_to_windows, args.resamples, args.seed)

    if cascade_name and "statcheck_repaired" in found_by_system:
        paired["cascade_vs_statcheck_repaired"] = paired_entry(
            cascade_name, found_by_system[cascade_name],
            "statcheck_repaired", found_by_system["statcheck_repaired"],
            gold_by_window, gold_meta, doc_ids, doc_to_windows, args.resamples, args.seed)
        mcnemar["cascade_vs_statcheck_repaired"] = mcnemar_entry(
            cascade_name, found_by_system[cascade_name],
            "statcheck_repaired", found_by_system["statcheck_repaired"], gold_by_window)

    if best_record and "statcheck_repaired" in found_by_system:
        paired["best_model_vs_statcheck_repaired"] = paired_entry(
            best_record["name"], found_by_system[best_record["name"]],
            "statcheck_repaired", found_by_system["statcheck_repaired"],
            gold_by_window, gold_meta, doc_ids, doc_to_windows, args.resamples, args.seed)
        mcnemar["best_model_vs_statcheck_repaired"] = mcnemar_entry(
            best_record["name"], found_by_system[best_record["name"]],
            "statcheck_repaired", found_by_system["statcheck_repaired"], gold_by_window)

    # --- family recall ---
    family = {}
    if "statcheck_repaired" in found_by_system:
        family["statcheck_repaired"] = ev.family_recall(
            found_by_system["statcheck_repaired"], gold_by_window, gold_meta)
    if best_record:
        family["best_model"] = ev.family_recall(
            found_by_system[best_record["name"]], gold_by_window, gold_meta)
    if cascade_name:
        family["cascade"] = ev.family_recall(
            found_by_system[cascade_name], gold_by_window, gold_meta)

    # --- provenance ---
    provenance = {
        "date": None,        # filled by the caller's log, not guessed here
        "git_sha": ev.git_sha(REPO),
        "seed": args.seed,
        "resamples": args.resamples,
        "systems": sorted(systems),
        "inputs": {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p):
                  ev.sha256_file(p) for p in input_files},
        # Gold results `align.as_number` cannot parse at all, so they never
        # enter value matching: no system can find them, and none is blamed
        # for missing them. See `align.as_number` for what it does and does
        # not repair.
        "gold_unparseable": {"count": len(gold_unparseable), "items": gold_unparseable},
    }

    report = {
        "provenance": provenance,
        "gold": {"windows": len(window_ids), "results": n_gold,
                "damaged": n_damaged, "checkable": n_checkable},
        "systems": systems,
        "seeds": seeds_report,
        "paired": paired,
        "mcnemar": mcnemar,
        "family_recall": family,
        "verdicts": verdicts,
        "notes": notes,
        "cascade_name": cascade_name,
        "best_model": best_record["name"] if best_record else None,
    }

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")

    print(f"{len(window_ids)} windows, {n_gold} gold results "
          f"({n_damaged} damaged, {n_checkable} checkable)\n")
    print(f"{'system':28s} {'P':>7s} {'R':>7s} {'F1':>7s} {'F1 95% CI':>17s} {'dmgR':>7s}")
    for name, s in systems.items():
        o, d = s.get("overall", {}), s.get("damaged", {})
        lo, hi = s.get("bootstrap", {}).get("overall", {}).get("f1", (0.0, 0.0))
        print(f"{name:28s} {o.get('p', 0):7.3f} {o.get('r', 0):7.3f} {o.get('f1', 0):7.3f} "
              f"[{lo:6.3f},{hi:6.3f}] {d.get('r', 0):7.3f}")
    for note in notes:
        print(f"note: {note}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
