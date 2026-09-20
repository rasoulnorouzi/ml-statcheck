"""Build `results/REPORT.md` from files other stages already wrote.

Every number in the report comes from a committed measurement file. Nothing
here computes a metric; it reads, counts, and fills a template. When an
input file does not exist yet (the model grid has not finished, say), the
table it feeds prints a one-line "not available" note instead of a gap that
would read like a zero. See `src/statcheck_ml/reportutil.py` for the
placeholder grammar and the table functions.

Usage:
  python pipeline/11_report.py --template docs/report_template.md \\
      --eval results/eval.json --agreement dataset/agreement \\
      --runs models/runs.json --export models/export.json \\
      --engines results/engines.json --dataset dataset \\
      --out results/REPORT.md [--readme-block results/readme_block.md]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]      # statcheck-ml
sys.path.insert(0, str(REPO / "src"))

from statcheck_ml import reportutil as ru  # noqa: E402

RATERS = ("haiku", "sonnet", "opus")
SETS = ("train", "holdout")


def resolve(p) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"), strict=False)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line, strict=False))
    return rows


# ----------------------------------------------------------- data reads --

def load_key(dataset_dir: Path) -> list:
    return read_json(dataset_dir / "windows" / "key.json") or []


def load_final(dataset_dir: Path, set_name: str) -> list:
    return read_json(dataset_dir / "annotations" / set_name / "final.json") or []


def load_rater(dataset_dir: Path, set_name: str, rater: str) -> list:
    return read_json(dataset_dir / "annotations" / set_name / f"{rater}.json") or []


def load_disputes(dataset_dir: Path, set_name: str):
    """The dispute list for one set, or None when disputes.json does not exist.

    Unlike the other loaders this keeps `None` distinct from an empty list: a
    set with zero disputes is real data, a missing file is not.
    """
    return read_json(dataset_dir / "annotations" / set_name / "disputes.json")


def pool_sentence(key_rows: list, set_name: str) -> str:
    counts = Counter(r["pool"] for r in key_rows if r.get("set") == set_name)
    if not counts:
        return "not available"
    return ", ".join(f"{pool} {n}" for pool, n in sorted(counts.items()))


def doc_count(key_rows: list, set_name: str) -> int:
    return len({r["doc"] for r in key_rows if r.get("set") == set_name})


def window_count(key_rows: list, set_name: str) -> int:
    return sum(1 for r in key_rows if r.get("set") == set_name)


def dataset_jsonl_stats(dataset_dir: Path, set_name: str) -> dict:
    rows = read_jsonl(dataset_dir / f"{set_name}.jsonl")
    results = [res for row in rows for res in row.get("results", [])]
    n = len(results)
    damaged = sum(1 for res in results if res.get("damaged"))
    return {"n_results": n, "damaged": damaged, "damaged_share": (damaged / n) if n else None}


def tier_and_dispute_counts(final_rows: list) -> dict:
    tiers = Counter()
    for row in final_rows:
        for res in row.get("results", []):
            tiers[res.get("tier")] += 1
    prov = final_rows[0].get("provenance", {}) if final_rows else {}
    return {"tiers": dict(tiers), "n_disputes": prov.get("n_disputes"),
            "n_adjudicated": prov.get("n_adjudicated")}


# --------------------------------------------------------- table rows ----

def build_dataset_counts_rows(dataset_dir: Path, key_rows: list) -> list:
    rows = []
    for set_name in SETS:
        stats = dataset_jsonl_stats(dataset_dir, set_name)
        w = window_count(key_rows, set_name)
        if w == 0 and stats["n_results"] == 0:
            continue
        rows.append({"set": set_name, "windows": w, "docs": doc_count(key_rows, set_name),
                    "results": stats["n_results"], "damaged": stats["damaged"],
                    "damaged_share": stats["damaged_share"]})
    return rows


def build_rater_yield_rows(dataset_dir: Path) -> list:
    rows = []
    for set_name in SETS:
        for rater in RATERS:
            recs = load_rater(dataset_dir, set_name, rater)
            if not recs:
                continue
            n_results = sum(len(r.get("results") or []) for r in recs)
            rows.append({"set": set_name, "rater": rater, "windows": len(recs),
                        "results": n_results})
    return rows


def build_disputes_by_set(dataset_dir: Path) -> dict:
    return {set_name: load_disputes(dataset_dir, set_name) for set_name in SETS}


def build_tiers_by_set(dataset_dir: Path) -> dict:
    out = {}
    for set_name in SETS:
        final_rows = load_final(dataset_dir, set_name)
        out[set_name] = ({"windows": len(final_rows),
                          "tiers": tier_and_dispute_counts(final_rows)["tiers"]}
                         if final_rows else None)
    return out


def build_benchmark_screen_rows(runs, eval_data) -> list:
    if not runs:
        return []
    systems = (eval_data or {}).get("systems", {})
    rows = []
    for r in runs:
        if r.get("seed") != 0 or "noaug" in (r.get("config") or ""):
            continue
        sysrec = systems.get(r["name"], {})
        rows.append({
            "config": r["config"], "params": r.get("params"), "dev_f1": r.get("dev_f1"),
            "holdout_f1": sysrec.get("overall", {}).get("f1"),
            "holdout_ci": sysrec.get("bootstrap", {}).get("overall", {}).get("f1"),
            "wall_min": (r["wall_seconds"] / 60) if r.get("wall_seconds") is not None else None,
        })
    return rows


# ------------------------------------------------------------- values ----

def build_values(dataset_dir: Path, agreement: dict, runs, export, engines, eval_data,
                 versions) -> dict:
    values: dict = {}
    key_rows = load_key(dataset_dir)

    for set_name in SETS:
        values[f"n_{set_name}_windows"] = window_count(key_rows, set_name)
        values[f"n_{set_name}_docs"] = doc_count(key_rows, set_name)
        values[f"pools_{set_name}"] = pool_sentence(key_rows, set_name)

        stats = dataset_jsonl_stats(dataset_dir, set_name)
        values[f"n_results_{set_name}"] = stats["n_results"]
        values[f"damaged_share_{set_name}"] = stats["damaged_share"]

        for rater in RATERS:
            recs = load_rater(dataset_dir, set_name, rater)
            values[f"results_{rater}_{set_name}"] = (
                sum(len(r.get("results") or []) for r in recs) if recs else None)

        final_rows = load_final(dataset_dir, set_name)
        td = tier_and_dispute_counts(final_rows)
        for tier in ("unanimous", "majority", "adjudicated"):
            values[f"tier_{tier}_{set_name}"] = td["tiers"].get(tier, 0) if final_rows else None
        values[f"n_disputes_{set_name}"] = td["n_disputes"]
        values[f"n_adjudicated_{set_name}"] = td["n_adjudicated"]

        disputes = load_disputes(dataset_dir, set_name)
        if disputes is not None:
            values[f"n_singletons_{set_name}"] = sum(1 for d in disputes
                                                      if d.get("kind") == "singleton")
            values[f"n_field_conflicts_{set_name}"] = sum(1 for d in disputes
                                                           if d.get("kind") == "field_conflict")
            values[f"n_kept_singletons_{set_name}"] = sum(
                1 for d in disputes if d.get("kind") == "singleton"
                and (d.get("decision") or {}).get("keep"))
        else:
            values[f"n_singletons_{set_name}"] = None
            values[f"n_field_conflicts_{set_name}"] = None
            values[f"n_kept_singletons_{set_name}"] = None

        agr = agreement.get(set_name) or {}
        w = agr.get("window", {})
        r_strict = agr.get("result", {}).get("strict", {}).get("mean", {})
        r_lenient = agr.get("result", {}).get("lenient", {}).get("mean", {})
        values[f"fleiss_{set_name}"] = w.get("fleiss", {}).get("value")
        values[f"fleiss_{set_name}_ci"] = w.get("fleiss", {}).get("ci")
        values[f"alpha_{set_name}"] = w.get("alpha", {}).get("value")
        values[f"alpha_{set_name}_ci"] = w.get("alpha", {}).get("ci")
        values[f"strict_mean_{set_name}"] = r_strict.get("value")
        values[f"strict_mean_{set_name}_ci"] = r_strict.get("ci")
        values[f"lenient_mean_{set_name}"] = r_lenient.get("value")
        values[f"lenient_mean_{set_name}_ci"] = r_lenient.get("ci")

        vs_final = agr.get("result", {}).get("vs_final") or {}
        for rater in RATERS:
            values[f"vsfinal_{rater}_{set_name}"] = vs_final.get(rater, {}).get("f1")

    values["r_version"] = (versions or {}).get("r_version")
    values["statcheck_version"] = (versions or {}).get("statcheck_version")
    values["engine_spread"] = (engines or {}).get("spread")
    values["engine_limit"] = (engines or {}).get("limit")

    provenance = (eval_data or {}).get("provenance", {})
    values["git_sha"] = provenance.get("git_sha")
    values["gold_unparseable"] = (provenance.get("gold_unparseable") or {}).get("count")

    # The report's own header names the commit and date it was generated at,
    # from HEAD, never from `datetime.now()`: two runs at the same commit
    # render the same bytes.
    git_sha_short, generated_at = ru.git_head_info(REPO)
    values["git_sha_short"] = git_sha_short or "unknown"
    values["generated_at"] = generated_at or "unknown"

    systems = (eval_data or {}).get("systems", {})

    def sysval(name, bucket, field):
        return systems.get(name, {}).get(bucket, {}).get(field) if name else None

    values["statcheck_raw_f1"] = sysval("statcheck_raw", "overall", "f1")
    values["statcheck_raw_r"] = sysval("statcheck_raw", "overall", "r")
    values["statcheck_repaired_f1"] = sysval("statcheck_repaired", "overall", "f1")
    values["statcheck_repaired_r"] = sysval("statcheck_repaired", "overall", "r")

    best_name = (eval_data or {}).get("best_model")
    runs_by_name = {r["name"]: r for r in (runs or [])}
    best_config = None
    if best_name:
        best_config = (runs_by_name[best_name]["config"] if best_name in runs_by_name
                       else re.sub(r"-s\d+$", "", best_name))
    values["best_config"] = best_config
    values["best_f1"] = sysval(best_name, "overall", "f1")
    values["best_ci"] = (systems.get(best_name, {}).get("bootstrap", {})
                         .get("overall", {}).get("f1")) if best_name else None

    zoo_names = [r["name"] for r in (export or []) if r.get("in_zoo")]
    if not zoo_names and runs:
        seed0 = [r for r in runs if r.get("seed") == 0 and "noaug" not in (r.get("config") or "")
                and r.get("dev_f1") is not None]
        zoo_names = [r["name"] for r in sorted(seed0, key=lambda r: -r["dev_f1"])[:3]]
    values["top3"] = (", ".join(runs_by_name.get(n, {}).get("config", n) for n in zoo_names)
                      or None)

    cascade_name = (eval_data or {}).get("cascade_name")
    values["cascade_f1"] = sysval(cascade_name, "overall", "f1")
    values["cascade_ci"] = (systems.get(cascade_name, {}).get("bootstrap", {})
                            .get("overall", {}).get("f1")) if cascade_name else None

    paired = (eval_data or {}).get("paired", {})
    pc = paired.get("cascade_vs_statcheck_repaired")
    values["paired_cascade_diff"] = pc.get("diff") if pc else None
    values["paired_cascade_ci"] = pc.get("ci") if pc else None
    values["paired_cascade_p"] = pc.get("p") if pc else None

    mcnemar = (eval_data or {}).get("mcnemar", {})
    mc = mcnemar.get("cascade_vs_statcheck_repaired")
    values["mcnemar_cascade_b"] = mc.get("b_only_a") if mc else None
    values["mcnemar_cascade_c"] = mc.get("b_only_b") if mc else None
    values["mcnemar_cascade_p"] = mc.get("p") if mc else None

    values["n_runs"] = len(runs) if runs else None
    values["total_train_hours"] = (
        sum(r["wall_seconds"] for r in runs if r.get("wall_seconds") is not None) / 3600
        if runs else None)

    # --- values the interpretation sentences need, all from the same files ---
    gold = (eval_data or {}).get("gold", {})
    values["eval_gold_results"] = gold.get("results")
    values["eval_gold_damaged"] = gold.get("damaged")
    values["eval_gold_checkable"] = gold.get("checkable")

    verdicts = (eval_data or {}).get("verdicts") or {}
    values["verdict_agree"] = verdicts.get("agree")
    values["verdict_disagree"] = verdicts.get("disagree")
    values["verdict_undecidable"] = verdicts.get("undecidable")

    for key in ("top1_vs_top2", "top1_vs_top3", "best_model_vs_statcheck_repaired"):
        entry = paired.get(key) or {}
        values[f"paired_{key}_a"] = entry.get("a")
        values[f"paired_{key}_b"] = entry.get("b")
        values[f"paired_{key}_diff"] = entry.get("diff")
        values[f"paired_{key}_ci"] = entry.get("ci")
        values[f"paired_{key}_p"] = entry.get("p")

    # The ablation: the best config trained without augmentation.
    ablation = next((r for r in (runs or []) if "noaug" in (r.get("config") or "")), None)
    values["ablation_name"] = ablation["name"] if ablation else None
    values["ablation_f1"] = sysval(ablation["name"], "overall", "f1") if ablation else None
    values["ablation_damaged_r"] = sysval(ablation["name"], "damaged", "r") if ablation else None
    values["best_damaged_r"] = sysval(best_name, "damaged", "r")

    # The seed-0 run with the highest holdout F1, which the protocol did not
    # use for selection, and whether it reached the zoo.
    seed0_runs = [r for r in (runs or []) if r.get("seed") == 0
                  and "noaug" not in (r.get("config") or "") and r["name"] in systems]
    top_holdout = max(seed0_runs, key=lambda r: systems[r["name"]]["overall"]["f1"],
                      default=None)
    values["holdout_top_config"] = top_holdout["config"] if top_holdout else None
    values["holdout_top_f1"] = (sysval(top_holdout["name"], "overall", "f1")
                                if top_holdout else None)
    values["holdout_top_dev_f1"] = top_holdout.get("dev_f1") if top_holdout else None
    values["holdout_top_in_zoo"] = ("yes" if top_holdout and top_holdout["config"] in
                                    (values.get("top3") or "") else "no")

    # Runs whose loss blew up: a tenfold rise between two epochs to above 100.
    divergent = []
    for r in (runs or []):
        report_path = REPO / "models" / r["name"] / "report.json"
        if not report_path.exists():
            continue
        history = json.loads(report_path.read_text(encoding="utf-8")).get("history", [])
        for prev, cur in zip(history, history[1:]):
            if cur["loss"] > 100 and cur["loss"] > 10 * prev["loss"]:
                divergent.append(f"{r['name']} at epoch {cur['epoch']} "
                                 f"(loss {prev['loss']:.1f} to {cur['loss']:.0f}, "
                                 f"dev F1 {cur['f1']:.3f})")
                break
    values["divergent_runs"] = "; ".join(divergent) if divergent else "none"
    values["n_divergent"] = len(divergent)

    return values


# --------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="docs/report_template.md")
    ap.add_argument("--eval", default="results/eval.json")
    ap.add_argument("--agreement", default="dataset/agreement")
    ap.add_argument("--runs", default="models/runs.json")
    ap.add_argument("--export", default="models/export.json")
    ap.add_argument("--engines", default="results/engines.json")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="results/REPORT.md")
    ap.add_argument("--readme-block", default=None)
    args = ap.parse_args()

    dataset_dir = resolve(args.dataset)
    agreement_dir = resolve(args.agreement)

    eval_data = read_json(resolve(args.eval))
    runs = read_json(resolve(args.runs))
    export = read_json(resolve(args.export))
    engines = read_json(resolve(args.engines))
    versions = read_json(dataset_dir / "baseline" / "versions.json")
    agreement = {s: read_json(agreement_dir / f"{s}.json") for s in SETS}

    key_rows = load_key(dataset_dir)
    values = build_values(dataset_dir, agreement, runs, export, engines, eval_data, versions)

    tables = {
        "dataset_counts": lambda: ru.dataset_counts(build_dataset_counts_rows(dataset_dir, key_rows)),
        "rater_yield": lambda: ru.rater_yield(build_rater_yield_rows(dataset_dir)),
        "agreement_window": lambda: ru.agreement_window(agreement),
        "agreement_result": lambda: ru.agreement_result(agreement),
        "agreement_field": lambda: ru.agreement_field(agreement),
        "rater_vs_final": lambda: ru.rater_vs_final(agreement),
        "adjudication": lambda: ru.adjudication(build_disputes_by_set(dataset_dir),
                                                build_tiers_by_set(dataset_dir)),
        "benchmark_screen": lambda: ru.benchmark_screen(build_benchmark_screen_rows(runs, eval_data)),
        "benchmark_seeds": lambda: ru.benchmark_seeds((eval_data or {}).get("seeds")),
        "systems_overall": lambda: ru.systems_overall((eval_data or {}).get("systems")),
        "systems_damaged": lambda: ru.systems_damaged((eval_data or {}).get("systems")),
        "family_recall": lambda: ru.family_recall_table((eval_data or {}).get("family_recall")),
        "paired_tests": lambda: ru.paired_tests((eval_data or {}).get("paired"),
                                                (eval_data or {}).get("mcnemar")),
        "zoo": lambda: ru.zoo(export),
        "engines": lambda: ru.engines_table(engines),
    }

    template_path = resolve(args.template)
    template_text = template_path.read_text(encoding="utf-8")
    try:
        rendered = ru.render(template_text, values, tables)
    except ru.ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"wrote {out_path}")

    if args.readme_block:
        systems = (eval_data or {}).get("systems", {})
        zoo_runs = [r["name"] for r in (export or []) if r.get("in_zoo")]
        readme_rows = (["statcheck_raw", "statcheck_repaired"] + sorted(zoo_runs)
                       + [n for n in systems if n.startswith("cascade")])
        block = ru.readme_block(values, ru.systems_overall(systems, only=readme_rows))
        block_path = resolve(args.readme_block)
        block_path.parent.mkdir(parents=True, exist_ok=True)
        block_path.write_text(block, encoding="utf-8")
        print(f"wrote {block_path}")


if __name__ == "__main__":
    main()
