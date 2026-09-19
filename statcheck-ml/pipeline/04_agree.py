"""Inter-rater agreement for one annotation set.

Three blind raters label the same windows. This measures agreement at each
level: whether a window holds a result, whether raters point at the same
span, and whether the fields of a matched result read the same. Every
number carries a bootstrap interval over the windows.

Usage:
  python pipeline/04_agree.py --annotations dataset/annotations/train \
      --raters haiku sonnet opus [--final dataset/annotations/train/final.json] \
      --out dataset/agreement/train.json [--resamples 1000] [--seed 0] \
      [--windows dataset/windows/train.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statcheck_ml.agreement import (  # noqa: E402
    match_results, pairwise_f1, window_agreement, result_agreement,
    field_agreement, CATEGORICAL_FIELDS, NUMERIC_FIELDS,
)
from statcheck_ml.align import align_quote  # noqa: E402

def read_json(path: Path) -> object:
    """Load JSON the way rater output needs: quotes carry raw control chars."""
    return json.loads(path.read_text(encoding="utf-8"), strict=False)

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_raters(annotations_dir: Path, rater_names: list[str]) -> dict:
    return {name: {rec["window_id"]: rec for rec in read_json(annotations_dir / f"{name}.json")}
            for name in rater_names}

def attach_spans_to_records(records: dict, windows: dict) -> None:
    """Fill `span` on results that lack it, using window text.

    A result whose window text is unavailable, or whose quote does not
    align, keeps `span: None`, and `match_results` falls back to the
    statistic.
    """
    for wid, rec in records.items():
        text = windows.get(wid)
        for res in rec.get("results", []):
            if res.get("span") is not None:
                continue
            span = align_quote(text, res.get("quote")) if text is not None else None
            res["span"] = span
            res["aligned"] = span is not None

def attach_missing_spans(raters: dict, windows_path: Path | None,
                          final: dict | None = None) -> None:
    """Attach spans to every rater, and to `final` when given.

    Both sides need spans, or a rater matched against an unaligned `final`
    would silently fall back to the weaker statistic-only match.
    """
    if windows_path is None:
        return
    windows = {w["window_id"]: w["text"] for w in read_json(windows_path)}
    for records in raters.values():
        attach_spans_to_records(records, windows)
    if final is not None:
        attach_spans_to_records(final, windows)

def resample(raters: dict, window_ids: list[str]) -> dict:
    """Rebuild the rater dicts over a (possibly repeated) list of window ids.

    Resampling draws ids with replacement, so a synthetic key per draw
    keeps repeats from colliding in the per-window dicts.
    """
    out = {name: {} for name in raters}
    for k, wid in enumerate(window_ids):
        for name, records in raters.items():
            out[name][f"{wid}__{k}"] = records[wid]
    return out

def bootstrap_many(window_ids: list[str], compute, n: int, seed: int, level: float = 0.95):
    """Bootstrap several related metrics in one pass over the resamples.

    `compute(ids)` returns a flat dict of metric name to value for one
    resample; returns the point estimate and percentile interval per metric.
    """
    point = compute(window_ids)
    rng = np.random.default_rng(seed)
    m = len(window_ids)
    if m == 0:
        return {k: {"value": v, "ci": [float("nan"), float("nan")]} for k, v in point.items()}
    samples = {k: np.empty(n) for k in point}
    for i in range(n):
        ids = [window_ids[j] for j in rng.integers(0, m, size=m)]
        for k, v in compute(ids).items():
            samples[k][i] = v
    alpha = (1 - level) / 2
    return {k: {"value": v, "ci": [float(np.quantile(samples[k], alpha)),
                                    float(np.quantile(samples[k], 1 - alpha))]}
            for k, v in point.items()}

def vs_final(raters: dict, final: dict) -> dict:
    """Precision, recall, and F1 of each rater against the adjudicated file.

    Lenient span matching. Not bootstrapped: `final` is a fixed reference,
    not a sample the raters were drawn from.
    """
    out = {}
    for name, records in raters.items():
        window_ids = sorted(set(records.keys()) & set(final.keys()))
        tp = fp = fn = 0
        for w in window_ids:
            a, b = records[w].get("results", []), final[w].get("results", [])
            pairs = match_results(a, b, mode="lenient")
            tp += len(pairs)
            fp += len(a) - len(pairs)
            fn += len(b) - len(pairs)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[name] = {"p": p, "r": r, "f1": f1}
    return out

def print_table(report: dict) -> None:
    def fmt(entry):
        v, ci = entry["value"], entry["ci"]
        return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

    print(f"set: {report['set']}  windows: {report['n_windows']}  raters: {', '.join(report['raters'])}")
    print("\nwindow agreement (contains_result)")
    w = report["window"]
    print(f"  raw pct agreement   {fmt(w['raw_pct'])}")
    for pair, entry in w["cohen"].items():
        print(f"  cohen kappa {pair:<20} {fmt(entry)}")
    print(f"  fleiss kappa        {fmt(w['fleiss'])}")
    print(f"  krippendorff alpha  {fmt(w['alpha'])}")
    for mode in ("strict", "lenient"):
        r = report["result"][mode]
        print(f"\nresult agreement, {mode} span match")
        for pair, entry in r["pairwise"].items():
            print(f"  F1 {pair:<24} {fmt(entry)}")
        print(f"  F1 mean{'':<20} {fmt(r['mean'])}")
    if report["result"]["vs_final"] is not None:
        print("\nrater vs final (lenient)")
        for name, m in report["result"]["vs_final"].items():
            print(f"  {name:<10} P {m['p']:.3f}  R {m['r']:.3f}  F1 {m['f1']:.3f}")
    print(f"\nfield agreement, over {report['field']['n_results_all_raters']} results every rater matched")
    for field, entry in report["field"].items():
        if field != "n_results_all_raters":
            print(f"  {field:<12} {fmt(entry)}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--raters", required=True, nargs="+")
    ap.add_argument("--final", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=Path, default=None)
    args = ap.parse_args()

    raters = load_raters(args.annotations, args.raters)
    final_records = ({rec["window_id"]: rec for rec in read_json(args.final)}
                      if args.final is not None else None)
    attach_missing_spans(raters, args.windows, final=final_records)
    window_ids = sorted(set.intersection(*(set(r.keys()) for r in raters.values())))

    def window_metrics(ids):
        wa = window_agreement(resample(raters, ids))
        out = {"raw_pct": wa["raw_pct"], "fleiss": wa["fleiss"], "alpha": wa["alpha"]}
        out.update({f"cohen::{pair}": v for pair, v in wa["cohen"].items()})
        return out

    def result_metrics(mode):
        def inner(ids):
            ra = result_agreement(resample(raters, ids), mode=mode)
            out = {"mean": ra["mean"]}
            out.update({f"pair::{pair}": v for pair, v in ra["pairwise"].items()})
            return out
        return inner

    def field_metrics(ids):
        fa = field_agreement(resample(raters, ids))
        return {k: v for k, v in fa.items() if k != "n_results_all_raters"}

    window_boot = bootstrap_many(window_ids, window_metrics, args.resamples, args.seed)
    strict_boot = bootstrap_many(window_ids, result_metrics("strict"), args.resamples, args.seed)
    lenient_boot = bootstrap_many(window_ids, result_metrics("lenient"), args.resamples, args.seed)
    field_boot = bootstrap_many(window_ids, field_metrics, args.resamples, args.seed)
    field_point = field_agreement(raters)

    def shape_window(boot):
        cohen = {k.split("::", 1)[1]: v for k, v in boot.items() if k.startswith("cohen::")}
        return {"raw_pct": boot["raw_pct"], "cohen": cohen, "fleiss": boot["fleiss"],
                "alpha": boot["alpha"]}

    def shape_result(boot):
        pairwise = {k.split("::", 1)[1]: v for k, v in boot.items() if k.startswith("pair::")}
        return {"pairwise": pairwise, "mean": boot["mean"]}

    inputs = {str(args.annotations / f"{r}.json"): sha256_of(args.annotations / f"{r}.json")
              for r in args.raters}
    final_scores = None
    if args.final is not None:
        final_scores = vs_final(raters, final_records)
        inputs[str(args.final)] = sha256_of(args.final)
    if args.windows is not None:
        inputs[str(args.windows)] = sha256_of(args.windows)

    field_report = {field: field_boot[field] for field in CATEGORICAL_FIELDS + NUMERIC_FIELDS}
    field_report["n_results_all_raters"] = field_point["n_results_all_raters"]

    report = {
        "set": args.annotations.name,
        "n_windows": len(window_ids),
        "raters": list(args.raters),
        "window": shape_window(window_boot),
        "result": {"strict": shape_result(strict_boot), "lenient": shape_result(lenient_boot),
                   "vs_final": final_scores},
        "field": field_report,
        "provenance": {"resamples": args.resamples, "seed": args.seed, "inputs": inputs},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print_table(report)

if __name__ == "__main__":
    main()
