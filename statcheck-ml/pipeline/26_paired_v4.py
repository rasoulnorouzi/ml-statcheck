"""Paired comparisons of the shipped cascade, the candidate cascade and R statcheck on the
gold-test rows, and one combined table. Reads only stored result files: no gold paper, no
model. Gold and bronze stay in separate columns.

Usage: python pipeline/26_paired_v4.py [--final results/v4/final] [--seed 0] [--resamples 2000]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from statcheck_ml.stats import mcnemar_exact, paired_bootstrap  # noqa: E402

FLAGGED = ("inconsistent", "decision_error")
PAIRS = [("cascade", "cascade_v4"), ("cascade", "statcheck_r_passages"),
         ("cascade", "statcheck_r_pdf"), ("cascade_v4", "statcheck_r_passages")]
# Table rows: label, verdict-file system, gold entry in summary, bronze entry.
SHIPPED_RUN, CANDIDATE_RUN = "old-gru-crf-s0", "lstm-softmax-none-s0"
TABLE = [
    ("shipped cascade (gru-crf-s0)", "cascade", ("runs", SHIPPED_RUN, "gold_cascade"), ("runs", SHIPPED_RUN, "bronze")),
    ("candidate cascade (lstm-softmax-none-s0)", "cascade_v4", ("runs", CANDIDATE_RUN, "gold_cascade"), ("runs", CANDIDATE_RUN, "bronze")),
    ("regex", "regex", ("baselines", "regex"), None),
    ("statcheck_r_fulltext_raw", "statcheck_r_fulltext_raw", ("baselines", "statcheck_r_fulltext_raw"), None),
    ("statcheck_r_fulltext_repaired", "statcheck_r_fulltext_repaired", ("baselines", "statcheck_r_fulltext_repaired"), None),
    ("statcheck_r_passages", "statcheck_r_passages", ("baselines", "statcheck_r_passages"), None),
    ("statcheck_r_pdf", "statcheck_r_pdf", ("baselines", "statcheck_r_pdf"), None),
    ("statcheck_raw (bronze windows only)", None, None, "statcheck_raw"),
    ("statcheck_repaired (bronze windows only)", None, None, "statcheck_repaired"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def found(row, system):
    return row["systems"].get(system) is not None


def flagged(row, system):
    return row["systems"].get(system) in FLAGGED


def paired(rows, a, b, hit, seed, resamples):
    """hit(row, system) -> bool. Rows are the population."""
    only_a = sum(1 for r in rows if hit(r, a) and not hit(r, b))
    only_b = sum(1 for r in rows if hit(r, b) and not hit(r, a))
    both = sum(1 for r in rows if hit(r, a) and hit(r, b))
    n = len(rows)
    by_paper = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        u = by_paper[r["doc"]]
        u[0] += hit(r, a)
        u[1] += hit(r, b)
        u[2] += 1
    units = [tuple(v) for _, v in sorted(by_paper.items())]

    def stat(i):
        return lambda us: (sum(u[i] for u in us) / sum(u[2] for u in us)) if sum(u[2] for u in us) else 0.0

    diff, (lo, hi), _ = paired_bootstrap(units, stat(0), stat(1), n=resamples, seed=seed)
    return {"a": a, "b": b, "n": n, "papers": len(units),
            "a_only": only_a, "b_only": only_b, "both": both, "neither": n - only_a - only_b - both,
            "recall_a": round((only_a + both) / n, 4) if n else None,
            "recall_b": round((only_b + both) / n, 4) if n else None,
            "mcnemar_p": mcnemar_exact(only_a, only_b),
            "recall_diff": round(diff, 4), "recall_diff_ci95": [round(lo, 4), round(hi, 4)]}


def dig(d, path):
    for k in path:
        d = d[k]
    return d


def build(final, seed, resamples):
    paths = {n: os.path.join(final, n) for n in ("verdicts.json", "summary.json", "bronze.json")}
    verdicts, summary, bronze = (json.load(open(paths[n], encoding="utf-8"))
                                 for n in ("verdicts.json", "summary.json", "bronze.json"))
    rows = verdicts["rows"]
    ref_flagged = [r for r in rows if r["reference"] in FLAGGED]
    out = {"n_rows": len(rows), "n_reference_flagged": len(ref_flagged), "seed": seed,
           "resamples": resamples,
           "found": [paired(rows, a, b, found, seed, resamples) for a, b in PAIRS],
           "reference_flagged": [paired(ref_flagged, a, b, flagged, seed, resamples) for a, b in PAIRS]}
    table = []
    for label, vsys, gpath, bpath in TABLE:
        row = {"system": label, "gold": None, "bronze": None}
        if gpath:
            g = dig(summary, gpath)
            row["gold"] = {"recall_named": g["R_named"], "recall_named_ci": g["R_named_ci"],
                           "finds_in_no_result_papers": g["FP"],
                           "recall_from_rows": round(sum(found(r, vsys) for r in rows) / len(rows), 3)}
        if bpath:
            b = dig(summary, bpath) if isinstance(bpath, tuple) else bronze["systems"][bpath]["overall"]
            if "precision" in b:
                row["bronze"] = {"precision": b["precision"], "recall": b["recall"], "f1": b["f1"]}
            else:
                row["bronze"] = {"precision": b["p"], "recall": b["r"], "f1": b["f1"]}
        table.append(row)
    out["table"] = table
    out["provenance"] = {"git_sha": git_sha(),
                         "inputs": {"results/v4/final/" + n: sha256(p) for n, p in paths.items()},
                         "command": "pipeline/26_paired_v4.py"}
    return out


def pct(x):
    return "—" if x is None else f"{x:.3f}"


def fmt_p(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def render(out):
    L = ["# Paired comparisons, version 4 gold test", "",
         "Source: stored files under `results/v4/final/`. No gold paper and no model is read.",
         f"Seed {out['seed']}, {out['resamples']} paper resamples.", "",
         "Rows within a paper are not independent. The paper bootstrap is therefore the main",
         "interval. The exact McNemar p treats rows as independent and is too small when results",
         "cluster in a few papers. Read it as a secondary figure.", ""]
    for key, title, unit in (("found", "Result found (verdict not null)", "gold rows"),
                             ("reference_flagged", "Reference inconsistency flagged", "gold rows with a reference verdict of inconsistent or decision_error")):
        n = out["n_rows"] if key == "found" else out["n_reference_flagged"]
        L += [f"## {title}", "", f"Population: {n} {unit}.", "",
              "| A | B | A only | B only | both | neither | recall A | recall B | diff A-B [95% paper CI] | McNemar p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for p in out[key]:
            lo, hi = p["recall_diff_ci95"]
            L.append(f"| {p['a']} | {p['b']} | {p['a_only']} | {p['b_only']} | {p['both']} | {p['neither']} | "
                     f"{pct(p['recall_a'])} | {pct(p['recall_b'])} | {p['recall_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {fmt_p(p['mcnemar_p'])} |")
        L.append("")
    L += ["## Combined table", "",
          "Gold columns come from the gold test; bronze columns come from the bronze holdout. The",
          "bronze labels are machine annotations, not human validation. The two are never pooled.",
          "Bronze figures for the two cascades are for the model alone, without the cascade.",
          "`statcheck_raw` and `statcheck_repaired` exist only on the bronze windows.", "",
          "| system | gold recall (named) [95% CI] | gold finds in no-result papers | bronze precision | bronze recall | bronze F1 |",
          "|---|---|---|---|---|---|"]
    for r in out["table"]:
        g, b = r["gold"], r["bronze"]
        gr = "—" if not g else f"{g['recall_named']:.3f} [{g['recall_named_ci'][0]:.3f}, {g['recall_named_ci'][1]:.3f}]"
        gf = "—" if not g else str(g["finds_in_no_result_papers"])
        L.append(f"| {r['system']} | {gr} | {gf} | " +
                 (" | ".join(pct(b[k]) for k in ("precision", "recall", "f1")) if b else "— | — | —") + " |")
    L += ["", "## Provenance", "", f"Git sha: `{out['provenance']['git_sha']}`", ""]
    L += [f"- `{k}`: `{v}`" for k, v in out["provenance"]["inputs"].items()]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", default=os.path.join(ROOT, "results/v4/final"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()
    out = build(args.final, args.seed, args.resamples)
    with open(os.path.join(args.final, "paired.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(args.final, "paired.md"), "w", encoding="utf-8") as fh:
        fh.write(render(out))


if __name__ == "__main__":
    main()
