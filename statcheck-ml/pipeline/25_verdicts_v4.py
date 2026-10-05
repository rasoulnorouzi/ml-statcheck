"""Verdicts on the gold rows: what a user sees, scored against R statcheck's own verdict.

14_gold_eval.py asks whether a system finds a result. This stage asks whether
the system then says the right thing about it: consistent, inconsistent, or a
decision error.

Reference. For each in-scope (t F r z chi2 Q), located gold row of the chosen
papers, an APA string is built from the coded values, for example
" F(1, 184) = 7.64, p < .01". statcheck's rounding tolerance depends on the
decimals of the statistic and of p, so each number is written as the coder's
copy (`raw`) prints it when `raw` holds it, else as the coded value. All
strings go through R statcheck::statcheck() in one Rscript call
(r/run_statcheck_rows.R); its `error` and `decision_error` are the reference.
Two conventions, both stated in the output:
  - The operators "≤" and "≥" are written "<" and ">". statcheck 1.5.0 does
    not read "≤" at all, and the project reads it as "<" (provenance.py).
  - The statistic is written without its sign. Every p-value statcheck
    computes depends on the absolute value only, and the coders did not
    always copy the sign.
Every string starts with a space, because statcheck reads "z = 2.50" only
after a non-letter character.

A row with no reference is counted by cause: a part the coders did not code,
or statcheck found nothing, raised an error, read another value, or gave no
verdict.

pvalue.py is checked against the reference on the same coded values and the
same printed decimals, through statcheck_ml.pvalue.check with the pipeline's
settings. The rows where the two disagree are listed. This stage reports
them and does not resolve them.

Systems. Finds are matched to gold rows by exactly the rule of
14_gold_eval.py (match_local, match_statcheck), over the same list of located
rows in the same order, so the hits are the same as there. match_served
applies that rule and also says which find served each row; the tests show
that its hits equal those of the two originals.
  cascade         the shipped model, as Pipeline() loads it
  cascade_v4      models/v4/lstm-softmax-none-s0/zoo, the run the rules named
  regex           the ported pattern after operator repair
  statcheck_r_*   the four R statcheck forms, from dataset/baseline/gold
A local system's verdict is the "verdict" of the matched entry of
Pipeline.run_text(text)["results"]. An R form's verdict is its error and
decision_error columns.

"Flagged" means the verdict is inconsistent or decision_error, the two
verdicts statcheck reports as an error.

Gold discipline. --papers gold_train reads the gold-train papers only, for
development, and writes no log line. --papers gold_test reads the frozen test
part once and appends one line to results/gold_eval_log.jsonl. Nothing here
feeds a choice.

Usage: python pipeline/25_verdicts_v4.py --papers gold_test
"""
import argparse
import collections
import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("gold_eval_14", ROOT / "pipeline" / "14_gold_eval.py")
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

from statcheck_ml.pvalue import (CONSISTENT, DECISION_ERROR, INCONSISTENT,  # noqa: E402
                                 UNDECIDABLE, Result, check, compute_p)

LOCAL = ["cascade", "cascade_v4", "regex"]
SYSTEMS = LOCAL + list(ge.STATCHECK)
FLAGS = {INCONSISTENT, DECISION_ERROR}
VERDICTS = [CONSISTENT, INCONSISTENT, DECISION_ERROR, UNDECIDABLE]
ALPHA = 0.05
V4_MODEL = "models/v4/lstm-softmax-none-s0/zoo"
R_SCRIPT = ROOT / "r" / "run_statcheck_rows.R"

OPERATORS = {"=": "=", "<": "<", ">": ">", "≤": "<", "≥": ">", "ns": "ns"}
R_TEST = {"t": "t", "F": "F", "r": "r", "Z": "z", "Chi2": "chi2", "Q": "Q"}
ONE_DF = {"t", "r", "chi2", "Q"}
# A number as printed: thousands groups, a bare leading point, or an integer.
NUMBER = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)")

# The coded values passed through a spreadsheet, so 7.64 can come back as
# 7.6400000000000006. Values that differ by less than this are the same
# printed number; two distinct printed numbers differ by far more.
SAME_REL = 1e-9
# scipy and R compute the tail probabilities with different algorithms. The
# verdict needs p to about five decimals; a relative difference above this is
# reported as a disagreement in the computed p even when the verdicts agree.
P_REL_TOL = 1e-6


def same(a, b):
    return math.isclose(a, b, rel_tol=SAME_REL, abs_tol=1e-12)


def plain(x):
    """A float in positional notation with no trailing zeros: 1e-05 -> 0.00001."""
    return np.format_float_positional(float(x), trim="-")


def decimals(text):
    return len(text.split(".", 1)[1]) if "." in text else 0


def numbers(raw):
    """[(position, text)] of every number in raw, thousands commas removed."""
    return [(m.start(), m.group().replace(",", "")) for m in NUMBER.finditer(raw or "")]


def stat_as_written(raw, stat):
    """(text, position in raw or -1, ambiguous) for the absolute statistic.

    Among the numbers in raw equal to the statistic, the one with the most
    decimals wins: a degree of freedom equal to the statistic is printed as
    an integer, the statistic usually is not. `ambiguous` is true when the
    candidates print with different decimals.
    """
    hits = [(pos, tok) for pos, tok in numbers(raw) if same(float(tok), abs(stat))]
    if not hits:
        return plain(abs(stat)), -1, False
    best = max(hits, key=lambda h: (decimals(h[1]), -h[0]))
    return best[1], best[0], len({decimals(t) for _, t in hits}) > 1


def p_as_written(raw, p, after):
    """(text, from raw) for p: the first equal number after the statistic, else the first."""
    hits = [(pos, tok) for pos, tok in numbers(raw) if same(float(tok), p)]
    later = [h for h in hits if h[0] > after]
    pick = (later or hits or [None])[0]
    return (pick[1], True) if pick else (plain(p), False)


def operator(op):
    if op is None:
        return None
    return OPERATORS.get(str(op).strip().strip('"').strip())


def build(g):
    """The APA string of one gold row and the values our check reads, or a cause."""
    t = g["test_type"]
    df_needed = {"F": ("df1", "df2"), "z": ()}.get(t, ("df",))
    if t in ONE_DF:
        # Coders follow statcheck: the df of t and r sit in df2, of chi2 and Q in df1.
        first, second = ("df2", "df1") if t in ("t", "r") else ("df1", "df2")
        df = g[first] if g[first] is not None else g[second]
        dfs = {"df": df}
    else:
        dfs = {"df1": g["df1"], "df2": g["df2"]}
    op = operator(g["p_operator"])
    out = {"string": None, "cause": None, "operator_mapped": g["p_operator"] in ("≤", "≥")}
    if g["stat"] is None:
        out["cause"] = "no statistic coded"
    elif any(dfs.get(k) is None for k in df_needed):
        out["cause"] = "no degrees of freedom coded"
    elif op != "ns" and g["p"] is None:
        out["cause"] = "no p-value coded"
    elif op is None:
        out["cause"] = "no p operator coded"
    if out["cause"]:
        return out
    stat_text, at, ambiguous = stat_as_written(g["raw"], g["stat"])
    d = {k: plain(v) for k, v in dfs.items() if v is not None}
    head = {"t": f"t({d.get('df')})", "r": f"r({d.get('df')})", "chi2": f"χ2({d.get('df')})",
            "Q": f"Q({d.get('df')})", "z": "z",
            "F": f"F({d.get('df1')}, {d.get('df2')})"}[t]
    if op == "ns":
        p_text, p_raw, tail = None, False, ", ns"
    else:
        p_text, p_raw = p_as_written(g["raw"], g["p"], at)
        tail = f", p {op} {p_text}"
    df1 = dfs["df"] if t in ONE_DF else dfs.get("df1")
    out.update(string=f" {head} = {stat_text}{tail}", stat_text=stat_text, p_text=p_text,
               stat_from_raw=at >= 0, p_from_raw=p_raw, stat_ambiguous=ambiguous,
               test=t, df1=df1, df2=None if t in ONE_DF else dfs.get("df2"), op=op)
    return out


def run_r(rows, rscript="Rscript"):
    """{id: R outcome} for [{"id", "text"}], in one Rscript call."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "rows.json"), os.path.join(tmp, "out.json")
        with open(src, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        subprocess.run([rscript, str(R_SCRIPT), src, dst], check=True,
                       capture_output=True, text=True)
        with open(dst, encoding="utf-8") as fh:
            return {o["id"]: o for o in json.load(fh)}


def r_verdict(error, decision_error):
    if error is None:
        return None
    return DECISION_ERROR if decision_error else INCONSISTENT if error else CONSISTENT


def reference(b, r):
    """(verdict, cause, R computed p) of one built row from its R outcome."""
    if r["status"] == "none":
        return None, "statcheck found no result", None
    if r["status"] == "error":
        return None, "statcheck raised an error", None
    if r["status"] == "several":
        return None, "statcheck found several results", None
    read_p = r["reported_p"] is None if b["op"] == "ns" else (
        r["reported_p"] is not None and same(r["reported_p"], float(b["p_text"])))
    if (R_TEST.get(r["test_type"]) != b["test"] or not same(abs(r["test_value"]), float(b["stat_text"]))
            or not read_p):
        return None, "statcheck read another value", None
    verdict = r_verdict(r["error"], r["decision_error"])
    if verdict is None:
        return None, "statcheck gave no verdict", r["computed_p"]
    return verdict, None, r["computed_p"]


def ours(b):
    """pvalue.py on the same values and decimals, with the pipeline's settings."""
    res = Result(test_type=b["test"], statistic=float(b["stat_text"]), df1=b["df1"], df2=b["df2"],
                 p_operator=b["op"], p_value=float(b["p_text"]) if b["p_text"] else None)
    c = check(res, alpha=ALPHA, reported_p_text=b["p_text"], statistic_text=b["stat_text"])
    return c.verdict, compute_p(res.test_type, res.statistic, res.df1, res.df2)


# ---------------- finds and matching ----------------

def local_finds(report, units):
    """finds_for of 14_gold_eval.py, named finds only, each with its verdict and its index.

    [first line, last line, statistic, test type, used, verdict, index in results]
    """
    ranges = [(u.start_line, u.end_line) for u in units]
    out = []
    for i, r in enumerate(report["results"]):
        if r["statistic"] is None:
            continue
        lo, hi = next(((a, b) for a, b in ranges if a <= r["line"] <= b), (r["line"], r["line"]))
        out.append([lo, hi, round(abs(r["statistic"]), 4), r["test_type"], False, r["verdict"], i])
    return out


def read_statcheck_verdicts(path):
    """read_statcheck of 14_gold_eval.py with the verdict appended to each find."""
    out = collections.defaultdict(list)
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                stat = round(abs(float(row["test_value"])), 4)
            except ValueError:
                continue
            unit, lo, hi = row["window_id"], None, None
            if "#" in unit:
                unit, span = unit.rsplit("#", 1)
                lo, hi = map(int, span.split("-"))
            kind = ge.TEST_NAMES.get(row["test_type"].lower(), "other")
            flag = {"TRUE": True, "FALSE": False}
            verdict = r_verdict(flag.get(row["error"]), flag.get(row["decision_error"]))
            out[unit].append([lo, hi, stat, kind, False, verdict or UNDECIDABLE])
    return out


def match_served(gold, finds, rule):
    """The index of the find that served each gold row, or None. Consumes finds in place.

    rule "local" is match_local of 14_gold_eval.py, rule "statcheck" is
    match_statcheck: a statcheck find with no lines serves any row of its paper.
    """
    served = []
    for g in gold:
        def fits(f):
            lines = (f[0] <= g["line"] <= f[1] if rule == "local"
                     else f[0] is None or f[0] <= g["line"] <= f[1])
            return not f[4] and f[2] in (g["stat"], g["anchor_value"]) and lines
        i = next((k for k, f in enumerate(finds) if fits(f)), None)
        if i is not None:
            finds[i][4] = True
        served.append(i)
    return served


_PIPES = {}


def find_paper(job):
    """Gold rows with their anchors, and every local system's finds, for one paper."""
    paper, clean_dir, models = job
    from statcheck_ml.normalize import normalize
    from statcheck_ml.pipeline import Pipeline

    for s, path in models.items():
        if s not in _PIPES:
            _PIPES[s] = Pipeline(model_path=path)
    raw = open(os.path.join(clean_dir, paper["doc"]), encoding="utf-8").read()
    base = _PIPES["regex"]
    text, _ = normalize(raw)
    fixed, _ = base.repair(text)
    starts = [0]
    for ln in fixed.split("\n"):
        starts.append(starts[-1] + len(ln) + 1)
    windows = list(base.prefilter.windows(fixed))
    units = list(base.prefilter.units(fixed))
    ranges = [(starts[w.start_line], starts[w.end_line + 1]) for w in windows]
    gold = []
    for g in paper["results"]:
        if not g["located"]:
            continue
        s, e = g["span"]
        a_s, a_e = g["anchor"]
        gold.append({"row": g["row"], "test_type": g["test_type"], "stat": g["stat"], "line": g["line"],
                     "anchor_value": round(abs(float(fixed[a_s:a_e])), 4) if fixed[a_s:a_e] else None,
                     "kept": any(a <= s and e <= b for a, b in ranges)})
    finds = {s: local_finds(_PIPES[s].run_text(raw), units) for s in models}
    return {"doc": paper["doc"], "has_results": paper["has_results"], "gold": gold, "finds": finds}


def score_paper(found, statcheck_finds):
    """Each system's verdict per gold row, and its flags on finds that serve no in-scope row."""
    finds = dict(found["finds"])
    finds.update(statcheck_finds)
    verdicts, outside = {}, {}
    for s, fs in finds.items():
        fs = [list(f[:4]) + [False] + list(f[5:]) for f in fs]
        served = match_served(found["gold"], fs, "local" if s in LOCAL else "statcheck")
        verdicts[s] = [fs[i][5] if i is not None else None for i in served]
        out_of_scope = collections.Counter(
            fs[i][5] for g, i in zip(found["gold"], served)
            if i is not None and g["test_type"] not in ge.IN_SCOPE)
        unmatched = collections.Counter(f[5] for f in fs if not f[4])
        outside[s] = {"unmatched": dict(unmatched), "out_of_scope": dict(out_of_scope)}
    return verdicts, outside


# ---------------- metrics ----------------

def flagged(v):
    return v in FLAGS


def system_metrics(rows, papers, system, resamples, seed):
    """The verdict metrics of one system.

    rows: in-scope rows with "doc", "reference" (a verdict or None) and
    "systems" {system: verdict, or None when not found}. papers: [{"doc",
    "has_results", "outside": {system: {"unmatched", "out_of_scope"}}}].
    """
    ref = [r for r in rows if r["reference"]]
    get = lambda r: r["systems"][system]

    def split(sel):
        return {"n": len(sel),
                "not_found": sum(get(r) is None for r in sel),
                "judged_consistent": sum(get(r) == CONSISTENT for r in sel),
                "not_checked": sum(get(r) == UNDECIDABLE for r in sel)}

    inc = [r for r in ref if r["reference"] in FLAGS]
    de = [r for r in ref if r["reference"] == DECISION_ERROR]
    cons = [r for r in ref if r["reference"] == CONSISTENT]
    m = {"recall": round(sum(get(r) is not None for r in rows) / len(rows), 3) if rows else None,
         "inconsistencies": {**split(inc), "flagged": sum(flagged(get(r)) for r in inc)},
         "decision_errors": {**split(de), "flagged": sum(get(r) == DECISION_ERROR for r in de),
                             "flagged_inconsistent_only": sum(get(r) == INCONSISTENT for r in de)},
         "false_flags": {"n_consistent": len(cons), "found": sum(get(r) is not None for r in cons),
                         "flagged": sum(flagged(get(r)) for r in cons),
                         "flagged_decision_error": sum(get(r) == DECISION_ERROR for r in cons)}}
    i = m["inconsistencies"]
    i["share_flagged"] = round(i["flagged"] / i["n"], 3) if i["n"] else None
    by_doc = collections.defaultdict(list)
    for r in inc:
        by_doc[r["doc"]].append({"k": True, system: flagged(get(r))})
    boot = [{"gold": by_doc.get(p["doc"], [])} for p in papers]
    i["share_flagged_ci"] = ge.bootstrap(boot, system, lambda g: g["k"], resamples, seed) if inc else None

    def count(group, key, which):
        return sum(v for p in papers if p["has_results"] == group
                   for k, v in p["outside"][system][key].items() if k in which)
    m["flags_outside_gold"] = {
        "papers_with_results": {"flagged": count(True, "unmatched", FLAGS),
                                "decision_errors": count(True, "unmatched", {DECISION_ERROR})},
        "papers_without_results": {"flagged": count(False, "unmatched", FLAGS),
                                   "decision_errors": count(False, "unmatched", {DECISION_ERROR})}}
    m["flags_on_gold_rows_without_reference"] = {
        "in_scope": sum(flagged(get(r)) for r in rows if not r["reference"]),
        "out_of_scope": sum(v for p in papers for k, v in p["outside"][system]["out_of_scope"].items()
                            if k in FLAGS)}
    hit_docs = {r["doc"] for r in inc}
    m["papers"] = {"with_inconsistency": len(hit_docs),
                   "flagged_correctly": len({r["doc"] for r in inc if flagged(get(r))})}
    conf = {v: collections.Counter() for v in VERDICTS[:3]}
    for r in ref:
        conf[r["reference"]][get(r) or "not_found"] += 1
    m["confusion"] = {k: dict(sorted(v.items())) for k, v in conf.items()}
    return m


def reference_summary(rows):
    ref = [r for r in rows if r["reference"]]
    built = [r for r in rows if r["string"]]
    return {"rows": len(rows), "with_reference": len(ref),
            "consistent": sum(r["reference"] == CONSISTENT for r in ref),
            "inconsistencies": sum(r["reference"] in FLAGS for r in ref),
            "decision_errors": sum(r["reference"] == DECISION_ERROR for r in ref),
            "no_reference_by_cause": dict(sorted(collections.Counter(
                r["cause"] for r in rows if not r["reference"]).items())),
            "strings_built": len(built),
            "operator_written_as_strict": sum(r["operator_mapped"] for r in built),
            "statistic_from_raw": sum(r["stat_from_raw"] for r in built),
            "p_from_raw": sum(r["p_from_raw"] for r in built),
            "p_written_ns": sum(r["p_text"] is None for r in built),
            "statistic_decimals_ambiguous": sum(r["stat_ambiguous"] for r in built),
            # A coded value that raw does not print is often a coding slip, such as
            # .15 coded for a printed .015, so its verdict is the coder's arithmetic.
            "coded_value_not_in_raw": sum(not_in_raw(r) for r in built),
            "inconsistencies_with_coded_value_not_in_raw": sum(not_in_raw(r) for r in ref
                                                                if r["reference"] in FLAGS)}


def not_in_raw(r):
    return not r["stat_from_raw"] or (r["p_text"] is not None and not r["p_from_raw"])


def pvalue_summary(rows):
    ref = [r for r in rows if r["reference"]]
    bad = [r for r in ref if r["ours"] != r["reference"]]
    diffs = [abs(r["ours_p"] - r["r_p"]) / r["r_p"] for r in ref
             if r["ours_p"] is not None and r["r_p"]]
    return {"rows": len(ref), "agree": len(ref) - len(bad),
            "disagree": [{k: r[k] for k in ("doc", "row", "string", "reference", "ours", "r_p", "ours_p")}
                         for r in bad],
            "p_relative_tolerance": P_REL_TOL,
            "p_max_relative_difference": float(f"{max(diffs):.3g}") if diffs else None,
            "p_beyond_tolerance": sum(d > P_REL_TOL for d in diffs)}


# ---------------- report ----------------

def write_md(path, out):
    ref, pv = out["reference"], out["pvalue_check"]
    f = lambda x: "-" if x is None else f"{x:.3f}" if isinstance(x, float) and x >= 0.001 else str(x)
    lines = [f"# Verdicts on the gold rows ({out['papers']})", "",
             "The reference is R statcheck 1.5.0 on one APA string per coded gold row. "
             "A system is scored on the rows that have a reference. "
             "\"Flagged\" means inconsistent or decision error.", "",
             "## Reference", "",
             "| rows | with reference | consistent | inconsistencies | decision errors |",
             "|---|---|---|---|---|",
             f"| {ref['rows']} | {ref['with_reference']} | {ref['consistent']} | "
             f"{ref['inconsistencies']} | {ref['decision_errors']} |", "",
             "Inconsistencies include the decision errors.", "",
             "| no reference, cause | rows |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in ref["no_reference_by_cause"].items()]
    lines += ["", f"Of {ref['strings_built']} strings, {ref['statistic_from_raw']} take the statistic "
              f"and {ref['p_from_raw']} take p as `raw` prints them; {ref['p_written_ns']} are \"ns\". "
              f"{ref['operator_written_as_strict']} write a coded ≤ or ≥ as < or >. "
              f"In {ref['coded_value_not_in_raw']} strings `raw` does not print a coded value, "
              f"so the coded value is used; {ref['inconsistencies_with_coded_value_not_in_raw']} "
              f"of the reference inconsistencies are such rows, and some of them are coding slips.", "",
              "## pvalue.py against R", "",
              f"pvalue.py gives the R verdict on {pv['agree']} of {pv['rows']} rows. "
              f"The largest relative difference in the computed p is {f(pv['p_max_relative_difference'])}; "
              f"{pv['p_beyond_tolerance']} rows exceed {pv['p_relative_tolerance']}.", ""]
    if pv["disagree"]:
        lines += ["| doc | row | string | R | pvalue.py |", "|---|---|---|---|---|"]
        lines += [f"| {d['doc']} | {d['row']} | `{d['string'].strip()}` | {d['reference']} | {d['ours']} |"
                  for d in pv["disagree"]]
        lines.append("")
    s = out["systems"]
    lines += ["## Reference inconsistencies", "",
              "| system | n | flagged | share [95% CI] | not found | found, judged consistent | found, not checked |",
              "|---|---|---|---|---|---|---|"]
    for name, m in s.items():
        i = m["inconsistencies"]
        ci = i["share_flagged_ci"] or [None, None]
        lines.append(f"| {name} | {i['n']} | {i['flagged']} | {f(i['share_flagged'])} "
                     f"[{f(ci[0])}, {f(ci[1])}] | {i['not_found']} | {i['judged_consistent']} | "
                     f"{i['not_checked']} |")
    lines += ["", "## Reference decision errors", "",
              "| system | n | flagged as decision error | flagged as inconsistent only | not found | "
              "found, judged consistent | found, not checked |", "|---|---|---|---|---|---|---|"]
    for name, m in s.items():
        d = m["decision_errors"]
        lines.append(f"| {name} | {d['n']} | {d['flagged']} | {d['flagged_inconsistent_only']} | "
                     f"{d['not_found']} | {d['judged_consistent']} | {d['not_checked']} |")
    lines += ["", "## Flags where the reference has no error", "",
              "A find outside gold is not always wrong: the coders followed statcheck's scope.", "",
              "| system | consistent rows | found | flagged | of which decision errors | "
              "flags on finds outside gold, papers with results | the same, papers with no result | "
              "flags on gold rows with no reference |", "|---|---|---|---|---|---|---|---|"]
    for name, m in s.items():
        ff, o, nr = m["false_flags"], m["flags_outside_gold"], m["flags_on_gold_rows_without_reference"]
        lines.append(f"| {name} | {ff['n_consistent']} | {ff['found']} | {ff['flagged']} | "
                     f"{ff['flagged_decision_error']} | {o['papers_with_results']['flagged']} | "
                     f"{o['papers_without_results']['flagged']} | {nr['in_scope'] + nr['out_of_scope']} |")
    lines += ["", "## Papers", "",
              "| system | papers with a reference inconsistency | papers flagged correctly at least once | "
              "recall of the in-scope rows |", "|---|---|---|---|"]
    for name, m in s.items():
        lines.append(f"| {name} | {m['papers']['with_inconsistency']} | {m['papers']['flagged_correctly']} | "
                     f"{f(m['recall'])} |")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def r_versions(rscript):
    code = "cat(as.character(getRversion()), as.character(packageVersion('statcheck')), sep='|')"
    r, sc = subprocess.run([rscript, "-e", code], check=True, capture_output=True,
                           text=True).stdout.strip().split("|")
    return {"r_version": r, "statcheck_version": sc}


def rel(path):
    return os.path.relpath(os.path.realpath(path), ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--papers", choices=["gold_train", "gold_test"], required=True,
                    help="gold_train for development; gold_test once, at the end")
    ap.add_argument("--gold", default="dataset/gold/gold.jsonl")
    ap.add_argument("--split", default="dataset/gold/split.json")
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--baseline-dir", default="dataset/baseline/gold")
    ap.add_argument("--v4-model", default=V4_MODEL)
    ap.add_argument("--out", default=None, help="default results/v4/final/verdicts.json for gold_test")
    ap.add_argument("--md", default=None, help="default: --out with .md")
    ap.add_argument("--log", default="results/gold_eval_log.jsonl")
    ap.add_argument("--label", default="v4-verdicts")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    if args.out is None:
        if args.papers != "gold_test":
            sys.exit("--papers gold_train needs --out, so a development run never overwrites the result")
        args.out = "results/v4/final/verdicts.json"
    args.md = args.md or os.path.splitext(args.out)[0] + ".md"

    from statcheck_ml.pipeline import bundled_model_path

    shipped = os.path.realpath(bundled_model_path())
    v4 = os.path.realpath(args.v4_model)
    models = {"cascade": shipped, "cascade_v4": v4, "regex": None}

    split = json.load(open(args.split))
    wanted = set(split[args.papers])
    # The filter comes before any text is read, so a paper of the other part is never opened.
    gold = [p for p in map(json.loads, open(args.gold, encoding="utf-8")) if p["doc"] in wanted]

    with ProcessPoolExecutor(args.workers) as ex:
        found = list(ex.map(find_paper, [(p, args.clean_dir, models) for p in gold], chunksize=2))

    baseline = {s: read_statcheck_verdicts(os.path.join(args.baseline_dir, f))
                for s, f in ge.STATCHECK.items()}
    rows, papers = [], []
    by_row = {(p["doc"], g["row"]): g for p in gold for g in p["results"]}
    for fp in found:
        verdicts, outside = score_paper(fp, {s: b.get(fp["doc"], []) for s, b in baseline.items()})
        papers.append({"doc": fp["doc"], "has_results": fp["has_results"], "outside": outside})
        for k, g in enumerate(fp["gold"]):
            if g["test_type"] not in ge.IN_SCOPE:
                continue
            b = build(by_row[(fp["doc"], g["row"])])
            rows.append({"doc": fp["doc"], "row": g["row"], "test_type": g["test_type"], "kept": g["kept"],
                         **b, "systems": {s: verdicts[s][k] for s in SYSTEMS}})

    outcome = run_r([{"id": f"{r['doc']}#{r['row']}", "text": r["string"]} for r in rows if r["string"]],
                    args.rscript)
    for r in rows:
        r["reference"], r["r_p"], r["ours"], r["ours_p"] = None, None, None, None
        if not r["string"]:
            continue
        r["reference"], cause, r["r_p"] = reference(r, outcome[f"{r['doc']}#{r['row']}"])
        r["cause"] = cause
        if r["reference"]:
            r["ours"], r["ours_p"] = ours(r)

    systems = {s: system_metrics(rows, papers, s, args.resamples, args.seed) for s in SYSTEMS}
    keep = ("doc", "row", "test_type", "string", "reference", "cause", "ours", "systems")
    model_files = sorted(os.path.join(m, f) for m in (shipped, v4) for f in os.listdir(m)
                         if os.path.isfile(os.path.join(m, f)))
    inputs = ([args.gold, args.split] + [os.path.join(args.baseline_dir, f) for f in ge.STATCHECK.values()]
              + [os.path.join(args.baseline_dir, "versions.json"), str(R_SCRIPT),
                 str(ROOT / "src/statcheck_ml/pvalue.py")] + model_files)
    in_scope = len(rows)
    out = {
        "note": "Reference: R statcheck on one APA string per coded, located, in-scope gold row. "
                "Systems are matched to gold rows by the rule of 14_gold_eval.py. Flagged means "
                "inconsistent or decision_error. The coded values are a human annotation; the "
                "reference verdict is statcheck's arithmetic on them.",
        "papers": args.papers,
        "paper_count": len(papers),
        "papers_with_results": sum(p["has_results"] for p in papers),
        "alpha": ALPHA,
        "reference": reference_summary(rows),
        "pvalue_check": pvalue_summary(rows),
        "systems": systems,
        "rows": [{k: r[k] for k in keep} for r in rows],
        "provenance": {
            "git_sha": ge.git_sha(), "label": args.label,
            "command": " ".join([rel(sys.argv[0])] + sys.argv[1:]),
            "seed": args.seed, "resamples": args.resamples, "systems": SYSTEMS,
            "models": {"cascade": rel(shipped), "cascade_v4": rel(v4)},
            **r_versions(args.rscript),
            "inputs": {rel(f): ge.sha256(f) for f in inputs},
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    write_md(args.md, out)

    pv = out["pvalue_check"]
    print(json.dumps({"reference": out["reference"],
                      "pvalue_check": {k: pv[k] for k in pv if k != "disagree"},
                      "disagree": len(pv["disagree"])}, indent=1, ensure_ascii=False))
    if args.papers != "gold_test":
        return
    kept = sum(r["kept"] for r in rows)
    line = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": out["provenance"]["git_sha"], "label": args.label,
            "model": shipped, "model_v4": v4, "systems": SYSTEMS,
            "gold_test_recall": {s: systems[s]["recall"] for s in SYSTEMS},
            "gold_test_papers": len(papers), "gold_in_scope": in_scope,
            "prefilter_keeps": round(kept / in_scope, 3) if in_scope else None,
            "reference": {k: out["reference"][k] for k in
                          ("with_reference", "inconsistencies", "decision_errors")},
            "gold_test": {s: {"inconsistencies_flagged": systems[s]["inconsistencies"]["flagged"],
                              "share_flagged": systems[s]["inconsistencies"]["share_flagged"],
                              "share_flagged_ci": systems[s]["inconsistencies"]["share_flagged_ci"],
                              "decision_errors_flagged": systems[s]["decision_errors"]["flagged"],
                              "false_flags": systems[s]["false_flags"]["flagged"],
                              "flags_in_papers_without_results":
                                  systems[s]["flags_outside_gold"]["papers_without_results"]["flagged"]}
                          for s in SYSTEMS}}
    with open(args.log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
