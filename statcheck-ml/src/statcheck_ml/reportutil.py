"""Helpers for `pipeline/11_report.py`: placeholder resolution and tables.

Every function here but one (`git_head_info`) is pure. It turns data already
loaded from a committed file into a string. Nothing here reads a file, and
nothing here computes a metric that `pipeline/09_evaluate.py` or
`pipeline/04_agree.py` did not already compute. The report generator's whole
job is arranging numbers that already exist, never producing a new one.

Placeholder grammar, inside a Markdown template:

  {{key}}            values[key], stringified
  {{key|3}}           values[key] formatted to 3 decimal places
  {{key|pct1}}        values[key] as a percentage, 1 decimal place
  {{key|int}}         values[key] rounded to the nearest integer
  {{key|ci}}          values[key] is a [lo, hi] pair -> "[0.912, 0.945]"
  {{table:<name>}}    replaced by the named table function's return value
  {{figure:<f>|<c>}}  an image link to figures/<f>, captioned <c>

A key missing from `values`, or a table name missing from `tables`, raises
`ReportError` so the caller can exit 1 and name the problem rather than ship
a report with a silent gap.
"""
from __future__ import annotations

import re
import subprocess
from typing import Callable, Dict, Optional, Sequence, Tuple


class ReportError(Exception):
    """An unresolved placeholder, an unknown table, or a bad format suffix."""


def git_head_info(repo=None) -> Tuple[Optional[str], Optional[str]]:
    """(short sha, commit date ISO 8601) of HEAD, the one impure helper here.

    Both come from a single `git log` of HEAD, not from `datetime.now()`, so
    two runs at the same commit render the same bytes.
    """
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h%x09%cI"], cwd=repo,
                             capture_output=True, text=True, timeout=10)
        line = out.stdout.strip()
        if not line or "\t" not in line:
            return None, None
        short_sha, date = line.split("\t", 1)
        return short_sha, date
    except Exception:
        return None, None


PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")


# --------------------------------------------------------- number format --

def _f(x, d: int = 3) -> str:
    """A float to `d` places, or a dash when it is not known."""
    return "-" if x is None else f"{float(x):.{d}f}"


def _pct(x, d: int = 1) -> str:
    return "-" if x is None else f"{float(x) * 100:.{d}f}%"


def _ci(pair, d: int = 3) -> str:
    if not pair:
        return "-"
    lo, hi = pair
    return f"[{float(lo):.{d}f}, {float(hi):.{d}f}]"


# --------------------------------------------------------------- values ---

def _stringify(value) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_value(value, suffix: Optional[str]) -> str:
    """Render one placeholder value, given its `|suffix` (or none)."""
    if suffix is None:
        return _stringify(value)
    if value is None:
        return "not available"
    if suffix == "int":
        return str(int(round(float(value))))
    if suffix == "pct1":
        return _pct(value, 1)
    if suffix == "ci":
        return _ci(value)
    if suffix.isdigit():
        return _f(value, int(suffix))
    raise ReportError(f"unknown format suffix: {suffix!r}")


# ----------------------------------------------------------- rendering ---

def render(template_text: str, values: dict, tables: Dict[str, Callable[[], str]],
          figures_dir: str = "figures") -> str:
    """Fill every `{{...}}` block in `template_text`.

    `tables` maps a table name to a zero-argument function that returns
    either a Markdown table or a one-line "not available" note.
    """

    def repl(m: re.Match) -> str:
        token = m.group(1).strip()

        if token.startswith("table:"):
            name = token[len("table:"):].strip()
            if name not in tables:
                raise ReportError(f"unknown table: {name}")
            return tables[name]()

        if token.startswith("figure:"):
            rest = token[len("figure:"):]
            if "|" not in rest:
                raise ReportError(f"malformed figure block: {token!r}")
            file_name, caption = rest.split("|", 1)
            file_name, caption = file_name.strip(), caption.strip()
            return f"![{caption}]({figures_dir}/{file_name})\n\n*{caption}*"

        if "|" in token:
            key, suffix = (part.strip() for part in token.split("|", 1))
        else:
            key, suffix = token, None
        if key not in values:
            raise ReportError(f"unresolved placeholder: {key}")
        return format_value(values[key], suffix)

    return PLACEHOLDER_RE.sub(repl, template_text)


# ------------------------------------------------------------ tables -----

def note(reason: str) -> str:
    return f"*not available: {reason}*"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return note("no rows to show")
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def dataset_counts(rows: Optional[Sequence[dict]]) -> str:
    """rows: [{"set", "windows", "docs", "results", "damaged", "damaged_share"}, ...]"""
    if not rows:
        return note("dataset window/result counts not found")
    headers = ["Set", "Windows", "Documents", "Results", "Damaged", "Damaged share"]
    body = [[r["set"], str(r["windows"]), str(r["docs"]), str(r["results"]),
             str(r["damaged"]), _pct(r["damaged_share"])] for r in rows]
    return md_table(headers, body)


def rater_yield(rows: Optional[Sequence[dict]]) -> str:
    """rows: [{"set", "rater", "windows", "results"}, ...]"""
    if not rows:
        return note("dataset/annotations/<set>/<rater>.json not found")
    headers = ["Set", "Rater", "Windows annotated", "Results found", "Results / window"]
    body = []
    for r in rows:
        per_window = r["results"] / r["windows"] if r["windows"] else 0.0
        body.append([r["set"], r["rater"], str(r["windows"]), str(r["results"]),
                    f"{per_window:.2f}"])
    return md_table(headers, body)


def _available_sets(by_set: Optional[Dict[str, Optional[dict]]]) -> Dict[str, dict]:
    by_set = by_set or {}
    return {k: v for k, v in by_set.items() if v}


def _missing_footnote(by_set: Optional[Dict[str, Optional[dict]]], avail: Dict[str, dict]) -> str:
    missing = sorted(set(by_set or {}) - set(avail))
    return f"\n\n*not available for: {', '.join(missing)}*" if missing else ""


def agreement_window(by_set: Optional[Dict[str, Optional[dict]]]) -> str:
    """by_set: {"train": agreement.json contents or None, "holdout": ...}"""
    avail = _available_sets(by_set)
    if not avail:
        return note("dataset/agreement/{train,holdout}.json not found")
    headers = ["Set", "Raw % agreement [CI]", "Fleiss kappa [CI]", "Krippendorff alpha [CI]"]
    body = []
    for set_name, agr in avail.items():
        w = agr.get("window", {})
        raw, fleiss, alpha = w.get("raw_pct", {}), w.get("fleiss", {}), w.get("alpha", {})
        body.append([set_name,
                     f"{_pct(raw.get('value'))} {_ci(raw.get('ci'))}",
                     f"{_f(fleiss.get('value'))} {_ci(fleiss.get('ci'))}",
                     f"{_f(alpha.get('value'))} {_ci(alpha.get('ci'))}"])
    return md_table(headers, body) + _missing_footnote(by_set, avail)


def agreement_result(by_set: Optional[Dict[str, Optional[dict]]]) -> str:
    avail = _available_sets(by_set)
    if not avail:
        return note("dataset/agreement/{train,holdout}.json not found")
    headers = ["Set", "Mode", "Pair", "F1 [CI]"]
    body = []
    for set_name, agr in avail.items():
        for mode in ("strict", "lenient"):
            r = agr.get("result", {}).get(mode, {})
            for pair, entry in sorted(r.get("pairwise", {}).items()):
                body.append([set_name, mode, pair, f"{_f(entry.get('value'))} {_ci(entry.get('ci'))}"])
            mean = r.get("mean", {})
            body.append([set_name, mode, "mean", f"{_f(mean.get('value'))} {_ci(mean.get('ci'))}"])
    return md_table(headers, body) + _missing_footnote(by_set, avail)


def agreement_field(by_set: Optional[Dict[str, Optional[dict]]]) -> str:
    avail = _available_sets(by_set)
    if not avail:
        return note("dataset/agreement/{train,holdout}.json not found")
    fields = sorted({f for agr in avail.values() for f in agr.get("field", {})
                     if f != "n_results_all_raters"})
    headers = ["Set", "Field", "Agreement [CI]"]
    body = []
    for set_name, agr in avail.items():
        field_data = agr.get("field", {})
        for field in fields:
            entry = field_data.get(field)
            if entry is None:
                continue
            body.append([set_name, field, f"{_pct(entry.get('value'))} {_ci(entry.get('ci'))}"])
    return md_table(headers, body) + _missing_footnote(by_set, avail)


def rater_vs_final(by_set: Optional[Dict[str, Optional[dict]]]) -> str:
    avail = {k: v for k, v in _available_sets(by_set).items()
             if v.get("result", {}).get("vs_final")}
    if not avail:
        return note("no vs_final scores in dataset/agreement/{train,holdout}.json")
    headers = ["Set", "Rater", "P", "R", "F1"]
    body = []
    for set_name, agr in avail.items():
        for rater, m in sorted(agr["result"]["vs_final"].items()):
            body.append([set_name, rater, _f(m.get("p")), _f(m.get("r")), _f(m.get("f1"))])
    return md_table(headers, body)


def adjudication(disputes_by_set: Optional[Dict[str, Optional[Sequence[dict]]]],
                 tiers_by_set: Optional[Dict[str, Optional[dict]]]) -> str:
    """Two small tables: disputes by kind, and kept-result tiers.

    `disputes_by_set`: {"train": [{"kind", "decision": {"keep", ...}}, ...] or
    None, ...}, straight from `dataset/annotations/<set>/disputes.json`.
    `tiers_by_set`: {"train": {"windows": n, "tiers": {tier: n}} or None, ...},
    built from the per-result "tier" field in `dataset/annotations/<set>/final.json`.
    """
    disputes_avail = {k: v for k, v in (disputes_by_set or {}).items() if v is not None}
    tiers_avail = {k: v for k, v in (tiers_by_set or {}).items() if v}
    if not disputes_avail and not tiers_avail:
        return note("dataset/annotations/{train,holdout}/disputes.json and "
                    "final.json not found")

    parts = []
    if disputes_avail:
        headers = ["Set", "Kind", "Total", "Kept", "Rejected"]
        body = []
        for set_name, disputes in disputes_avail.items():
            by_kind: Dict[str, list] = {}
            for d in disputes:
                by_kind.setdefault(d.get("kind"), []).append(d)
            for kind in sorted(by_kind):
                items = by_kind[kind]
                kept = sum(1 for d in items if (d.get("decision") or {}).get("keep"))
                body.append([set_name, kind, str(len(items)), str(kept), str(len(items) - kept)])
        parts.append("**Disputes by kind**\n\n" + md_table(headers, body))
    else:
        parts.append(note("dataset/annotations/{train,holdout}/disputes.json not found"))

    if tiers_avail:
        tiers = sorted({t for v in tiers_avail.values() for t in v.get("tiers", {})})
        headers = ["Set", "Windows"] + [f"Tier: {t}" for t in tiers]
        body = []
        for set_name, v in tiers_avail.items():
            row = [set_name, str(v.get("windows", "-"))]
            row += [str(v.get("tiers", {}).get(t, 0)) for t in tiers]
            body.append(row)
        parts.append("**Kept-result tiers**\n\n" + md_table(headers, body))
    else:
        parts.append(note("dataset/annotations/{train,holdout}/final.json not found"))

    return "\n\n".join(parts)


def benchmark_screen(rows: Optional[Sequence[dict]]) -> str:
    """rows: [{"config", "params", "dev_f1", "holdout_f1", "holdout_ci", "wall_min"}, ...]"""
    if not rows:
        return note("models/runs.json has no seed-0 screen runs, or results/eval.json "
                    "has not scored them yet")
    headers = ["Config", "Params", "Dev F1", "Holdout F1 [CI]", "Wall min"]
    body = []
    for r in sorted(rows, key=lambda r: -(r["dev_f1"] if r["dev_f1"] is not None else -1)):
        params = "-" if r.get("params") is None else f"{r['params']:,}"
        wall = "-" if r.get("wall_min") is None else f"{r['wall_min']:.1f}"
        body.append([r["config"], params, _f(r.get("dev_f1")),
                    f"{_f(r.get('holdout_f1'))} {_ci(r.get('holdout_ci'))}", wall])
    return md_table(headers, body)


def benchmark_seeds(seeds: Optional[dict]) -> str:
    """seeds: eval.json["seeds"]: {config: {"mean_f1", "sd_f1", "values"}}"""
    if not seeds:
        return note("results/eval.json seeds is empty (models not evaluated yet)")
    headers = ["Config", "Mean F1 (+/- sd)", "Values"]
    body = []
    for config, s in sorted(seeds.items()):
        values_str = ", ".join(f"{v:.3f}" for v in s.get("values", []))
        body.append([config, f"{_f(s.get('mean_f1'))} (+/- {_f(s.get('sd_f1'))})", values_str])
    return md_table(headers, body)


def systems_overall(systems: Optional[dict]) -> str:
    """systems: eval.json["systems"]"""
    if not systems:
        return note("results/eval.json has no systems")
    headers = ["System", "P", "R", "F1 [CI]"]
    body = []
    for name, s in systems.items():
        o = s.get("overall", {})
        ci = s.get("bootstrap", {}).get("overall", {}).get("f1")
        body.append([name, _f(o.get("p")), _f(o.get("r")), f"{_f(o.get('f1'))} {_ci(ci)}"])
    return md_table(headers, body)


def systems_damaged(systems: Optional[dict]) -> str:
    if not systems:
        return note("results/eval.json has no systems")
    headers = ["System", "P", "R", "F1 [CI]"]
    body = []
    for name, s in systems.items():
        d = s.get("damaged", {})
        ci = s.get("bootstrap", {}).get("damaged", {}).get("f1")
        body.append([name, _f(d.get("p")), _f(d.get("r")), f"{_f(d.get('f1'))} {_ci(ci)}"])
    return md_table(headers, body)


def family_recall_table(family: Optional[dict]) -> str:
    """family: eval.json["family_recall"]: {system: {family: {k, n, r, wilson}}}"""
    if not family:
        return note("results/eval.json family_recall is empty (no models scored)")
    groups = sorted(family.keys())
    fams = sorted({f for g in groups for f in family[g]})
    headers = ["Damage family"] + groups
    body = []
    for fam in fams:
        row = [fam]
        for g in groups:
            entry = family[g].get(fam)
            if entry is None:
                row.append("-")
            else:
                row.append(f"{_f(entry.get('r'))} (n={entry.get('n')}) {_ci(entry.get('wilson'))}")
        body.append(row)
    return md_table(headers, body)


def paired_tests(paired: Optional[dict], mcnemar: Optional[dict]) -> str:
    if not paired:
        return note("results/eval.json paired is empty (fewer than two seed-0 models scored)")
    headers = ["Comparison", "delta F1", "CI", "p", "McNemar b", "McNemar c", "McNemar p"]
    body = []
    for name, entry in sorted(paired.items()):
        m = (mcnemar or {}).get(name)
        b = str(m["b_only_a"]) if m else "-"
        c = str(m["b_only_b"]) if m else "-"
        mp = _f(m["p"], 4) if m else "-"
        body.append([name, _f(entry.get("diff")), _ci(entry.get("ci")), _f(entry.get("p"), 4),
                    b, c, mp])
    return md_table(headers, body)


def zoo(records: Optional[Sequence[dict]]) -> str:
    """records: models/export.json"""
    if not records:
        return note("models/export.json not found")
    headers = ["Config", "ONNX MB", "int8 MB", "Latency ms", "Dev F1", "Quant delta F1",
              "Parity", "In zoo"]
    body = []
    for r in sorted(records, key=lambda r: -(r.get("dev_f1") if r.get("dev_f1") is not None else -1)):
        onnx_mb = r.get("onnx_bytes")
        quant_mb = r.get("quant_bytes")
        dq = None
        if r.get("dev_f1") is not None and r.get("dev_f1_quant") is not None:
            dq = r["dev_f1_quant"] - r["dev_f1"]
        body.append([r["name"],
                    "-" if onnx_mb is None else f"{onnx_mb / 1e6:.2f}",
                    "-" if quant_mb is None else f"{quant_mb / 1e6:.2f}",
                    _f(r.get("latency_ms_median"), 1),
                    _f(r.get("dev_f1")),
                    _f(dq),
                    str(bool(r.get("parity"))),
                    "yes" if r.get("in_zoo") else "no"])
    return md_table(headers, body)


def engines_table(engines: Optional[dict]) -> str:
    """engines: results/engines.json"""
    if not engines or not engines.get("recall"):
        return note("results/engines.json not found")
    recall, in_text = engines["recall"], engines.get("in_text", {})
    headers = ["Engine", "Recall", "In-text"]
    body = [[name, _f(recall.get(name)), _f(in_text.get(name))]
           for name in sorted(recall, key=lambda n: -recall[n])]
    table = md_table(headers, body)
    return table + (f"\n\nSpread {_f(engines.get('spread'))} against a limit of "
                    f"{_f(engines.get('limit'))}, over {engines.get('documents', '-')} "
                    f"documents and {engines.get('gold', '-')} gold results.")


def readme_block(values: dict, systems_table_md: str) -> str:
    """A short Markdown block for README.md: the systems table and one sentence."""
    cascade_f1, repaired_f1 = values.get("cascade_f1"), values.get("statcheck_repaired_f1")
    if cascade_f1 is not None and repaired_f1 is not None:
        headline = (f"The cascade reaches holdout F1 {cascade_f1:.3f}, against "
                    f"{repaired_f1:.3f} for statcheck alone.")
    else:
        headline = ("The model grid has not finished training, so the cascade result "
                    "is not yet available.")
    return "## Results\n\n" + systems_table_md + "\n\n" + headline + "\n"
