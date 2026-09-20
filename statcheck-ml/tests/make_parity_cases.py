"""Write the parity cases, and the answer the Python reference gives for each.

Nine stages exist once, in Python (`normalize.py`, `repair_validated.py`,
`prefilter.py`, `extract.py`, `onnx_runtime.py`, `evalutil.py`, `pvalue.py`,
`pipeline.py`), and two more ports -- R and a browser -- will be written
against this file. A port that drifts silently is worse than no port, because
the same paper then gives a different answer in Python, in R and in a
browser.

Every case is built from something committed: the 20 normalisation cases
already in this file, `dataset/holdout.jsonl`, hand-written text for the
edges the corpus may not reach, and `examples/sample_paper.pdf`. Nothing here
reads the original corpus, which is why the file can be regenerated on a
clean checkout.

Run this only when a spec or a stage changes. Commit the result, so the
JavaScript and R checks, and the Python self-test, can run without it.

Usage: python tests/make_parity_cases.py [out.json]
"""
from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_lib as pl  # noqa: E402

ROOT = pl.ROOT
HOLDOUT_PATH = ROOT / "dataset" / "holdout.jsonl"
SAMPLE_PDF = ROOT / "examples" / "sample_paper.pdf"
DEFAULT_OUT = Path(__file__).parent / "parity_cases.json"

SEED = 7

# ======================================================================
# Section: normalise
#
# Kept short on purpose. The file is committed, and a long case proves no
# more than a short one.
# ======================================================================

#: Unchanged from the version-1 generator. A change here changes the 10 hand
#: cases; the 10 corpus cases are loaded from the file already committed,
#: because the corpus folders (`data/clean/txt_*`) no longer exist.
NORMALISE_HAND_CASES = [
    ("empty", ""),
    ("short", "F(1, 17) = 3.5, p = .05"),
    # PyMuPDF writes a destroyed operator as a control character. The stage
    # must leave these alone, or it would regress the engine the models were
    # trained on.
    ("pymupdf damage", "F(1, 17) \x03 35.72, p \x04 .0005"),
    # Poppler writes the same damage as letters in the Greek and Coptic block.
    ("poppler damage", "F(1, 17) ϭ 35.72, p Ͻ .0005"),
    ("mixed damage",
     "a ϭ 1.0 and b Ͻ 2.0 and c Ϫ 3.0 and d ␩ 4.0"),
    # These symbols carry meaning inside a statistic and must never be
    # renamed.
    ("protected greek", "χ2(1) = 8.69, p = .003; ϕ = −.20"),
    ("long joined line",
     "word " * 60 + "F(2, 30) ϭ 4.11, p ϭ .03 " + "tail " * 30),
    ("crlf", "one\r\ntwo\rthree\n"),
    ("no spaces to cut", "x" * 400),
    ("every slot taken",
     "".join(chr(c) for c in range(1, 9)) + " t(9) ϭ 2.0"),
]


def _load_existing(path: Path):
    """Read whatever is already at `path`, in either format this script has
    written. `None` when the file does not exist yet."""
    if not path.exists():
        return None
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_normalise_section(existing) -> list:
    """The 20 committed cases: 10 from the corpus, regenerated fresh only for
    the 10 hand cases.

    The corpus folders that built the original 10 are gone
    (`data/clean/txt_*` does not exist), so those cases are carried forward
    from the file already committed rather than regenerated from text this
    checkout does not have. Every corpus case is re-run through `normalize()`
    and asserted unchanged, so a spec change is still caught.

    Corpus case names are not unique on their own -- `d004.txt` is both a
    `pdfjs` case and an `rpdftools` case in the original file -- so each name
    is namespaced by its engine here, the same way `parity.mjs` already
    prints failures (`f"{engine}/{name}"`).
    """
    corpus_rows = []
    if existing is None:
        pass
    elif isinstance(existing, list):
        # The version-1 flat file: every row whose engine is not "hand".
        for row in existing:
            if row.get("engine") == "hand":
                continue
            corpus_rows.append({
                "name": f"{row['engine']}__{row['name']}",
                "engine": row["engine"],
                "text": row["text"],
                "expected_reflow": row.get("expected_reflow"),
                "renamed": row.get("renamed"),
            })
    else:
        # Our own nested format, from a previous run of this script: the
        # names are already namespaced, so use them as they are.
        for row in existing.get("sections", {}).get("normalise", []):
            if row.get("engine") == "hand":
                continue
            corpus_rows.append(row)

    rows = list(corpus_rows)
    for name, text in NORMALISE_HAND_CASES:
        expected = pl.normalise_case(text)
        rows.append({
            "name": name, "engine": "hand", "text": text,
            "expected": expected,
        })

    # Every corpus row keeps its recorded `expected`, but is checked against
    # a fresh call so a spec change cannot go unnoticed just because the
    # corpus text is gone.
    for row in corpus_rows:
        fresh = pl.normalise_case(row["text"])
        if "expected" in row and row["expected"] != fresh:
            raise SystemExit(
                f"normalise regression: {row['name']} no longer matches "
                f"normalize(); regenerate from a corpus checkout")
        row["expected"] = fresh

    names = [r["name"] for r in rows]
    assert len(names) == len(set(names)), "normalise: duplicate case name"
    rows.sort(key=lambda r: r["name"])
    return rows


# ======================================================================
# Holdout sampling, shared by repair / extract / model / group
# ======================================================================

#: The six edges the corpus may not reliably show. Named exactly as the task
#: asks: a clean APA sentence, a result split across a line break, a
#: control-character operator, a fraction sign for "=", a letter "b" for
#: "<", and a page range / citation year that must yield nothing.
HAND_WINDOWS = [
    ("hand-clean-apa",
     "The main effect was significant, F(2, 87) = 4.11, p = .03."),
    ("hand-line-break",
     "The effect was significant, t(28) = 2.87,\np = .006."),
    ("hand-control-operator",
     "F(1, 40) \x02 6.20, p = .016"),
    ("hand-fraction-for-eq",
     "r(38) ¼ .42, p ¼ .007"),
    ("hand-letter-b-for-lt",
     "t(24) = 3.10, p b .01"),
    ("hand-citation-year",
     "See pages 12-34 for details (Smith, 2015-2016)."),
]


def load_holdout_rows() -> list:
    rows = []
    with io.open(HOLDOUT_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _has_damaged(row: dict) -> bool:
    return any(r.get("damaged") for r in row.get("results") or [])


def _is_empty(row: dict) -> bool:
    return not row.get("results")


def _test_types(row: dict) -> set:
    return {(r.get("test_type") or "").lower() for r in row.get("results") or []}


def select_holdout_windows(rows: list, k: int = 25, seed: int = SEED) -> list:
    """25 windows: covering t/F/r/chi2/z, >=10 with a damaged result, >=5
    with no result, the rest filled from a `random.Random(seed)` shuffle.

    Each requirement is satisfied by walking the same shuffled order and
    taking the next row that still helps, so the selection is reproducible
    (fixed seed, fixed code) without depending on the order dataset rows
    happen to be sampled for another requirement.
    """
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)

    chosen: list = []
    chosen_set: set = set()

    def take(idx):
        if idx not in chosen_set:
            chosen.append(idx)
            chosen_set.add(idx)

    for tt in ("t", "f", "r", "chi2", "z"):
        for idx in order:
            if idx in chosen_set:
                continue
            if tt in _test_types(rows[idx]):
                take(idx)
                break

    for idx in order:
        if sum(1 for i in chosen_set if _has_damaged(rows[i])) >= 10:
            break
        if idx not in chosen_set and _has_damaged(rows[idx]):
            take(idx)

    for idx in order:
        if sum(1 for i in chosen_set if _is_empty(rows[i])) >= 5:
            break
        if idx not in chosen_set and _is_empty(rows[idx]):
            take(idx)

    for idx in order:
        if len(chosen) >= k:
            break
        if idx not in chosen_set:
            take(idx)

    chosen = sorted(chosen[:k])
    picked = [rows[i] for i in chosen]

    n_damaged = sum(1 for r in picked if _has_damaged(r))
    n_empty = sum(1 for r in picked if _is_empty(r))
    covered = set().union(*(_test_types(r) for r in picked)) if picked else set()
    assert n_damaged >= 10, f"only {n_damaged} damaged windows"
    assert n_empty >= 5, f"only {n_empty} empty windows"
    assert {"t", "f", "r", "chi2", "z"} <= covered, f"missing test types: {covered}"
    return picked


def build_repair_section(units: list) -> list:
    rows = []
    for name, text in units:
        rows.append({"name": name, "text": text, "expected": pl.repair_case(text)})
    rows.sort(key=lambda r: r["name"])
    return rows


def build_extract_section(units: list) -> list:
    rows = []
    for name, text in units:
        rows.append({"name": name, "text": text, "expected": pl.extract_case(text)})
    rows.sort(key=lambda r: r["name"])
    return rows


def build_model_and_group_sections(units: list) -> tuple:
    model_rows, group_rows = [], []
    for name, text in units:
        mc = pl.model_case(text)
        model_rows.append({"name": name, "text": mc["text"], "expected": mc["tags"]})
        group_rows.append({
            "name": name, "text": mc["text"], "tags": mc["tags"],
            "expected": pl.group_case(mc["text"], mc["tags"]),
        })
    model_rows.sort(key=lambda r: r["name"])
    group_rows.sort(key=lambda r: r["name"])
    return model_rows, group_rows


# ======================================================================
# Section: model_logits
# ======================================================================

MODEL_LOGITS_CASES = [
    ("logits-clean", "F(2, 30) = 4.11, p = .03"),
    ("logits-damaged", "F(1, 17) \x03 35.72, p \x04 .0005"),
    ("logits-no-result",
     "Participants completed a brief demographic questionnaire first."),
]


def build_model_logits_section() -> list:
    rows = []
    for name, text in MODEL_LOGITS_CASES:
        assert len(text) <= 120, f"{name} is {len(text)} chars, must be <= 120"
        lc = pl.model_logits_case(text)
        rows.append({"name": name, "text": lc["text"], "expected": lc["logits"]})
    rows.sort(key=lambda r: r["name"])
    return rows


# ======================================================================
# Section: prefilter / pipeline documents
# ======================================================================

def _build_concatenated_doc_no_result(rng: random.Random, pool: list, k: int,
                                      lo_lines=20, hi_lines=80) -> str:
    """`k` windows of at most 4 lines, joined by a blank line -- every one
    of the corpus's short windows carries no result, so this document has
    none either. See `_join_windows` for the line-count reasoning.
    """
    sample = rng.sample(pool, k)
    text = "\n\n".join(row["text"] for row in sample)
    lines = text.count("\n") + 1
    assert 15 <= k <= 30, f"{k} windows, want 15-30"
    assert lo_lines <= lines <= hi_lines, f"{lines} lines, want {lo_lines}-{hi_lines}"
    return text


def _join_windows(rng: random.Random, result_pool: list, short_pool: list,
                  n_result: int, n_short: int,
                  lo_lines=20, hi_lines=80) -> str:
    """`n_result` result-bearing windows (always 5 lines in this corpus:
    every one of the 209 holdout windows with a result is exactly 5 lines)
    plus `n_short` windows of at most 4 lines, joined by a blank line each.

    `dataset/holdout.jsonl` windows are almost all exactly 5 lines (519 of
    576); only 57 are 3 or 4 lines, and every one of those 57 has no result.
    A document built only from 5-line windows blows past 80 lines well
    before 15 of them (measured: 15 windows already gives 87 lines), so
    every document here mixes in enough of the short, result-free windows to
    reach a window count in "15-30" while staying under the 80-line bound.
    This is a property of the corpus, not a rule this file invents; see the
    final report.
    """
    sample = rng.sample(result_pool, n_result) + rng.sample(short_pool, n_short)
    rng.shuffle(sample)
    text = "\n\n".join(row["text"] for row in sample)
    lines = text.count("\n") + 1
    k = n_result + n_short
    assert 15 <= k <= 30, f"{k} windows, want 15-30"
    assert lo_lines <= lines <= hi_lines, f"{lines} lines, want {lo_lines}-{hi_lines}"
    return text


def build_documents(rows: list) -> list:
    """Four documents: three concatenated from the holdout, one from
    `examples/sample_paper.pdf` (skipped if `pymupdf` cannot read it).

    The three holdout documents are not identical in shape, so the prefilter
    and the pipeline are each exercised on more than one scenario:

    - `doc-holdout-1` holds no result at all, built only from the corpus's
      57 short (no-result) windows. This is the "finds nothing, and nothing
      spurious" case.
    - `doc-holdout-2` is biased towards damaged results, so the pipeline's
      repair stage has real work to do inside a full document.
    - `doc-holdout-3` is a general mix of results and result-free text.
    """
    rng = random.Random(SEED)
    short_pool = [r for r in rows
                 if r.get("text", "").strip() and r["text"].count("\n") + 1 <= 4]
    result_pool = [r for r in rows if r.get("results")]
    damaged_pool = [r for r in result_pool if _has_damaged(r)]

    docs = [
        ("doc-holdout-1", _build_concatenated_doc_no_result(rng, short_pool, 16)),
        ("doc-holdout-2", _join_windows(rng, damaged_pool, short_pool, 6, 9)),
        ("doc-holdout-3", _join_windows(rng, result_pool, short_pool, 8, 7)),
    ]

    try:
        import pymupdf

        pdf = pymupdf.open(str(SAMPLE_PDF))
        text = "".join(page.get_text() for page in pdf)
        pdf.close()
        lines = text.count("\n") + 1
        if 20 <= lines <= 80:
            docs.append(("doc-sample-paper", text))
        else:
            print(f"skipping doc-sample-paper: {lines} lines, outside 20-80")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"skipping doc-sample-paper: pymupdf unavailable ({exc})")

    return docs


def build_prefilter_section(docs: list) -> list:
    rows = []
    for name, text in docs:
        rows.append({"name": name, "text": text, "expected": pl.prefilter_case(text)})
    rows.sort(key=lambda r: r["name"])
    return rows


def build_pipeline_section(docs: list) -> list:
    rows = []
    for name, text in docs:
        rows.append({"name": name, "text": text, "expected": pl.pipeline_case(text)})
    rows.sort(key=lambda r: r["name"])
    return rows


# ======================================================================
# Section: pvalue
# ======================================================================

def pvalue_rows() -> list:
    """(name, test_type, statistic, df1, df2, p_operator, p_text) tuples.

    Covers t, F, r, z, chi2, q with a spread of statistics and degrees of
    freedom, plus the edges the task names by name: df1=1, df2=1, df=1000, a
    huge statistic (t=40), r=0.999, r=1.0 (undecidable), p_text with 1-4
    decimals, p_text "0", ".000" and "<.001", every operator, a p_value of
    None, a statistic of None, and a decision-error row.
    """
    rows = [
        # ---- t: a spread of df, plus df1=1 and df=1000 ----
        ("t-basic-small-df", "t", 2.10, 8, None, "=", ".06"),
        ("t-basic-mid-df", "t", 2.50, 18, None, "=", ".02"),
        ("t-basic-large-df", "t", 1.80, 60, None, "=", ".08"),
        ("t-negative-stat", "t", -3.00, 15, None, "=", ".005"),
        ("t-df1-eq-1", "t", 12.0, 1, None, "=", ".05"),
        ("t-df-eq-1000", "t", 1.96, 1000, None, "=", ".05"),
        ("t-huge-stat-40", "t", 40.0, 5, None, "<", ".001"),
        ("t-operator-lt", "t", 3.20, 20, None, "<", ".01"),
        ("t-operator-gt", "t", 0.50, 20, None, ">", ".10"),
        ("t-p-one-decimal", "t", 1.50, 20, None, "=", ".1"),
        ("t-p-two-decimal", "t", 1.50, 20, None, "=", ".14"),
        ("t-p-three-decimal", "t", 1.50, 20, None, "=", ".149"),
        ("t-p-four-decimal", "t", 1.50, 20, None, "=", ".1492"),
        ("t-p-text-zero", "t", 0.02, 20, None, "=", "0"),
        ("t-p-text-dot-000", "t", 6.00, 20, None, "=", ".000"),
        ("t-p-text-lt-dot001", "t", 6.00, 20, None, "<", "<.001"),
        ("t-decision-error", "t", 1.50, 20, None, "=", ".01"),
        ("t-statistic-none", "t", None, 10, None, "=", ".05"),
        ("t-p-value-none", "t", 2.00, 10, None, None, None),

        # ---- F: df1 and df2 both vary, df1=1, df2=1, df2=1000 ----
        ("f-basic-small", "f", 4.11, 2, 30, "=", ".03"),
        ("f-basic-mid", "f", 6.20, 3, 60, "=", ".001"),
        ("f-df1-eq-1", "f", 5.00, 1, 25, "=", ".03"),
        ("f-df2-eq-1", "f", 5.00, 2, 1, "=", ".30"),
        ("f-df2-eq-1000", "f", 2.00, 3, 1000, "=", ".11"),
        ("f-operator-lt", "f", 9.00, 2, 40, "<", ".001"),
        ("f-operator-gt", "f", 0.80, 2, 40, ">", ".10"),
        ("f-huge-stat", "f", 150.0, 1, 30, "<", ".001"),
        ("f-p-text-zero", "f", 20.0, 2, 30, "=", "0"),
        ("f-df2-none", "f", 5.00, 2, None, "=", ".05"),
        ("f-decision-error", "f", 1.20, 2, 40, "=", ".01"),

        # ---- r: including 0.999 and 1.0 (undecidable) ----
        ("r-basic-positive", "r", 0.50, 30, None, "=", ".005"),
        ("r-basic-negative", "r", -0.60, 20, None, "=", ".005"),
        ("r-df1-eq-1", "r", 0.90, 1, None, "=", ".29"),
        ("r-df-eq-1000", "r", 0.05, 1000, None, "=", ".11"),
        ("r-near-one-0999", "r", 0.999, 10, None, "<", ".001"),
        ("r-equals-one-undecidable", "r", 1.0, 15, None, "=", ".05"),
        ("r-equals-negative-one", "r", -1.0, 15, None, "=", ".05"),
        ("r-operator-lt", "r", 0.40, 25, None, "<", ".05"),
        ("r-operator-gt", "r", 0.05, 25, None, ">", ".50"),
        ("r-statistic-none", "r", None, 20, None, "=", ".05"),

        # ---- z: no degrees of freedom ----
        ("z-basic-significant", "z", 1.96, None, None, "=", ".05"),
        ("z-basic-strong", "z", 2.58, None, None, "=", ".01"),
        ("z-not-significant", "z", 0.50, None, None, "=", ".62"),
        ("z-negative", "z", -1.64, None, None, "=", ".10"),
        ("z-operator-lt", "z", 3.00, None, None, "<", ".01"),
        ("z-operator-gt", "z", 0.10, None, None, ">", ".50"),
        ("z-p-value-none", "z", 2.00, None, None, None, None),
        ("z-p-text-lt-dot001", "z", 5.00, None, None, "<", "<.001"),

        # ---- chi2: including df1=1, df=1000, and df1=0 (skipped: NaN) ----
        ("chi2-df1-eq-1", "chi2", 3.84, 1, None, "=", ".05"),
        ("chi2-basic-mid", "chi2", 11.07, 5, None, "=", ".05"),
        ("chi2-df-eq-1000", "chi2", 1050.0, 1000, None, "=", ".10"),
        ("chi2-operator-lt", "chi2", 20.0, 4, None, "<", ".001"),
        ("chi2-operator-gt", "chi2", 1.00, 4, None, ">", ".50"),
        ("chi2-huge-stat", "chi2", 500.0, 10, None, "<", ".001"),
        ("chi2-p-text-zero", "chi2", 30.0, 3, None, "=", "0"),
        ("chi2-df1-eq-0-nan", "chi2", 5.0, 0, None, "=", ".05"),
        ("chi2-decision-error", "chi2", 2.00, 4, None, "=", ".01"),

        # ---- q (Cochran's Q, chi-square distributed) ----
        ("q-basic-small", "q", 3.84, 1, None, "=", ".05"),
        ("q-basic-mid", "q", 10.00, 5, None, "=", ".075"),
        ("q-df1-eq-1", "q", 6.63, 1, None, "=", ".01"),
        ("q-operator-lt", "q", 25.0, 8, None, "<", ".01"),
        ("q-operator-gt", "q", 2.00, 8, None, ">", ".90"),
        ("q-p-text-dot000", "q", 40.0, 6, None, "=", ".000"),
        ("q-statistic-none", "q", None, 5, None, "=", ".05"),
    ]
    assert len(rows) == len({r[0] for r in rows}), "pvalue: duplicate case name"
    return rows


def build_pvalue_section() -> list:
    rows = []
    skipped = []
    for name, test_type, statistic, df1, df2, p_operator, p_text in pvalue_rows():
        outcome = pl.pvalue_case(test_type, statistic, df1, df2, p_operator, p_text)
        expected = pl.pvalue_expected(outcome)
        if expected is None:
            skipped.append(name)
            continue
        rows.append({
            "name": name, "test_type": test_type, "statistic": statistic,
            "df1": df1, "df2": df2, "p_operator": p_operator, "p_text": p_text,
            "expected": expected,
        })
    if skipped:
        print(f"pvalue: skipped {len(skipped)} case(s) with a NaN computed_p: "
              f"{', '.join(skipped)}")
    rows.sort(key=lambda r: r["name"])
    return rows


# ======================================================================
# Assembly
# ======================================================================

def main(out_path: str | None = None):
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    existing = _load_existing(out_path)

    normalise_rows = build_normalise_section(existing)

    holdout_rows = load_holdout_rows()
    sample = select_holdout_windows(holdout_rows)
    units = [(f"holdout-{row['window_id']}", row["text"]) for row in sample]
    units += HAND_WINDOWS
    names = [n for n, _ in units]
    assert len(names) == len(set(names)), "repair/extract/model/group: duplicate name"

    repair_rows = build_repair_section(units)
    extract_rows = build_extract_section(units)
    model_rows, group_rows = build_model_and_group_sections(units)
    model_logits_rows = build_model_logits_section()

    docs = build_documents(holdout_rows)
    prefilter_rows = build_prefilter_section(docs)
    pipeline_rows = build_pipeline_section(docs)

    pvalue_section_rows = build_pvalue_section()

    payload = {
        "version": 2,
        "generated_by": "tests/make_parity_cases.py",
        "model": "gru-crf",
        "model_input_note": (
            "The `model` and `model_logits` sections tag the text after "
            "`statcheck_ml.data.normalise` -- the whitespace-folding pass -- "
            "not after `statcheck_ml.normalize.normalize` (the reflow and "
            "operator-canonicalisation pass used by `Pipeline.run_text`). "
            "This mirrors `evalutil.model_predictions`, which is what the "
            "phase-11 evaluator scores the model with. Each case's `text` "
            "field is the string the model actually saw."
        ),
        "logit_tolerance": pl.LOGIT_TOLERANCE,
        "logit_note": pl.LOGIT_NOTE,
        "pvalue_rounding_rule": pl.PVALUE_ROUNDING_RULE,
        "sections": {
            "normalise": normalise_rows,
            "repair": repair_rows,
            "prefilter": prefilter_rows,
            "extract": extract_rows,
            "model": model_rows,
            "model_logits": model_logits_rows,
            "group": group_rows,
            "pvalue": pvalue_section_rows,
            "pipeline": pipeline_rows,
        },
    }

    with io.open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.write("\n")

    counts = {k: len(v) for k, v in payload["sections"].items()}
    print(f"wrote {out_path} ({sum(counts.values())} cases)")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
