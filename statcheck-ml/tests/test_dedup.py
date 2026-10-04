"""Position-based deduplication in Pipeline.run_text. No model is loaded."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parity_lib as pl  # noqa: E402,F401
from statcheck_ml import pipeline as pipeline_mod  # noqa: E402
from statcheck_ml.pipeline import Found, Pipeline  # noqa: E402

FILL = pl._FILL


def run(text, model_finds=None):
    p = Pipeline(model_path=None)
    if model_finds is not None:
        p.model = object()
        p.find_with_model = lambda wtext, line: [
            Found(**{**f, "line": line}) for f in model_finds(wtext)]
    return p.run_text(text)["results"]


def test_same_statistic_twice_gives_two_results():
    text = dict(pl.DEDUP_DOCS)["dedup-same-statistic"]
    rows = run(text)
    assert [r["test_type"] for r in rows] == ["t", "f"]
    assert all(r["statistic"] == 2.1 for r in rows)
    spans = [r["statistic_span"] for r in rows]
    assert text[spans[0][0]:spans[0][1]] == "2.10"
    assert text[spans[1][0]:spans[1][1]] == "2.10"
    assert spans[0] != spans[1]


def test_overlapping_windows_give_one_result():
    text = dict(pl.DEDUP_DOCS)["dedup-window-overlap"]
    windows = list(Pipeline(model_path=None).prefilter.windows(text))
    assert len(windows) >= 2
    rows = run(text)
    assert len(rows) == 1
    s, e = rows[0]["statistic_span"]
    assert text[s:e] == "2.10"


def test_pattern_and_model_find_give_one_pattern_result():
    text = "Intro line that is long enough to sit alone here.\nWe found t(28) = 2.10, p = .04 in the sample.\n"

    def model(wtext):
        i = wtext.index("2.10")
        return [dict(test_type="t", statistic=2.1, df1=28.0, p_operator="=",
                     p_value=0.04, source="model",
                     statistic_span=(i, i + 4))]

    rows = run(text, model)
    assert len(rows) == 1
    assert rows[0]["source"] == "pattern"


def test_model_find_with_same_value_elsewhere_is_kept():
    text = ("We found t(28) = 2.10, p = .04 in the sample of students.\n"
            "A separate line with the value 2.10 and some digits 12345 here.\n")

    def model(wtext):
        i = wtext.rindex("2.10")
        if i < wtext.index("\n"):
            return []
        return [dict(test_type="t", statistic=2.1, source="model",
                     statistic_span=(i, i + 4))]

    rows = run(text, model)
    assert [r["source"] for r in rows] == ["pattern", "model"]
    assert rows[0]["statistic_span"] != rows[1]["statistic_span"]


def test_rule_text_matches_docstring_and_parity_constant():
    doc = " ".join(pipeline_mod.Pipeline.run_text.__doc__.split())
    assert pl.DEDUP_RULE in doc
    assert pl.LINE_RULE in doc


def test_line_is_the_line_of_the_statistic_in_a_multi_line_unit():
    text = "\n".join([
        FILL,
        "The groups differed on the first measure, which was reliable and",
        "stable across sessions, t(28) = 2.10, p = .04, in the expected way.",
        FILL])
    rows = run(text)
    assert len(rows) == 1
    assert rows[0]["line"] == 2
    s, e = rows[0]["statistic_span"]
    assert text[:s].count("\n") == rows[0]["line"]


def test_line_of_a_fragment_is_its_own_line():
    # The unit starts on line 0; the fragment sits on line 1.
    text = "\n".join([
        FILL,
        "Elsewhere the difference was 2.31 in the first sample of students, as shown.",
        FILL])

    def model(wtext):
        i = wtext.index("2.31")
        return [dict(test_type=None, statistic=2.31, source="model",
                     statistic_span=(i, i + 4))]

    p = Pipeline(model_path=None, use_pattern=False)
    p.model = object()
    p.find_with_model = lambda wtext, line: [
        Found(**{**f, "line": line}) for f in model(wtext)]
    report = p.run_text(text)
    assert [f["line"] for f in report["fragments"]] == [1]


def test_unit_text_holds_end_minus_start_newlines():
    pf = Pipeline(model_path=None).prefilter
    texts = [t for _, t in pl.DEDUP_DOCS] + [pl.FRAGMENT_DOCS[0][1]]
    data = json.load(open(Path(pl.ROOT) / "tests" / "parity_cases.json", encoding="utf-8"))
    texts += [c["text"] for c in data["sections"]["pipeline"] if c["name"].startswith("doc-")]
    seen = 0
    for t in texts:
        for u in pf.units(t):
            assert u.text.count("\n") == u.end_line - u.start_line
            seen += 1
    assert seen > 0
