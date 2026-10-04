"""Finds with no test name go to `fragments`. No model is loaded."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from statcheck_ml.pipeline import Found, Pipeline, summarise  # noqa: E402

LINE = "Elsewhere the difference was 2.31 in the first sample of students, as shown.\n"


def run(text, model_finds, use_pattern=False):
    pipe = Pipeline(model_path=None, use_pattern=use_pattern)
    pipe.model = object()

    def find(wtext, line):
        return [Found(**{**f, "line": line}) for f in model_finds(wtext)]

    pipe.find_with_model = find
    return pipe.run_text(text)


def find_at(wtext, name, value="2.31"):
    i = wtext.index(value)
    return dict(test_type=name, statistic=float(value), p_operator="=", p_value=0.02,
                source="model", statistic_span=(i, i + len(value)))


def test_nameless_find_is_a_fragment_with_no_verdict():
    report = run(LINE, lambda w: [find_at(w, None)])
    assert report["results"] == []
    assert len(report["fragments"]) == 1
    frag = report["fragments"][0]
    assert frag["statistic"] == 2.31 and frag["test_type"] is None
    for key in ("verdict", "computed_p", "reason", "missing"):
        assert key not in frag
    assert "statistic_span" in frag and "quote" in frag
    assert report["stages"]["find"]["fragments"] == 1
    assert report["stages"]["check"] == {}


def test_named_find_wins_when_it_comes_first():
    def finds(w):
        return [find_at(w, "t"), find_at(w, None)]

    report = run(LINE, finds)
    assert [r["test_type"] for r in report["results"]] == ["t"]
    assert report["fragments"] == []


def test_fragment_first_removes_the_later_named_find():
    def finds(w):
        return [find_at(w, None), find_at(w, "t")]

    report = run(LINE, finds)
    assert report["results"] == []
    assert len(report["fragments"]) == 1


def test_pattern_find_of_a_bare_statistic_stays_out_of_results():
    report = run("The difference was 2.31, p = .02.\n", lambda w: [], use_pattern=True)
    assert report["results"] == []


def test_summarise_counts_fragments_apart():
    report = run(LINE, lambda w: [find_at(w, None)])
    report["characters"] = 0
    text = summarise(report)
    assert "fragments         : 1 (no test name)" in text
    assert "found by model    : 1" in text


def test_parity_cases_hold_the_reference_fragments():
    import json

    import parity_lib as pl

    cases = json.load(open(Path(__file__).resolve().parent / "parity_cases.json"))
    for case in cases["sections"]["pipeline"]:
        assert case["expected_fragments"] == json.loads(
            json.dumps(pl.pipeline_fragments_case(case["text"]))), case["name"]
