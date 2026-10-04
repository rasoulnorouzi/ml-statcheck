from statcheck_ml.prefilter import Prefilter, Window


def w(lo, hi):
    return Window("", lo, lo, hi)


def test_overlapping_and_touching_windows_merge():
    runs = Prefilter().passages([w(10, 14), w(0, 3), w(4, 6), w(12, 20)])
    assert runs == [[0, 6], [10, 20]]


def test_gap_of_one_line_keeps_runs_apart():
    assert Prefilter().passages([w(0, 2), w(4, 5)]) == [[0, 2], [4, 5]]


def test_contained_window_does_not_shrink_the_run():
    assert Prefilter().passages([w(0, 9), w(2, 3)]) == [[0, 9]]


def test_no_windows():
    assert Prefilter().passages([]) == []


def test_matches_real_windows_on_text():
    pf = Prefilter()
    text = "\n".join(["words only here"] * 5 + ["t(20) = 2.31, p = .03 (n = 40)"] + ["x"] * 30)
    runs = pf.passages(pf.windows(text))
    assert len(runs) == 1 and runs[0][0] <= 5 <= runs[0][1]


RESULT = "t(20) = 2.31, p = .03 (n = 40)"


def spec_with(unit):
    import json
    from statcheck_ml.prefilter import SPEC_PATH

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec["unit"] = unit
    return spec


def doc_with_two_touching_windows():
    # Two kept lines three lines apart: their 2-line contexts touch.
    return "\n".join(["words only here"] * 6 + [RESULT, "filler words only", "more filler words",
                                                "F(1, 40) = 5.20, p = .03 (n = 41)"]
                     + ["words only here"] * 10)


def test_touching_windows_give_one_unit_with_joined_lines():
    pf = Prefilter(spec_with("passage"))
    text = doc_with_two_touching_windows()
    windows = list(pf.windows(text))
    units = list(pf.units(text))
    assert len(windows) >= 2 and len(units) == 1
    u, lines = units[0], text.split("\n")
    assert u.text == "\n".join(lines[u.start_line:u.end_line + 1])
    assert u.line == u.start_line == min(w.start_line for w in windows)
    assert u.end_line == max(w.end_line for w in windows)


def test_unit_window_returns_the_windows_unchanged():
    pf = Prefilter(spec_with("window"))
    text = doc_with_two_touching_windows()
    assert list(pf.units(text)) == list(pf.windows(text))


def test_units_never_read_more_than_windows():
    text = doc_with_two_touching_windows()
    pf = Prefilter(spec_with("passage"))
    assert sum(len(u.text) for u in pf.units(text)) < sum(len(w.text) for w in pf.windows(text))


def test_shipped_spec_uses_passages():
    assert Prefilter().spec["unit"] == "passage"


def test_run_text_returns_a_straddling_result_once_with_its_line():
    from statcheck_ml.pipeline import Pipeline

    text = "\n".join(["words only here"] * 8 + ["The test was t(20) = 2.31,", "p = .03 (n = 40) and more"]
                     + ["words only here"] * 8)
    pipe = Pipeline(model_path=None)
    report = pipe.run_text(text)
    stats = [r["statistic"] for r in report["results"]]
    assert stats.count(2.31) == 1
    # The line of a find is the first line of its unit, the key finds_for maps back to a range.
    unit = list(pipe.prefilter.units(text))[0]
    assert report["results"][0]["line"] == unit.start_line
    assert unit.start_line <= 8 and unit.end_line >= 9
    assert report["stages"]["prefilter"]["units_kept"] == 1
    assert report["stages"]["prefilter"]["windows_kept"] >= 2
    assert report["stages"]["prefilter"]["characters_read"] > 0
