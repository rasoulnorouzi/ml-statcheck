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
