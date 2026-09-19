import math

from statcheck_ml.agreement import (cohen_kappa, fleiss_kappa, krippendorff_alpha,
                                    match_results, pairwise_f1, bootstrap_ci)


def test_cohen_kappa_textbook():
    # 50 items: both yes 20, both no 15, A yes B no 5, A no B yes 10
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert math.isclose(cohen_kappa(a, b), 0.4, abs_tol=1e-9)


def test_fleiss_kappa_wikipedia():
    # 10 subjects, 14 raters, 5 categories; counts per subject
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
             [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
             [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert math.isclose(fleiss_kappa(table), 0.210, abs_tol=0.001)


def test_krippendorff_perfect_and_two_rater_matches_cohen_direction():
    a = ["y", "n", "y", "n", "y", "n", "y", "y", "n", "n"]
    assert math.isclose(krippendorff_alpha([a, a]), 1.0, abs_tol=1e-9)
    b = ["y", "n", "n", "n", "y", "y", "y", "y", "n", "n"]
    assert 0 < krippendorff_alpha([a, b]) < 1


def test_match_results_strict_and_lenient():
    a = [{"span": (0, 21), "statistic": "2.45"}]
    b = [{"span": (0, 20), "statistic": "2.45"}]
    assert match_results(a, b, mode="strict") == []
    assert match_results(a, b, mode="lenient") == [(0, 0)]


def test_pairwise_f1_symmetric():
    a = {"w1": [{"span": (0, 5)}], "w2": []}
    b = {"w1": [{"span": (0, 5)}, {"span": (10, 15)}], "w2": []}
    p = pairwise_f1(a, b, mode="strict")
    q = pairwise_f1(b, a, mode="strict")
    assert math.isclose(p, q) and math.isclose(p, 2 / 3)


def test_bootstrap_ci_contains_point():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    lo, hi = bootstrap_ci(values, lambda v: sum(v) / len(v), n=500, seed=0)
    assert lo <= 0.7 <= hi
