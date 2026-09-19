import math

from statcheck_ml.stats import wilson, mcnemar_exact, paired_bootstrap


def test_wilson_known():
    lo, hi = wilson(k=8, n=10)
    assert math.isclose(lo, 0.4902, abs_tol=0.001) and math.isclose(hi, 0.9433, abs_tol=0.001)


def test_mcnemar_exact_symmetric_is_one():
    assert math.isclose(mcnemar_exact(b=5, c=5), 1.0)


def test_mcnemar_exact_one_sided_counts():
    assert math.isclose(mcnemar_exact(b=10, c=0), 2 * 0.5 ** 10)


def test_paired_bootstrap_zero_diff():
    units = [(1, 1)] * 20
    d, (lo, hi), p = paired_bootstrap(units, lambda u: sum(x for x, _ in u) / len(u),
                                       lambda u: sum(y for _, y in u) / len(u), n=200, seed=0)
    assert d == 0 and lo == 0 and hi == 0
