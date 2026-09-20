import math

from statcheck_ml.pvalue import CONSISTENT, DECISION_ERROR, INCONSISTENT, UNDECIDABLE, Result, check, compute_p


def test_known_values():
    # R: 2*pt(2.45, 34, lower=FALSE) = 0.0195887; pf(5.1, 2, 30, lower=FALSE) = 0.0124002;
    # pchisq(8.69, 1, lower=FALSE) = 0.0031996; 2*pnorm(1.96, lower=FALSE) = 0.0499958;
    # r = .42, df 30: t = 2.535, p = 0.0167016
    assert math.isclose(compute_p("t", 2.45, 34), 0.0195887, rel_tol=1e-5)
    assert math.isclose(compute_p("f", 5.1, 2, 30), 0.0124002, rel_tol=1e-5)
    assert math.isclose(compute_p("chi2", 8.69, 1), 0.0031996, rel_tol=1e-4)
    assert math.isclose(compute_p("z", 1.96), 0.0499958, rel_tol=1e-5)
    assert math.isclose(compute_p("r", 0.42, 30), 0.0167016, rel_tol=1e-5)


def test_non_positive_df_gives_no_p_not_nan():
    # scipy returns nan silently for df 0; the parity generator found a
    # verdict being issued on it.
    for args in (("chi2", 8.69, 0), ("t", 2.0, 0), ("f", 3.0, 0, 10), ("f", 3.0, 2, 0),
                 ("t", 2.0, -1)):
        assert compute_p(*args) is None
    res = Result(test_type="chi2", statistic=8.69, df1=0, df2=None, p_operator="=", p_value=0.003)
    assert check(res, reported_p_text=".003").verdict == UNDECIDABLE


def test_verdicts():
    def v(test, stat, df1, df2, op, p_text):
        res = Result(test_type=test, statistic=stat, df1=df1, df2=df2, p_operator=op,
                     p_value=float(p_text.lstrip("<>= ")) if p_text else None)
        return check(res, reported_p_text=p_text).verdict
    assert v("t", 2.45, 34, None, "=", ".020") == CONSISTENT
    assert v("t", 2.45, 34, None, "=", ".022") == INCONSISTENT
    assert v("t", 1.8, 46, None, "=", ".04") == DECISION_ERROR
    assert v("t", 5.59, 99, None, "<", ".01") == CONSISTENT
    assert v("r", 1.0, 30, None, "=", ".001") == UNDECIDABLE
