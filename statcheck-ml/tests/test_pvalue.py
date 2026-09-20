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


def test_the_statistic_rounding_widens_the_comparison():
    # t(67) = 1.48 implies p = .1436, and a paper that writes p = .143 is not
    # wrong: the statistic itself was rounded, so the true p lies in a range.
    # statcheck 1.5.0 accepts this row; before the rule matched statcheck this
    # project called it inconsistent.
    res = Result(test_type="t", statistic=1.48, df1=67, p_operator="=", p_value=0.143)
    assert check(res, reported_p_text=".143", statistic_text="1.48").verdict == CONSISTENT
    # A statistic printed to more decimals leaves less room, and the same
    # reported p is then an error.
    res = Result(test_type="t", statistic=1.4800, df1=67, p_operator="=", p_value=0.143)
    assert check(res, reported_p_text=".143", statistic_text="1.4800").verdict == INCONSISTENT


def test_a_reported_p_of_zero_is_an_error():
    # No test gives exactly zero, whatever the computed value is.
    res = Result(test_type="t", statistic=12.0, df1=30, p_operator="=", p_value=0.0)
    assert check(res, reported_p_text=".000", statistic_text="12.0").verdict == INCONSISTENT


def test_ns_reads_as_greater_than_alpha():
    res = Result(test_type="t", statistic=0.54, df1=178, p_operator="ns", p_value=None)
    assert check(res, statistic_text="0.54").verdict == CONSISTENT
    # A statistic that is significant contradicts the claim of no significance.
    res = Result(test_type="t", statistic=5.0, df1=178, p_operator="ns", p_value=None)
    assert check(res, statistic_text="5.0").verdict == DECISION_ERROR


def test_operators_use_the_interval():
    # p < .01 with a computed .0099: the paper's bound holds.
    res = Result(test_type="t", statistic=2.70, df1=90, p_operator="<", p_value=0.01)
    assert check(res, reported_p_text=".01", statistic_text="2.70").verdict == CONSISTENT
    # p < .001 with a computed .008 does not. Both sit under alpha, so the
    # paper's conclusion still holds and the verdict is not a decision error.
    res = Result(test_type="t", statistic=2.70, df1=90, p_operator="<", p_value=0.001)
    assert check(res, reported_p_text=".001", statistic_text="2.70").verdict == INCONSISTENT
    # p < .05 claimed where the statistic gives .38 flips the conclusion.
    res = Result(test_type="t", statistic=0.88, df1=90, p_operator="<", p_value=0.05)
    assert check(res, reported_p_text=".05", statistic_text="0.88").verdict == DECISION_ERROR


def test_damaged_statistic_text_is_ignored():
    # A model span can hand over the damaged characters it read. The text is
    # used only when it reads back as the same number.
    res = Result(test_type="t", statistic=1.48, df1=67, p_operator="=", p_value=0.143)
    good = check(res, reported_p_text=".143", statistic_text="1.48").verdict
    junk = check(res, reported_p_text=".143", statistic_text="1\x0348").verdict
    assert good == junk == CONSISTENT
    # And a text that says more decimals than the number has is not trusted
    # either, because it cannot be what was printed.
    res = Result(test_type="t", statistic=1.48, df1=67, p_operator="=", p_value=0.143)
    assert check(res, reported_p_text=".143", statistic_text="1.4899").verdict == CONSISTENT
