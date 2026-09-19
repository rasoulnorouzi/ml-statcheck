from statcheck_ml.consensus import consensus

def r(span, **f):
    d = {"test_type": "t", "statistic": "2.45", "df1": "23", "df2": None, "n": None,
         "p_operator": "=", "p_value": ".02", "quote": "t(23) = 2.45, p = .02",
         "damaged": False, "confidence": "high", "span": span, "aligned": True}
    d.update(f); return d

def test_three_agree_is_unanimous():
    out = consensus({"a": [r((0, 21))], "b": [r((0, 21))], "c": [r((0, 21))]})
    assert out["kept"][0]["tier"] == "unanimous" and out["disputes"] == []

def test_two_agree_is_majority_and_field_by_vote():
    out = consensus({"a": [r((0, 21))], "b": [r((0, 21), p_operator="<")], "c": []})
    assert out["kept"][0]["tier"] == "majority"
    assert out["kept"][0]["p_operator"] == "="

def test_singleton_is_dispute():
    out = consensus({"a": [r((0, 21))], "b": [], "c": []})
    assert out["kept"] == [] and out["disputes"][0]["kind"] == "singleton"

def test_three_way_field_conflict_is_dispute():
    out = consensus({"a": [r((0, 21), p_operator="=")], "b": [r((0, 21), p_operator="<")],
                     "c": [r((0, 21), p_operator=">")]})
    assert out["disputes"][0]["kind"] == "field_conflict"
    assert out["disputes"][0]["field"] == "p_operator"

def test_numeric_fields_compare_as_numbers():
    out = consensus({"a": [r((0, 21), statistic="2.45")], "b": [r((0, 21), statistic="2.450")],
                     "c": [r((0, 21), statistic="2.45")]})
    assert out["kept"][0]["tier"] == "unanimous"
