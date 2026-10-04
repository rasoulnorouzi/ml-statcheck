"""The four-vote rule of docs/PROTOCOL.md version 3, one test per branch."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("consensus4", ROOT / "pipeline" / "19_consensus.py")
c4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4)

H, S, O, C = c4.VOTERS
TEXT = "As expected, t(23) = 2.45, p = .02, and the second test gave F(1, 184) = 7.64, p < .01."
T_QUOTE = "t(23) = 2.45, p = .02"
F_QUOTE = "F(1, 184) = 7.64, p < .01"


def span_of(quote, text=TEXT):
    i = text.index(quote)
    return [i, i + len(quote)]


def rater(quote=T_QUOTE, **fields):
    base = {"test_type": "t", "statistic": "2.45", "df1": "23", "df2": None, "n": None,
            "p_operator": "=", "p_value": ".02", "quote": quote, "damaged": False,
            "confidence": "high"}
    if quote == F_QUOTE:
        base.update(test_type="F", statistic="7.64", df1="1", df2="184",
                    p_operator="<", p_value=".01")
    base.update(fields)
    return c4.rater_result(base, TEXT)


def coder_row(quote=T_QUOTE, located=True, row=7, **fields):
    """A coder row in the shape of dataset/passages/inventory.jsonl."""
    base = {"test_type": "t", "test_type_coded": "t", "in_scope": True,
            "statistic": "2.45", "df1": "23", "df2": None, "n": None,
            "p_operator": "=", "p_value": "0.02", "quote": "t (23)=2.45 , p=0.02",
            "damaged": False, "confidence": "high", "coder": "MS", "row": row,
            "located": True, "span_in_passage": span_of(quote) if located else None,
            "statistic_span_in_passage": None, "on_blanked_line": not located}
    if quote == F_QUOTE:
        base.update(test_type="F", statistic="7.64", df1="1", df2="184",
                    p_operator="<", p_value="0.01")
    base.update(fields)
    return base


def vote(haiku=(), sonnet=(), opus=(), coders=()):
    voting, _ = c4.coder_votes(list(coders), TEXT)
    return c4.four_vote({H: list(haiku), S: list(sonnet), O: list(opus), C: voting})


# --------------------------------------------------------------- branches --

def test_two_raters_of_four_keep_a_result():
    out = vote(haiku=[rater()], sonnet=[rater()])
    assert out["disputes"] == []
    (kept,) = out["kept"]
    assert kept["voters"] == [H, S]
    assert kept["tier"] == "majority"
    assert kept["coder_vote"] is None
    assert kept["span"] == span_of(T_QUOTE)


def test_a_coder_and_one_rater_keep_a_result():
    # The coders never vote on n, so the rater's n stands without a conflict.
    out = vote(opus=[rater(n="50", confidence="low")], coders=[coder_row()])
    assert out["disputes"] == []
    (kept,) = out["kept"]
    assert kept["voters"] == [O, C]
    assert kept["coder_vote"] == "span"
    assert kept["coder_row"] == 7
    assert kept["n"] == "50"
    assert kept["quote"] == T_QUOTE
    assert kept["p_value"] == ".02"  # the rater's written form, not the coders' 0.02
    assert kept["confidence"] == "low"  # one rater of one wrote low
    assert kept["tier"] == "majority"


def test_a_singleton_rater_is_a_dispute():
    out = vote(sonnet=[rater()])
    assert out["kept"] == []
    (d,) = out["disputes"]
    assert d["kind"] == "singleton"
    assert [v for v, _ in d["members"]] == [S]


def test_a_singleton_coder_is_a_dispute():
    out = vote(coders=[coder_row()])
    assert out["kept"] == []
    (d,) = out["disputes"]
    assert d["kind"] == "singleton"
    assert [v for v, _ in d["members"]] == [C]


def test_a_coder_row_outside_the_six_names_does_not_vote():
    # The inventory writes test_type null for an out-of-scope row; "b" is a raw name.
    rows = [coder_row(test_type=None, in_scope=False), coder_row(test_type="b", row=8)]
    voting, counts = c4.coder_votes(rows, TEXT)
    assert voting == []
    assert counts["out_of_scope"] == 2
    out = vote(haiku=[rater()], coders=rows)
    (d,) = out["disputes"]
    assert d["kind"] == "singleton" and [v for v, _ in d["members"]] == [H]
    assert out["kept"] == []


def _with_span(result, span):
    result = dict(result)
    result["span"] = span
    return result


def test_overlap_of_half_the_shorter_span_matches_and_less_does_not():
    shorter = _with_span(rater(), [10, 20])
    at_half = c4.four_vote({H: [shorter], S: [], O: [], C: [_with_span(rater(), [15, 30])]})
    assert len(at_half["kept"]) == 1 and at_half["disputes"] == []

    below = c4.four_vote({H: [shorter], S: [], O: [], C: [_with_span(rater(), [16, 30])]})
    assert below["kept"] == []
    assert [d["kind"] for d in below["disputes"]] == ["singleton", "singleton"]


def test_an_unlocated_coder_row_matches_by_statistic_value():
    out = vote(haiku=[rater(), rater(F_QUOTE)],
               coders=[coder_row(F_QUOTE, located=False, statistic="7.64")])
    (single, kept) = (out["disputes"], out["kept"])
    (kept,) = kept
    assert kept["voters"] == [H, C]
    assert kept["coder_vote"] == "value"
    assert kept["quote"] == F_QUOTE
    assert [d["kind"] for d in single] == ["singleton"]


def test_an_unlocated_coder_row_that_matches_nothing_is_dropped_not_a_singleton():
    out = vote(coders=[coder_row(located=False, statistic="9.99", row=11)])
    assert out["kept"] == [] and out["disputes"] == []
    assert out["dropped"] == [11]


def test_a_two_two_split_is_a_field_conflict():
    out = vote(haiku=[rater()], sonnet=[rater()],
               opus=[rater(p_operator="<")], coders=[coder_row(p_operator="<")])
    (kept,) = out["kept"]
    assert kept["p_operator"] is None
    assert kept["voters"] == [H, S, O, C]
    (d,) = out["disputes"]
    assert d["kind"] == "field_conflict" and d["field"] == "p_operator"
    assert len(d["members"]) == 4  # the coder result is a candidate too


def test_a_one_one_split_is_a_field_conflict():
    # Version 2 let the first rater win this split; version 3 does not.
    out = vote(haiku=[rater()], coders=[coder_row(test_type="F")])
    (kept,) = out["kept"]
    assert kept["test_type"] is None
    (d,) = out["disputes"]
    assert d["kind"] == "field_conflict" and d["field"] == "test_type"


def test_each_field_in_conflict_raises_its_own_dispute():
    out = vote(haiku=[rater()], sonnet=[rater(df1="22", p_value=".03")])
    assert sorted(d["field"] for d in out["disputes"]) == ["df1", "p_value"]


def test_three_one_split_takes_the_majority_and_a_coder_dissent_breaks_unanimous():
    out = vote(haiku=[rater()], sonnet=[rater()], opus=[rater()],
               coders=[coder_row(df1="24")])
    (kept,) = out["kept"]
    assert out["disputes"] == []
    assert kept["df1"] == "23"
    assert kept["votes"]["df1"] == 3
    assert kept["tier"] == "majority"


def test_voters_coder_vote_and_tier_are_recorded():
    # Numbers compare as numbers, so the coders' 0.02 agrees with .02.
    out = vote(haiku=[rater()], sonnet=[rater()], opus=[rater()], coders=[coder_row()])
    (kept,) = out["kept"]
    assert kept["voters"] == [H, S, O, C]
    assert kept["coder_vote"] == "span"
    assert kept["tier"] == "unanimous"
    assert kept["pending"] == {}

    two = vote(haiku=[rater()], opus=[rater()], coders=[coder_row()])["kept"][0]
    assert two["voters"] == [H, O, C] and two["tier"] == "majority"


def test_span_shared_by_two_raters_wins_and_brings_its_quote():
    wide = T_QUOTE + ", and"
    out = vote(haiku=[rater(wide)], sonnet=[rater()], opus=[rater()])
    (kept,) = out["kept"]
    assert kept["span"] == span_of(T_QUOTE) and kept["quote"] == T_QUOTE


def test_coder_candidate_quotes_the_passage_and_takes_the_written_form():
    voting, _ = c4.coder_votes([coder_row()], TEXT)
    cand = c4.coder_candidate(voting[0], TEXT, None, None)
    assert set(cand) == set(c4.CANDIDATE_KEYS)
    assert cand["quote"] == T_QUOTE  # never the coders' transcription
    assert cand["p_value"] == ".02"
    assert cand["n"] is None and cand["confidence"] == "high"


def test_unlocated_coder_candidate_in_a_conflict_quotes_the_kept_span():
    out = vote(haiku=[rater()], coders=[coder_row(located=False, p_operator="<")])
    (kept,) = out["kept"]
    (d,) = out["disputes"]
    assert d["field"] == "p_operator"
    coder = next(r for v, r in d["members"] if v == C)
    cand = c4.candidate(C, coder, TEXT, kept)
    assert cand["quote"] == T_QUOTE and cand["p_operator"] == "<"


# ------------------------------------------------------------ end to end --

def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _env(tmp_path, n_passages):
    """Passages that each hold a haiku singleton and a coder singleton."""
    passages = [{"window_id": f"gpdead:{i}:{i}", "text": TEXT} for i in range(n_passages)]
    batches, raters = tmp_path / "batches", tmp_path / "raters"
    per_rater = {"haiku": [], "sonnet": [], "opus": []}
    for p in passages:
        per_rater["haiku"].append({"window_id": p["window_id"], "contains_result": True,
                                   "results": [{k: rater()[k] for k in c4.CANDIDATE_KEYS}]})
        for name in ("sonnet", "opus"):
            per_rater[name].append({"window_id": p["window_id"], "contains_result": False,
                                    "results": []})
    for b in range(0, n_passages, 10):
        _write(batches / f"batch_{b // 10}.json", passages[b:b + 10])
        for name, recs in per_rater.items():
            _write(raters / name / f"batch_{b // 10}.json", recs[b:b + 10])
    sha = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    lines = [json.dumps({"passage_id": p["window_id"], "text_sha256": sha,
                         "coder_results": [coder_row(F_QUOTE, row=100 + i)]})
             for i, p in enumerate(passages)]
    (tmp_path / "inventory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return batches, raters, tmp_path / "inventory.jsonl"


def _run(tmp_path, tag, env):
    batches, raters, inventory = env
    out, ddir = tmp_path / tag / "consensus.json", tmp_path / tag / "disputes"
    summary = c4.run(batches, raters, inventory, out, ddir, seed=0)
    return summary, out, ddir


def test_a_dispute_batch_names_no_voter(tmp_path):
    summary, out, ddir = _run(tmp_path, "a", _env(tmp_path, 11))
    assert summary["disputes"] == 22 and summary["dispute_batches"] == 2
    assert summary["singletons_by_voter"] == {H: 11, S: 0, O: 0, C: 11}
    assert summary["label_tier"] == "bronze"

    first = json.loads((ddir / "batch_0.json").read_text(encoding="utf-8"))
    second = json.loads((ddir / "batch_1.json").read_text(encoding="utf-8"))
    assert len(first) == 20 and len(second) == 2
    raw = (ddir / "batch_0.json").read_text(encoding="utf-8")
    for word in ("annotator", "haiku", "sonnet", "opus", "coder", "voter", "MS", "row", "votes"):
        assert word not in raw
    for d in first + second:
        assert set(d) == {"dispute_id", "window_id", "text", "kind", "candidates"}
        for cand in d["candidates"]:
            assert set(cand) == set(c4.CANDIDATE_KEYS)
    quotes = {c["quote"] for d in first for c in d["candidates"]}
    assert quotes == {T_QUOTE, F_QUOTE}

    # The voters stay in the consensus file, for the merge and the audit.
    record = json.loads(out.read_text(encoding="utf-8"))["passages"][0]
    assert sorted(v for d in record["disputes"] for v in d["voters"]) == [H, C]


def test_field_conflict_dispute_names_its_field(tmp_path):
    batches, raters, inventory = _env(tmp_path, 1)
    _write(raters / "sonnet" / "batch_0.json",
           [{"window_id": "gpdead:0:0", "contains_result": True,
             "results": [{k: rater(p_value=".03")[k] for k in c4.CANDIDATE_KEYS}]}])
    _, out, ddir = _run(tmp_path, "b", (batches, raters, inventory))
    disputes = json.loads((ddir / "batch_0.json").read_text(encoding="utf-8"))
    conflict = next(d for d in disputes if d["kind"] == "field_conflict")
    assert conflict["field"] == "p_value" and len(conflict["candidates"]) == 2
    kept = json.loads(out.read_text(encoding="utf-8"))["passages"][0]["results"][0]
    assert kept["pending"] == {"p_value": conflict["dispute_id"]}


def test_two_runs_write_identical_bytes(tmp_path):
    env = _env(tmp_path, 13)
    _, out_a, dir_a = _run(tmp_path, "a", env)
    _, out_b, dir_b = _run(tmp_path, "b", env)
    assert out_a.read_bytes() == out_b.read_bytes()
    names = sorted(p.name for p in dir_a.iterdir())
    assert names == sorted(p.name for p in dir_b.iterdir())
    for name in names:
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()


def test_a_missing_rater_file_stops_the_stage(tmp_path):
    batches, raters, inventory = _env(tmp_path, 3)
    (raters / "opus" / "batch_0.json").unlink()
    with pytest.raises(SystemExit, match="missing or invalid"):
        _run(tmp_path, "c", (batches, raters, inventory))


def test_a_wide_coder_span_matches_by_its_statistic():
    # The gold locator can span both results of the passage; the statistic
    # position still names the F test, so the coders vote for the F test only.
    stat = span_of("7.64")
    wide = coder_row(F_QUOTE, span_in_passage=[0, len(TEXT)], statistic_span_in_passage=stat)
    voting, counts = c4.coder_votes([wide], TEXT)
    assert voting[0]["span"] == stat
    assert counts["quote_span_from_line"] == 0  # the wide span still holds the statistic
    out = vote(haiku=[rater(), rater(F_QUOTE)], coders=[wide])
    f_kept = [k for k in out["kept"] if k["test_type"] == "F"]
    assert len(f_kept) == 1 and f_kept[0]["voters"] == [H, C]
    assert not [k for k in out["kept"] if k["test_type"] == "t" and C in k["voters"]]


def test_a_coder_span_that_misses_its_statistic_quotes_the_line():
    stat = span_of("2.45")
    clamped = coder_row(span_in_passage=[-229, stat[1]], statistic_span_in_passage=stat)
    voting, counts = c4.coder_votes([clamped], TEXT)
    assert counts["span_clamped"] == 1 and counts["quote_span_from_line"] == 0
    assert voting[0]["span"] == stat and voting[0]["quote_span"] == [0, stat[1]]
    missing = coder_row(span_in_passage=[0, 5], statistic_span_in_passage=stat)
    voting, counts = c4.coder_votes([missing], TEXT)
    assert counts["quote_span_from_line"] == 1
    assert voting[0]["span"] == stat
    assert voting[0]["quote_span"] == [0, len(TEXT)]
