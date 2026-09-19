import pytest

from statcheck_ml.reportutil import (ReportError, adjudication, dataset_counts,
                                     engines_table, family_recall_table, format_value,
                                     git_head_info, md_table, note, rater_yield, render)


# ------------------------------------------------------------- format_value

def test_format_value_default_stringifies():
    assert format_value("cnn-crf", None) == "cnn-crf"
    assert format_value(None, None) == "not available"


def test_format_value_decimal_suffix():
    assert format_value(0.123456, "3") == "0.123"
    assert format_value(0.1, "5") == "0.10000"


def test_format_value_pct1_suffix():
    assert format_value(0.4562, "pct1") == "45.6%"


def test_format_value_int_suffix():
    assert format_value(41.6, "int") == "42"


def test_format_value_ci_suffix():
    assert format_value([0.9123, 0.9451], "ci") == "[0.912, 0.945]"


def test_format_value_missing_value_is_not_available_even_with_suffix():
    assert format_value(None, "3") == "not available"
    assert format_value(None, "ci") == "not available"


def test_format_value_unknown_suffix_raises():
    with pytest.raises(ReportError):
        format_value(1.0, "bogus")


# ------------------------------------------------------------------ render

def test_render_plain_and_suffixed_placeholders():
    values = {"n": 3, "f": 0.5}
    out = render("count {{n|int}}, half {{f|3}}", values, {})
    assert out == "count 3, half 0.500"


def test_render_unresolved_placeholder_raises_with_name():
    with pytest.raises(ReportError, match="missing_key"):
        render("{{missing_key}}", {}, {})


def test_render_unknown_table_raises_with_name():
    with pytest.raises(ReportError, match="mystery"):
        render("{{table:mystery}}", {}, {})


def test_render_known_table_is_substituted():
    tables = {"greeting": lambda: "| a |\n|---|\n| b |"}
    out = render("before\n{{table:greeting}}\nafter", {}, tables)
    assert "| a |" in out and "before" in out and "after" in out


def test_render_figure_block():
    out = render("{{figure:plot.png|A caption}}", {}, {})
    assert "![A caption](figures/plot.png)" in out
    assert "*A caption*" in out


def test_render_is_deterministic():
    values = {"n": 3, "f": 0.5}
    tables = {"t": lambda: md_table(["h"], [["1"]])}
    template = "{{n|int}} {{f|3}} {{table:t}}"
    assert render(template, values, tables) == render(template, values, tables)


# ------------------------------------------------------------------ tables

def test_dataset_counts_on_synthetic_input():
    rows = [
        {"set": "train", "windows": 10, "docs": 4, "results": 6, "damaged": 3,
         "damaged_share": 0.5},
        {"set": "holdout", "windows": 5, "docs": 2, "results": 2, "damaged": 0,
         "damaged_share": 0.0},
    ]
    table = dataset_counts(rows)
    assert table.startswith("| Set |")
    assert "| train | 10 | 4 | 6 | 3 | 50.0% |" in table
    assert "| holdout | 5 | 2 | 2 | 0 | 0.0% |" in table


def test_rater_yield_on_synthetic_input():
    rows = [{"set": "train", "rater": "haiku", "windows": 4, "results": 2}]
    table = rater_yield(rows)
    assert "| train | haiku | 4 | 2 | 0.50 |" in table


def test_engines_table_on_synthetic_input():
    engines = {"recall": {"pymupdf": 0.93, "poppler": 0.88}, "in_text": {"pymupdf": 0.93,
              "poppler": 0.88}, "spread": 0.05, "limit": 0.06, "documents": 10, "gold": 20}
    table = engines_table(engines)
    assert "| pymupdf | 0.930 | 0.930 |" in table
    assert "Spread 0.050" in table


def test_family_recall_table_on_synthetic_input():
    family = {"statcheck_repaired": {"control character": {"k": 2, "n": 4, "r": 0.5,
                                                            "wilson": [0.1, 0.9]}}}
    table = family_recall_table(family)
    assert "control character" in table
    assert "n=4" in table


# --------------------------------------------------------- missing inputs

def test_dataset_counts_missing_input_is_a_note():
    result = dataset_counts(None)
    assert result == note("dataset window/result counts not found")
    assert "not available" in result


def test_rater_yield_missing_input_is_a_note():
    assert "not available" in rater_yield([])


def test_family_recall_table_missing_input_is_a_note():
    assert "not available" in family_recall_table(None)


def test_engines_table_missing_input_is_a_note():
    assert "not available" in engines_table(None)
    assert "not available" in engines_table({"recall": {}})


def test_md_table_no_rows_is_a_note():
    assert "not available" in md_table(["h"], [])


# --------------------------------------------------------------- adjudication

def _tiny_disputes():
    return {
        "train": [
            {"kind": "singleton", "decision": {"keep": True}},
            {"kind": "singleton", "decision": {"keep": False}},
            {"kind": "field_conflict", "decision": {"keep": True}},
        ],
        "holdout": [
            {"kind": "singleton", "decision": {"keep": True}},
        ],
    }


def _tiny_tiers():
    return {
        "train": {"windows": 5, "tiers": {"unanimous": 3, "majority": 1, "adjudicated": 1}},
        "holdout": {"windows": 2, "tiers": {"unanimous": 2}},
    }


def test_adjudication_on_synthetic_input_breaks_down_by_kind():
    table = adjudication(_tiny_disputes(), _tiny_tiers())
    assert "Disputes by kind" in table
    assert "| train | field_conflict | 1 | 1 | 0 |" in table
    assert "| train | singleton | 2 | 1 | 1 |" in table
    assert "| holdout | singleton | 1 | 1 | 0 |" in table
    assert "Kept-result tiers" in table
    # Tier columns are alphabetical: adjudicated, majority, unanimous.
    assert "| train | 5 | 1 | 1 | 3 |" in table
    # No stale claim that the kind breakdown is unavailable.
    assert "not committed" not in table


def test_adjudication_missing_disputes_still_shows_tiers():
    table = adjudication({"train": None, "holdout": None}, _tiny_tiers())
    assert "not available: dataset/annotations/{train,holdout}/disputes.json not found" in table
    assert "Kept-result tiers" in table


def test_adjudication_missing_everything_is_a_note():
    result = adjudication(None, None)
    assert "not available" in result


# -------------------------------------------------------------- git_head_info

def test_git_head_info_returns_none_pair_for_a_non_git_directory(tmp_path):
    short_sha, date = git_head_info(repo=str(tmp_path))
    assert (short_sha, date) == (None, None)
