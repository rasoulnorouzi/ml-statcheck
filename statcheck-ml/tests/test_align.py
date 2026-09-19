"""Tests for align module."""
import pytest

from statcheck_ml.align import squeeze, find_span, align_quote, as_number


def test_squeeze():
    """Test squeeze function."""
    text = "hello  world\n  test"
    flat, idx = squeeze(text)
    # squeeze collapses runs of whitespace to single spaces
    assert flat == "hello world test"
    assert len(idx) == len(flat)
    # Verify index map: each position in flat maps back to original text
    # Non-whitespace chars map exactly; spaces map to first whitespace in the run
    for i, ch in enumerate(flat):
        orig_ch = text[idx[i]]
        if ch == ' ':
            # Space in flat maps to first whitespace in original run
            assert orig_ch.isspace()
        else:
            # Non-whitespace chars map exactly
            assert orig_ch == ch


def test_align_quote_simple():
    """Test align_quote with simple text."""
    text = "F(2, 89) = 6.14, p < .01"
    quote = "F(2, 89) = 6.14"
    span = align_quote(text, quote)
    assert span is not None
    start, end = span
    assert text[start:end] == quote


def test_align_quote_with_newline():
    """Test align_quote when quote is split by newline."""
    text = "F(2, 89) = 6.14,\np < .01"
    quote = "F(2, 89) = 6.14, p < .01"
    span = align_quote(text, quote)
    assert span is not None
    # Should find it despite the newline


def test_align_quote_absent():
    """Test align_quote when quote is not in text."""
    text = "Some other text"
    quote = "F(2, 89) = 6.14"
    span = align_quote(text, quote)
    assert span is None


def test_align_quote_with_control_character():
    """Test align_quote with control character in quote."""
    text = "F(1,184) \x02 7.64, p \x03 .01"
    # The \x02 is a control character that replaced the equals sign
    quote = "F(1,184) \x02 7.64"
    span = align_quote(text, quote)
    assert span is not None


def test_as_number_simple():
    """Test as_number with simple numbers."""
    assert as_number("123") == 123.0
    assert as_number("123.45") == 123.45
    assert as_number(".02") == 0.02
    assert as_number("0.5") == 0.5


def test_as_number_negative():
    """Test as_number with negative numbers."""
    assert as_number("-2.45") == -2.45
    assert as_number("−2.45") == -2.45  # en-dash


def test_as_number_rounding():
    """Test as_number rounds to 4 decimal places."""
    assert as_number("0.123456") == 0.1235
    assert as_number("1.23456789") == 1.2346


def test_as_number_invalid():
    """Test as_number returns None for invalid input."""
    assert as_number("abc") is None
    assert as_number("") is None
    assert as_number(None) is None
    assert as_number("   ") is None
