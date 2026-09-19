"""Text alignment utilities: squeeze whitespace, find spans, convert numbers."""
import re
from typing import Optional


def squeeze(text):
    """Collapse each run of whitespace to one space.

    Returns the collapsed string and a list that maps every collapsed index
    back to its index in the original string.
    """
    out, idx, prev_ws = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_ws:
                out.append(' ')
                idx.append(i)
            prev_ws = True
        else:
            out.append(ch)
            idx.append(i)
            prev_ws = False
    return ''.join(out), idx


def find_span(flat, imap, needle, lo=0, hi=None):
    """Find `needle` while ignoring differences in whitespace.

    `lo` and `hi` bound the search in collapsed coordinates. Returns
    (start, end, flat_start, flat_end) where start and end are original
    coordinates, or None.
    """
    if needle in (None, ''):
        return None
    nflat = ' '.join(str(needle).split())
    if not nflat:
        return None
    hi = len(flat) if hi is None else hi
    j = flat.find(nflat, lo, hi)
    if j < 0:
        return None
    return imap[j], imap[j + len(nflat) - 1] + 1, j, j + len(nflat)


def align_quote(text: str, quote: str) -> Optional[tuple[int, int]]:
    """Find the character span of quote inside text, ignoring whitespace.

    Returns [start, end) or None if not found. Whitespace is ignored on
    both sides when matching.
    """
    flat, imap = squeeze(text)
    result = find_span(flat, imap, quote)
    if result is None:
        return None
    start, end, _, _ = result
    return (start, end)


# ASCII hyphen, MINUS SIGN (U+2212), and EN DASH (U+2013). A damaged PDF
# conversion writes any of the three where a reported value was negative.
_MINUS_CHARS = ("-", "−", "–")

# \x00-\x1f: a font whose glyph table has no minus sign, or no decimal
# point, sometimes leaves a raw control character in its place.
_CONTROL_RE = re.compile(r"[\x00-\x1f]")

# A thousands separator sits between a digit and exactly three digits, and
# the group is not followed by a fourth digit. "1,234.5" is 1234.5; "2,45"
# is left alone, because this corpus writes its decimals with a period, not
# a comma, and two digits after the comma is not a thousands group.
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def _parse_clean(s: str) -> Optional[float]:
    """Parse a string that has already had any control character removed."""
    is_negative = False
    if s[:1] in _MINUS_CHARS:
        is_negative = True
        s = s[1:].strip()
    s = _THOUSANDS_RE.sub("", s)
    if not s:
        return None
    try:
        val = float(s)
    except (ValueError, TypeError):
        return None
    return -val if is_negative else val


def as_number(text: str) -> Optional[float]:
    """Parse text as a number, handling various formats.

    Strips whitespace, handles a leading minus, en dash, or minus sign,
    strips a thousands comma, parses as float, and rounds to 4 decimal
    places. Returns None on failure.

    A leading or trailing control character (\\x00-\\x1f) is a common stand-in
    for a damaged minus sign or a damaged decimal point in this corpus. When
    the ordinary parse fails, one is stripped from each end and the parse is
    retried; the sign such a character carried is not recoverable, so the
    retry always returns the positive magnitude.
    """
    if not text:
        return None

    s = str(text).strip()
    if not s:
        return None

    val = _parse_clean(s)
    if val is not None:
        return round(val, 4)

    trimmed = _CONTROL_RE.sub("", s[:1]) + s[1:-1] + _CONTROL_RE.sub("", s[-1:])
    trimmed = trimmed.strip()
    if trimmed and trimmed != s:
        val = _parse_clean(trimmed)
        if val is not None:
            return round(abs(val), 4)

    return None
