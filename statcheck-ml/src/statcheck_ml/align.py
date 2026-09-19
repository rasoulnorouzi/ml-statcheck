"""Text alignment utilities: squeeze whitespace, find spans, convert numbers."""
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


def as_number(text: str) -> Optional[float]:
    """Parse text as a number, handling various formats.

    Strips whitespace, handles leading minus/en-dash, parses as float,
    and rounds to 4 decimal places.
    Returns None on failure.
    """
    if not text:
        return None

    s = str(text).strip()
    if not s:
        return None

    # Track if negative
    is_negative = False
    if s[0] in ('-', '−'):
        is_negative = True
        s = s[1:].strip()

    try:
        val = float(s)
        if is_negative:
            val = -val
        # Round to 4 decimal places
        return round(val, 4)
    except (ValueError, TypeError):
        return None
