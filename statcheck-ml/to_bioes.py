"""Align annotation values onto window text and emit character-level tags.

The annotator reports values and an exact quote. It does not report offsets,
because a language model counts characters unreliably. This script finds the
quote in the text, finds each value inside the quote, and produces two tag
layers plus a window flag.

The search ignores differences in spaces and line breaks. A result is often
split by a line break, and a strict search fails on exactly those cases.

Usage: python to_bioes.py <sample_dir>
"""
import sys, json, os, collections

D = sys.argv[1]
win = {w['window_id']: w['text'] for w in json.load(open(os.path.join(D, 'windows.json'), encoding='utf-8'))}
ann = json.load(open(os.path.join(D, 'annotations_pass1.json'), encoding='utf-8'))
key = {k['window_id']: k for k in json.load(open(os.path.join(D, 'key.json'), encoding='utf-8'))}

PARTS = [('test_type', 'TEST'), ('statistic', 'STAT'), ('df1', 'DF1'),
         ('df2', 'DF2'), ('n', 'N'), ('p_operator', 'POP'), ('p_value', 'PVAL')]


def squeeze(text):
    """Collapse every run of whitespace to one space.

    Returns the collapsed string and a list mapping each collapsed index back
    to its index in the original string.
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


def find_span(text, flat, imap, needle, lo=0, hi=None):
    """Locate `needle` in `text`, tolerating whitespace differences.

    `lo` and `hi` are bounds in the collapsed coordinate system.
    Returns (start, end) in original coordinates, or None.
    """
    if not needle:
        return None
    nflat = ' '.join(str(needle).split())
    hi = len(flat) if hi is None else hi
    j = flat.find(nflat, lo, hi)
    if j < 0:
        return None
    start = imap[j]
    end = imap[j + len(nflat) - 1] + 1
    return start, end, j, j + len(nflat)


def bioes(n, spans):
    tags = ['O'] * n
    for s, e, lab in spans:
        if e - s <= 0:
            continue
        if e - s == 1:
            tags[s] = 'S-' + lab
        else:
            tags[s] = 'B-' + lab
            for i in range(s + 1, e - 1):
                tags[i] = 'I-' + lab
            tags[e - 1] = 'E-' + lab
    return tags


st = collections.Counter()
missing = []
out = []

for a in ann:
    wid = a['window_id']
    text = win.get(wid, '')
    flat, imap = squeeze(text)
    st['windows'] += 1
    block_spans, part_spans = [], []

    for r in a.get('results') or []:
        st['results'] += 1
        hit = find_span(text, flat, imap, r.get('quote'))
        if not hit:
            st['quote_missing'] += 1
            if len(missing) < 5:
                missing.append((wid, str(r.get('quote'))[:70]))
            continue
        st['quote_found'] += 1
        qs, qe, fs, fe = hit
        block_spans.append((qs, qe, 'RESULT'))
        cur = fs
        for field, lab in PARTS:
            val = r.get(field)
            if val in (None, ''):
                continue
            st['parts_total'] += 1
            p = find_span(text, flat, imap, val, cur, fe) or find_span(text, flat, imap, val, fs, fe)
            if not p:
                st['parts_missing'] += 1
                continue
            st['parts_aligned'] += 1
            part_spans.append((p[0], p[1], lab))
            cur = p[3]

    out.append({'window_id': wid, 'pool': key.get(wid, {}).get('pool'),
                'journal': key.get(wid, {}).get('journal'),
                'text': text,
                'block_tags': bioes(len(text), block_spans),
                'part_tags': bioes(len(text), part_spans),
                'contains_result': bool(block_spans)})

json.dump(out, open(os.path.join(D, 'tagged.json'), 'w', encoding='utf-8'), ensure_ascii=False)

qf, qm = st['quote_found'], st['quote_missing']
pa, pt = st['parts_aligned'], st['parts_total']
print('ALIGNMENT REPORT')
print(f"  windows            : {st['windows']}")
print(f"  results annotated  : {st['results']}")
print(f"  quote aligned      : {qf} / {qf + qm}  ({100*qf/max(qf+qm,1):.1f}%)")
print(f"  parts aligned      : {pa} / {pt}  ({100*pa/max(pt,1):.1f}%)")
for wid, q in missing:
    print(f'    unaligned: {wid}  {q!r}')
