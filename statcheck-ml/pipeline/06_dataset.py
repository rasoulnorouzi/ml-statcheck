"""Turn one annotation round into the shareable dataset file.

The dataset holds short passages and their labels. It does not hold the
papers. A passage is one candidate line with the line before and after, so it
is an excerpt of a few hundred characters. Every row records the document and
the line it came from, so anyone with the same corpus can rebuild the passage.

Labels are stored as character spans, never as per-character tag arrays. The
tag arrays are about ten times the size of the text and they produce an
unreadable difference in version control. `to_bioes.py` expands the spans into
tags at training time.

Usage: python pipeline/06_dataset.py <sample_dir> <dataset_dir> <round_name>
"""
import sys, os, json


PARTS = [('test_type', 'TEST'), ('statistic', 'STAT'), ('df1', 'DF1'),
         ('df2', 'DF2'), ('n', 'N'), ('p_operator', 'POP'), ('p_value', 'PVAL')]


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


def build(sample_dir, out_dir, name):
    def load(fn):
        # strict=False allows raw control characters inside strings. A damaged
        # operator in this corpus IS a control character, so rejecting them
        # would discard the results the project exists to find.
        with open(os.path.join(sample_dir, fn), encoding='utf-8') as fh:
            return json.loads(fh.read(), strict=False)

    windows = {w['window_id']: w['text'] for w in load('windows.json')}
    key = {k['window_id']: k for k in load('key.json')}
    ann = load('annotations_pass1.json')

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + '.jsonl')

    n_rows = n_res = n_aligned = n_parts = n_parts_ok = 0
    with open(path, 'w', encoding='utf-8') as fh:
        for a in ann:
            wid = a['window_id']
            text = windows.get(wid, '')
            k = key.get(wid, {})
            flat, imap = squeeze(text)
            results = []

            for r in a.get('results') or []:
                n_res += 1
                hit = find_span(flat, imap, r.get('quote'))
                if not hit:
                    continue          # unaligned results are dropped, and counted
                n_aligned += 1
                qs, qe, fs, fe = hit
                spans, cur = {}, fs
                for field, lab in PARTS:
                    val = r.get(field)
                    if val in (None, ''):
                        continue
                    n_parts += 1
                    p = find_span(flat, imap, val, cur, fe) or find_span(flat, imap, val, fs, fe)
                    if not p:
                        continue
                    n_parts_ok += 1
                    spans[lab] = [p[0], p[1]]
                    cur = p[3]

                # A result can be checked only when the arithmetic has every
                # value it needs.
                checkable = bool(r.get('statistic')) and bool(r.get('df1')) and bool(r.get('p_value'))
                results.append({
                    'test_type': r.get('test_type'), 'statistic': r.get('statistic'),
                    'df1': r.get('df1'), 'df2': r.get('df2'), 'n': r.get('n'),
                    'p_operator': r.get('p_operator'), 'p_value': r.get('p_value'),
                    'quote': r.get('quote'), 'damaged': bool(r.get('damaged')),
                    'confidence': r.get('confidence'),
                    'block_span': [qs, qe], 'part_spans': spans,
                    'checkable': checkable,
                })

            row = {
                'window_id': wid,
                'round': name,
                'pool': k.get('pool'),
                'journal': k.get('journal'),
                'source_doc': k.get('doc'),
                'source_line': k.get('line'),
                'text': text,
                'contains_result': bool(results),
                'label_tier': 'bronze',        # produced by a language model
                'results': results,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            n_rows += 1

    size = os.path.getsize(path)
    print(f'wrote {path}')
    print(f'  windows            : {n_rows}')
    print(f'  results kept       : {n_aligned} of {n_res}')
    print(f'  part spans located : {n_parts_ok} of {n_parts}')
    print(f'  file size          : {size/1024:.1f} KB  ({size/max(n_rows,1):.0f} bytes per window)')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2], sys.argv[3])
