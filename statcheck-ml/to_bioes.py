"""Align annotation values onto window text and emit character-level tags.

The annotator reports values and an exact quote. It does not report offsets,
because a language model counts characters unreliably. This script finds the
quote in the text, then finds each part inside the quote, and produces two
tag layers:

  layer 1 (block) : one span per whole result, so the parts of one result stay
                    grouped without a separate linking model
  layer 2 (part)  : TEST, STAT, DF1, DF2, N, POP, PVAL

Both layers use BIOES tags over characters. Short spans such as "23" benefit
from the explicit S and E tags.

Usage: python to_bioes.py <sample_dir>
"""
import sys, json, os

D = sys.argv[1]
win = {w['window_id']: w['text'] for w in json.load(open(os.path.join(D, 'windows.json'), encoding='utf-8'))}
ann = json.load(open(os.path.join(D, 'annotations_pass1.json'), encoding='utf-8'))
key = {k['window_id']: k for k in json.load(open(os.path.join(D, 'key.json'), encoding='utf-8'))}

PARTS = [('test_type', 'TEST'), ('statistic', 'STAT'), ('df1', 'DF1'),
         ('df2', 'DF2'), ('n', 'N'), ('p_operator', 'POP'), ('p_value', 'PVAL')]


def bioes(n, spans):
    """Build a BIOES tag sequence of length n from (start, end, label) spans."""
    tags = ['O'] * n
    for s, e, lab in spans:
        if e - s == 1:
            tags[s] = 'S-' + lab
        else:
            tags[s] = 'B-' + lab
            for i in range(s + 1, e - 1):
                tags[i] = 'I-' + lab
            tags[e - 1] = 'E-' + lab
    return tags


stats = {'windows': 0, 'results': 0, 'quote_found': 0, 'quote_missing': 0,
         'parts_total': 0, 'parts_aligned': 0}
missing_examples = []
out = []

for a in ann:
    wid = a['window_id']
    text = win.get(wid, '')
    stats['windows'] += 1
    block_spans, part_spans = [], []

    for r in a.get('results') or []:
        stats['results'] += 1
        quote = r.get('quote') or ''
        qi = text.find(quote) if quote else -1
        if qi < 0:
            # try a whitespace-tolerant match before giving up
            norm = ' '.join(quote.split())
            flat = ' '.join(text.split())
            if norm and norm in flat:
                qi = -2
            stats['quote_missing'] += 1
            if len(missing_examples) < 4:
                missing_examples.append((wid, quote[:70]))
            continue
        stats['quote_found'] += 1
        block_spans.append((qi, qi + len(quote), 'RESULT'))
        cursor = qi
        for field, lab in PARTS:
            val = r.get(field)
            if val in (None, ''):
                continue
            stats['parts_total'] += 1
            pi = text.find(str(val), cursor, qi + len(quote))
            if pi < 0:
                pi = text.find(str(val), qi, qi + len(quote))
            if pi < 0:
                continue
            stats['parts_aligned'] += 1
            part_spans.append((pi, pi + len(str(val)), lab))
            cursor = pi + len(str(val))

    out.append({'window_id': wid, 'pool': key.get(wid, {}).get('pool'),
                'text': text,
                'block_tags': bioes(len(text), block_spans),
                'part_tags': bioes(len(text), part_spans),
                'contains_result': bool(block_spans)})

with open(os.path.join(D, 'tagged.json'), 'w', encoding='utf-8') as fh:
    json.dump(out, fh, ensure_ascii=False)

print('ALIGNMENT REPORT')
print(f"  windows              : {stats['windows']}")
print(f"  results annotated    : {stats['results']}")
print(f"  quote located in text: {stats['quote_found']}")
print(f"  quote NOT located    : {stats['quote_missing']}")
print(f"  parts aligned        : {stats['parts_aligned']} / {stats['parts_total']}")
for wid, q in missing_examples:
    print(f'    missing: {wid}  {q!r}')
