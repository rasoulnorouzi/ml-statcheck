"""Expand the dataset spans into character-level tags for training.

The dataset stores spans, because a per-character tag array is about ten times
the size of the text and it produces an unreadable difference in version
control. This module builds the tag arrays in memory when a model trains.

Four outputs are produced for each window, which match `SCHEMA.md`:

  part_tags  BIOES over TEST, STAT, DF1, DF2, N, POP, PVAL
  block_tags BIOES over RESULT, so the parts of one result stay grouped
  has_result one label for the whole window
  operators  the named operator for each POP span, so the model can recover an
             operator whose character was destroyed by the conversion

Usage:
    python to_bioes.py statcheck-ml/dataset/round1.jsonl      # report only

    from to_bioes import load_dataset
    rows = load_dataset('statcheck-ml/dataset/round1.jsonl')
"""
import sys, json, glob, collections

PART_LABELS = ['TEST', 'STAT', 'DF1', 'DF2', 'N', 'POP', 'PVAL']
OPERATORS = ['=', '<', '>']


def bioes(length, spans):
    """Build a BIOES tag sequence from (start, end, label) spans."""
    tags = ['O'] * length
    for start, end, label in spans:
        if end - start <= 0:
            continue
        if end - start == 1:
            tags[start] = 'S-' + label
        else:
            tags[start] = 'B-' + label
            for i in range(start + 1, end - 1):
                tags[i] = 'I-' + label
            tags[end - 1] = 'E-' + label
    return tags


def expand(row):
    """Turn one dataset row into training tensors, as plain Python lists."""
    text = row['text']
    block_spans, part_spans, operators = [], [], []

    for res in row.get('results') or []:
        bs = res.get('block_span')
        if bs:
            block_spans.append((bs[0], bs[1], 'RESULT'))
        for label, span in (res.get('part_spans') or {}).items():
            part_spans.append((span[0], span[1], label))
            if label == 'POP':
                # The character at this span may be damaged, so the target is
                # the operator the annotator named, not the character present.
                op = res.get('p_operator')
                operators.append({'span': span, 'operator': op if op in OPERATORS else None})

    return {
        'window_id': row['window_id'],
        'text': text,
        'chars': list(text),
        'part_tags': bioes(len(text), part_spans),
        'block_tags': bioes(len(text), block_spans),
        'has_result': bool(block_spans),
        'operators': operators,
        'pool': row.get('pool'),
        'journal': row.get('journal'),
        'source_doc': row.get('source_doc'),
    }


def load_dataset(*patterns):
    """Load one or more dataset files. Accepts glob patterns."""
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(expand(json.loads(line)))
    return rows


def main():
    rows = load_dataset(*sys.argv[1:])
    if not rows:
        print('no rows loaded')
        return

    n_chars = sum(len(r['text']) for r in rows)
    tag_counts = collections.Counter()
    for r in rows:
        for t in r['part_tags']:
            if t != 'O':
                tag_counts[t.split('-', 1)[1]] += 1

    ops = collections.Counter(o['operator'] for r in rows for o in r['operators'])
    pools = collections.Counter(r['pool'] for r in rows)

    print(f'windows        : {len(rows)}')
    print(f'characters     : {n_chars}')
    print(f'with a result  : {sum(1 for r in rows if r["has_result"])}')
    print(f'pools          : ' + ', '.join(f'{k}={v}' for k, v in sorted(pools.items(), key=lambda x: str(x[0]))))
    print(f'tagged chars   : {sum(tag_counts.values())} '
          f'({100*sum(tag_counts.values())/max(n_chars,1):.1f}% of characters)')
    print('per label      : ' + ', '.join(f'{k}={tag_counts[k]}' for k in PART_LABELS))
    print('operators      : ' + ', '.join(f'{k!r}={v}' for k, v in ops.most_common()))


if __name__ == '__main__':
    main()
