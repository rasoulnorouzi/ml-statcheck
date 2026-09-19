"""Turn final annotations into the dataset file.

The dataset holds short passages and their labels. It does not hold the papers.
A passage is one candidate line with the line before and after, so it is an
excerpt of a few hundred characters. Every row records the document and the line
it came from, so anyone with the same corpus can rebuild the passage.

Labels are stored as character spans, never as per-character tag arrays. The tag
arrays are about ten times the size of the text and they produce an unreadable
difference in version control.

Usage:
    python pipeline/06_dataset.py --windows <path> --key <path> --final <path>
        --set <train|holdout> --out <path>
        [--splits <path> --seed 0 --dev-share 0.15]
        [--holdout-windows <path>]
"""
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statcheck_ml.align import squeeze, find_span, align_quote
from statcheck_ml.splits import make_splits


PARTS = [('test_type', 'TEST'), ('statistic', 'STAT'), ('df1', 'DF1'),
         ('df2', 'DF2'), ('n', 'N'), ('p_operator', 'POP'), ('p_value', 'PVAL')]


def load_json(path):
    """Load JSON, allowing control characters."""
    with open(path, encoding='utf-8') as fh:
        return json.load(fh, strict=False)


def align_result(text, result):
    """Align a result to its block_span and part_spans.

    Returns (block_span, part_spans) or (None, None) if alignment fails.
    """
    quote = result.get('quote')
    if not quote:
        return None, None

    # Use provided span if available, otherwise align the quote
    if result.get('span'):
        block_span = result['span']
    else:
        block_span = align_quote(text, quote)

    if not block_span:
        return None, None

    # Find part_spans within the block
    flat, imap = squeeze(text)
    fs, fe = block_span  # original coordinates
    # Convert to flat coordinates for searching
    flat_quote = align_quote(text, quote)
    if not flat_quote:
        return None, None

    # Find in flat coordinates
    flat_result = find_span(flat, imap, quote)
    if not flat_result:
        return None, None

    _, _, ffs, ffe = flat_result

    part_spans = {}
    cur = ffs
    for field, lab in PARTS:
        val = result.get(field)
        if val in (None, ''):
            continue
        p = find_span(flat, imap, val, cur, ffe) or find_span(flat, imap, val, ffs, ffe)
        if not p:
            continue
        part_spans[lab] = [p[0], p[1]]
        cur = p[3]

    return block_span, part_spans


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--windows', required=True, help='Path to windows.json')
    ap.add_argument('--key', required=True, help='Path to key.json')
    ap.add_argument('--final', required=True, help='Path to final.json (annotations)')
    ap.add_argument('--set', required=True, choices=['train', 'holdout'],
                    help='Dataset set (train or holdout)')
    ap.add_argument('--out', required=True, help='Output JSONL path')
    ap.add_argument('--splits', help='Optional splits.json to write')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    ap.add_argument('--dev-share', type=float, default=0.15, help='Dev share')
    ap.add_argument('--holdout-windows', help='Path to holdout windows.json for validation')

    args = ap.parse_args()

    # Load data
    windows = {w['window_id']: w['text'] for w in load_json(args.windows)}
    key = {k['window_id']: k for k in load_json(args.key)}
    annotations = load_json(args.final)

    if args.holdout_windows:
        holdout_windows = {w['window_id'] for w in load_json(args.holdout_windows)}
    else:
        holdout_windows = set()

    # Build rows
    rows = []
    n_results = 0
    n_aligned = 0
    dropped_results = []

    for ann in annotations:
        wid = ann['window_id']
        text = windows.get(wid, '')
        k = key.get(wid, {})

        if args.set == 'train' and k.get('set') != 'train':
            continue
        if args.set == 'holdout' and k.get('set') != 'holdout':
            continue

        results = []
        for res in ann.get('results') or []:
            n_results += 1
            block_span, part_spans = align_result(text, res)

            if not block_span:
                dropped_results.append({
                    'window_id': wid,
                    'quote': res.get('quote'),
                })
                continue

            n_aligned += 1
            checkable = bool(res.get('statistic')) and bool(res.get('df1')) and bool(res.get('p_value'))

            results.append({
                'test_type': res.get('test_type'),
                'statistic': res.get('statistic'),
                'df1': res.get('df1'),
                'df2': res.get('df2'),
                'n': res.get('n'),
                'p_operator': res.get('p_operator'),
                'p_value': res.get('p_value'),
                'quote': res.get('quote'),
                'damaged': bool(res.get('damaged')),
                'confidence': res.get('confidence'),
                'block_span': block_span,
                'part_spans': part_spans,
                'checkable': checkable,
                'tier': res.get('tier'),
                'raters': res.get('raters'),
            })

        # Count tier distribution
        tier_counts = Counter(r.get('tier') for r in results)

        row = {
            'window_id': wid,
            'round': 'v2',
            'pool': k.get('pool'),
            'journal': k.get('journal'),
            'source_doc': k.get('doc'),
            'source_line': k.get('line'),
            'text': text,
            'contains_result': bool(results),
            'label_tier': 'bronze',
            'results': results,
            'set': args.set,
            'tier_counts': dict(tier_counts),
        }
        rows.append(row)

    # Check alignment gate
    if n_results > 0:
        aligned_share = n_aligned / n_results
        if aligned_share < 0.98:
            print(f"Alignment below 98%: {aligned_share:.1%} ({n_aligned}/{n_results})", file=sys.stderr)
            print(f"Failed alignments:", file=sys.stderr)
            for item in dropped_results[:20]:
                print(f"  {item['window_id']}: {item['quote']!r}", file=sys.stderr)
            if len(dropped_results) > 20:
                print(f"  ... and {len(dropped_results) - 20} more", file=sys.stderr)
            sys.exit(1)

    # Print warnings for dropped results
    for item in dropped_results:
        print(f"Warning: Dropped result {item['window_id']}: {item['quote']!r}", file=sys.stderr)

    # Sort by window order from windows file
    window_order = {w['window_id']: i for i, w in enumerate(load_json(args.windows))}
    rows.sort(key=lambda r: window_order.get(r['window_id'], 999999))

    # Make splits if requested
    if args.splits:
        unique_docs = sorted(set(r['source_doc'] for r in rows if r['source_doc']))
        splits = make_splits(unique_docs, dev_share=args.dev_share, seed=args.seed)

        # Validate holdout disjointness
        if args.holdout_windows:
            holdout_docs = set(k.get('doc') for k in key.values() if k.get('set') == 'holdout')
            train_docs = set(splits['documents'].keys())
            overlap = holdout_docs & train_docs
            if overlap:
                print(f"Error: Holdout documents overlap with training documents: {overlap}", file=sys.stderr)
                sys.exit(1)

        # Write splits
        with open(args.splits, 'w', encoding='utf-8') as fh:
            json.dump(splits, fh, ensure_ascii=False, indent=2)
            fh.write('\n')

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'Wrote {out_path}')
    print(f'  rows: {len(rows)}')
    print(f'  results kept: {n_aligned} of {n_results}')
    if n_results > 0:
        print(f'  aligned share: {n_aligned/n_results:.1%}')
    print(f'  dropped results: {len(dropped_results)}')


if __name__ == '__main__':
    main()
