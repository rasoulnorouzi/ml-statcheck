"""Build candidate windows from the clean corpus and draw a stratified sample.

A window is a candidate line plus the line before and the line after. This
shape is deliberate: 18.4% of results are separated from their p-value by a
line break, so a single line is not a safe unit.

The sample is drawn from three pools. The pool label is written to a separate
key file and never into the file the annotator reads, so annotation stays blind.

Usage: python sample_windows.py <clean_dir> <out_dir> [n_per_pool]
"""
import sys, os, re, json, random, hashlib

CLEAN, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8

# Same permissive pattern used for the corpus probe. Pool A is what it finds.
STAT = re.compile(
    r'\b(t|F|r|z|Z|Q|Qw|Qb|chi2|c2|X2|\u03c72|\u03c72)\s*\(\s*[0-9]+(?:\s*,\s*[0-9.]+)?'
    r'(?:\s*,\s*N\s*=\s*[0-9]+)?\s*\)\s*[=<>]\s*-?[0-9]*\.?[0-9]+', re.IGNORECASE)
# A near miss: looks statistical, but the full pattern did not match.
NEAR = re.compile(r'\b(t|F|r|z|Z|Q|chi|\u03c7|p|df|SD|M)\b[^\n]{0,20}[\(=<>]', re.IGNORECASE)
DIGIT = re.compile(r'[0-9]')


def density(line):
    """Share of characters that are not letters."""
    return 1 - sum(c.isalpha() for c in line) / len(line) if line else 0.0


def windows_of(path, rel):
    """Yield (window_text, pool) for every candidate line in one document."""
    try:
        lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
    except Exception:
        return
    for i, line in enumerate(lines):
        if len(line) < 20:
            continue
        d = density(line)
        has_digit = bool(DIGIT.search(line))
        kept = d >= 0.20 and has_digit
        text = '\n'.join(lines[max(0, i - 1):i + 2])
        if kept and STAT.search(line):
            pool = 'A'
        elif kept and NEAR.search(line):
            pool = 'B'
        elif not kept:
            pool = 'C'
        else:
            continue
        yield text, pool, rel, i


def main():
    files = []
    for dp, _, fns in os.walk(CLEAN):
        for fn in fns:
            if fn.endswith('.txt'):
                files.append(os.path.join(dp, fn))
    files.sort()
    random.seed(7)
    # Draw documents from across journals, not from one place.
    picks = random.sample(files, min(400, len(files)))

    pools = {'A': [], 'B': [], 'C': []}
    for path in picks:
        rel = os.path.relpath(path, CLEAN).replace(os.sep, '/')
        for text, pool, r, ln in windows_of(path, rel):
            if len(pools[pool]) < 4000:
                pools[pool].append({'doc': r, 'journal': r.split('/')[0], 'line': ln, 'text': text})

    os.makedirs(OUT, exist_ok=True)
    sample, key = [], []
    for pool in ('A', 'B', 'C'):
        rows = pools[pool]
        random.shuffle(rows)
        # One window per document, so no journal or paper dominates.
        seen, chosen = set(), []
        for r in rows:
            if r['doc'] in seen:
                continue
            seen.add(r['doc'])
            chosen.append(r)
            if len(chosen) >= N:
                break
        for r in chosen:
            wid = hashlib.sha1((r['doc'] + str(r['line'])).encode()).hexdigest()[:10]
            sample.append({'window_id': wid, 'text': r['text']})
            key.append({'window_id': wid, 'pool': pool, 'doc': r['doc'],
                        'journal': r['journal'], 'line': r['line']})

    random.shuffle(sample)  # hide pool order from the annotator
    with open(os.path.join(OUT, 'windows.json'), 'w', encoding='utf-8') as fh:
        json.dump(sample, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, 'key.json'), 'w', encoding='utf-8') as fh:
        json.dump(key, fh, ensure_ascii=False, indent=1)

    print(f'pool sizes found: ' + ', '.join(f'{k}={len(v)}' for k, v in pools.items()))
    print(f'sampled {len(sample)} windows to {OUT}/windows.json')
    print(f'key written to {OUT}/key.json (pool labels kept out of the annotator file)')


if __name__ == '__main__':
    main()
