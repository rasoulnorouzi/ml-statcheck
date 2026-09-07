"""Build candidate windows from the clean corpus and draw a stratified sample.

A window is a candidate line plus the line before and the line after. This shape
is deliberate: 18.4% of results are separated from their p-value by a line break,
so a single line is not a safe unit.

Reference lists are removed first. A reference entry has a high density of digits
and punctuation, so a density filter treats it as a candidate. In the first sample
six of eight pool B windows were reference entries, which wasted the budget.

Pools:
  A   the pattern matched a complete result
  B1  a p-value is present but no complete result matched   <- highest value
  B2  a test letter and digits are present, no p-value      <- lower value
  C   the density filter rejected the line

The pool label goes to a key file, never to the file the annotator reads.

Usage: python sample_windows.py <clean_dir> <out_dir> [n_per_pool]
"""
import sys, os, re, json, random, hashlib

CLEAN, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8

STAT = re.compile(
    r'\b(t|F|r|z|Z|Q|Qw|Qb|chi2|c2|X2|\u03c72)\s*\(\s*[0-9]+(?:\s*,\s*[0-9.]+)?'
    r'(?:\s*,\s*N\s*=\s*[0-9]+)?\s*\)\s*[=<>]\s*-?[0-9]*\.?[0-9]+', re.IGNORECASE)
PVAL = re.compile(r'\bp\s*[=<>]\s*\.?[0-9]', re.IGNORECASE)
TESTISH = re.compile(r'\b(t|F|r|z|Z|Q|chi2|\u03c72|M|SD|OR|CI)\s*[\(=]\s*[-.0-9]')
DIGIT = re.compile(r'[0-9]')

REF_HEAD = re.compile(r'^\s*(references|bibliography|works cited|literature cited)\s*$', re.I)
# A reference entry shows a year in parentheses, a volume with an issue and a page
# range, a digital object identifier, or a run of author initials.
REF_LINE = [
    re.compile(r'\(\s*(19|20)\d\d[a-z]?\s*\)'),
    re.compile(r'\b\d+\s*\(\s*\d+\s*\)\s*,\s*\d+\s*[-\u2013]\s*\d+'),
    re.compile(r'\b(doi:|https?://|dx\.doi\.org)', re.I),
    re.compile(r'\b[A-Z]\.\s*,?\s*[A-Z]\.'),
    re.compile(r',\s*\d+\s*[-\u2013]\s*\d+\s*\.\s*$'),
]


def strip_references(lines):
    """Drop the reference section, then drop stray reference-like lines."""
    cut = len(lines)
    for i in range(int(len(lines) * 0.55), len(lines)):
        if REF_HEAD.match(lines[i]):
            cut = i
            break
    body = lines[:cut]
    keep, dropped = [], 0
    for ln in body:
        if any(p.search(ln) for p in REF_LINE):
            keep.append('')      # keep the index stable, blank the content
            dropped += 1
        else:
            keep.append(ln)
    return keep, len(lines) - cut, dropped


def density(line):
    return 1 - sum(c.isalpha() for c in line) / len(line) if line else 0.0


def windows_of(path):
    raw = open(path, encoding='utf-8', errors='replace').read().split('\n')
    lines, n_ref_sec, n_ref_line = strip_references(raw)
    for i, line in enumerate(lines):
        if len(line) < 20:
            continue
        kept = density(line) >= 0.20 and DIGIT.search(line)
        text = '\n'.join(x for x in lines[max(0, i - 1):i + 2])
        joined = ' '.join(text.split())
        if kept and STAT.search(line):
            pool = 'A'
        elif kept and PVAL.search(joined) and not STAT.search(joined):
            pool = 'B1'
        elif kept and TESTISH.search(line):
            pool = 'B2'
        elif not kept:
            pool = 'C'
        else:
            continue
        yield text, pool, i
    return


def main():
    files = []
    for dp, _, fns in os.walk(CLEAN):
        for fn in fns:
            if fn.endswith('.txt'):
                files.append(os.path.join(dp, fn))
    files.sort()
    random.seed(13)
    picks = random.sample(files, min(700, len(files)))

    pools = {'A': [], 'B1': [], 'B2': [], 'C': []}
    for path in picks:
        rel = os.path.relpath(path, CLEAN).replace(os.sep, '/')
        try:
            for text, pool, ln in windows_of(path):
                if len(pools[pool]) < 6000:
                    pools[pool].append({'doc': rel, 'journal': rel.split('/')[0],
                                        'line': ln, 'text': text})
        except Exception:
            continue

    os.makedirs(OUT, exist_ok=True)
    sample, key = [], []
    for pool in ('A', 'B1', 'B2', 'C'):
        rows = pools[pool]
        random.shuffle(rows)
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

    random.shuffle(sample)
    json.dump(sample, open(os.path.join(OUT, 'windows.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump(key, open(os.path.join(OUT, 'key.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print('candidate windows found: ' + ', '.join(f'{k}={len(v)}' for k, v in pools.items()))
    print(f'sampled {len(sample)} windows to {OUT}/windows.json')


if __name__ == '__main__':
    main()
