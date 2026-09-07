"""Build candidate windows from the clean corpus and draw a stratified sample.

A window is a candidate line with two lines before it and two lines after it.
Two lines are used rather than one because a result can start before the
candidate line. In round 1 one window began in the middle of a result, which
gives the annotator an impossible task and produces a broken label.

Reference lists are removed first. A reference entry has the same density of
digits and punctuation as a result, so a density filter treats it as a
candidate. In round 0 six of eight pool B windows were reference entries.

Pools, and why each one exists:

  A    the pattern matched a complete result
       Used for quality control only. The pattern already labels these for
       free, so annotation here buys little.
  AN   the pattern matched, and the sample size is inside the parentheses
       Drawn on its own because rounds 0 and 1 sampled no chi-square of this
       shape at all, so the model never saw it.
  B1   a p-value is present but no complete result matched
  B2   a test letter and digits are present, no p-value
       This pool holds the results whose operator became a control character.
       It gave the best yield in round 1 and a pattern can never read it.
  C    the density filter rejected the line
       The only pool drawn from rejected text, so the only place a filter
       mistake can be seen.

The pool label goes to a key file, never to the file the annotator reads.

Usage: python sample_windows.py <clean_dir> <out_dir> [total_windows]
"""
import sys, os, re, json, random, hashlib

CLEAN = sys.argv[1]
OUT = sys.argv[2]
TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

# Share of the sample given to each pool. Pool A is small on purpose.
WEIGHTS = {'A': 0.06, 'AN': 0.04, 'B1': 0.30, 'B2': 0.45, 'C': 0.15}

CONTEXT_LINES = 2

STAT = re.compile(
    r'\b(t|F|r|z|Z|Q|Qw|Qb|chi2|c2|X2|χ2)\s*\(\s*[0-9]+(?:\s*,\s*[0-9.]+)?'
    r'(?:\s*,\s*N\s*=\s*[0-9]+)?\s*\)\s*[=<>]\s*-?[0-9]*\.?[0-9]+', re.IGNORECASE)
STAT_N = re.compile(r'\(\s*[0-9]+\s*,\s*N\s*[=<>]\s*[0-9, ]+\)', re.IGNORECASE)
PVAL = re.compile(r'\bp\s*[=<>]\s*\.?[0-9]', re.IGNORECASE)
TESTISH = re.compile(r'\b(t|F|r|z|Z|Q|chi2|χ2|M|SD|OR|CI)\s*[\(=]\s*[-.0-9]')
DIGIT = re.compile(r'[0-9]')
# A control character where an operator belongs. These results are invisible
# to every pattern, so a window holding one is worth annotating.
CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

REF_HEAD = re.compile(r'^\s*(references|bibliography|works cited|literature cited)\s*$', re.I)
REF_LINE = [
    re.compile(r'\(\s*(19|20)\d\d[a-z]?\s*\)'),
    re.compile(r'\b\d+\s*\(\s*\d+\s*\)\s*,\s*\d+\s*[-–]\s*\d+'),
    re.compile(r'\b(doi:|https?://|dx\.doi\.org)', re.I),
    re.compile(r'\b[A-Z]\.\s*,?\s*[A-Z]\.'),
    re.compile(r',\s*\d+\s*[-–]\s*\d+\s*\.\s*$'),
]


def strip_references(lines):
    """Drop the reference section, then blank stray reference-like lines."""
    cut = len(lines)
    for i in range(int(len(lines) * 0.55), len(lines)):
        if REF_HEAD.match(lines[i]):
            cut = i
            break
    kept = []
    for ln in lines[:cut]:
        kept.append('' if any(p.search(ln) for p in REF_LINE) else ln)
    return kept


def density(line):
    return 1 - sum(c.isalpha() for c in line) / len(line) if line else 0.0


def classify(line, window_text):
    """Return the pool for a candidate line, or None to skip it."""
    kept = density(line) >= 0.20 and DIGIT.search(line)
    if not kept:
        return 'C'
    if STAT.search(line):
        return 'AN' if STAT_N.search(line) else 'A'
    if CTRL.search(line) and TESTISH.search(line):
        return 'B2'
    flat = ' '.join(window_text.split())
    if PVAL.search(flat) and not STAT.search(flat):
        return 'B1'
    if TESTISH.search(line):
        return 'B2'
    return None


def windows_of(path):
    lines = strip_references(open(path, encoding='utf-8', errors='replace').read().split('\n'))
    for i, line in enumerate(lines):
        if len(line) < 20:
            continue
        lo = max(0, i - CONTEXT_LINES)
        hi = min(len(lines), i + CONTEXT_LINES + 1)
        text = '\n'.join(lines[lo:hi])
        pool = classify(line, text)
        if pool:
            yield text, pool, i


def main():
    files = []
    for dp, _, fns in os.walk(CLEAN):
        for fn in fns:
            if fn.endswith('.txt'):
                files.append(os.path.join(dp, fn))
    files.sort()
    random.seed(29)
    # Visit documents in random order. Filling a pool in alphabetical order
    # draws every window from the first few documents, and the per-document
    # rule below then discards almost all of them.
    random.shuffle(files)

    targets = {p: int(TOTAL * w) for p, w in WEIGHTS.items()}
    cap = {p: max(n * 40, 4000) for p, n in targets.items()}
    pools = {p: [] for p in WEIGHTS}
    # Collect at most this many windows of one pool from one document, so a
    # common pool such as C spreads over many papers instead of a handful.
    PER_DOC_COLLECT = 3

    for path in files:
        rel = os.path.relpath(path, CLEAN).replace(os.sep, '/')
        if all(len(pools[p]) >= cap[p] for p in pools):
            break
        taken = {p: 0 for p in WEIGHTS}
        try:
            for text, pool, ln in windows_of(path):
                if len(pools[pool]) >= cap[pool] or taken[pool] >= PER_DOC_COLLECT:
                    continue
                taken[pool] += 1
                pools[pool].append({'doc': rel, 'journal': rel.split('/')[0],
                                    'line': ln, 'text': text})
        except Exception:
            continue

    os.makedirs(OUT, exist_ok=True)
    sample, key = [], []
    per_doc = {}
    for pool in WEIGHTS:
        rows = pools[pool]
        random.shuffle(rows)
        chosen = []
        for r in rows:
            # At most three windows from one document, so no paper dominates.
            if per_doc.get((pool, r['doc']), 0) >= 3:
                continue
            per_doc[(pool, r['doc'])] = per_doc.get((pool, r['doc']), 0) + 1
            chosen.append(r)
            if len(chosen) >= targets[pool]:
                break
        for r in chosen:
            wid = hashlib.sha1((r['doc'] + str(r['line'])).encode()).hexdigest()[:10]
            sample.append({'window_id': wid, 'text': r['text']})
            key.append({'window_id': wid, 'pool': pool, 'doc': r['doc'],
                        'journal': r['journal'], 'line': r['line']})

    order = list(range(len(sample)))
    random.shuffle(order)
    sample = [sample[i] for i in order]

    json.dump(sample, open(os.path.join(OUT, 'windows.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump(key, open(os.path.join(OUT, 'key.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    found = ', '.join(f'{k}={len(v)}' for k, v in pools.items())
    drawn = ', '.join(f'{k}={sum(1 for x in key if x["pool"] == k)}' for k in WEIGHTS)
    print(f'candidates found : {found}')
    print(f'drawn            : {drawn}')
    print(f'total            : {len(sample)} windows -> {OUT}/windows.json')
    print(f'documents used   : {len(set(x["doc"] for x in key))}')
    print(f'journals covered : {len(set(x["journal"] for x in key))}')


if __name__ == '__main__':
    main()
