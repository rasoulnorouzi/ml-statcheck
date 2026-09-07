"""Merge the annotated chunks of one round, and check that nothing was lost.

Annotation runs as many agents in parallel, one for each chunk. An agent can
skip a passage or return them in a different order, and a silent gap becomes a
window with no label, which trains the model to find nothing there.

So this script checks coverage before it merges. It reports every window that
no chunk annotated, and every window that two chunks annotated.

Usage: python merge_chunks.py <sample_dir>
"""
import sys, os, json, glob, collections


def main(sample_dir):
    windows = json.load(open(os.path.join(sample_dir, 'windows.json'), encoding='utf-8'))
    expected = [w['window_id'] for w in windows]
    expected_set = set(expected)

    chunk_dir = os.path.join(sample_dir, 'chunks')
    files = sorted(glob.glob(os.path.join(chunk_dir, 'ann_*.json')))

    seen = collections.Counter()
    merged = {}
    bad_files = []
    invented = 0

    for path in files:
        try:
            # strict=False allows raw control characters inside strings. The
            # annotator quotes text verbatim, and a damaged operator in this
            # corpus IS a control character, so refusing them would throw away
            # exactly the results this project exists to find.
            rows = json.loads(open(path, encoding='utf-8').read(), strict=False)
        except Exception as exc:
            bad_files.append((os.path.basename(path), str(exc)[:80]))
            continue
        if not isinstance(rows, list):
            bad_files.append((os.path.basename(path), 'not a JSON array'))
            continue
        for r in rows:
            wid = r.get('window_id')
            if wid not in expected_set:
                invented += 1     # an id no chunk was given; discard it
                continue
            seen[wid] += 1
            if wid not in merged:
                merged[wid] = r

    missing = [w for w in expected if seen[w] == 0]
    duplicated = [w for w, n in seen.items() if n > 1]

    print(f'chunk files found : {len(files)}')
    print(f'windows expected  : {len(expected)}')
    print(f'windows annotated : {len(merged)}  ({100*len(merged)/max(len(expected),1):.1f}%)')
    print(f'missing           : {len(missing)}')
    print(f'annotated twice   : {len(duplicated)}')
    print(f'unknown window ids: {invented}')
    for name, err in bad_files:
        print(f'  UNREADABLE {name}: {err}')
    if missing[:5]:
        print('  first missing ids: ' + ', '.join(missing[:5]))

    # Write the merged file in the order of windows.json, so the dataset row
    # order is stable between runs.
    out = [merged[w] for w in expected if w in merged]
    dest = os.path.join(sample_dir, 'annotations_pass1.json')
    json.dump(out, open(dest, 'w', encoding='utf-8'), ensure_ascii=False)

    n_res = sum(len(r.get('results') or []) for r in out)
    n_with = sum(1 for r in out if r.get('results'))
    print(f'\nwrote {dest}')
    print(f'  windows with a result : {n_with}')
    print(f'  results total         : {n_res}')

    if missing:
        # A missing window is not fatal, but it must not be silently treated
        # as a window that holds nothing.
        json.dump(missing, open(os.path.join(sample_dir, 'missing.json'), 'w'), indent=1)
        print(f'  missing ids written to {sample_dir}/missing.json')


if __name__ == '__main__':
    main(sys.argv[1])
