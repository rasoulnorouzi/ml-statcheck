"""Convert every PDF in the corpus archive to text with PyMuPDF.

The text supplied with the corpus lost all Greek letters, so the chi-square
symbol and the effect-size symbols are gone. This conversion recovers them.

Both versions are kept. The supplied text becomes an aligned damaged copy of
each document, which phase 4 uses as real rather than simulated corruption.

Usage: python statcheck-ml/convert.py <archive.zip> <output_dir> [workers]
"""
import sys, os, zipfile, time, json
from concurrent.futures import ProcessPoolExecutor


def convert_one(args):
    """Extract one PDF from the archive. Returns a manifest row."""
    archive, name, out_dir = args
    import fitz
    rel = name[len('02_pdfs/'):] if name.startswith('02_pdfs/') else name
    dest = os.path.join(out_dir, rel[:-4] + '.txt')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as z:
            data = z.read(name)
        doc = fitz.open(stream=data, filetype='pdf')
        pages = [p.get_text() for p in doc]
        n_pages = len(pages)
        doc.close()
        text = ''.join(pages)
        with open(dest, 'w', encoding='utf-8') as fh:
            fh.write(text)
        return {
            'pdf': name, 'txt': os.path.relpath(dest, out_dir).replace(os.sep, '/'),
            'journal': rel.split('/')[0], 'pages': n_pages, 'chars': len(text),
            'non_ascii': sum(1 for c in text if ord(c) > 127), 'ok': True,
        }
    except Exception as exc:
        return {'pdf': name, 'journal': rel.split('/')[0], 'ok': False, 'error': str(exc)[:200]}


def main():
    archive, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 4)

    with zipfile.ZipFile(archive) as z:
        pdfs = sorted(n for n in z.namelist() if n.lower().endswith('.pdf'))

    os.makedirs(out_dir, exist_ok=True)
    print(f'converting {len(pdfs)} PDFs with {workers} workers')

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(convert_one, [(archive, n, out_dir) for n in pdfs], chunksize=8), 1):
            rows.append(row)
            if i % 500 == 0:
                print(f'  {i}/{len(pdfs)}  ({time.time()-t0:.0f}s)')

    with open(os.path.join(out_dir, 'manifest.jsonl'), 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')

    ok = [r for r in rows if r.get('ok')]
    bad = [r for r in rows if not r.get('ok')]
    with_greek = sum(1 for r in ok if r['non_ascii'] > 0)
    print(f'\ndone in {time.time()-t0:.0f}s')
    print(f'  converted : {len(ok)}')
    print(f'  failed    : {len(bad)}')
    print(f'  with non-ASCII recovered: {with_greek} ({100*with_greek/max(len(ok),1):.1f}%)')
    print(f'  total chars: {sum(r["chars"] for r in ok)/1e6:.0f} M')
    for r in bad[:5]:
        print('  FAIL', r['pdf'], r.get('error'))


if __name__ == '__main__':
    main()
