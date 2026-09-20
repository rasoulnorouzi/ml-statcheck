#!/usr/bin/env python
"""Chunk windows into batches for annotation.

Usage:
  python pipeline/02_chunk.py --windows WINDOWS_FILE --out OUT_DIR --size BATCH_SIZE
  python pipeline/02_chunk.py --manifest MANIFEST_FILE
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statcheck_ml.provenance import write_manifest


def chunk(windows_file, out_dir, batch_size=20):
    """Chunk windows into batches.

    Reads a windows.json file and writes batch_NNN.json files to out_dir.
    Returns the number of batches created.
    """
    # Read windows
    with open(windows_file, 'r', encoding='utf-8') as f:
        windows = json.load(f)

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write batches
    batch_count = 0
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i+batch_size]
        batch_file = out_path / f"batch_{batch_count:03d}.json"

        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False, indent=1)
            f.write('\n')

        batch_count += 1

    return batch_count, len(windows)


def main():
    parser = argparse.ArgumentParser(description='Chunk windows into batches for annotation')
    parser.add_argument('--windows', type=str, help='Path to windows.json file')
    parser.add_argument('--out', type=str, help='Output directory for batch files')
    parser.add_argument('--size', type=int, default=20, help='Batch size (default 20)')
    parser.add_argument('--manifest', type=str, help='Path to manifest file to generate')
    parser.add_argument('--manifest-root', type=str, help='Root directory for recursive manifest')

    args = parser.parse_args()

    if args.manifest:
        # Generate manifest for dataset/windows/
        windows_dir = Path('dataset/windows')
        window_files = list(windows_dir.glob('*.json'))

        cmd = f"python pipeline/02_chunk.py --manifest dataset/MANIFEST.json"
        write_manifest(window_files, args.manifest, cmd)
        print(f"Wrote manifest to {args.manifest}")

    elif args.manifest_root:
        # Generate manifest for entire root directory recursively
        root_dir = Path(args.manifest_root)
        manifest_path = root_dir / 'MANIFEST.json'

        # Find all files recursively except MANIFEST.json
        all_files = []
        # Data files only. A Markdown file is documentation, and git rewrites
        # its line endings on a Windows checkout, which would change its hash.
        for file_path in root_dir.rglob('*'):
            if (file_path.is_file() and file_path.name != 'MANIFEST.json'
                    and file_path.suffix != '.md'):
                all_files.append(file_path)

        cmd = f"python pipeline/02_chunk.py --manifest-root {args.manifest_root}"
        write_manifest(all_files, str(manifest_path), cmd)
        print(f"Wrote manifest to {manifest_path}")

    elif args.windows and args.out:
        batch_count, total = chunk(args.windows, args.out, args.size)
        print(f"Wrote {batch_count} batches ({total} windows)")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
