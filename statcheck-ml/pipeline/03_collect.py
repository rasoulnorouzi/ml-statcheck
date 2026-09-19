#!/usr/bin/env python
"""Collect and validate rater outputs.

Usage:
  python pipeline/03_collect.py --batches BATCHES_DIR --outputs OUTPUTS_DIR \\
    --rater RATER --agent AGENT_PATH --guideline GUIDELINE_PATH --out OUT_FILE \\
    [--retries RETRIES_FILE] [--agent-kind agent|general-purpose]
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statcheck_ml.provenance import (
    sha256_file, validate_batch, normalise_test_type, normalise_operator, stamp
)
from statcheck_ml.align import align_quote


# Model IDs for each rater
MODEL_IDS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}


def collect(batches_dir, outputs_dir, rater, agent_path, guideline_path, retries_file=None, agent_kind="agent"):
    """Collect and validate rater outputs.

    Returns (records, failures) where:
    - records: list of merged window records with provenance
    - failures: list of {'batch_id': str, 'errors': [str]} for failed batches
    """
    batches_dir = Path(batches_dir)
    outputs_dir = Path(outputs_dir)

    # Load retries if provided
    retries = {}
    if retries_file:
        with open(retries_file, 'r', encoding='utf-8') as f:
            retries = json.load(f)

    # Get SHA256 hashes
    guideline_sha = sha256_file(guideline_path)
    agent_sha = sha256_file(agent_path)
    model_id = MODEL_IDS.get(rater, rater)
    collected_at = datetime.now(timezone.utc).isoformat()

    # Find all batch files
    batch_files = sorted(batches_dir.glob('batch_*.json'))

    records = []
    failures = []

    for batch_file in batch_files:
        batch_id = batch_file.stem  # "batch_000", "batch_001", etc.

        # Load batch
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch = json.load(f)

        # Load output
        output_file = outputs_dir / f"{batch_id}.json"
        if not output_file.exists():
            failures.append({
                'batch_id': batch_id,
                'errors': [f'output file not found: {output_file}']
            })
            continue

        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                # strict=False allows control characters
                output = json.loads(f.read(), strict=False)
        except json.JSONDecodeError as e:
            failures.append({
                'batch_id': batch_id,
                'errors': [f'JSON parse error: {e}']
            })
            continue

        # Validate
        errors = validate_batch(batch, output)
        if errors:
            failures.append({
                'batch_id': batch_id,
                'errors': errors
            })
            continue

        # Merge and add provenance
        retry_count = retries.get(batch_id, 0)

        for batch_rec, out_rec in zip(batch, output):
            window_id = batch_rec['window_id']
            text = batch_rec['text']

            # Process results: add span and aligned flag, normalise test_type
            results = []
            for res in out_rec.get('results', []):
                quote = res.get('quote', '')
                span = align_quote(text, quote)

                # Normalise test_type and operator
                tt = res.get('test_type')
                norm_tt = normalise_test_type(tt)

                op = res.get('p_operator')
                norm_op = normalise_operator(op)

                result = {
                    'test_type': norm_tt,
                    'statistic': res.get('statistic'),
                    'df1': res.get('df1'),
                    'df2': res.get('df2'),
                    'n': res.get('n'),
                    'p_operator': norm_op,
                    'p_value': res.get('p_value'),
                    'quote': res.get('quote'),
                    'damaged': res.get('damaged'),
                    'confidence': res.get('confidence'),
                    'span': list(span) if span else None,
                    'aligned': span is not None,
                }
                results.append(result)

            record = {
                'window_id': window_id,
                'contains_result': out_rec.get('contains_result'),
                'results': results,
            }

            # Add provenance
            provenance = {
                'rater': rater,
                'model_alias': rater,
                'model_id': model_id,
                'guideline_sha': guideline_sha,
                'agent_sha': agent_sha,
                'agent_kind': agent_kind,
                'batch_id': batch_id,
                'collected_at': collected_at,
                'retries': retry_count,
            }
            stamp(record, **provenance)

            records.append(record)

    return records, failures


def main():
    parser = argparse.ArgumentParser(description='Collect and validate rater outputs')
    parser.add_argument('--batches', type=str, required=True, help='Batches directory')
    parser.add_argument('--outputs', type=str, required=True, help='Outputs directory')
    parser.add_argument('--rater', type=str, required=True, help='Rater name (haiku, sonnet, opus)')
    parser.add_argument('--agent', type=str, required=True, help='Path to agent file')
    parser.add_argument('--guideline', type=str, required=True, help='Path to guideline file')
    parser.add_argument('--out', type=str, required=True, help='Output file for merged records')
    parser.add_argument('--retries', type=str, help='Retries JSON file')
    parser.add_argument('--agent-kind', type=str, default='agent', help='Agent kind')

    args = parser.parse_args()

    records, failures = collect(
        args.batches,
        args.outputs,
        args.rater,
        args.agent,
        args.guideline,
        retries_file=args.retries,
        agent_kind=args.agent_kind
    )

    if failures:
        # Write failures file
        failures_path = Path(args.out).parent / (Path(args.out).stem + '.failed.json')
        with open(failures_path, 'w', encoding='utf-8') as f:
            json.dump(failures, f, ensure_ascii=False, indent=1)
            f.write('\n')

        print(f"Failed batches: {len(failures)}")
        for failure in failures:
            print(f"  {failure['batch_id']}: {failure['errors'][0]}")
        sys.exit(1)

    # Write merged output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
        f.write('\n')

    # Compute stats
    total_windows = len(records)
    total_results = sum(len(r['results']) for r in records)
    aligned_results = sum(sum(1 for res in r['results'] if res['aligned']) for r in records)
    aligned_share = aligned_results / total_results if total_results > 0 else 0

    print(f"windows {total_windows}, results {total_results}, aligned share {aligned_share:.3f}")


if __name__ == '__main__':
    main()
