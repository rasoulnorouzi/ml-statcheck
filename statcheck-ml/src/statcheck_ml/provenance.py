"""Provenance tracking: SHA256 hashes, manifests, batch validation, provenance stamping."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


TEXT_SUFFIXES = {'.json', '.jsonl', '.csv', '.txt', '.md'}


def sha256_file(path) -> str:
    """Return the SHA256 hash of a file.

    A text file is hashed with its line endings folded to LF, so the hash is
    the same whether git checked the file out on Windows (CRLF) or on Linux
    (LF), and whether a stage rewrote it on either platform. A binary file is
    hashed as it is.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        if Path(path).suffix.lower() in TEXT_SUFFIXES:
            h.update(f.read().replace(b'\r\n', b'\n'))
        else:
            for chunk in iter(lambda: f.read(4096), b''):
                h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA256 hash of a text string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_manifest(paths, out_path, command: str, seed=None) -> dict:
    """Write a manifest file tracking provenance of input files.

    Args:
        paths: list of file paths to hash
        out_path: path where the manifest file will be written (used to compute relative paths)
        command: the command that generated these files
        seed: optional random seed used

    Returns:
        the manifest dict that was written

    The manifest is written to out_path as JSON with:
    - generated_at: UTC ISO timestamp
    - command: the command string
    - seed: the seed (if provided)
    - files: dict of {relative_path: sha256_hash} sorted by path
    """
    out_dir = Path(out_path).parent

    files = {}
    for p in sorted(paths):
        path_obj = Path(p)
        try:
            rel_path = path_obj.relative_to(out_dir)
            # Use POSIX-style paths (forward slashes) for platform independence
            files[rel_path.as_posix()] = sha256_file(p)
        except ValueError:
            # If p is not relative to out_dir, use absolute path with POSIX style
            files[Path(p).as_posix()] = sha256_file(p)

    manifest = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'command': command,
        'files': files,
    }
    if seed is not None:
        manifest['seed'] = seed

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write('\n')

    return manifest


def verify_manifest(path) -> list[str]:
    """Verify that all files listed in a manifest still have the same hashes.

    Returns a list of file paths (relative to the manifest's parent directory)
    whose hash differs or which are missing.
    """
    with open(path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    manifest_dir = Path(path).parent
    errors = []

    for rel_path, expected_hash in manifest['files'].items():
        # Convert POSIX path to Path object (works on all platforms)
        file_path = manifest_dir / Path(rel_path)
        if not file_path.exists():
            errors.append(rel_path)
        else:
            actual_hash = sha256_file(file_path)
            if actual_hash != expected_hash:
                errors.append(rel_path)

    return errors


def stamp(record: dict, **fields) -> dict:
    """Add provenance fields to a record dict and return it.

    Sets record['provenance'] to a dict of the provided fields.
    """
    record['provenance'] = fields
    return record


def normalise_test_type(value) -> Optional[str]:
    """Normalise a test type string to the canonical form.

    Returns the normalised form (t, F, r, z, chi2, Q) or None.
    Accepts case-insensitive input and aliases like "chi-square", "χ2", "x2", etc.
    """
    if value is None:
        return None

    s = str(value).strip()
    lower = s.lower()

    # Direct matches
    if lower in ('t', 'f', 'r', 'z', 'q'):
        return lower.upper() if lower == 'f' else (lower.upper() if lower == 'q' else lower)

    # Chi-square variants
    if lower in ('chi2', 'chi²', 'chi-square', 'x2', 'χ2'):
        return 'chi2'

    # F-test variants (just 'f' is canonical)
    if lower == 'f':
        return 'F'

    # Q-test variant
    if lower == 'q':
        return 'Q'

    return None


def normalise_operator(value) -> Optional[str]:
    """Normalise a p-value operator string.

    Maps "≤" → "<", "≥" → ">", keeps "=", "<", ">", None.
    Returns the normalised form or None on invalid input.
    """
    if value is None:
        return None

    s = str(value).strip()

    # Normalise variants
    if s in ('≤', 'le'):
        return '<'
    if s in ('≥', 'ge'):
        return '>'

    # Keep standard forms
    if s in ('=', '<', '>'):
        return s

    return None


def validate_batch(batch: list[dict], output) -> list[str]:
    """Validate that rater output matches the batch specification.

    Args:
        batch: list of batch records with window_id and text
        output: the rater output (should be parsed JSON)

    Returns:
        list of error strings (empty when valid)
    """
    errors = []

    # Output must be a list
    if not isinstance(output, list):
        return ['output is not a list']

    # Output must have same length as batch
    if len(output) != len(batch):
        errors.append(f'output length {len(output)} != batch length {len(batch)}')

    # Check each record
    for i, (batch_rec, out_rec) in enumerate(zip(batch, output)):
        batch_wid = batch_rec.get('window_id')

        # window_id must match in order
        out_wid = out_rec.get('window_id')
        if out_wid != batch_wid:
            errors.append(f'record {i}: window_id {out_wid} != batch {batch_wid}')

        # contains_result must be bool
        if not isinstance(out_rec.get('contains_result'), bool):
            errors.append(f'record {i}: contains_result is not bool')

        # results must be a list
        if not isinstance(out_rec.get('results'), list):
            errors.append(f'record {i}: results is not a list')
        else:
            # Check each result
            for j, res in enumerate(out_rec.get('results', [])):
                # Must be a dict
                if not isinstance(res, dict):
                    errors.append(f'record {i} result {j}: not a dict')
                    continue

                # Must have exactly these keys
                required_keys = {'test_type', 'statistic', 'df1', 'df2', 'n', 'p_operator', 'p_value', 'quote', 'damaged', 'confidence'}
                res_keys = set(res.keys())
                if res_keys != required_keys:
                    missing = required_keys - res_keys
                    extra = res_keys - required_keys
                    if missing:
                        errors.append(f'record {i} result {j}: missing keys {missing}')
                    if extra:
                        errors.append(f'record {i} result {j}: extra keys {extra}')

                # p_operator must be one of =, <, >, or None (normalise first)
                op = res.get('p_operator')
                norm_op = normalise_operator(op)
                if norm_op is None and op is not None:
                    errors.append(f'record {i} result {j}: p_operator {op} not recognised')

                # test_type must be normalised to t, F, r, z, chi2, Q, or None
                tt = res.get('test_type')
                if tt is not None:
                    norm_tt = normalise_test_type(tt)
                    if norm_tt is None:
                        errors.append(f'record {i} result {j}: test_type {tt} not recognised')

                # damaged must be bool
                if not isinstance(res.get('damaged'), bool):
                    errors.append(f'record {i} result {j}: damaged is not bool')

                # confidence must be high or low
                conf = res.get('confidence')
                if conf not in ('high', 'low'):
                    errors.append(f'record {i} result {j}: confidence {conf} not in {{high, low}}')

                # quote must be non-empty string when results is non-empty
                quote = res.get('quote')
                if results_list := out_rec.get('results'):
                    if j < len(results_list) and not isinstance(quote, str):
                        errors.append(f'record {i} result {j}: quote is not a string')
                    if isinstance(quote, str) and not quote:
                        errors.append(f'record {i} result {j}: quote is empty string')

    return errors
