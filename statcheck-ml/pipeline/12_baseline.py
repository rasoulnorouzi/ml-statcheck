"""Regenerate the R statcheck baseline the project is measured against.

Writes the repaired windows, then runs the real R package twice: once on the
raw holdout text and once on the text after the operator repair. The Python
port in `extract.py` is a convenience; only the R package says what statcheck
actually finds.

The repair function is `repair_validated` from `repair_validated.py`, the
arithmetic-validated repair (it tries every plausible operator mapping and
keeps the one whose recomputed p-values agree with the reported ones most
often). Version 1 used this same function to produce `windows_repaired2.json`
and `statcheck_repaired2.csv` (commit fe05255: "on the holdout the real
statcheck goes from 60 results to 159 once the text is repaired"; 159 matches
the row count of `statcheck_repaired2.csv`). The simpler, unvalidated
`repair.py::repair_document` produced version 1's plain `windows_repaired.json`
/ `statcheck_repaired.csv`, which this script does not reproduce. Version 2
keeps only the validated repair and drops the "2" suffix, because
`pipeline.py` (the reproducible cascade) already names `repair_validated` as
the project's one repair stage.

Usage:
  python pipeline/12_baseline.py --windows dataset/windows/holdout.json \
      --out-dir dataset/baseline --rscript "<path to Rscript.exe>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statcheck_ml.repair_validated import repair_validated

REPAIR_SPEC = Path(__file__).resolve().parents[1] / "src" / "statcheck_ml" / "spec" / "repair.json"
RUN_STATCHECK = Path(__file__).resolve().parent.parent / "r" / "run_statcheck.R"


def read_windows(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"), strict=False)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r_versions(rscript: str) -> dict:
    """R, statcheck and jsonlite versions, from a single Rscript call."""
    code = (
        "cat(as.character(getRversion()), '|', "
        "as.character(packageVersion('statcheck')), '|', "
        "as.character(packageVersion('jsonlite')), sep='')"
    )
    out = subprocess.run([rscript, "-e", code], check=True,
                          capture_output=True, text=True, encoding="utf-8")
    r_version, sc_version, jsonlite_version = out.stdout.strip().split("|")
    return {"r_version": r_version, "statcheck_version": sc_version,
            "jsonlite_version": jsonlite_version}


def run_statcheck(rscript: str, windows_path: Path, out_csv: Path) -> None:
    cmd = [rscript, str(RUN_STATCHECK), str(windows_path), str(out_csv)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rscript", required=True)
    args = ap.parse_args()

    windows_path = Path(args.windows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Repair every window's text with the validated repair.
    windows = read_windows(windows_path)
    repaired = []
    total_replacements = 0
    for w in windows:
        fixed, info = repair_validated(w["text"])
        total_replacements += info.get("replacements", 0)
        repaired.append({"window_id": w["window_id"], "text": fixed})

    repaired_path = out_dir / "windows_repaired.json"
    repaired_path.write_text(
        json.dumps(repaired, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {repaired_path} ({len(repaired)} windows, "
          f"{total_replacements} operator replacements)")

    # 2. Run the real R package, raw and repaired.
    raw_csv = out_dir / "statcheck_r.csv"
    repaired_csv = out_dir / "statcheck_repaired.csv"
    run_statcheck(args.rscript, windows_path, raw_csv)
    run_statcheck(args.rscript, repaired_path, repaired_csv)

    # 3. Record versions and provenance.
    versions = r_versions(args.rscript)
    versions["repair_spec_sha256"] = sha256_of(REPAIR_SPEC)
    versions["repair_function"] = "statcheck_ml.repair_validated.repair_validated"
    # Paths relative to the project, so the record carries no machine name.
    versions["command"] = " ".join([
        "python", "pipeline/12_baseline.py",
        "--windows", args.windows, "--out-dir", args.out_dir,
        "--rscript", "<Rscript>",
    ])
    versions_path = out_dir / "versions.json"
    versions_path.write_text(json.dumps(versions, indent=1), encoding="utf-8")
    print(f"wrote {versions_path}: {versions}")


if __name__ == "__main__":
    main()
