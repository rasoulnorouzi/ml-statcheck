"""Copy the shared rules into a port repository, with a manifest.

The Python port, the R port and the browser port will live in three separate
repositories. Each one needs the same rules, and a rule that is restated in a
port is a rule that will drift.

This copies every shared file into a port repository and writes a manifest that
records the version and the checksum of each one. A port checks the manifest at
load time. When a checksum does not match, the port knows its copy is stale
before it produces a wrong answer, rather than after.

What travels:

  spec/*.json              the rules themselves
  tests/parity_cases.json  the cases, and the answer Python gives for each

The parity cases travel because a port that cannot prove it matches Python is
worse than no port. Three real faults were found this way on the first day: a
search window one character too wide in JavaScript, an index counted from one
instead of zero in R, and a tie broken by locale collation in R.

Usage:
    python pipeline/08_port_kit.py <target_dir> [--name statcheck-ml-r]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SPEC_DIR = HERE / "src" / "statcheck_ml" / "spec"
PARITY_CASES = HERE / "tests" / "parity_cases.json"

MANIFEST = "statcheck-ml-manifest.json"


def digest(path: Path) -> str:
    """The SHA-256 of one file, so a port can detect a stale copy."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_version(path: Path) -> int | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("version")
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="the port repository to copy into")
    ap.add_argument("--name", default=None,
                    help="the name of the port, recorded in the manifest")
    ap.add_argument("--model", default=None,
                    help="a model directory holding tagger.onnx and "
                         "decoder.json, such as models/final-crf-aug")
    args = ap.parse_args(argv)

    target = Path(args.target)
    spec_out = target / "spec"
    tests_out = target / "tests"
    spec_out.mkdir(parents=True, exist_ok=True)
    tests_out.mkdir(parents=True, exist_ok=True)

    files = {}

    if args.model:
        model_dir = Path(args.model)
        model_out = target / "model"
        model_out.mkdir(parents=True, exist_ok=True)
        for name in ("tagger.onnx", "decoder.json"):
            src = model_dir / name
            if not src.exists():
                print(f"  WARNING: {src} is missing. Run "
                      f"`python -m statcheck_ml.export` first.")
                continue
            # A loose weight file is the way a port ends up with a model that
            # has no weights in it. The exporter writes one file for that
            # reason, and this refuses to copy a split pair.
            if (model_dir / (name + ".data")).exists():
                raise SystemExit(
                    f"{src} keeps its weights in a separate .data file. Export "
                    f"it again: the browser runtime cannot follow that link.")
            shutil.copyfile(src, model_out / name)
            files[f"model/{name}"] = {
                "sha256": digest(src), "bytes": src.stat().st_size}
            print(f"  model/{name}")

    for path in sorted(SPEC_DIR.glob("*.json")):
        dest = spec_out / path.name
        shutil.copyfile(path, dest)
        files[f"spec/{path.name}"] = {
            "version": spec_version(path),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        print(f"  spec/{path.name}")

    if PARITY_CASES.exists():
        dest = tests_out / PARITY_CASES.name
        shutil.copyfile(PARITY_CASES, dest)
        cases = json.loads(PARITY_CASES.read_text(encoding="utf-8"))
        files[f"tests/{PARITY_CASES.name}"] = {
            "cases": len(cases),
            "sha256": digest(PARITY_CASES),
            "bytes": PARITY_CASES.stat().st_size,
        }
        print(f"  tests/{PARITY_CASES.name}  ({len(cases)} cases)")
    else:
        print("  WARNING: no parity cases found. Run "
              "tests/make_parity_cases.py first.")

    manifest = {
        "_comment": (
            "Written by export_port_kit.py in the statcheck-ml repository. "
            "Do not edit these files here. Change them in that repository and "
            "export again, or the ports will drift. Check every checksum when "
            "the port loads."),
        "port": args.name or target.name,
        "exported": date.today().isoformat(),
        "files": files,
    }
    (target / MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  {MANIFEST}")
    print(f"\nexported {len(files)} files to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
