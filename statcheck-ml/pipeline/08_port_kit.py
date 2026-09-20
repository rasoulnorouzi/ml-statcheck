"""Copy the shared rules and a shipped model into a port repository.

The R port, the web port and the Python reference all read the same prefilter,
normalisation and p-value rules, and all three must reproduce the same model.
This writes that shared material into one directory as a fixed tree, with a
manifest of every file's checksum. A port checks the manifest at load time and
refuses a stale copy before it produces a wrong answer.

The kit tree:

    spec/prefilter.json  normalize.json  repair.json  charmap.json  font_table.json
    model/<config>/tagger.onnx  decoder.json  charmap.json  weights.json
    parity/cases.json
    manifest.json
    README.md

`manifest.json` records the mother repository's commit, whether it was dirty
at export time, and the sha256 of every other file in the kit. A text file is
hashed with its line endings folded to LF, so the same file checked out on
Windows or on Linux hashes the same way.

Usage:
    python pipeline/08_port_kit.py <target_dir> [--config gru-crf ...] [--name statcheck-ml-r]
    python pipeline/08_port_kit.py --verify <kit_dir>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]     # statcheck-ml
sys.path.insert(0, str(REPO / "src"))

from statcheck_ml.provenance import sha256_file, verify_manifest  # noqa: E402

SPEC_DIR = REPO / "src" / "statcheck_ml" / "spec"
SPEC_FILES = ("prefilter.json", "normalize.json", "repair.json", "charmap.json",
              "font_table.json")
MODEL_FILES = ("tagger.onnx", "decoder.json", "charmap.json", "weights.json")
ZOO_DIR = REPO / "models" / "zoo"
PARITY_CASES = REPO / "tests" / "parity_cases.json"
RUNS_JSON = REPO / "models" / "runs.json"
EXPORT_JSON = REPO / "models" / "export.json"
EVAL_JSON = REPO / "results" / "eval.json"
INIT_PY = REPO / "src" / "statcheck_ml" / "__init__.py"

MOTHER_REPOSITORY = "https://github.com/rasoulnorouzi/ml-statcheck"


def kit_version() -> str:
    # Parsed from the source text rather than imported: importing the package
    # pulls in torch and onnxruntime for no reason this script needs them.
    match = re.search(r'__version__\s*=\s*"([^"]+)"', INIT_PY.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"no __version__ found in {INIT_PY}")
    return match.group(1)


def git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


def mother_commit() -> str:
    return git(["rev-parse", "HEAD"]).strip()


def mother_dirty() -> bool:
    # "." from a cwd of REPO is statcheck-ml itself, so a change anywhere
    # else in the enclosing repository does not mark this kit stale.
    return bool(git(["status", "--porcelain", "--", "."]).strip())


def default_config() -> str:
    eval_data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    best_model = eval_data["best_model"]
    runs = json.loads(RUNS_JSON.read_text(encoding="utf-8"))
    for record in runs:
        if record["name"] == best_model:
            return record["config"]
    raise SystemExit(f"results/eval.json best_model {best_model!r} is not in models/runs.json")


def check_configs_in_zoo(configs: list[str]) -> None:
    available = sorted(p.name for p in ZOO_DIR.iterdir() if p.is_dir())
    missing = [c for c in configs if c not in available]
    if missing:
        raise SystemExit(
            f"--config {missing} not in models/zoo/. Available: {available}")


def check_parity_model_shipped(configs: list[str]) -> dict:
    parity = json.loads(PARITY_CASES.read_text(encoding="utf-8"))
    parity_model = parity["model"]
    if parity_model not in configs:
        raise SystemExit(
            f"tests/parity_cases.json was made with model {parity_model!r}, but "
            f"--config only ships {configs}. A port built from this kit could not "
            f"test itself against the model it does not have. Add {parity_model!r} "
            f"to --config.")
    return parity


def model_stats(configs: list[str]) -> list[dict]:
    """Dev F1 (from the export report) and holdout F1 with its CI (from eval.json)."""
    runs = json.loads(RUNS_JSON.read_text(encoding="utf-8"))
    name_to_config = {r["name"]: r["config"] for r in runs}
    export_records = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
    shipped = {}
    for rec in export_records:
        if rec.get("in_zoo") and rec["name"] in name_to_config:
            shipped[name_to_config[rec["name"]]] = rec

    eval_data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    stats = []
    for config in configs:
        rec = shipped.get(config)
        if rec is None:
            raise SystemExit(f"models/export.json has no in_zoo entry for config {config!r}")
        sys_entry = eval_data["systems"].get(rec["name"])
        if sys_entry is None:
            raise SystemExit(f"results/eval.json has no system {rec['name']!r} "
                             f"for config {config!r}")
        lo, hi = sys_entry["bootstrap"]["overall"]["f1"]
        stats.append({
            "config": config,
            "run_name": rec["name"],
            "dev_f1": rec["dev_f1"],
            "holdout_f1": sys_entry["overall"]["f1"],
            "holdout_ci": (lo, hi),
        })
    return stats


def build_readme(version: str, commit: str, configs: list[str], stats: list[dict],
                  port_name: str | None) -> str:
    lines = [
        "# statcheck-ml port kit",
        "",
        "This kit holds the rules and the trained model a statcheck-ml port needs: "
        "the prefilter, the character vocabulary, the p-value constants, and one "
        "or more shipped models. The mother repository, statcheck-ml, writes it.",
        "",
        "Do not edit a file in this kit here. Change it in the mother repository "
        "and export the kit again, or this copy will drift from the other ports.",
        "",
        f"Kit version: {version}",
        f"Mother repository: {MOTHER_REPOSITORY}",
        f"Mother commit: {commit}",
        "",
        "## Models",
        "",
        "| config | dev F1 | holdout F1 | holdout 95% CI |",
        "|---|---|---|---|",
    ]
    for s in stats:
        lo, hi = s["holdout_ci"]
        lines.append(f"| {s['config']} | {s['dev_f1']:.3f} | {s['holdout_f1']:.3f} | "
                     f"[{lo:.3f}, {hi:.3f}] |")
    lines += [
        "",
        "## Verification",
        "",
        "`manifest.json` lists every other file in this kit with its sha256 hash. "
        "A port checks each hash before it trusts the kit. A text file is hashed "
        "with its line endings folded to LF first, so the hash does not depend on "
        "the platform that checked the file out.",
        "",
        "Verify from the mother repository:",
        "",
        "    python pipeline/08_port_kit.py --verify <kit_dir>",
        "",
        "## Regeneration",
        "",
    ]
    regen = ["python pipeline/08_port_kit.py <target_dir>", "--config", *configs]
    if port_name:
        regen += ["--name", port_name]
    lines.append(f"    {' '.join(regen)}")
    lines.append("")
    return "\n".join(lines)


def export_kit(target: Path, configs: list[str] | None, name: str | None) -> int:
    configs = configs or [default_config()]
    check_configs_in_zoo(configs)
    check_parity_model_shipped(configs)

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    spec_out = target / "spec"
    spec_out.mkdir()
    for fname in SPEC_FILES:
        shutil.copyfile(SPEC_DIR / fname, spec_out / fname)
        print(f"  spec/{fname}")

    for config in configs:
        model_out = target / "model" / config
        model_out.mkdir(parents=True)
        for fname in MODEL_FILES:
            shutil.copyfile(ZOO_DIR / config / fname, model_out / fname)
            print(f"  model/{config}/{fname}")

    parity_out = target / "parity"
    parity_out.mkdir()
    shutil.copyfile(PARITY_CASES, parity_out / "cases.json")
    print("  parity/cases.json")

    readme_text = build_readme(kit_version(), mother_commit(), configs, model_stats(configs), name)
    (target / "README.md").write_text(readme_text, encoding="utf-8")
    print("  README.md")

    # Every file is hashed only after the tree is complete, so manifest.json
    # is the last file written and never hashes itself.
    files = {}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            files[path.relative_to(target).as_posix()] = sha256_file(path)

    manifest = {
        "kit_version": kit_version(),
        "mother_repository": MOTHER_REPOSITORY,
        "mother_commit": mother_commit(),
        "mother_dirty": mother_dirty(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "port_name": name,
        "model_default": configs[0],
        "models": configs,
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                          encoding="utf-8")
    print("  manifest.json")
    print(f"\nexported {len(files)} files to {target}")
    return 0


def verify_kit(kit_dir: Path) -> int:
    manifest_path = kit_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest.json in {kit_dir}")
        return 1
    errors = verify_manifest(manifest_path)
    if errors:
        print(f"{len(errors)} file(s) do not match manifest.json:")
        for err in errors:
            print(f"  {err}")
        return 1
    print(f"{kit_dir}: every file matches manifest.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default=None,
                    help="the port repository directory to write the kit into")
    ap.add_argument("--config", nargs="+", default=None,
                    help="one or more models/zoo/ configs to ship; default is the "
                         "config of results/eval.json's best_model")
    ap.add_argument("--name", default=None, help="the port's name, recorded in the manifest")
    ap.add_argument("--verify", metavar="KIT_DIR", default=None,
                    help="verify an exported kit instead of exporting one")
    args = ap.parse_args(argv)

    if args.verify:
        return verify_kit(Path(args.verify))
    if not args.target:
        ap.error("target is required unless --verify is given")
    return export_kit(Path(args.target), args.config, args.name)


if __name__ == "__main__":
    raise SystemExit(main())
