"""Ship a version 4 run as the packaged model: the owner's decision of 2026-10-05.

The pre-registered rule 3 kept gru-crf-s0 (docs/v4/preregistration.md). After the gold
test was read, the owner chose to ship a version 4 model; the session recommended
gate-none (gru-crf, no noise, seed 0), the same architecture as the old model. This
stage makes that swap repeatable:

  1. copies <run>/zoo (tagger.onnx, decoder.json, charmap.json) into the packaged zoo
     src/statcheck_ml/zoo/gru-crf and into models/zoo/gru-crf, where the parity
     generator and 08_port_kit.py read it, with weights.json for the R forward pass;
  2. promotes the run's charmap.json to src/statcheck_ml/spec/charmap.json, because
     the shared spec and the shipped model must use one vocabulary;
  3. scores the run and its cascade on the bronze holdout (09_evaluate.py; no gold paper)
     and writes results/v4/shipped.json with the scores 08_port_kit.py prints.

The gold-test scores in shipped.json come from results/v4/final/summary.json, read
before this decision; they are not a clean held-out score for the shipped model.

Usage: python pipeline/28_ship_v4.py [--run gate-none]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("tagger.onnx", "decoder.json", "charmap.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="gate-none")
    ap.add_argument("--models", default="models/v4")
    ap.add_argument("--out", default="results/v4/shipped.json")
    args = ap.parse_args()

    run = ROOT / args.models / args.run
    report = json.loads((run / "report.json").read_text())
    if not report.get("crf") or report.get("unit") != "gru":
        sys.exit(f"{args.run} is not a gru-crf run; the ports load model/gru-crf")
    packaged, zoo = ROOT / "src/statcheck_ml/zoo/gru-crf", ROOT / "models/zoo/gru-crf"
    for dest in (packaged, zoo):
        dest.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copyfile(run / "zoo" / name, dest / name)
    shutil.copyfile(run / "zoo" / "export_report.json", zoo / "export_report.json")
    shutil.copyfile(run / "zoo" / "charmap.json", ROOT / "src/statcheck_ml/spec/charmap.json")
    py = sys.executable
    subprocess.run([py, "-m", "statcheck_ml.weights", str(run / "model.pt"), str(zoo / "weights.json")],
                   cwd=ROOT, check=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), stdout=subprocess.DEVNULL)

    view = ROOT / args.models / "_ship_view"
    view.mkdir(exist_ok=True)
    link = view / args.run
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((run / "zoo").resolve())
    (view / "runs.json").write_text(json.dumps([{"name": args.run, "status": "done", "config": "gru-crf",
                                                 "seed": report["seed"],
                                                 "dev_f1": report["dev"]["overall"]["f1"]}]) + "\n")
    bronze_path = ROOT / "results/v4/final/shipped_bronze.json"
    subprocess.run([py, "pipeline/09_evaluate.py", "--runs", str(view / "runs.json"), "--export", "/nonexistent",
                    "--out", str(bronze_path), "--resamples", "2000", "--seed", "0"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    systems = json.loads(bronze_path.read_text())["systems"]
    model, cascade = systems[args.run], systems["cascade_gru-crf-s0"]
    gold = json.loads((ROOT / "results/v4/final/summary.json").read_text())["runs"][args.run]

    def block(s):
        return {"precision": s["overall"]["p"], "recall": s["overall"]["r"], "f1": s["overall"]["f1"],
                "f1_ci": s["bootstrap"]["overall"]["f1"]}

    shipped = {
        "run": args.run, "config": "gru-crf", "seed": report["seed"],
        "decision": "owner, 2026-10-05, after the gold test was read; rule 3 had kept gru-crf-s0",
        "dev_f1": report["dev"]["overall"]["f1"],
        "dev_note": "F1 on the dev side of dataset/splits_v4.json (bronze and passages), the early-stopping set",
        "bronze_holdout": {"model": block(model), "cascade": block(cascade)},
        "gold_test": {"model": gold["gold_model"], "cascade": gold["gold_cascade"],
                      "note": "read before the decision; not a clean held-out score for this model"},
        "files": {f"zoo/gru-crf/{name}": sha256(packaged / name) for name in FILES},
    }
    (ROOT / args.out).write_text(json.dumps(shipped, indent=1) + "\n")
    print(f"shipped {args.run}: bronze model F1 {model['overall']['f1']:.3f}, "
          f"cascade F1 {cascade['overall']['f1']:.3f}; wrote {args.out}")


if __name__ == "__main__":
    main()
