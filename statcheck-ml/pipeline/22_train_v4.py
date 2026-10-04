"""Train the 23 runs of the version 4 pre-registration (docs/v4/preregistration.md).

The runs are listed here, not chosen at run time, so the set of runs is fixed before
any of them is seen. A run whose report.json exists is skipped, so the script resumes
after an interruption. It is written for the owner's GPU; --device cpu also works.

Usage: python pipeline/22_train_v4.py --device cuda [--parallel 4] [--epochs 30] [--only NAME,...]

One epoch of one run takes 409 s on an 8-core CPU, so the plan needs a GPU. The
models are small, so several runs share one GPU with --parallel.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The version 2 default noise: perturbed copies with the families on by default.
OLD = ["--augment", "2", "--hard-negatives", "400"]
NONE = ["--augment", "0", "--hard-negatives", "0"]
NEW = ["font_map", "negative_damage", "spaced_digits", "operator_drop"]
UNITS = {"gru-crf": ["--unit", "gru", "--crf"], "gru-softmax": ["--unit", "gru"],
         "lstm-softmax": ["--unit", "lstm"]}


def runs():
    out = {}
    # Noise gate: gru-crf seed 0, one family at a time on top of the old default.
    out["gate-none"] = UNITS["gru-crf"] + NONE + ["--seed", "0"]
    out["gate-old"] = UNITS["gru-crf"] + OLD + ["--seed", "0"]
    for fam in NEW:
        out[f"gate-old+{fam}"] = UNITS["gru-crf"] + OLD + ["--families", fam, "--seed", "0"]
    out["gate-old+all"] = UNITS["gru-crf"] + OLD + ["--families", ",".join(NEW), "--seed", "0"]
    # Grid: three configurations, three seeds, the two noise settings the rule compares.
    for cfg, unit in UNITS.items():
        for seed in (0, 1, 2):
            for noise, flags in (("none", NONE), ("all", OLD + ["--families", ",".join(NEW)])):
                name = f"{cfg}-{noise}-s{seed}"
                if cfg == "gru-crf" and seed == 0:
                    continue  # the same runs as gate-none and gate-old+all
                out[name] = unit + flags + ["--seed", str(seed)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data", default="dataset/train_v4.jsonl")
    ap.add_argument("--splits", default="dataset/splits_v4.json")
    ap.add_argument("--models", default="models/v4")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--parallel", type=int, default=1, help="runs that train at the same time")
    ap.add_argument("--list", action="store_true", help="print the run names and stop")
    args = ap.parse_args()

    plan = runs()
    if args.list:
        print("\n".join(plan))
        print(f"{len(plan)} runs")
        return
    names = args.only.split(",") if args.only else list(plan)
    unknown = [n for n in names if n not in plan]
    if unknown:
        sys.exit(f"unknown runs {unknown}; see --list")

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    todo = []
    for name in names:
        out = ROOT / args.models / name
        if (out / "report.json").exists():
            print(f"{name}: done before, skipped")
        else:
            todo.append((name, out))
    running, failed = [], []

    def start(name, out):
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "statcheck_ml.train", "--device", args.device,
               "--data", args.data, "--splits", args.splits, "--epochs", str(args.epochs),
               "--out", str(out)] + plan[name]
        print(f"start {name}: {' '.join(cmd[3:])}", flush=True)
        log = open(out / "train.log", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        running.append((name, out, proc, log, time.time()))

    while todo or running:
        while todo and len(running) < args.parallel:
            start(*todo.pop(0))
        time.sleep(5)
        for item in list(running):
            name, out, proc, log, started = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            print(f"done  {name}: exit {proc.returncode} in {time.time() - started:.0f} s", flush=True)
            if proc.returncode:
                failed.append(name)
    if failed:
        sys.exit(f"failed runs {failed}; see models/v4/<run>/train.log")
    print(f"all {len(names)} runs have report.json")

if __name__ == "__main__":
    main()
