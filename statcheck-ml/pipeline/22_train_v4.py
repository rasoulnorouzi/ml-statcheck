"""Train the 35 runs of the version 4 pre-registration (docs/v4/preregistration.md).

The runs are listed here, not chosen at run time, so the set of runs is fixed before
any of them is seen. A run whose report.json exists is skipped, so the script resumes
after an interruption. It is written for the owner's GPU; --device cpu also works.

Usage: python pipeline/22_train_v4.py --device cuda [--parallel 4] [--epochs 30] [--only NAME,...]

On a CPU, use one thread per run and run several at once: with eight threads one
epoch on 600 rows took 771 s, with one thread 42 s, because the small recurrent
model spends eight threads waiting on each other.
"""
import argparse
import json
import re
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


SEEDS = (0, 1, 2, 3, 4)


def runs():
    out = {}
    # Noise gate: gru-crf seed 0, one family at a time on top of the old default.
    out["gate-none"] = UNITS["gru-crf"] + NONE + ["--seed", "0"]
    out["gate-old"] = UNITS["gru-crf"] + OLD + ["--seed", "0"]
    for fam in NEW:
        out[f"gate-old+{fam}"] = UNITS["gru-crf"] + OLD + ["--families", fam, "--seed", "0"]
    out["gate-old+all"] = UNITS["gru-crf"] + OLD + ["--families", ",".join(NEW), "--seed", "0"]
    # Grid: three configurations, five seeds, the two noise settings the rules compare.
    for cfg, unit in UNITS.items():
        for seed in SEEDS:
            for noise, flags in (("none", NONE), ("all", OLD + ["--families", ",".join(NEW)])):
                name = f"{cfg}-{noise}-s{seed}"
                if cfg == "gru-crf" and seed == 0:
                    continue  # the same runs as gate-none and gate-old+all
                out[name] = unit + flags + ["--seed", str(seed)]
    return out


NOISE = {"none": NONE, "old": OLD,
         **{f"old+{fam}": OLD + ["--families", fam] for fam in NEW},
         "all": OLD + ["--families", ",".join(NEW)]}


def extended_runs():
    """Every noise setting on every configuration for seeds 0 to 2, plus the 35 runs.

    Exploratory. The pre-registered rules read only the runs of runs(), which this
    plan contains under the same names, so the extra runs can never pick the model.
    """
    out = dict(runs())
    for cfg, unit in UNITS.items():
        for seed in (0, 1, 2):
            for noise, flags in NOISE.items():
                if cfg == "gru-crf" and seed == 0:
                    name = "gate-old+all" if noise == "all" else f"gate-{noise}"
                else:
                    name = f"{cfg}-{noise}-s{seed}"
                out.setdefault(name, unit + flags + ["--seed", str(seed)])
    assert all(out[n] == runs()[n] for n in runs())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=30)
    # 64, not the version 2 default of 16: the CRF forward pass steps through every
    # character, and four times fewer batches is four times fewer GPU launches.
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data", default="dataset/train_v4.jsonl")
    ap.add_argument("--splits", default="dataset/splits_v4.json")
    ap.add_argument("--models", default="models/v4")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--parallel", type=int, default=1, help="runs that train at the same time")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads per run (OMP_NUM_THREADS)")
    ap.add_argument("--list", action="store_true", help="print the run names and stop")
    ap.add_argument("--progress-every", type=int, default=60, help="seconds between progress tables")
    ap.add_argument("--plan", default="prereg", choices=["prereg", "extended"],
                    help="prereg: the 35 runs the rules read; extended: 75 runs, exploratory")
    args = ap.parse_args()

    plan = runs() if args.plan == "prereg" else extended_runs()
    if args.list:
        print("\n".join(plan))
        print(f"{len(plan)} runs")
        return
    names = args.only.split(",") if args.only else list(plan)
    unknown = [n for n in names if n not in plan]
    if unknown:
        sys.exit(f"unknown runs {unknown}; see --list")

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    if args.threads:
        env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    def command(name, out):
        return [sys.executable, "-u", "-m", "statcheck_ml.train", "--device", args.device,
                "--data", args.data, "--splits", args.splits, "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size), "--out", str(out)] + plan[name]

    def same_settings(name, out):
        # A run finished under other settings (an older kit, another batch size) must
        # train again; only the interpreter path may differ between machines.
        try:
            done = json.loads((out / "command.json").read_text())
        except (OSError, ValueError):
            return False
        return done[3:] == command(name, out)[3:]

    todo = []
    for name in names:
        out = ROOT / args.models / name
        if (out / "report.json").exists() and same_settings(name, out):
            print(f"{name}: done before with the same settings, skipped")
        elif (out / "report.json").exists():
            print(f"{name}: done before with other settings, trains again")
            todo.append((name, out))
        else:
            todo.append((name, out))
    running, failed = [], []

    def start(name, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").unlink(missing_ok=True)
        cmd = command(name, out)
        (out / "command.json").write_text(json.dumps(cmd))
        print(f"start {name}: {' '.join(cmd[3:])}", flush=True)
        log = open(out / "train.log", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        running.append((name, out, proc, log, time.time()))

    began, last_table, finished_secs, total = time.time(), 0.0, [], len(todo)
    epoch_line = re.compile(r"epoch\s+(\d+).*F1 ([0-9.]+)")

    def table():
        # A notebook cell shows only what this loop prints, so the state of every
        # running run is read back from its log and printed here.
        lines = [f"[{time.strftime('%H:%M:%S')}] done {total - len(todo) - len(running)}/{total}, "
                 f"running {len(running)}, waiting {len(todo)}, elapsed {(time.time() - began) / 60:.0f} min"]
        for name, out, _, _, started in running:
            try:
                found = epoch_line.findall((out / "train.log").read_text(errors="replace"))
            except OSError:
                found = []
            ep, f1 = (found[-1][0], found[-1][1]) if found else ("0", "-")
            lines.append(f"    {name:28s} epoch {ep:>2}/{args.epochs}  dev F1 {f1}  {(time.time() - started) / 60:.0f} min")
        if finished_secs:
            mean = sum(finished_secs) / len(finished_secs)
            left = (len(todo) + len(running)) * mean / max(1, args.parallel)
            lines.append(f"    mean run {mean / 60:.0f} min; about {left / 60:.0f} min left")
        print("\n".join(lines), flush=True)

    while todo or running:
        while todo and len(running) < args.parallel:
            start(*todo.pop(0))
        time.sleep(5)
        if time.time() - last_table >= args.progress_every:
            table()
            last_table = time.time()
        for item in list(running):
            name, out, proc, log, started = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            finished_secs.append(time.time() - started)
            print(f"done  {name}: exit {proc.returncode} in {time.time() - started:.0f} s", flush=True)
            if proc.returncode:
                failed.append(name)
    if failed:
        sys.exit(f"failed runs {failed}; see models/v4/<run>/train.log")
    print(f"all {len(names)} runs have report.json")

if __name__ == "__main__":
    main()
