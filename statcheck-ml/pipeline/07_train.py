"""Run the character-tagger grid: screen, reseed the winners, and ablate.

Three stages, always in this order when `--stage all` is given:

  screen    every unit/decoder combination in `grid.json`, seed 0.
  seeds     the top `top_k` screen configs by dev F1, at seeds 1 and 2.
  ablation  the single best screen config, with augmentation removed, seed 0.

Each run is a subprocess of `statcheck_ml.train`, so one crash does not stop
the grid. A run whose `report.json` already exists is skipped, so a killed
grid resumes where it left off.

Usage:
    python pipeline/07_train.py --grid pipeline/grid.json --stage screen \
        --parallel 3 --threads 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]     # statcheck-ml


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return "unknown"


def override_common(common: list, epochs=None, data=None, splits=None) -> list:
    out = list(common)

    def set_flag(flag, value):
        if value is None:
            return
        for i in range(len(out) - 1):
            if out[i] == flag:
                out[i + 1] = str(value)
                return
        out.extend([flag, str(value)])

    set_flag("--epochs", epochs)
    set_flag("--data", data)
    set_flag("--splits", splits)
    return out


def drop_args(common: list, drop: list) -> list:
    """Remove flag/value pairs named in `drop` from `common`."""
    out = list(common)
    for i in range(0, len(drop) - 1, 2):
        flag, value = drop[i], drop[i + 1]
        for j in range(len(out) - 1):
            if out[j] == flag and out[j + 1] == value:
                del out[j:j + 2]
                break
    return out


def make_run(config: str, config_args: list, seed: int, common: list) -> dict:
    return {"name": f"{config}-s{seed}", "config": config, "seed": seed,
            "args": list(common) + list(config_args) + ["--seed", str(seed)]}


def rank_configs(config_names, models_dir: Path) -> list:
    """Screen configs sorted by dev F1, best first, read from their s0 report."""
    scored = []
    for cfg in config_names:
        report_path = models_dir / f"{cfg}-s0" / "report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scored.append((report["dev"]["overall"]["f1"], cfg))
    if not scored:
        raise SystemExit("no screen report.json files found; run --stage screen first")
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [cfg for _, cfg in scored]


def stage_runs(stage: str, grid: dict, common: list, models_dir: Path, only) -> list:
    if stage == "screen":
        return [make_run(cfg, args, 0, common) for cfg, args in grid["screen"].items()
                if not only or cfg == only]
    if stage == "seeds":
        top = rank_configs(grid["screen"].keys(), models_dir)[:grid["top_k"]]
        runs = []
        for cfg in top:
            if only and cfg != only:
                continue
            for seed in grid["seeds"]:
                runs.append(make_run(cfg, grid["screen"][cfg], seed, common))
        return runs
    if stage == "ablation":
        cfg = rank_configs(grid["screen"].keys(), models_dir)[0]
        if only and cfg != only:
            return []
        new_common = drop_args(common, grid["ablation"]["drop"])
        suffix = grid["ablation"]["suffix"]
        name = f"{cfg}-{suffix}-s0"
        return [{"name": name, "config": f"{cfg}-{suffix}", "seed": 0,
                 "args": new_common + list(grid["screen"][cfg]) + ["--seed", "0"]}]
    raise ValueError(f"unknown stage {stage!r}")


def build_record(run: dict, out_dir: Path, wall_seconds, started_iso, sha, tv) -> dict:
    record = {"name": run["name"], "config": run["config"], "seed": run["seed"],
              "args": run["args"], "torch": tv, "git_sha": sha,
              "started": started_iso, "wall_seconds": wall_seconds,
              "best_epoch": None, "dev_f1": None, "dev_p": None, "dev_r": None,
              "params": None, "status": "failed"}
    report_path = out_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        overall = report.get("dev", {}).get("overall", {})
        record.update(best_epoch=report.get("best_epoch"), dev_f1=overall.get("f1"),
                      dev_p=overall.get("precision"), dev_r=overall.get("recall"),
                      params=report.get("parameters"), status="done")
    return record


def run_one(run: dict, models_dir: Path, threads: int, py: str, sha: str) -> dict:
    name = run["name"]
    out_dir = models_dir / name
    cmd = [py, "-u", "-m", "statcheck_ml.train"] + run["args"] + ["--out", str(out_dir)]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["PYTHONIOENCODING"] = "utf-8"
    started_iso = datetime.now(timezone.utc).isoformat()
    started = time.time()
    with open(models_dir / f"{name}.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = round(time.time() - started, 1)
    return build_record(run, out_dir, wall, started_iso, sha, torch_version())


def write_records(path: Path, records: dict) -> None:
    ordered = [records[name] for name in sorted(records)]
    path.write_text(json.dumps(ordered, indent=1), encoding="utf-8")


def print_table(records: dict) -> None:
    print(f"\n{'name':28s} {'status':8s} {'dev_f1':>7s} {'epoch':>6s} {'wall_s':>8s} {'params':>10s}")
    for name in sorted(records):
        r = records[name]
        f1 = f"{r['dev_f1']:.3f}" if r["dev_f1"] is not None else "-"
        params = f"{r['params']:,}" if r["params"] is not None else "-"
        wall = f"{r['wall_seconds']:.0f}" if r["wall_seconds"] is not None else "-"
        print(f"{name:28s} {r['status']:8s} {f1:>7s} {str(r['best_epoch']):>6s} "
              f"{wall:>8s} {params:>10s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="pipeline/grid.json")
    ap.add_argument("--stage", choices=["screen", "seeds", "ablation", "all"], required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--models", default="models")
    ap.add_argument("--only", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override, for smoke tests")
    ap.add_argument("--data", default=None, help="override, for smoke tests")
    ap.add_argument("--splits", default=None, help="override, for smoke tests")
    args = ap.parse_args()

    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    common = override_common(grid["common"], args.epochs, args.data, args.splits)

    models_dir = Path(args.models)
    if not models_dir.is_absolute():
        models_dir = REPO / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    runs_path = models_dir / "runs.json"
    records = {}
    if runs_path.exists():
        for r in json.loads(runs_path.read_text(encoding="utf-8")):
            records[r["name"]] = r

    sha = git_sha()
    py = sys.executable
    stages = ["screen", "seeds", "ablation"] if args.stage == "all" else [args.stage]
    for stage in stages:
        runs = stage_runs(stage, grid, common, models_dir, args.only)
        if not runs:
            print(f"{stage}: nothing to run")
            continue

        to_run, to_skip = [], []
        for run in runs:
            report_path = models_dir / run["name"] / "report.json"
            (to_skip if report_path.exists() else to_run).append(run)

        for run in to_skip:
            print(f"skip {run['name']}: report.json exists")
            existing = records.get(run["name"])
            records[run["name"]] = existing or build_record(
                run, models_dir / run["name"], None, None, sha, torch_version())
            write_records(runs_path, records)

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(run_one, run, models_dir, args.threads, py, sha): run
                      for run in to_run}
            for future in as_completed(futures):
                record = future.result()
                print(f"done {record['name']}: {record['status']}, "
                      f"dev F1 {record['dev_f1']}, {record['wall_seconds']}s")
                records[record["name"]] = record
                write_records(runs_path, records)

    print_table(records)


if __name__ == "__main__":
    main()
