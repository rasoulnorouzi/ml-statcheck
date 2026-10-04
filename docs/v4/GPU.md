# Version 4 — training on a GPU

The 23 pre-registered runs (`docs/v4/preregistration.md`) need a GPU. One epoch of one
run takes 409 s on the 8-core development machine, so 23 runs of 30 epochs need about
78 hours on its CPU.

## What to copy to the GPU machine

`data/v4_gpu_kit.tar.gz` (3.3 MB, not in git). It holds `statcheck-ml/src`,
`pipeline/22_train_v4.py`, `run_v4_gpu.sh`, `requirements.txt`, `pyproject.toml`,
`dataset/train_v4.jsonl` and `dataset/splits_v4.json` and `run_v4_gpu.sh` from commit `1b213cf`.

## Commands on the GPU machine

```
tar xzf v4_gpu_kit.tar.gz
cd statcheck-ml
bash run_v4_gpu.sh 4
```

`run_v4_gpu.sh` checks that PyTorch sees the GPU and stops if it does not, installs
numpy and scipy if they are missing, trains the 23 runs with 4 at a time on the GPU,
and writes `v4_models.tar.gz`. The number after the script is how many runs share
the GPU. If the session stops, run the same command again: finished runs are skipped.

## Bring the results back

Copy `v4_models.tar.gz` to `data/v4_models.tar.gz` in this repository on the development
machine. The session unpacks it into `statcheck-ml/models/v4/`, exports every run,
scores the dev papers, and applies the pre-registered rule.
