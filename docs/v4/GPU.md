# Version 4 — training on a GPU

The 35 pre-registered runs (`docs/v4/preregistration.md`) need a GPU. One epoch of one
run takes 409 s on the 8-core development machine, so 35 runs of 30 epochs need about
120 hours on its CPU. On the A100, 12 runs at once train about one epoch a minute each.

## What to copy to the GPU machine

`data/v4_gpu_kit.tar.gz` (3.3 MB, not in git). It holds `statcheck-ml/src`,
`pipeline/22_train_v4.py`, `run_v4_gpu.sh`, `requirements.txt`, `pyproject.toml`,
`dataset/train_v4.jsonl` and `dataset/splits_v4.json` and `run_v4_gpu.sh` from commit `075f446`.

## Commands on the GPU machine

```
tar xzf v4_gpu_kit.tar.gz
cd statcheck-ml
bash run_v4_gpu.sh smoke          # 2 epochs of one run: checks the GPU, times an epoch
bash run_v4_gpu.sh prereg 12      # the 35 pre-registered runs, 12 at a time
bash run_v4_gpu.sh extended 12    # optional: all 75 runs, exploratory (the 35 are skipped)
bash run_v4_gpu.sh only gate-none,gru-softmax-all-s0 4   # retrain named runs
bash run_v4_gpu.sh pack           # writes v4_models.tar.gz
```

Every scenario checks that PyTorch sees the GPU and stops if it does not. A run that
has `report.json` is skipped, so the same command continues after an interruption. The
number after the scenario is how many runs share the GPU.

## Bring the results back

Copy `v4_models.tar.gz` to `data/v4_models.tar.gz` in this repository on the development
machine. The session unpacks it into `statcheck-ml/models/v4/`, exports every run,
scores the dev papers, and applies the pre-registered rule.
