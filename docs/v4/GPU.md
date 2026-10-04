# Version 4 — training on a GPU

The 23 pre-registered runs (`docs/v4/preregistration.md`) need a GPU. One epoch of one
run takes 409 s on the 8-core development machine, so 23 runs of 30 epochs need about
78 hours on its CPU.

## What to copy to the GPU machine

`data/v4_gpu_kit.tar.gz` (3.3 MB, not in git). It holds `statcheck-ml/src`,
`pipeline/22_train_v4.py`, `requirements.txt`, `pyproject.toml`,
`dataset/train_v4.jsonl` and `dataset/splits_v4.json` from commit `e0d289f`.

## Commands on the GPU machine

1. Unpack the kit and go into it.

   ```
   tar xzf v4_gpu_kit.tar.gz
   cd statcheck-ml
   ```

2. Make sure that PyTorch sees the GPU. The second value must be `True`.

   ```
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   If PyTorch is missing, install the CUDA build that matches `nvidia-smi`, for example
   `pip install torch --index-url https://download.pytorch.org/whl/cu126`. Then install
   `pip install numpy scipy`.

3. Train the 23 runs. Four runs share the GPU. A run that has `report.json` is skipped,
   so the same command continues after an interruption.

   ```
   python pipeline/22_train_v4.py --device cuda --parallel 4 2>&1 | tee train_v4.out
   ```

4. Pack the results.

   ```
   tar czf v4_models.tar.gz models/v4 train_v4.out
   ```

## Bring the results back

Copy `v4_models.tar.gz` to `data/v4_models.tar.gz` in this repository on the development
machine. The session unpacks it into `statcheck-ml/models/v4/`, exports every run,
scores the dev papers, and applies the pre-registered rule.
