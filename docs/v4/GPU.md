# Training on a GPU

Run these commands in a terminal on the machine with the GPU.

## 1. Get the branch

```
git clone <repository-url> ml-statcheck
cd ml-statcheck
git checkout v4-gold
```

Or copy the whole `ml-statcheck` folder, including `statcheck-ml/dataset/`.

## 2. Install a CUDA build of torch

`requirements.txt` pins `torch==2.14.0+cpu`. Install the same version with a CUDA build.
Read the CUDA version in the first lines of `nvidia-smi`. Replace `cu126` with the matching tag.

```
python -m venv .venv
source .venv/bin/activate
pip install -r statcheck-ml/requirements.txt
pip install --force-reinstall --no-deps torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

## 3. Check that CUDA is visible

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The second value must be `True`. If it is `False`, do not continue.

## 4. Train one configuration

```
cd statcheck-ml
PYTHONPATH=src python -u -m statcheck_ml.train --device cuda \
  --data dataset/train.jsonl --splits dataset/splits.json \
  --augment 2 --hard-negatives 400 --epochs 30 --unit lstm --crf --seed 0 \
  --out models/lstm-crf-s0
```

Add `--no-default-families --families ocr,unicode,font_map` to choose the noise families.
An unknown family name stops the run and prints the valid names.

## 5. Train the grid

```
cd statcheck-ml
PYTHONPATH=src python pipeline/07_train.py --stage screen --device cuda --parallel 1
PYTHONPATH=src python pipeline/07_train.py --stage seeds --device cuda --parallel 1
```

Use `--only lstm-crf,gru-crf` to run a subset. A run with a `report.json` is skipped,
so you can restart a stopped grid.

## 6. Bring the models back

Each run writes `models/<run>/` with `model.pt`, `charmap.json` and `report.json`.
Checkpoints hold CPU tensors, so they load on any machine. `report.json` records the
device, the torch version and the seed.

```
cd statcheck-ml
tar czf runs.tar.gz models/<run> models/runs.json models/<run>.log
```

Copy `runs.tar.gz` to the owner's machine and unpack it in `statcheck-ml/`.
