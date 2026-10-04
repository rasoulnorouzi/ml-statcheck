#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# The interpreter: the sibling virtual environment by default, or PY=<path>.
if [ -z "${PY:-}" ]; then
  if [ -f ../.venv/bin/python ]; then PY=../.venv/bin/python; else PY=../.venv/Scripts/python.exe; fi
fi
export PYTHONIOENCODING=utf-8

# Parse command-line arguments
TRAIN_MODE=false
FROM_STAGE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --train)
      TRAIN_MODE=true
      shift
      ;;
    --from)
      FROM_STAGE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Define stages
stages=("agree" "dataset" "train" "export" "evaluate" "gold" "compare" "figures" "report")

# Determine starting index
start_idx=0
if [ -n "$FROM_STAGE" ]; then
  for i in "${!stages[@]}"; do
    if [ "${stages[$i]}" == "$FROM_STAGE" ]; then
      start_idx=$i
      break
    fi
  done
fi

# agree stage
if [ $start_idx -le 0 ]; then
  echo "Running: agree stage..."
  echo "$PY pipeline/04_agree.py --annotations dataset/annotations/train --raters haiku sonnet opus --final dataset/annotations/train/final.json --out dataset/agreement/train.json"
  $PY pipeline/04_agree.py --annotations dataset/annotations/train --raters haiku sonnet opus --final dataset/annotations/train/final.json --out dataset/agreement/train.json

  echo "$PY pipeline/04_agree.py --annotations dataset/annotations/holdout --raters haiku sonnet opus --final dataset/annotations/holdout/final.json --out dataset/agreement/holdout.json"
  $PY pipeline/04_agree.py --annotations dataset/annotations/holdout --raters haiku sonnet opus --final dataset/annotations/holdout/final.json --out dataset/agreement/holdout.json
fi

# dataset stage
if [ $start_idx -le 1 ]; then
  echo "Running: dataset stage..."
  echo "$PY pipeline/06_dataset.py --windows dataset/windows/train.json --key dataset/windows/key.json --final dataset/annotations/train/final.json --set train --out dataset/train.jsonl --splits dataset/splits.json --seed 0 --holdout-windows dataset/windows/holdout.json"
  $PY pipeline/06_dataset.py --windows dataset/windows/train.json --key dataset/windows/key.json --final dataset/annotations/train/final.json --set train --out dataset/train.jsonl --splits dataset/splits.json --seed 0 --holdout-windows dataset/windows/holdout.json

  echo "$PY pipeline/06_dataset.py --windows dataset/windows/holdout.json --key dataset/windows/key.json --final dataset/annotations/holdout/final.json --set holdout --out dataset/holdout.jsonl"
  $PY pipeline/06_dataset.py --windows dataset/windows/holdout.json --key dataset/windows/key.json --final dataset/annotations/holdout/final.json --set holdout --out dataset/holdout.jsonl
fi

# train stage
if [ $TRAIN_MODE = true ] && [ $start_idx -le 2 ]; then
  echo "Running: train stage..."
  echo "$PY pipeline/07_train.py --grid pipeline/grid.json --stage all --parallel 3 --threads 4 --models models"
  $PY pipeline/07_train.py --grid pipeline/grid.json --stage all --parallel 3 --threads 4 --models models
fi

# export stage
# --update-spec is the one path that changes the shared spec: it promotes the
# top-1 zoo config's charmap.json, which the three ports read as the known set.
if [ $TRAIN_MODE = true ] && [ $start_idx -le 3 ]; then
  echo "Running: export stage..."
  echo "$PY pipeline/08_export.py --runs models/runs.json --models models --zoo models/zoo --dev dataset/train.jsonl --splits dataset/splits.json --out models/export.json --update-spec"
  $PY pipeline/08_export.py --runs models/runs.json --models models --zoo models/zoo --dev dataset/train.jsonl --splits dataset/splits.json --out models/export.json --update-spec
fi

# evaluate stage
if [ $start_idx -le 4 ]; then
  echo "Running: evaluate stage..."
  echo "$PY pipeline/09_evaluate.py --windows dataset/holdout.jsonl --statcheck dataset/baseline/statcheck_r.csv --repaired dataset/baseline/statcheck_repaired.csv --runs models/runs.json --zoo models/zoo --export models/export.json --out results/eval.json --resamples 2000 --seed 0"
  $PY pipeline/09_evaluate.py --windows dataset/holdout.jsonl --statcheck dataset/baseline/statcheck_r.csv --repaired dataset/baseline/statcheck_repaired.csv --runs models/runs.json --zoo models/zoo --export models/export.json --out results/eval.json --resamples 2000 --seed 0
fi

# gold stage
if [ $start_idx -le 5 ]; then
  echo "Running: gold stage..."
  echo "$PY pipeline/14_gold_split.py --seed 0"
  $PY pipeline/14_gold_split.py --seed 0
  echo "$PY pipeline/14_gold_dev_split.py --seed 0"
  $PY pipeline/14_gold_dev_split.py --seed 0
  if [ -f data/clean/manifest.jsonl ]; then
    echo "$PY pipeline/17_passage_inventory.py data/clean/ --seed 0"
    $PY pipeline/17_passage_inventory.py data/clean/ --seed 0
    echo "$PY pipeline/18_sample_passages.py data/clean/ --seed 0"
    $PY pipeline/18_sample_passages.py data/clean/ --seed 0
    echo "$PY pipeline/14_gold_eval.py --resamples 2000 --seed 0 --label reproduce"
    $PY pipeline/14_gold_eval.py --resamples 2000 --seed 0 --label reproduce
  fi
  # The batches come from the committed sample, so they need no corpus.
  echo "$PY pipeline/02_chunk.py --windows dataset/passages/sample.json --out dataset/passages/batches --size 10 --unpadded --manifest-out dataset/passages/batches/MANIFEST.json --seed 0"
  $PY pipeline/02_chunk.py --windows dataset/passages/sample.json --out dataset/passages/batches --size 10 --unpadded --manifest-out dataset/passages/batches/MANIFEST.json --seed 0
  if [ ! -f data/clean/manifest.jsonl ]; then
    echo "gold evaluation skipped: needs the corpus; the committed results/gold_eval.json stands"
    echo "passage inventory skipped: needs the corpus"
  fi
fi

# compare stage
if [ $start_idx -le 6 ]; then
  echo "Running: compare stage..."
  echo "$PY pipeline/16_compare.py"
  $PY pipeline/16_compare.py
fi

# figures stage
if [ $start_idx -le 7 ]; then
  echo "Running: figures stage..."
  echo "$PY pipeline/10_figures.py --eval results/eval.json --agreement dataset/agreement --runs models/runs.json --export models/export.json --out results/figures"
  $PY pipeline/10_figures.py --eval results/eval.json --agreement dataset/agreement --runs models/runs.json --export models/export.json --out results/figures
fi

# report stage
if [ $start_idx -le 8 ]; then
  echo "Running: report stage..."
  echo "$PY pipeline/11_report.py --template docs/report_template.md --eval results/eval.json --agreement dataset/agreement --runs models/runs.json --export models/export.json --engines results/engines.json --out results/REPORT.md --readme-block results/readme_block.md"
  $PY pipeline/11_report.py --template docs/report_template.md --eval results/eval.json --agreement dataset/agreement --runs models/runs.json --export models/export.json --engines results/engines.json --out results/REPORT.md --readme-block results/readme_block.md
fi

# Verify artifacts
echo "Verifying the dataset manifest (regenerated files must match the committed hashes)..."
$PY -c "import sys; sys.path.insert(0, 'src'); from statcheck_ml.provenance import verify_manifest; bad = verify_manifest('dataset/MANIFEST.json'); print('manifest: ok' if not bad else 'manifest: MISMATCH ' + ', '.join(bad)); sys.exit(1 if bad else 0)"
echo "Verifying artifacts..."
sha256sum results/eval.json results/gold_eval.json results/v4/compare.json dataset/gold/split.json dataset/gold/dev.json dataset/passages/inventory.jsonl dataset/passages/inventory_summary.json results/REPORT.md results/readme_block.md results/figures/*.png
