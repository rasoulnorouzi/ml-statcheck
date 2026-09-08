#!/usr/bin/env bash
# Train every configuration in the plan, one after another.
#
# Runs are sequential on purpose. Several trainings at once contend for the
# same cores, which does not finish sooner and makes the recorded epoch times
# useless for the cost table.
#
# Each run writes its own log and its own directory, so a failure stops that
# run and not the chain.
set -u

PY="./.venv/Scripts/python.exe"
DATA="statcheck-ml/dataset/*.jsonl"
M="statcheck-ml/models"

run () {           # run <name> <extra args...>
  local name="$1"; shift
  if [ -f "$M/$name/model.pt" ] && [ -f "$M/$name/report.json" ]; then
    echo "SKIP $name (already trained)"
    return
  fi
  echo "START $name  $(date +%H:%M:%S)"
  PYTHONIOENCODING=utf-8 $PY -u -m statcheck_ml.train \
      --data "$DATA" --out "$M/$name" "$@" > "$M/$name.log" 2>&1
  echo "DONE  $name  $(date +%H:%M:%S)  $(grep -m1 '^best epoch' "$M/$name.log" || echo 'no result')"
}

run_bert () {      # run_bert <name> <arch> <extra args...>
  local name="$1"; local arch="$2"; shift 2
  if [ -f "$M/$name/report.json" ]; then
    echo "SKIP $name (already trained)"
    return
  fi
  echo "START $name  $(date +%H:%M:%S)"
  PYTHONIOENCODING=utf-8 $PY -u -m statcheck_ml.train_bert \
      --data "$DATA" --out "$M/$name" --arch "$arch" "$@" > "$M/$name.log" 2>&1
  echo "DONE  $name  $(date +%H:%M:%S)  $(grep -m1 '^best epoch' "$M/$name.log" || echo 'no result')"
}

echo "=== character models ==="
run final-softmax      --epochs 30
run final-crf          --epochs 25 --crf
run final-aug          --epochs 30 --augment 2 --hard-negatives 400
run final-gru          --epochs 30 --unit gru
run final-gru-crf      --epochs 25 --unit gru --crf
run final-crf-aug      --epochs 25 --crf --augment 2 --hard-negatives 400
run final-gru-aug      --epochs 30 --unit gru --augment 2 --hard-negatives 400

echo "=== ablations, on the best character architecture ==="
run abl-hardneg-only   --epochs 25 --crf --hard-negatives 400
run abl-perturb-only   --epochs 25 --crf --augment 2

echo "=== transformers, a research baseline ==="
run_bert final-mobilebert  mobilebert --epochs 4
run_bert final-distilbert  distilbert --epochs 4
run_bert final-distilbert-aug distilbert --epochs 4 --augment 1 --hard-negatives 200

echo "ALL TRAINING COMPLETE $(date +%H:%M:%S)"
