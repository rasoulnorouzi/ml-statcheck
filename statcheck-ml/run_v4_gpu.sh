#!/bin/bash
# Version 4 training on a GPU. Run it inside the unpacked kit.
#
#   bash run_v4_gpu.sh smoke                 2 epochs of one run: checks the GPU and times an epoch
#   bash run_v4_gpu.sh prereg   [PARALLEL]   the 35 pre-registered runs; the selection reads only these
#   bash run_v4_gpu.sh extended [PARALLEL]   all 75 runs (the 35 included): every family on every
#                                            configuration and seed; exploratory, reported only
#   bash run_v4_gpu.sh only NAME[,NAME] [PARALLEL]   train or retrain the named runs
#   bash run_v4_gpu.sh pack                  pack models/v4 and the logs into v4_models.tar.gz
#
# PARALLEL is how many runs share the GPU at once (default 12; an A100 holds more). A run that already has
# report.json and the same settings is skipped, so any scenario continues after an
# interruption; a run trained with other settings trains again. The 35 runs of
# prereg are part of extended, so prereg first and extended later trains nothing twice.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
SCENARIO="${1:-}"

check_gpu() {
  $PY - <<'EOF'
import sys
try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed. Install a CUDA build that matches nvidia-smi, "
             "for example: pip install torch --index-url https://download.pytorch.org/whl/cu126")
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("PyTorch does not see a GPU; stop here rather than train on the CPU.")
print("GPU:", torch.cuda.get_device_name(0))
EOF
  $PY -c "import numpy, scipy" 2>/dev/null || $PY -m pip install -q numpy scipy
}

train() {  # train <log> <args...>
  local log="$1"; shift
  # One CPU thread per run: parallel runs otherwise fight over the cores for data work.
  $PY pipeline/22_train_v4.py --device cuda --threads 1 "$@" 2>&1 | tee -a "$log"
}

case "$SCENARIO" in
  smoke)
    check_gpu
    started=$(date +%s)
    train train_smoke.out --only gate-old+all --epochs 2 --models models/v4-smoke --progress-every 20
    echo "2 epochs took $(( $(date +%s) - started )) s. The smoke run is in models/v4-smoke, apart from models/v4."
    ;;
  prereg)
    check_gpu
    train train_v4.out --plan prereg --parallel "${2:-12}"
    ;;
  extended)
    check_gpu
    train train_v4_extended.out --plan extended --parallel "${2:-12}"
    ;;
  only)
    [ -n "${2:-}" ] || { echo "usage: bash run_v4_gpu.sh only NAME[,NAME] [PARALLEL]"; exit 2; }
    check_gpu
    train train_v4_only.out --plan extended --only "$2" --parallel "${3:-4}"
    ;;
  pack)
    tar czf v4_models.tar.gz models/v4 train_v4*.out 2>/dev/null || tar czf v4_models.tar.gz models/v4
    ls -la v4_models.tar.gz
    echo "Copy v4_models.tar.gz into ml-statcheck/data/ on the development machine."
    ;;
  *)
    sed -n '2,15p' "$0"
    exit 2
    ;;
esac
