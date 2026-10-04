#!/bin/bash
# Train the 23 pre-registered version 4 runs on a GPU, then pack the results.
# Usage, inside the unpacked kit:  bash run_v4_gpu.sh [parallel runs, default 4]
# A run that already has report.json is skipped, so the same command resumes.
set -euo pipefail
cd "$(dirname "$0")"
PARALLEL="${1:-4}"
PY="${PY:-python}"

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

$PY pipeline/22_train_v4.py --list | tail -1
$PY pipeline/22_train_v4.py --device cuda --parallel "$PARALLEL" 2>&1 | tee -a train_v4.out

tar czf v4_models.tar.gz models/v4 train_v4.out
echo "Done. Copy v4_models.tar.gz back into ml-statcheck/data/ on the development machine."
