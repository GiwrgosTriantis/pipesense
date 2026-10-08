#!/bin/bash
# Leak pipeline — the job watcher on the analytics VM launches this for every requested job.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source ~/.bashrc  # the VM relies on it for conda activation; sourced before strict mode
set -euo pipefail  # stop at the first failed stage instead of running later stages on stale files
cd "$SCRIPT_DIR"

export DATA_DIR=./data/raw/fairport
export WORK_DIR=./output

echo "[run_all] starting pipeline at $(date)"

python pipeline/prepare_data.py
python pipeline/train_model.py
python pipeline/predict.py

echo "[run_all] pipeline completed at $(date)"
