#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Override CONFIG, PROFILE, GPU and extra pipeline CLI options as needed.
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source .venv/bin/activate
python -u -m scripts.run_pipeline --config "${CONFIG:-configs/source_only.yaml}" --profile "${PROFILE:-source_only}" "$@"
