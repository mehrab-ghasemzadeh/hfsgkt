#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install -e .
python -c "import torch,torchvision; print('torch:',torch.__version__,'vision:',torchvision.__version__,'CUDA:',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is unavailable; check NVIDIA driver and selected torch build'"
python -m unittest discover -s tests -v
printf '\nInstallation complete. Activate with: source .venv/bin/activate\n'
