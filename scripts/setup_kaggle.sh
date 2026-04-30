#!/usr/bin/env bash
# Bootstrap a Kaggle notebook environment for NeuroVal-3D.
set -euo pipefail

echo "=== NeuroVal-3D Kaggle bootstrap ==="

pip install --quiet --upgrade pip
pip install --quiet -e ".[dev,eval,nlp,rag]"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pytest -q

echo "=== Try: python -m neuroval3d.cli benchmark --synthetic --n-samples 80 ==="
