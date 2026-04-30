#!/usr/bin/env bash
# Bootstrap a Colab Pro environment for NeuroVal-3D.
#
# Usage on Colab:
#     !bash scripts/setup_colab.sh
#
set -euo pipefail

echo "=== NeuroVal-3D Colab bootstrap ==="

# Install in editable mode with the most useful optional groups.
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev,eval,nlp,rag]"

python - <<'PY'
import torch, sys
print(f"python: {sys.version.split()[0]}")
print(f"torch:  {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

echo "=== Run smoke tests ==="
pytest -q

echo "=== Done. Try: python -m neuroval3d.cli benchmark --synthetic --n-samples 80 ==="
