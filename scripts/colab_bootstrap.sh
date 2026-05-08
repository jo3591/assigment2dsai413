#!/usr/bin/env bash
# One-shot Colab setup. Mount Drive *before* running this so persistent storage works.
#   from google.colab import drive; drive.mount('/content/drive')
#   !bash scripts/colab_bootstrap.sh

set -euo pipefail

echo "==> Installing pinned deps"
pip install -q -r requirements-colab.txt

echo "==> Logging in to Hugging Face (needed for gated MedGemma)"
python - <<'PY'
import os
from huggingface_hub import login
tok = os.getenv("HF_TOKEN")
if tok:
    login(tok)
    print("HF login OK")
else:
    print("WARN: HF_TOKEN not set — MedGemma gated models will fail to load")
PY

echo "==> Setting up Kaggle credentials"
mkdir -p ~/.kaggle
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  echo "Kaggle credentials written"
else
  echo "WARN: KAGGLE_USERNAME / KAGGLE_KEY not set — uploads via kaggle.json required"
fi

echo "==> Installing this package in editable mode"
pip install -q -e .

echo "==> Bootstrap complete"
