#!/usr/bin/env bash
# Analyses added after the main pipeline was already running.
# Fold these into run_pipeline.sh once it is idle. Same checkpoint/skip rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe
S=ai_post_analysis/src
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(llama31_8b llama31_70b llama33_70b)

for M in "${MODELS[@]}"; do
  if [ -f "ai_post_analysis/results/${M}_scores.npz" ]; then
    echo "=== $M within-template contract contrast ==="
    $PY $S/contract_analysis.py --model "$M" --n-boot 2000
  else
    echo "=== $M skipped: scores not exported yet ==="
  fi
done

echo "=== cross-model agreement ==="
$PY $S/cross_model.py
