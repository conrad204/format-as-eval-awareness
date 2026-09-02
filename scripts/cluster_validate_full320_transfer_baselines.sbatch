#!/bin/bash
#SBATCH --job-name=tae_cxfer_gate
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_cxfer_gate-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_cxfer_gate-%j.err
set -euo pipefail

: "${SOURCE_ROOT:?set immutable transfer source snapshot}"
: "${CONFIG:?set frozen transfer config}"
: "${PLAN_JSON:?set immutable transfer plan JSON}"
: "${PLAN_NPZ:?set immutable transfer plan NPZ}"
: "${MODEL_ID:?set frozen model id}"
: "${OUT:?set new shakedown artifact path}"
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/src"
PYTHON=/bighome/jppatton/langea/runtime/vllm/bin/python

if [[ -e "$OUT" ]]; then
  echo "refusing occupied shakedown artifact: $OUT" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
"$PYTHON" "$SOURCE_ROOT/scripts/validate_full320_transfer_baselines.py" \
  --config "$CONFIG" \
  --plan-json "$PLAN_JSON" \
  --plan-npz "$PLAN_NPZ" \
  --model-id "$MODEL_ID" \
  --rows 8 \
  --out "$OUT"
sha256sum "$OUT"
