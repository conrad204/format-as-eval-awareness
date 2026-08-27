#!/bin/bash
#SBATCH --job-name=tae_cxfer_31_70b
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_cxfer-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_cxfer-%j.err
set -euo pipefail

: "${SOURCE_ROOT:?set immutable transfer source snapshot}"
: "${CONFIG:?set frozen transfer config}"
: "${PLAN_JSON:?set immutable transfer plan JSON}"
: "${PLAN_NPZ:?set immutable transfer plan NPZ}"
: "${MODEL_ID:?set frozen model id}"
: "${OUT_DIR:?set immutable transfer output directory}"

SOURCE_ROOT="$(realpath -e "$SOURCE_ROOT")"
CONFIG="$(realpath -e "$CONFIG")"
PLAN_JSON="$(realpath -e "$PLAN_JSON")"
PLAN_NPZ="$(realpath -e "$PLAN_NPZ")"
OUT_PARENT="$(realpath -e "$(dirname "$OUT_DIR")")"
OUT_DIR="$OUT_PARENT/$(basename "$OUT_DIR")"
for path in "$SOURCE_ROOT" "$CONFIG" "$PLAN_JSON" "$PLAN_NPZ" "$OUT_DIR"; do
  case "$path" in
    /scratch/jppatton/langea/*) ;;
    *) echo "path outside immutable scratch boundary: $path" >&2; exit 2 ;;
  esac
done
if [[ -e "$OUT_DIR/runner_manifest.json" || -e "$OUT_DIR/INVALID.json" ]]; then
  echo "refusing completed or invalid immutable output: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

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

nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
RESUME=()
if [[ -s "$OUT_DIR/causal_raw.jsonl" ]]; then
  RESUME=(--resume)
fi
"$PYTHON" "$SOURCE_ROOT/scripts/run_full320_orthc_model_extension.py" \
  --config "$CONFIG" \
  --plan-json "$PLAN_JSON" \
  --plan-npz "$PLAN_NPZ" \
  --model-id "$MODEL_ID" \
  --out-dir "$OUT_DIR" \
  --batch-size 1 \
  --baseline-batch-size 1 \
  "${RESUME[@]}"
sha256sum "$OUT_DIR/causal_raw.jsonl" "$OUT_DIR/runner_manifest.json"
