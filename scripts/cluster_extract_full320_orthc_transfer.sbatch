#!/bin/bash
#SBATCH --job-name=tae_cxfer_extract
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_cxfer_extract-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_cxfer_extract-%j.err
set -euo pipefail

: "${SOURCE_ROOT:?set immutable source snapshot under /scratch/jppatton/langea}"
: "${MODEL_PATH:?set exact immutable local model snapshot}"
: "${MODEL_REVISION:?set exact model revision}"
: "${ITEMS_PATH:?set frozen transfer-bank items.jsonl}"
: "${OUT_DIR:?set new immutable extraction output directory}"
: "${PRIMARY_LAYER:?set compatibility archive layer}"
: "${EXPECTED_ITEMS_SHA256:?set frozen item hash}"

SOURCE_ROOT="$(realpath -e "$SOURCE_ROOT")"
MODEL_PATH="$(realpath -e "$MODEL_PATH")"
ITEMS_PATH="$(realpath -e "$ITEMS_PATH")"
OUT_PARENT="$(realpath -e "$(dirname "$OUT_DIR")")"
OUT_DIR="$OUT_PARENT/$(basename "$OUT_DIR")"
for path in "$SOURCE_ROOT" "$MODEL_PATH" "$ITEMS_PATH" "$OUT_DIR"; do
  case "$path" in
    /scratch/jppatton/langea/*) ;;
    *) echo "path outside immutable scratch boundary: $path" >&2; exit 2 ;;
  esac
done
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing occupied immutable output: $OUT_DIR" >&2
  exit 2
fi
if [[ "$(sha256sum "$ITEMS_PATH" | cut -d' ' -f1)" != "$EXPECTED_ITEMS_SHA256" ]]; then
  echo "item hash mismatch" >&2
  exit 2
fi
mkdir "$OUT_DIR"

export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$SOURCE_ROOT/src"
PYTHON=/bighome/jppatton/langea/runtime/vllm/bin/python

nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
"$PYTHON" "$SOURCE_ROOT/scripts/extract_activations.py" \
  --items "$ITEMS_PATH" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL_PATH" \
  --revision "$MODEL_REVISION" \
  --layer "$PRIMARY_LAYER" \
  --batch-size 1 \
  --max-length 2048 \
  --device-map auto \
  --all-layers
sha256sum "$OUT_DIR"/activations.npz "$OUT_DIR"/activations_all_layers.npz "$OUT_DIR"/meta.jsonl "$OUT_DIR"/config.json
