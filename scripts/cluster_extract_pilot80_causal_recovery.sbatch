#!/bin/bash
#SBATCH --job-name=tae_pilot80_reextract
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_pilot80_reextract-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_pilot80_reextract-%j.err
set -euo pipefail
: "${SOURCE_ROOT:?set SOURCE_ROOT under /scratch/jppatton/langea}"
: "${MODEL_PATH:?set immutable local model snapshot}"
: "${ITEMS_PATH:?set frozen pilot80 meta.jsonl}"
: "${OUT_DIR:?set fresh output directory}"
: "${PRIMARY_LAYER:?set an in-range primary layer}"
for path in "$SOURCE_ROOT" "$MODEL_PATH" "$ITEMS_PATH" "$OUT_DIR"; do
  case "$path" in
    /scratch/jppatton/langea/*) ;;
    *) echo "project path outside /scratch/jppatton/langea: $path" >&2; exit 2 ;;
  esac
done
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing occupied immutable output: $OUT_DIR" >&2
  exit 2
fi
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SOURCE_ROOT/src"
PYTHON=/bighome/jppatton/langea/runtime/vllm/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
"$PYTHON" -m eval_format_mvp.cli extract \
  --items "$ITEMS_PATH" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL_PATH" \
  --layer "$PRIMARY_LAYER" \
  --batch-size 1 \
  --max-length 2048 \
  --device-map auto \
  --all-layers
sha256sum "$ITEMS_PATH" "$OUT_DIR/activations.npz" "$OUT_DIR/activations_all_layers.npz" "$OUT_DIR/meta.jsonl" "$OUT_DIR/config.json"
