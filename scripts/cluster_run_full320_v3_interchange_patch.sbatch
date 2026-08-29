#!/bin/bash
#SBATCH --job-name=tae_ixpatch
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_ixpatch-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_ixpatch-%j.err
set -euo pipefail

: "${MODEL:?set model key: llama31_8b | llama31_70b | llama33_70b}"
: "${OUT:?set output jsonl path under /scratch/jppatton/langea}"
RANK="${RANK:-16}"
LAYERS="${LAYERS:-}"
MAX_PAIRS_PER_FOLD="${MAX_PAIRS_PER_FOLD:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RESUME_FLAG="${RESUME_FLAG:-}"

case "$OUT" in
  /scratch/jppatton/langea/*) ;;
  *) echo "refusing output outside /scratch/jppatton/langea: $OUT" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$OUT")"

export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap

ARGS=(--model "$MODEL" --rank "$RANK" --batch-size "$BATCH_SIZE" --out "$OUT")
[[ -n "$LAYERS" ]] && ARGS+=(--layers "$LAYERS")
[[ -n "$MAX_PAIRS_PER_FOLD" ]] && ARGS+=(--max-pairs-per-fold "$MAX_PAIRS_PER_FOLD")
[[ -n "$RESUME_FLAG" ]] && ARGS+=(--resume)

/bighome/jppatton/langea/runtime/vllm/bin/python \
  scripts/full320_v3_interchange_patch_full_layer.py "${ARGS[@]}"
