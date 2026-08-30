#!/bin/bash
#SBATCH --job-name=tae_implv2_70b_all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_implv2_70b_all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_implv2_70b_all-%j.err
# All-layer (80 blocks) implicit_v2 extraction for the 70B checkpoints, the
# all-layer counterpart to cluster_extract_implicit_v2_70b_l18.sbatch. Same
# frozen bank r3, same extraction semantics. Output tensor is
# (3840, 80, 8192) float32 ~= 10.1 GB before npz deflate.
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY to llama31_70b or llama33_70b}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
BANK=$ROOT/results/implicit_v2_frozen_r3
EXPECTED=ba29c2cb55c96664a957aa135801343d4c77409b65d1251ccee5d0626a0b4743
actual=$(sha256sum "$BANK/items.jsonl" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "frozen bank hash mismatch: $actual" >&2; exit 2; }
case "$MODEL_KEY" in
  llama31_70b)
    MODEL=/scratch/jppatton/langea/huggingface/transformers/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b ;;
  llama33_70b)
    MODEL=/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/implicit_v2_${MODEL_KEY}_all_layers
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Refuse to clobber a completed run; retries must use a fresh directory.
if [[ -f "$OUT/activations_all_layers.npz" ]]; then
  echo "completed output already present at $OUT; move it aside to re-extract" >&2
  exit 3
fi
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/extract_activations.py \
  --items results/implicit_v2_frozen_r3/items.jsonl --out-dir "$OUT" \
  --model "$MODEL" --layer 18 --all-layers --batch-size 8 \
  --device-map auto --max-length 2048
