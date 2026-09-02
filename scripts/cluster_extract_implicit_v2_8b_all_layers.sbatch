#!/bin/bash
#SBATCH --job-name=tae_implv2_all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_implv2_all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_implv2_all-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
BANK=$ROOT/results/implicit_v2_frozen_r3
OUT=/scratch/jppatton/langea/results/implicit_v2_8b_all_layers
EXPECTED=ba29c2cb55c96664a957aa135801343d4c77409b65d1251ccee5d0626a0b4743
actual=$(sha256sum "$BANK/items.jsonl" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "frozen bank hash mismatch: $actual" >&2; exit 2; }
[[ -f "$BANK/FROZEN.sha256" ]] || { echo "freeze marker missing" >&2; exit 2; }
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/extract_activations.py \
  --items results/implicit_v2_frozen_r3/items.jsonl \
  --out-dir "$OUT" \
  --model /scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2 \
  --layer 8 --all-layers --batch-size 8 --max-length 2048
