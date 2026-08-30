#!/bin/bash
#SBATCH --job-name=tae_implicit8
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_implicit8-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_implicit8-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
OUT=/scratch/jppatton/langea/results/implicit_purpose_fixed_probe_8b_l8
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/extract_activations.py \
  --items results/implicit_purpose_fixed_probe/items.jsonl \
  --out-dir "$OUT" \
  --model /scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2 \
  --layer 8 --batch-size 16 --max-length 2048
