#!/bin/bash
#SBATCH --job-name=tae_bind_8b_all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind_8b_all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind_8b_all-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
BANK=$ROOT/results/binding_v1_frozen
OUT=/scratch/jppatton/langea/results/binding_v1_8b_all_layers
EXPECTED=bdfd3d4360d1704511f7c3451432c5127b1dbecad75e73654ab4d301f1018efd
actual=$(sha256sum "$BANK/items.jsonl" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "frozen binding_v1 hash mismatch: $actual" >&2; exit 2; }
[[ -f "$BANK/FROZEN.sha256" ]] || { echo "freeze marker missing" >&2; exit 2; }
grep -q '"status": "FROZEN_BEFORE_ACTIVATIONS"' "$BANK/freeze_manifest.json" \
  || { echo "freeze status wrong" >&2; exit 2; }
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/extract_activations.py \
  --items results/binding_v1_frozen/items.jsonl \
  --out-dir "$OUT" \
  --model /scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2 \
  --layer 8 --all-layers --batch-size 8 --max-length 2048
