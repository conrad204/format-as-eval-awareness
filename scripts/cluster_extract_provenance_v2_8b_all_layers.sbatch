#!/bin/bash
#SBATCH --job-name=tae_prov2_8b_all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_prov2_8b_all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_prov2_8b_all-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
BANK=$ROOT/results/provenance_v2_frozen
OUT=/scratch/jppatton/langea/results/provenance_v2_8b_all_layers
EXPECTED=7b34fa5ef4cfc80150b9d9fc6d78854344541b1609d1b137fa58d15651a1036f
actual=$(sha256sum "$BANK/items.jsonl" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "frozen provenance_v2 hash mismatch: $actual" >&2; exit 2; }
grep -q '"status": "FROZEN_BEFORE_ACTIVATIONS"' "$BANK/freeze_manifest.json" || exit 2
if [[ -f "$OUT/activations_all_layers.npz" ]]; then echo "output exists" >&2; exit 3; fi
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/extract_activations.py \
  --items results/provenance_v2_frozen/items.jsonl --out-dir "$OUT" \
  --model /scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2 \
  --layer 14 --all-layers --batch-size 8 --max-length 2048
