#!/bin/bash
#SBATCH --job-name=tae_bind2_8b_all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind2_8b_all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind2_8b_all-%j.err
# binding_v2 all-layer extraction, Llama-3.1-8B. 7680 items x 32 layers x 4096.
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
BANK=$ROOT/results/binding_v2_frozen
OUT=/scratch/jppatton/langea/results/binding_v2_8b_all_layers
EXPECTED=d7ad085991345750384f8877a380e6ca5c23e93dccffa1cf0712137063824cc2
actual=$(sha256sum "$BANK/items.jsonl" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "frozen binding_v2 hash mismatch: $actual" >&2; exit 2; }
grep -q '"status": "FROZEN_BEFORE_ACTIVATIONS"' "$BANK/freeze_manifest.json" \
  || { echo "freeze status wrong" >&2; exit 2; }
if [[ -f "$OUT/activations_all_layers.npz" ]]; then
  echo "completed output already present at $OUT; move it aside to re-extract" >&2
  exit 3
fi
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/extract_activations.py \
  --items results/binding_v2_frozen/items.jsonl --out-dir "$OUT" \
  --model /scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2 \
  --layer 14 --all-layers --batch-size 8 --max-length 2048
