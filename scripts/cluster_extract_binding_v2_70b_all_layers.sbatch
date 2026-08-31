#!/bin/bash
#SBATCH --job-name=tae_bind2_70b_all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind2_70b_all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind2_70b_all-%j.err
# binding_v2 all-layer extraction for the 70B checkpoints.
# 7680 items x 80 layers x 8192 float32 ~= 20.1 GB in RAM before npz deflate,
# hence --mem=220G (the 4096-item binding_v1 run peaked at 31 GB RSS for a
# 10.7 GB tensor).
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY to llama31_70b or llama33_70b}"
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
BANK=$ROOT/results/binding_v2_frozen
EXPECTED=d7ad085991345750384f8877a380e6ca5c23e93dccffa1cf0712137063824cc2
actual=$(sha256sum "$BANK/items.jsonl" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "frozen binding_v2 hash mismatch: $actual" >&2; exit 2; }
grep -q '"status": "FROZEN_BEFORE_ACTIVATIONS"' "$BANK/freeze_manifest.json" \
  || { echo "freeze status wrong" >&2; exit 2; }
case "$MODEL_KEY" in
  llama31_70b)
    MODEL=/scratch/jppatton/langea/huggingface/transformers/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b ;;
  llama33_70b)
    MODEL=/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/binding_v2_${MODEL_KEY}_all_layers
if [[ -f "$OUT/activations_all_layers.npz" ]]; then
  echo "completed output already present at $OUT; move it aside to re-extract" >&2
  exit 3
fi
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/extract_activations.py \
  --items results/binding_v2_frozen/items.jsonl --out-dir "$OUT" \
  --model "$MODEL" --layer 18 --all-layers --batch-size 8 \
  --device-map auto --max-length 2048
