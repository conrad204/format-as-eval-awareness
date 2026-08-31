#!/bin/bash
#SBATCH --job-name=tae_prov3_all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
# Default sized for the 70B checkpoints; override at submit time for the 8B:
#   sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL_KEY=llama31_8b ...
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_prov3_all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_prov3_all-%j.err
# provenance_v3 all-layer extraction. MODEL_KEY selects the checkpoint.
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY to llama31_8b, llama31_70b or llama33_70b}"
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
BANK=$ROOT/results/provenance_v3_frozen
EXPECTED=2a574bf1bf0cd5cc8ec53c97e2bf69e9246316b1a3d5364e370acc0f8fcddc7e
actual=$(sha256sum "$BANK/items.jsonl" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "frozen provenance_v3 hash mismatch: $actual" >&2; exit 2; }
grep -q '"status": "FROZEN_BEFORE_ACTIVATIONS"' "$BANK/freeze_manifest.json" || exit 2
case "$MODEL_KEY" in
  llama31_8b)
    MODEL=/scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2
    LAYER=14; EXTRA="" ;;
  llama31_70b)
    MODEL=/scratch/jppatton/langea/huggingface/transformers/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
    LAYER=18; EXTRA="--device-map auto" ;;
  llama33_70b)
    MODEL=/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
    LAYER=18; EXTRA="--device-map auto" ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/provenance_v3_${MODEL_KEY}_all_layers
if [[ -f "$OUT/activations_all_layers.npz" ]]; then
  echo "completed output already present at $OUT" >&2; exit 3
fi
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/extract_activations.py \
  --items results/provenance_v3_frozen/items.jsonl --out-dir "$OUT" \
  --model "$MODEL" --layer "$LAYER" --all-layers --batch-size 8 --max-length 2048 $EXTRA
