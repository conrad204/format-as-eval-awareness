#!/bin/bash
#SBATCH --job-name=tae_hc_ea
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_hc_ea-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_hc_ea-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
case "$MODEL_KEY" in
  llama31_8b)
    MODEL=/scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2; BATCH=16 ;;
  llama31_70b)
    MODEL=/scratch/jppatton/langea/huggingface/transformers/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b; BATCH=8 ;;
  llama33_70b)
    MODEL=/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b; BATCH=8 ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"
ITEMS=${ITEMS:-results/behavioral_ea_hidden_channel_frozen/items.jsonl}
MANIFEST=${MANIFEST:-results/behavioral_ea_hidden_channel_frozen/manifest.json}
EXPECTED=${EXPECTED:-0bcd4b48cb95e56ac09c74ae3fe5eaa9739d0b97ae96ebc8f843d5985740b32c}
OUT_TAG=${OUT_TAG:-behavioral_ea_hidden_channel}
actual=$(sha256sum "$ITEMS" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "bank hash mismatch $actual" >&2; exit 2; }
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/run_behavioral_ea_hidden_channel.py \
  --model "$MODEL" --model-key "$MODEL_KEY" --items "$ITEMS" --manifest "$MANIFEST" \
  --out "/scratch/jppatton/langea/results/${OUT_TAG}_${MODEL_KEY}.json" \
  --batch-size "$BATCH"
