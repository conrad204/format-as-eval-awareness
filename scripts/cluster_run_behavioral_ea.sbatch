#!/bin/bash
#SBATCH --job-name=tae_beh_ea
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_beh_ea-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_beh_ea-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
case "$MODEL_KEY" in
  llama31_8b)
    MODEL=/scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2
    BATCH=32
    ;;
  llama31_70b)
    MODEL=/scratch/jppatton/langea/huggingface/transformers/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
    BATCH=8
    ;;
  llama33_70b)
    MODEL=/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
    BATCH=8
    ;;
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
ITEMS=${ITEMS:-results/behavioral_ea_frozen/items.jsonl}
MANIFEST=${MANIFEST:-results/behavioral_ea_frozen/manifest.json}
EXPECTED=${EXPECTED:-c71de20dda7bf0466e95bc9da425026c8f822646969ac9da7e3f1c7ecbdde4d4}
OUT_TAG=${OUT_TAG:-behavioral_ea}
actual=$(sha256sum "$ITEMS" | cut -d' ' -f1)
[[ "$actual" == "$EXPECTED" ]] || { echo "bank hash mismatch $actual" >&2; exit 2; }
EXTRA=()
[[ "${QUANT8:-0}" == 1 ]] && EXTRA+=(--load-in-8bit)
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/run_behavioral_ea.py \
  --model "$MODEL" --model-key "$MODEL_KEY" --items "$ITEMS" --manifest "$MANIFEST" \
  --out "/scratch/jppatton/langea/results/${OUT_TAG}_${MODEL_KEY}.json" --batch-size "$BATCH" "${EXTRA[@]}"
