#!/bin/bash
#SBATCH --job-name=tae_confirm_8b
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_confirm_8b-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_confirm_8b-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/scratch/jppatton/langea/confirmation_work/tae_2026
work=/scratch/jppatton/langea/confirmation_work
bank=$work/confirmation_candidate_20260816
if [ ! -f "$bank/sealed_manifest.json" ]; then echo 'sealed manifest missing' >&2; exit 2; fi
/bighome/jppatton/langea/runtime/vllm/bin/python /scratch/jppatton/langea/confirmation_work/tae_2026/extract_confirmatory.py \
  --model-key llama31_8b --items "$bank/rendered_items.jsonl" \
  --output "$work/activations_confirm_8b" --batch-size 2 --max-length 2048
