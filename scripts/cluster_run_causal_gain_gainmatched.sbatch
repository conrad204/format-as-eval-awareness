#!/bin/bash
#SBATCH --job-name=tae_causal_gain
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_causal_gain_gainmatched-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_causal_gain_gainmatched-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/bighome/jppatton/langea/runtime/vllm/bin/python /scratch/jppatton/langea/confirmation_work/tae_2026/run_causal_gain_intervention.py --model /scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2 --tokenizer /scratch/jppatton/langea/confirmation_work/llama31_8b_tokenizer_0e9e39f2 --revision 0e9e39f249a16976918f6564b8830bc894c89659 --items /scratch/jppatton/langea/confirmation_work/tae_2026/rendered_items_v2_pilot80_8-16-2026.jsonl --vectors /scratch/jppatton/langea/confirmation_work/tae_2026/causal_directions_100.npz --lens /scratch/jppatton/langea/runs/jlens_100_20260817_01/lens.pt --out /scratch/jppatton/langea/runs/causal_gain_100_20260818_02/results.json --max-items 20 --normalization equal_predicted_gain
