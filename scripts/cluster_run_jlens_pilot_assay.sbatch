#!/bin/bash
#SBATCH --job-name=tae_jlens_assay
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_jlens_assay-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_jlens_assay-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
SOURCE_DIR=${SOURCE_DIR:-/scratch/jppatton/langea/confirmation_work/tae_2026}
JLENS_DIR=${JLENS_DIR:-/scratch/jppatton/langea/confirmation_work/jlens-581d3986}
MODEL=${MODEL:-/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
TOKENIZER=${TOKENIZER:-/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
MODEL_REVISION=${MODEL_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}
LENS=${LENS:?}
ITEMS=${ITEMS:-/scratch/jppatton/langea/confirmation_work/tae_2026/rendered_items_v2_pilot80_8-16-2026.jsonl}
ASSAY_SPEC=${ASSAY_SPEC:?}
OUT=${OUT:?}
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/run_jlens_pilot_assay.py" \
  --jlens-dir "$JLENS_DIR" --lens "$LENS" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --revision "$MODEL_REVISION" --items "$ITEMS" --assay-spec "$ASSAY_SPEC" --out "$OUT"
