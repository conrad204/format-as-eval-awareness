#!/bin/bash
#SBATCH --job-name=tae_jlens_8b
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_jlens_8b-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_jlens_8b-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN_DIR=${RUN_DIR:?}
SOURCE_DIR=${SOURCE_DIR:-/scratch/jppatton/langea/confirmation_work/tae_2026}
JLENS_DIR=${JLENS_DIR:-/scratch/jppatton/langea/confirmation_work/jlens-581d3986}
MODEL=${MODEL:-/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
TOKENIZER=${TOKENIZER:-/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
MODEL_REVISION=${MODEL_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}
mkdir -p "$RUN_DIR"
RAW_CORPUS=${RAW_CORPUS:-/scratch/jppatton/langea/confirmation_work/jlens_fineweb_1000_seed20260817/raw.jsonl}
# Validate the exact corpus with the exact staged tokenizer before fitting.
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/tokenize_jlens_corpus.py" \
  --raw "$RAW_CORPUS" --out "$RUN_DIR/tokenized.jsonl" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --revision "$MODEL_REVISION" --max-seq-len 128
# The official fit module is imported from the pinned checkout.
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/fit_jlens_8b.py" \
  --jlens-dir "$JLENS_DIR" --model "$MODEL" --tokenizer "$TOKENIZER" --revision "$MODEL_REVISION" \
  --source-layers 4 9 12 15 20 24 --target-layer 31 \
  --dim-batch 2 --max-seq-len 128 --skip-first 16 \
  --input "$RAW_CORPUS" --out-dir "$RUN_DIR" --seed 20260817
