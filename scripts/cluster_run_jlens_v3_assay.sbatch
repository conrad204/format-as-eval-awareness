#!/bin/bash
#SBATCH --job-name=tae_jlens_v3_8b
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_jlens_v3_8b-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_jlens_v3_8b-%j.err
set -euo pipefail

export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SOURCE_DIR=${SOURCE_DIR:-/scratch/jppatton/langea/confirmation_work/tae_2026}
JLENS_DIR=${JLENS_DIR:-/scratch/jppatton/langea/confirmation_work/jlens-581d3986}
MODEL=${MODEL:-/scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2}
TOKENIZER=${TOKENIZER:-/scratch/jppatton/langea/confirmation_work/llama31_8b_tokenizer_0e9e39f2}
MODEL_REVISION=${MODEL_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}
LENS=${LENS:?set LENS to a pinned lens checkpoint}
ITEMS=${ITEMS:?set ITEMS to the staged full320 rendered JSONL}
ASSAY_SPEC=${ASSAY_SPEC:?set ASSAY_SPEC to the staged assay spec}
OUT=${OUT:?set OUT to a new immutable result path}

mkdir -p "$(dirname "$OUT")"
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/run_jlens_v3_assay.py" \
  --jlens-dir "$JLENS_DIR" \
  --lens "$LENS" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --revision "$MODEL_REVISION" \
  --items "$ITEMS" \
  --assay-spec "$ASSAY_SPEC" \
  --out "$OUT" \
  --expected-rows 11520 \
  --expected-blocks 320 \
  --expected-canonical-rows 10560 \
  --expected-purpose-rows 7040 \
  --expected-format-rows 3520 \
  --max-seq-len 512
