#!/bin/bash
#SBATCH --job-name=tae_full320_equaln
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_full320_equaln-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_full320_equaln-%j.err
set -euo pipefail

: "${ITEMS:?absolute rendered full320 JSONL path required}"
: "${ACTIVATIONS:?absolute all-layer NPZ path required}"
: "${OUT:?absolute output JSON path required}"
: "${MODEL:?model identifier required}"
: "${MODEL_REVISION:?immutable model revision required}"
: "${LAYERS:?space-separated layer list required}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp

/bighome/jppatton/langea/runtime/vllm/bin/python /scratch/jppatton/langea/deploy/tae_2026/full320_equal_n.py \
  --items "$ITEMS" \
  --activations "$ACTIVATIONS" \
  --out "$OUT" \
  --layers $LAYERS \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --seeds 20260822 20260823 20260824 20260825 \
  --bootstrap 2000
