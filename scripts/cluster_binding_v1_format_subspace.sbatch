#!/bin/bash
#SBATCH --job-name=tae_bind_fmt
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind_fmt-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind_fmt-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
R=/scratch/jppatton/langea/results
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/binding_v1_format_subspace.py \
  --model-key llama31_8b --items results/binding_v1_frozen/items.jsonl \
  --activations $R/binding_v1_8b_all_layers/activations_all_layers.npz \
  --raw-analysis $R/binding_v1_llama31_8b_analysis.json \
  --layers 4,8,12,16,20,24,28,31 --ranks 1,4,16,64 \
  --out $R/binding_v1_llama31_8b_format_subspace.json --jobs 8
