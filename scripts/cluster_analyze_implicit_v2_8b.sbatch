#!/bin/bash
#SBATCH --job-name=tae_implv2_analysis
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_implv2_analysis-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_implv2_analysis-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/implicit_v2_analyze.py \
  --items results/implicit_v2_frozen_r3/items.jsonl \
  --activations /scratch/jppatton/langea/results/implicit_v2_8b_all_layers/activations_all_layers.npz \
  --explicit-bank /scratch/jppatton/langea/results/implicit_v2_explicit_probe_bank_8b_all_layers.npz \
  --out /scratch/jppatton/langea/results/implicit_v2_8b_analysis.json \
  --jobs 12
