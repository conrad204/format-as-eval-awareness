#!/bin/bash
#SBATCH --job-name=tae_bind_ana70all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind_ana70all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind_ana70all-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
R=/scratch/jppatton/langea/results
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/binding_v1_analyze.py \
  --model-key "$MODEL_KEY" --items results/binding_v1_frozen/items.jsonl \
  --activations $R/binding_v1_${MODEL_KEY}_all_layers/activations_all_layers.npz \
  --explicit-bank $R/binding_v1_${MODEL_KEY}_explicit_probe_bank_all_layers.npz \
  --out $R/binding_v1_${MODEL_KEY}_all_layers_analysis.json --jobs 14
