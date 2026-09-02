#!/bin/bash
#SBATCH --job-name=tae_implv2_70b18ana
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_implv2_70b18ana-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_implv2_70b18ana-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/implicit_v2_analyze_fixed_layer.py \
 --model-key "$MODEL_KEY" --layer 18 --items results/implicit_v2_frozen_r3/items.jsonl \
 --activations "/scratch/jppatton/langea/results/implicit_v2_${MODEL_KEY}_l18/activations.npz" \
 --explicit-bank "/scratch/jppatton/langea/results/implicit_v2_${MODEL_KEY}_explicit_probe_bank_l18.npz" \
 --out "/scratch/jppatton/langea/results/implicit_v2_${MODEL_KEY}_l18_analysis.json" --jobs 12
