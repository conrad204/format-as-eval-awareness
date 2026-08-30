#!/bin/bash
#SBATCH --job-name=tae_implv2_nocue
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_implv2_nocue-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_implv2_nocue-%j.err
set -euo pipefail
: "${RANK:?set RANK}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/implicit_v2_nocue_subspace_control.py \
 --rank "$RANK" --original-items v3-activations/rendered_items_v3_full320_8-17-2026.jsonl \
 --original-activations v3-activations/activations/llama31_8b/activations_all_layers.npz \
 --implicit-items results/implicit_v2_frozen_r3/items.jsonl \
 --implicit-activations /scratch/jppatton/langea/results/implicit_v2_8b_all_layers/activations_all_layers.npz \
 --raw-analysis /scratch/jppatton/langea/results/implicit_v2_8b_analysis.json \
 --out "/scratch/jppatton/langea/results/implicit_v2_nocue_control_r${RANK}.json" --jobs 12
