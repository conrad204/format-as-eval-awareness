#!/bin/bash
#SBATCH --job-name=tae_nocue_fmt
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_nocue_fmt-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_nocue_fmt-%j.err
set -euo pipefail
: "${MODEL:?set MODEL}"; : "${RANK:?set RANK}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export N_JOBS=16
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/full320_v3_nocue_format_subspace.py \
 --model "$MODEL" --rank "$RANK" --n-jobs 16 --n-bootstrap 1000 \
 --scores "/scratch/jppatton/langea/results/nocue_format_subspace_scores_${MODEL}_r${RANK}.npz" \
 --out-prefix "/scratch/jppatton/langea/results/nocue_format_subspace_${MODEL}_r${RANK}"
