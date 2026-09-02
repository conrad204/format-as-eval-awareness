#!/bin/bash
#SBATCH --job-name=tae_gap_llama33_70b
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_gap_llama33_70b-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_gap_llama33_70b-%j.err
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export N_JOBS="${SLURM_CPUS_PER_TASK:-28}"
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export TMPDIR=/scratch/jppatton/langea/tmp

cd /scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/full_layer_gap_erasure_sweep.py --model llama33_70b --out results/full_layer_gap_erasure_scores_llama33_70b.npz
