#!/bin/bash
#SBATCH --job-name=tae_erasure_layersweep
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_erasure_layersweep-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_erasure_layersweep-%j.err
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export N_JOBS="${SLURM_CPUS_PER_TASK:-28}"
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export TMPDIR=/scratch/jppatton/langea/tmp

/bighome/jppatton/langea/runtime/vllm/bin/python \
  /scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap/scripts/full320_v3_erasure_full_layer_sweep.py \
  /scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap/results/full320_v3_erasure_full_layer_sweep.json
