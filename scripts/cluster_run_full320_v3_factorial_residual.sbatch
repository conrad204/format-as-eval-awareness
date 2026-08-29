#!/bin/bash
#SBATCH --job-name=tae_factorial_resid
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_factorial_resid-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_factorial_resid-%j.err
set -euo pipefail

: "${MODEL:?set MODEL to llama31_8b, llama31_70b, or llama33_70b}"
case "$MODEL" in
  llama31_8b|llama31_70b|llama33_70b) ;;
  *) echo "unsupported MODEL=$MODEL" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export TMPDIR=/scratch/jppatton/langea/tmp

ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
OUT="$ROOT/results/full320_v3_factorial_residual_scores_${MODEL}.npz"
SANITY="$ROOT/results/full320_v3_factorial_residual_scores_${MODEL}.sanity.json"
case "$OUT" in /scratch/jppatton/langea/*) ;; *) echo "unsafe output path" >&2; exit 2 ;; esac
if [[ -e "$OUT" || -e "$SANITY" ]]; then
  echo "immutable output already exists" >&2
  exit 2
fi

cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python \
  scripts/full320_v3_factorial_residual_sweep.py \
  --model "$MODEL" --n-jobs "${SLURM_CPUS_PER_TASK:-32}" \
  --out "$OUT" --sanity-out "$SANITY"
