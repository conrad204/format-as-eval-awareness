#!/bin/bash
#SBATCH --job-name=tae_implv2_expbank
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_implv2_expbank-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_implv2_expbank-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
OUT=/scratch/jppatton/langea/results/implicit_v2_explicit_probe_bank_8b_all_layers.npz
rm -f "$OUT" "${OUT%.npz}.manifest.json"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/implicit_v2_freeze_explicit_probes.py \
  --items v3-activations/rendered_items_v3_full320_8-17-2026.jsonl \
  --activations v3-activations/activations/llama31_8b/activations_all_layers.npz \
  --out "$OUT" --jobs 24
