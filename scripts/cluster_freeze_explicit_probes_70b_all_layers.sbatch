#!/bin/bash
#SBATCH --job-name=tae_bind_bank70all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind_bank70all-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind_bank70all-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
R=/scratch/jppatton/langea/results
case "$MODEL_KEY" in
 llama31_70b) NPZ=$ROOT/v3-activations/activations/llama31_70b/activations_all_layers.npz ;;
 llama33_70b) NPZ=$ROOT/v3-activations/activations/llama33_70b/activations_all_layers.npz ;;
 *) exit 2 ;;
esac
OUT=$R/binding_v1_${MODEL_KEY}_explicit_probe_bank_all_layers.npz
rm -f "$OUT" "${OUT%.npz}.manifest.json"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/implicit_v2_freeze_explicit_probes.py \
  --items v3-activations/rendered_items_v3_full320_8-17-2026.jsonl \
  --activations "$NPZ" --out "$OUT" --jobs 14 --layer-chunk 4
