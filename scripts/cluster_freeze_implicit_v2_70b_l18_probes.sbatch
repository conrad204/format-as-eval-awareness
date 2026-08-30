#!/bin/bash
#SBATCH --job-name=tae_implv2_70b18bank
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_implv2_70b18bank-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_implv2_70b18bank-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
case "$MODEL_KEY" in
 llama31_70b) NPZ=$ROOT/v3-activations/activations/llama31_70b/activations_all_layers.npz ;;
 llama33_70b) NPZ=$ROOT/v3-activations/activations/llama33_70b/activations_all_layers.npz ;;
 *) exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/implicit_v2_${MODEL_KEY}_explicit_probe_bank_l18.npz
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
rm -f "$OUT" "${OUT%.npz}.manifest.json"
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/implicit_v2_freeze_explicit_probes_fixed_layer.py \
 --model-key "$MODEL_KEY" --items v3-activations/rendered_items_v3_full320_8-17-2026.jsonl \
 --activations "$NPZ" --layer 18 --out "$OUT" --jobs 15
