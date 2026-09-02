#!/bin/bash
#SBATCH --job-name=tae_prov_probe70
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=18:00:00
#SBATCH --constraint=avx2
#SBATCH --output=/scratch/jppatton/langea/logs/tae_prov_probe70-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_prov_probe70-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY to llama31_70b or llama33_70b}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
ITEMS=$ROOT/v3-activations/rendered_items_v3_full320_8-17-2026.jsonl
case "$MODEL_KEY" in
  llama31_70b) ACT=$ROOT/v3-activations/activations/llama31_70b/activations_all_layers.npz ;;
  llama33_70b) ACT=$ROOT/v3-activations/activations/llama33_70b/activations_all_layers.npz ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/provenance_v1_${MODEL_KEY}_explicit_probe_bank_all_layers.npz
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/implicit_v2_freeze_explicit_probes.py \
  --items "$ITEMS" --activations "$ACT" --out "$OUT" --jobs 20 --layer-chunk 16
