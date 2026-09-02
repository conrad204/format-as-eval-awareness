#!/bin/bash
#SBATCH --job-name=tae_prov2_qnc
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=18:00:00
#SBATCH --constraint=avx2
#SBATCH --output=/scratch/jppatton/langea/logs/tae_prov2_qnc-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_prov2_qnc-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
REF_ITEMS=$ROOT/v3-activations/rendered_items_v3_full320_8-17-2026.jsonl
PROV_ITEMS=$ROOT/results/provenance_v2_frozen/items.jsonl
case "$MODEL_KEY" in
 llama31_8b)
  REF_ACT=$ROOT/v3-activations/activations/llama31_8b/activations_all_layers.npz
  PROV_ACT=/scratch/jppatton/langea/results/provenance_v2_8b_all_layers/activations_all_layers.npz
  BANK=$ROOT/results/implicit_v2_explicit_probe_bank_8b_all_layers.npz
  LAYERS=14,31 ;;
 llama31_70b)
  REF_ACT=$ROOT/v3-activations/activations/llama31_70b/activations_all_layers.npz
  PROV_ACT=/scratch/jppatton/langea/results/provenance_v2_llama31_70b_all_layers/activations_all_layers.npz
  BANK=/scratch/jppatton/langea/results/provenance_v1_llama31_70b_explicit_probe_bank_all_layers.npz
  LAYERS=18 ;;
 llama33_70b)
  REF_ACT=$ROOT/v3-activations/activations/llama33_70b/activations_all_layers.npz
  PROV_ACT=/scratch/jppatton/langea/results/provenance_v2_llama33_70b_all_layers/activations_all_layers.npz
  BANK=/scratch/jppatton/langea/results/provenance_v1_llama33_70b_explicit_probe_bank_all_layers.npz
  LAYERS=18 ;;
 *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/provenance_v2_${MODEL_KEY}_qnocue_r64.json
cd "$ROOT"
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28 OPENBLAS_NUM_THREADS=28
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/provenance_v2_format_projection.py \
 --model-key "$MODEL_KEY" --reference-items "$REF_ITEMS" --reference-activations "$REF_ACT" \
 --provenance-items "$PROV_ITEMS" --provenance-activations "$PROV_ACT" \
 --probe-bank "$BANK" --layers "$LAYERS" --rank 64 --n-bootstrap 1000 --out "$OUT"
