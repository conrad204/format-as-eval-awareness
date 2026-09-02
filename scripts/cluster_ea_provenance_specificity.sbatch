#!/bin/bash
#SBATCH --job-name=tae_easpec
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --constraint=avx2
#SBATCH --output=/scratch/jppatton/langea/logs/tae_easpec-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_easpec-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
TRAIN_ITEMS=$ROOT/v3-activations/rendered_items_v3_full320_8-17-2026.jsonl
PROV_ITEMS=$ROOT/results/provenance_v3_frozen/items.jsonl
PROV_ACT=/scratch/jppatton/langea/results/provenance_v3_${MODEL_KEY}_all_layers/activations_all_layers.npz
case "$MODEL_KEY" in
 llama31_8b) LAYER=8 ;;
 llama31_70b|llama33_70b) LAYER=18 ;;
 *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
TRAIN_ACT=$ROOT/v3-activations/activations/$MODEL_KEY/activations_all_layers.npz
OUT=/scratch/jppatton/langea/results/ea_provenance_specificity_${MODEL_KEY}.json
cd "$ROOT"
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/ea_provenance_direction_specificity.py \
 --model-key "$MODEL_KEY" --layer "$LAYER" \
 --train-items "$TRAIN_ITEMS" --train-activations "$TRAIN_ACT" \
 --provenance-items "$PROV_ITEMS" --provenance-activations "$PROV_ACT" \
 --n-null 100 --n-bootstrap 1000 --out "$OUT"
