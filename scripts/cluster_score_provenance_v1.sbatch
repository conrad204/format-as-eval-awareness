#!/bin/bash
#SBATCH --job-name=tae_prov1_score
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --constraint=avx2
#SBATCH --output=/scratch/jppatton/langea/logs/tae_prov1_score-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_prov1_score-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
ITEMS=$ROOT/results/provenance_v1_frozen/items.jsonl
case "$MODEL_KEY" in
  llama31_8b)
    ACT=/scratch/jppatton/langea/results/provenance_v1_8b_all_layers/activations_all_layers.npz
    BANK=$ROOT/results/implicit_v2_explicit_probe_bank_8b_all_layers.npz
    LAYERS=all ;;
  llama31_70b)
    ACT=/scratch/jppatton/langea/results/provenance_v1_llama31_70b_all_layers/activations_all_layers.npz
    BANK=/scratch/jppatton/langea/results/provenance_v1_llama31_70b_explicit_probe_bank_all_layers.npz
    LAYERS=all ;;
  llama33_70b)
    ACT=/scratch/jppatton/langea/results/provenance_v1_llama33_70b_all_layers/activations_all_layers.npz
    BANK=/scratch/jppatton/langea/results/provenance_v1_llama33_70b_explicit_probe_bank_all_layers.npz
    LAYERS=all ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/provenance_v1_${MODEL_KEY}_frozen_probe.json
SCORES=/scratch/jppatton/langea/results/provenance_v1_${MODEL_KEY}_frozen_probe_scores.npz
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/provenance_v1_score.py \
  --model-key "$MODEL_KEY" --items "$ITEMS" --activations "$ACT" \
  --probe-bank "$BANK" --layers "$LAYERS" --n-bootstrap 1000 \
  --out "$OUT" --scores-out "$SCORES"
