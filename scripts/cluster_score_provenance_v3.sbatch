#!/bin/bash
#SBATCH --job-name=tae_prov3_score
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --constraint=avx2
#SBATCH --output=/scratch/jppatton/langea/logs/tae_prov3_score-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_prov3_score-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
ITEMS=$ROOT/results/provenance_v3_frozen/items.jsonl
ACT=/scratch/jppatton/langea/results/provenance_v3_${MODEL_KEY}_all_layers/activations_all_layers.npz
case "$MODEL_KEY" in
 llama31_8b) BANK=$ROOT/results/implicit_v2_explicit_probe_bank_8b_all_layers.npz ;;
 llama31_70b|llama33_70b)
  BANK=/scratch/jppatton/langea/results/provenance_v1_${MODEL_KEY}_explicit_probe_bank_all_layers.npz ;;
 *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/provenance_v3_${MODEL_KEY}_frozen_probe.json
SCORES=/scratch/jppatton/langea/results/provenance_v3_${MODEL_KEY}_frozen_probe_scores.npz
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/provenance_v1_score.py \
 --model-key "$MODEL_KEY" --items "$ITEMS" --activations "$ACT" --probe-bank "$BANK" \
 --expected-items-sha256 2a574bf1bf0cd5cc8ec53c97e2bf69e9246316b1a3d5364e370acc0f8fcddc7e \
 --layers all --n-bootstrap 1000 --out "$OUT" --scores-out "$SCORES"
