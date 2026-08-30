#!/bin/bash
#SBATCH --job-name=tae_bind_ana
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind_ana-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind_ana-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
R=/scratch/jppatton/langea/results
cd "$ROOT"
if [[ "$MODEL_KEY" == "llama31_8b" ]]; then
  ACT=$R/binding_v1_8b_all_layers/activations_all_layers.npz
  BANK=$R/implicit_v2_explicit_probe_bank_8b_all_layers.npz
  EXTRA=()
else
  ACT=$R/binding_v1_${MODEL_KEY}_l18/activations.npz
  BANK=$R/implicit_v2_${MODEL_KEY}_explicit_probe_bank_l18.npz
  EXTRA=(--fixed-layer 18)
fi
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/binding_v1_analyze.py \
  --model-key "$MODEL_KEY" --items results/binding_v1_frozen/items.jsonl \
  --activations "$ACT" --explicit-bank "$BANK" \
  --out "$R/binding_v1_${MODEL_KEY}_analysis.json" --jobs 12 "${EXTRA[@]}"
