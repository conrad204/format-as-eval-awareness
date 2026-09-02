#!/bin/bash
#SBATCH --job-name=tae_bind2_analysis
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --constraint=avx2
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind2_analysis-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind2_analysis-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
ITEMS=$ROOT/results/binding_v2_frozen/items.jsonl
case "$MODEL_KEY" in
  llama31_8b)
    ACT=/scratch/jppatton/langea/results/binding_v2_8b_all_layers/activations_all_layers.npz
    BANK=$ROOT/results/implicit_v2_explicit_probe_bank_8b_all_layers.npz
    LAYERS=all ;;
  llama31_70b)
    ACT=/scratch/jppatton/langea/results/binding_v2_llama31_70b_all_layers/activations_all_layers.npz
    BANK=$ROOT/results/implicit_v2_llama31_70b_explicit_probe_bank_l18.npz
    LAYERS=18 ;;
  llama33_70b)
    ACT=/scratch/jppatton/langea/results/binding_v2_llama33_70b_all_layers/activations_all_layers.npz
    BANK=$ROOT/results/implicit_v2_llama33_70b_explicit_probe_bank_l18.npz
    LAYERS=18 ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/binding_v2_${MODEL_KEY}_analysis.json
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/binding_v2_analyze.py \
  --model-key "$MODEL_KEY" --items "$ITEMS" --activations "$ACT" \
  --explicit-bank "$BANK" --layers "$LAYERS" --jobs 28 --out "$OUT"
