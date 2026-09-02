#!/bin/bash
#SBATCH --job-name=tae_bind2_tmplq
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=18:00:00
#SBATCH --constraint=avx2
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind2_tmplq-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind2_tmplq-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
ITEMS=$ROOT/results/binding_v2_frozen/items.jsonl
case "$MODEL_KEY" in
  llama31_8b)
    ACT=/scratch/jppatton/langea/results/binding_v2_8b_all_layers/activations_all_layers.npz
    LAYERS=14,31 ;;
  llama31_70b)
    ACT=/scratch/jppatton/langea/results/binding_v2_llama31_70b_all_layers/activations_all_layers.npz
    LAYERS=18 ;;
  llama33_70b)
    ACT=/scratch/jppatton/langea/results/binding_v2_llama33_70b_all_layers/activations_all_layers.npz
    LAYERS=18 ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
OUT=/scratch/jppatton/langea/results/binding_v2_${MODEL_KEY}_template_subspace.json
cd "$ROOT"
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28 OPENBLAS_NUM_THREADS=28
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/binding_v2_format_subspace.py \
  --model-key "$MODEL_KEY" --items "$ITEMS" --activations "$ACT" \
  --layers "$LAYERS" --ranks 64 --out "$OUT"
