#!/bin/bash
#SBATCH --job-name=tae_bind_cnull
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_bind_cnull-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_bind_cnull-%j.err
set -euo pipefail
: "${MODEL_KEY:=llama31_8b}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/scratch/jppatton/langea/tmp
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
R=/scratch/jppatton/langea/results
cd "$ROOT"
if [[ "$MODEL_KEY" == "llama31_8b" ]]; then
  ACT=$R/binding_v1_8b_all_layers/activations_all_layers.npz
  EXTRA=(--layers 12,16,24,31)
else
  ACT=$R/binding_v1_${MODEL_KEY}_all_layers/activations_all_layers.npz
  EXTRA=(--layers 30,40,56,72)
fi
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/binding_v1_erasure_conditional_null.py \
  --model-key "$MODEL_KEY" --items results/binding_v1_frozen/items.jsonl \
  --activations "$ACT" --ranks 4,16,64 --n-draws 10 --jobs 8 \
  --out "$R/binding_v1_${MODEL_KEY}_erasure_conditional_null.json" "${EXTRA[@]}"
