#!/bin/bash
#SBATCH --job-name=tae_prov3_logc
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --constraint=avx2
#SBATCH --output=/scratch/jppatton/langea/logs/tae_prov3_logc-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_prov3_logc-%j.err
set -euo pipefail
: "${MODEL_KEY:?set MODEL_KEY}"
: "${LAYER_STRIDE:=4}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
SCORES=/scratch/jppatton/langea/results/provenance_v3_${MODEL_KEY}_frozen_probe_scores.npz
OUT=/scratch/jppatton/langea/results/provenance_v3_${MODEL_KEY}_log_corpus_contrast.json
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/provenance_v3_log_corpus_contrast.py \
  --scores "$SCORES" --model-key "$MODEL_KEY" --n-bootstrap 1000 \
  --layer-stride "$LAYER_STRIDE" --out "$OUT"
