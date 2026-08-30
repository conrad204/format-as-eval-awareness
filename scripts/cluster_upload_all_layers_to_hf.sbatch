#!/bin/bash
#SBATCH --job-name=tae_hf_upload
#SBATCH --partition=main
#SBATCH --account=sdibbo_grp
#SBATCH --qos=main
# The runtime's numpy is compiled with AVX2; uahpc-compute061 (Ivy Bridge
# e7-4860v2) has no AVX2 and faults on import with SIGILL.
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_hf_upload-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_hf_upload-%j.err
# Push the 70B all-layer tensors from /scratch to the format-EA HF org.
# Submit with DATASET=binding_v1 or DATASET=implicit_v2; gate the implicit_v2
# upload on its extraction jobs with --dependency=afterok:<jobids>.
set -euo pipefail
: "${DATASET:?set DATASET to binding_v1 or implicit_v2}"
ROOT=/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_ENABLE_HF_TRANSFER=0
# Approved cluster credential file; value is never echoed.
set -a
. /home/jppatton/langea/credentials/huggingface.env
set +a
cd "$ROOT"
/bighome/jppatton/langea/runtime/vllm/bin/python scripts/cluster_upload_all_layers_to_hf.py \
  --dataset "$DATASET"
