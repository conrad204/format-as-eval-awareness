#!/bin/bash
#SBATCH --job-name=tae_analysis_70b
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_analysis_70b-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_analysis_70b-%j.err
set -euo pipefail
work=/scratch/jppatton/langea/confirmation_work
bank=$work/confirmation_candidate_20260816
model_key="${MODEL_KEY:?MODEL_KEY must be set}"
activation_dir="${ACTIVATION_DIR:?ACTIVATION_DIR must be set}"
out="${OUTPUT:?OUTPUT must be set}"
layer="${LAYER:?LAYER must be set}"
case "$model_key" in
  llama31_70b) model=meta-llama/Llama-3.1-70B-Instruct; revision=1605565b47bb9346c5515c34102e054115b4f98b; n_layers=80 ;;
  llama33_70b) model=meta-llama/Llama-3.3-70B-Instruct; revision=6f6073b423013f6a7d4d9f39144961bfbfbc386b; n_layers=80 ;;
  *) echo "unknown model key: $model_key" >&2; exit 2 ;;
esac
/bighome/jppatton/langea/runtime/vllm/bin/python "$work/tae_2026/prepare_confirmatory_manifest.py" \
  --base "$bank/sealed_manifest.json" --out "$work/confirmation_run_${model_key}.json" \
  --items "$bank/rendered_items.jsonl" --activations "$work/$activation_dir/activations.npz" \
  --output "$work/$out" --sentinel "$work/${out%.json}.sentinel.json" \
  --analysis-script "$work/tae_2026/confirmation_analysis.canonical.py" --wrapper "$work/tae_2026/run_sealed_confirmation.py" \
  --model "$model" --revision "$revision" --layer "$layer" --n-layers "$n_layers"
/bighome/jppatton/langea/runtime/vllm/bin/python "$work/tae_2026/run_sealed_confirmation.py" --manifest "$work/confirmation_run_${model_key}.json"
