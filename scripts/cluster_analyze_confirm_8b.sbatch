#!/bin/bash
#SBATCH --job-name=tae_analysis_8b
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_analysis_8b-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_analysis_8b-%j.err
set -euo pipefail
work=/scratch/jppatton/langea/confirmation_work
bank=$work/confirmation_candidate_20260816
/bighome/jppatton/langea/runtime/vllm/bin/python "$work/tae_2026/prepare_confirmatory_manifest.py" \
  --base "$bank/sealed_manifest.json" --out "$work/confirmation_run_8b.json" \
  --items "$bank/rendered_items.jsonl" --activations "$work/activations_confirm_8b/activations.npz" \
  --output "$work/confirmatory_analysis_8b.json" --sentinel "$work/confirmatory_analysis_8b.sentinel.json" \
  --analysis-script "$work/tae_2026/confirmation_analysis.canonical.py" --wrapper "$work/tae_2026/run_sealed_confirmation.py" \
  --model meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 --layer 12 --n-layers 32
/bighome/jppatton/langea/runtime/vllm/bin/python "$work/tae_2026/run_sealed_confirmation.py" --manifest "$work/confirmation_run_8b.json"
