#!/bin/bash
#SBATCH --job-name=tae_confirm_seal
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_confirm_seal-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_confirm_seal-%j.err
set -euo pipefail
RUN_DIR=${RUN_DIR:?sbatch --export=RUN_DIR=... is required}
CANDIDATE_DIR=${CANDIDATE_DIR:?sbatch --export=CANDIDATE_DIR=... is required}
SOURCE_DIR=${SOURCE_DIR:-/scratch/jppatton/langea/confirmation_work/tae_2026}
ANALYSIS_SCRIPT=${ANALYSIS_SCRIPT:-$SOURCE_DIR/confirmation_analysis.canonical.py}
if [ ! -f "$CANDIDATE_DIR/structural_variation.json" ]; then echo 'structural gate artifact missing' >&2; exit 2; fi
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/seal_confirmation_bank.py" \
  --candidate-dir "$CANDIDATE_DIR" \
  --analysis-script "$ANALYSIS_SCRIPT" \
  --renderer "$SOURCE_DIR/build_confirmation_bank.py" \
  --builder "$SOURCE_DIR/assemble_confirmation_bank.py" \
  --lexical-gate-script "$SOURCE_DIR/lexical_gates_v2.py" \
  --subword-gate-script "$SOURCE_DIR/subword_gates.py" \
  --structural-gate-script "$SOURCE_DIR/structural_variation_gate.py" \
  --out "$CANDIDATE_DIR/sealed_manifest.json"
