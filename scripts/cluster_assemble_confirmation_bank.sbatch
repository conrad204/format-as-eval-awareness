#!/bin/bash
#SBATCH --job-name=tae_confirm_bank
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_confirm_bank-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_confirm_bank-%j.err
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN_DIR=${RUN_DIR:?sbatch --export=RUN_DIR=... is required}
CANDIDATE_DIR=${CANDIDATE_DIR:?sbatch --export=CANDIDATE_DIR=... is required}
SOURCE_DIR=${SOURCE_DIR:-/scratch/jppatton/langea/confirmation_work/tae_2026}
FORBIDDEN=${FORBIDDEN:?approved forbidden-lexicon artifact is required}
if [ -e "$CANDIDATE_DIR/.assembly_started" ]; then echo "immutable candidate run already started: $CANDIDATE_DIR" >&2; exit 2; fi
if [ ! -s "$RUN_DIR/payloads_candidates.json" ] || [ ! -s "$RUN_DIR/frames_candidates.json" ]; then echo 'candidate inputs missing' >&2; exit 2; fi
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/assemble_confirmation_bank.py" \
  --payloads "$RUN_DIR/payloads_candidates.json" --frames "$RUN_DIR/frames_candidates.json" \
  --forbidden "$FORBIDDEN" --out-dir "$CANDIDATE_DIR"
touch "$CANDIDATE_DIR/.assembly_started"
cp "$RUN_DIR"/*_generation_log.jsonl "$CANDIDATE_DIR"/
cp "$RUN_DIR"/*_semantic_validation.jsonl "$CANDIDATE_DIR"/
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/lexical_gates_v2.py" \
  "$CANDIDATE_DIR/rendered_items.jsonl" "$CANDIDATE_DIR/lexical_gate.json"
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/subword_gates.py" \
  "$CANDIDATE_DIR/rendered_items.jsonl" "$CANDIDATE_DIR/subword_gate.json" \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/surface_balance.py" \
  "$CANDIDATE_DIR/frames.json" "$CANDIDATE_DIR/surface_balance.json"
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/structural_variation_gate.py" \
  --frames "$CANDIDATE_DIR/frames.json" --out "$CANDIDATE_DIR/structural_variation.json"
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/red_team_lexical_audit.py" \
  "$CANDIDATE_DIR/rendered_items.jsonl" "$CANDIDATE_DIR/red_team_lexical_report.json"
