#!/usr/bin/env bash
# Full-depth emergency run for one model on one box, then its bootstrap.
#
#   ./emergency_pod_run.sh <model> [n_jobs]
#
# Resume-safe: re-running picks up every atomically published layer checkpoint.
set -euo pipefail

MODEL="${1:?usage: emergency_pod_run.sh <model> [n_jobs]}"
NJOBS="${2:-30}"
ROOT="${TAE_ROOT:-/root/tae}"
ACT="$ROOT/v3-activations/activations/$MODEL/activations_all_layers.npz"
ITEMS="$ROOT/v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
SCORES="$ROOT/results/emergency_cperp_scores_$MODEL.npz"

cd "$ROOT"
mkdir -p results runlogs

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 scripts/full320_v3_emergency_cperp_sweep.py \
  --model "$MODEL" \
  --npz "$ACT" \
  --items "$ITEMS" \
  --shard-dir "$ROOT/shards_$MODEL" \
  --out "$SCORES" \
  --partial-dir "$SCORES.partial" \
  --timing-out "$ROOT/results/emergency_cperp_timing_$MODEL.json" \
  --resume --n-jobs "$NJOBS" --reference-mode report
python3 scripts/full320_v3_emergency_cperp_bootstrap.py \
  --scores "$SCORES" \
  --out "$ROOT/results/emergency_cperp_summary_$MODEL.json" \
  --csv "$ROOT/results/emergency_cperp_summary_$MODEL.csv" \
  --n-boot 1000

echo "=== $MODEL complete ==="
