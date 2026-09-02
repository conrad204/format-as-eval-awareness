#!/usr/bin/env bash
# Within-Q_B C-alignment test for one model on one box, then its paired bootstrap.
#
#   ./qb_rotation_pod_run.sh <model> [n_jobs]
#
# Requires the emergency run's score archive for the same model (the reused raw
# baseline) and its float32 layer shards. Resume-safe.
set -euo pipefail

MODEL="${1:?usage: qb_rotation_pod_run.sh <model> [n_jobs]}"
NJOBS="${2:-30}"
ROOT="${TAE_ROOT:-/root/tae}"
ACT="$ROOT/v3-activations/activations/$MODEL/activations_all_layers.npz"
ITEMS="$ROOT/v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
EMERGENCY="$ROOT/results/emergency_cperp_scores_$MODEL.npz"
SCORES="$ROOT/results/qb_rotation_scores_$MODEL.npz"

test -s "$EMERGENCY" || { echo "missing reused raw baseline: $EMERGENCY" >&2; exit 2; }
cd "$ROOT"
mkdir -p results runlogs

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 scripts/full320_v3_qb_rotation_sweep.py \
  --model "$MODEL" \
  --npz "$ACT" \
  --items "$ITEMS" \
  --shard-dir "$ROOT/shards_$MODEL" \
  --out "$SCORES" \
  --partial-dir "$SCORES.partial" \
  --timing-out "$ROOT/results/qb_rotation_timing_$MODEL.json" \
  --resume --n-jobs "$NJOBS"

python3 scripts/full320_v3_qb_rotation_bootstrap.py \
  --emergency-scores "$EMERGENCY" \
  --rotation-scores "$SCORES" \
  --out "$ROOT/results/qb_rotation_summary_$MODEL.json" \
  --n-boot 1000

echo "=== $MODEL rotation test complete ==="
