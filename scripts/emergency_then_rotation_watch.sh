#!/usr/bin/env bash
# Wait for a running emergency C_perp sweep to exit, then finish that model
# end-to-end without supervision:
#
#   1. re-run assembly from the atomic layer checkpoints with the legacy
#      reference recorded rather than gated, then its block bootstrap
#   2. run the within-Q_B C-alignment rotation test, then its paired bootstrap
#
#   ./emergency_then_rotation_watch.sh <model> [n_jobs]
set -euo pipefail

MODEL="${1:?usage: emergency_then_rotation_watch.sh <model> [n_jobs]}"
NJOBS="${2:-30}"
ROOT="${TAE_ROOT:-/root/tae}"
cd "$ROOT"
mkdir -p runlogs

echo "watcher: waiting for the $MODEL C_perp sweep to exit"
while pgrep -f "[f]ull320_v3_emergency_cperp_sweep.py --model $MODEL" >/dev/null 2>&1; do
  sleep 60
done
echo "watcher: sweep process gone; assembling $MODEL"

# Resume-safe: reads every published layer checkpoint, writes the final artifact
# and the 1000-rep bootstrap.
bash scripts/emergency_pod_run.sh "$MODEL" "$NJOBS"

echo "watcher: starting the within-Q_B rotation test for $MODEL"
bash scripts/qb_rotation_pod_run.sh "$MODEL" "$NJOBS"

echo "watcher: $MODEL fully complete"
