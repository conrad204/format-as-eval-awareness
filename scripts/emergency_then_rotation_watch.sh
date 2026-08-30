#!/usr/bin/env bash
# Wait for a running emergency C_perp sweep to exit, then finish that model's
# C_perp result without supervision:
#
#   1. re-run assembly from the atomic layer checkpoints with the legacy
#      reference recorded rather than gated
#   2. run its 1000-rep payload-block bootstrap
#
# The within-Q_B rotation test is deliberately NOT chained here: it only runs on
# the 70B models once the 8B rotation result shows a clear signal. Launch it
# explicitly with scripts/qb_rotation_pod_run.sh when that gate is met.
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

echo "watcher: $MODEL C_perp result complete; rotation test not started (gated on the 8B signal)"
