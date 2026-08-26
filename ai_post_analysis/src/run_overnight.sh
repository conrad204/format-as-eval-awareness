#!/usr/bin/env bash
# Overnight queue: estimator robustness, then a larger permutation null.
#
# Ordered so partial completion is still useful: the 8B estimator check comes
# first because it is the one that could invalidate the headline result, then
# the null extension, then the expensive 70B estimator checks.
#
# Everything is checkpointed. Ctrl-C and re-run to continue.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe
S=ai_post_analysis/src

derived () {  # $1 model, $2 probe
  $PY $S/template_offset.py    --model "$1" --probe "$2" --n-boot 10000
  $PY $S/contract_analysis.py  --model "$1" --probe "$2" --n-boot 2000
  $PY $S/cue_responsiveness.py --model "$1" --probe "$2" --n-boot 10000
}

echo "########## 1. 8B estimator robustness ##########"
for P in dlda svc slda; do
  echo "=== llama31_8b / $P ==="
  $PY $S/score_export.py --model llama31_8b --probe "$P" --jobs 6
  $PY $S/run_analysis.py --model llama31_8b --probe "$P" --stage main --jobs 6 --n-boot 10000
  $PY $S/template_analysis.py --model llama31_8b --probe "$P" --n-boot 10000
  derived llama31_8b "$P"
done

echo "########## 2. permutation null 20 -> 100 draws (8B) ##########"
# Additive: the existing 20 draws are already checkpointed per cell and are reused.
$PY $S/permutation_null.py --model llama31_8b --layers 7 11 15 23 31 \
    --n-perm 100 --jobs 6

echo "########## 3. 70B estimator robustness (svc only) ##########"
# Shrinkage LDA needs a full covariance at 8192 dims and costs ~4.4h per model,
# which does not fit alongside the rest; the max-margin comparator carries the
# cross-scale check instead.
for M in llama31_70b llama33_70b; do
  echo "=== $M / svc ==="
  $PY $S/score_export.py --model "$M" --probe svc --jobs 6
  derived "$M" svc
done

echo "OVERNIGHT QUEUE COMPLETE"
