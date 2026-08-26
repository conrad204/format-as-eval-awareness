#!/usr/bin/env bash
# Full Direction 1 pipeline. Safe to interrupt and re-run.
#
# Every stage checkpoints, so re-running resumes where the last run stopped:
# the layer-wise stages keep a shard per finished layer (per finished
# permutation draw, for the null) under results/_partial/, and the cheap
# post-processing stages skip when their output file already exists. Nothing
# is recomputed unless you pass --force to that stage.
#
#   bash ai_post_analysis/src/run_pipeline.sh              # everything
#   bash ai_post_analysis/src/run_pipeline.sh llama31_8b   # one model's core stages
#
# Scope note: the INLP control stage and the permutation null are run on the
# 8B only. Both scale with hidden size squared, so on the 70Bs they cost more
# than they inform -- the erasure control was already inconclusive at 4096
# dims, and the load-bearing control (the cued-vs-no-cue differential) is in
# the per-model core and runs on all three.
#
# CPU only. Ctrl-C at any point; re-run to continue.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe
S=ai_post_analysis/src
MODELS=("$@")
ALL=0
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(llama31_8b llama31_70b llama33_70b); ALL=1
fi

for M in "${MODELS[@]}"; do
  echo "=============== $M core ==============="
  $PY $S/prep_activations.py --models "$M" --no-verify
  $PY $S/run_analysis.py      --model "$M" --stage main --jobs 6 --n-boot 10000
  $PY $S/score_export.py      --model "$M" --jobs 6
  $PY $S/template_analysis.py --model "$M" --n-boot 10000
  $PY $S/template_offset.py   --model "$M" --n-boot 10000
  $PY $S/cue_responsiveness.py --model "$M" --n-boot 10000
  $PY $S/nocue_regression.py  --model "$M" --n-boot 2000
  $PY $S/nuisance_axis.py     --model "$M" --jobs 6
  echo "=============== $M core done ==============="
done

if [ "$ALL" -eq 1 ]; then
  echo "=============== 8b control stage ==============="
  $PY $S/run_analysis.py --model llama31_8b --stage controls --jobs 5 --n-boot 10000

  echo "=============== 8b permutation null ==============="
  $PY $S/permutation_null.py --model llama31_8b --layers 7 11 15 23 31 \
      --n-perm 20 --jobs 5
fi

echo "=============== figures ==============="
$PY $S/make_figures.py
echo "PIPELINE COMPLETE"
