#!/bin/bash
#SBATCH --job-name=tae_determinism_shakedown
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_determinism_shakedown-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_determinism_shakedown-%j.err
set -euo pipefail
: "${SOURCE_ROOT:?set SOURCE_ROOT under /scratch/jppatton/langea}"
: "${MODEL_PATH:?set immutable local model snapshot}"
: "${ITEMS_PATH:?set the small shakedown items jsonl}"
: "${OUT_DIR:?set fresh output directory}"
: "${PRIMARY_LAYER:?set the failing source layer to focus on}"
for path in "$SOURCE_ROOT" "$MODEL_PATH" "$ITEMS_PATH" "$OUT_DIR"; do
  case "$path" in
    /scratch/jppatton/langea/*) ;;
    *) echo "project path outside /scratch/jppatton/langea: $path" >&2; exit 2 ;;
  esac
done
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing occupied immutable output: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Deterministic cuBLAS reduction order (paired with torch.use_deterministic_algorithms
# in extract.py); set at the process level so it predates CUDA context init
# regardless of Python import order.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$SOURCE_ROOT/src"
PYTHON=/bighome/jppatton/langea/runtime/vllm/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

# Two independent extraction runs of the same rows under the same patched
# code, back to back in one job so both share hardware/driver state. This is
# an engineering determinism check only: it compares raw activation vectors
# between run_a and run_b, never any purpose/format/causal readout score.
for tag in run_a run_b; do
  "$PYTHON" "$SOURCE_ROOT/scripts/extract_activations.py" \
    --items "$ITEMS_PATH" \
    --out-dir "$OUT_DIR/$tag" \
    --model "$MODEL_PATH" \
    --layer "$PRIMARY_LAYER" \
    --batch-size 1 \
    --max-length 2048 \
    --device-map auto \
    --all-layers
done

sha256sum "$ITEMS_PATH" "$OUT_DIR"/run_a/activations_all_layers.npz "$OUT_DIR"/run_b/activations_all_layers.npz \
  "$OUT_DIR"/run_a/config.json "$OUT_DIR"/run_b/config.json

"$PYTHON" - "$OUT_DIR" "$PRIMARY_LAYER" <<'PYEOF'
import json
import sys
from pathlib import Path

import numpy as np

out_dir = Path(sys.argv[1])
primary_layer = int(sys.argv[2])

a = np.load(out_dir / "run_a" / "activations_all_layers.npz", allow_pickle=False)
b = np.load(out_dir / "run_b" / "activations_all_layers.npz", allow_pickle=False)
assert a["item_ids"].tolist() == b["item_ids"].tolist(), "item order differs between runs"
assert a["layers"].tolist() == b["layers"].tolist(), "layer set differs between runs"

Xa, Xb = a["X"], b["X"]
diff = np.abs(Xa - Xb)
per_layer_max_abs = diff.max(axis=(0, 2))
layers = a["layers"].tolist()
primary_index = layers.index(primary_layer)

def cosine_rows(u, v):
    num = np.sum(u * v, axis=-1)
    den = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    return np.where(den > 0, num / den, np.nan)

primary_cosine = cosine_rows(Xa[:, primary_index, :], Xb[:, primary_index, :])

report = {
    "schema_version": 1,
    "purpose": "engineering_determinism_check_only",
    "content_inspected": "numerical_agreement_between_two_extraction_runs_only",
    "primary_layer": primary_layer,
    "n_items": int(Xa.shape[0]),
    "global_max_abs_diff": float(diff.max()),
    "global_mean_abs_diff": float(diff.mean()),
    "primary_layer_max_abs_diff": float(per_layer_max_abs[primary_index]),
    "primary_layer_min_cosine": float(np.nanmin(primary_cosine)),
    "primary_layer_mean_cosine": float(np.nanmean(primary_cosine)),
    "per_layer_max_abs_diff": {str(l): float(m) for l, m in zip(layers, per_layer_max_abs)},
    "frozen_gate_for_reference": {"max_abs_lte": 0.125, "cosine_gte": 0.999},
}
(out_dir / "shakedown_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
PYEOF
