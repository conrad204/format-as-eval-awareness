#!/usr/bin/env bash
# Prepare one RunPod CPU box for the emergency rank-64 C_perp_B run.
#
#   ./emergency_pod_setup.sh <model>            # deps + local download + 1-layer benchmark
#
# The pinned full320-v3 freeze is fetched from Hugging Face straight onto this
# box's local disk (never network storage) and hash-verified against the
# provenance manifest before any compute. Requires the HF token in
# /root/.hf_token (never echoed).
set -euo pipefail

MODEL="${1:?usage: emergency_pod_setup.sh <llama31_8b|llama31_70b|llama33_70b>}"
ROOT="${TAE_ROOT:-/root/tae}"
REPO="format-EA/full320-v3"
REV="d6c3be7be83c469d9a6a2aa4e797f3a7b538efde"

case "$MODEL" in
  llama31_8b)  ACT_SHA=e6a71be599e27fd846211e8e6ef215e4fb695a76c3415dc157c1450a134cd1b4 ;;
  llama31_70b) ACT_SHA=02018659ad8f68101d8ec35467321e791831e05b4d8d628f21e8fe222f6da8e2 ;;
  llama33_70b) ACT_SHA=d455140f2dce01e1bd4e1a8333454a494b1e5de0df0428779a2faa43bc80c031 ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
ITEMS_SHA=5528819d74e696d1a595062047a829836dfde1d2a45287b06a5ac1ba5d1b6c67

echo "=== host ==="
nproc
free -g | sed -n '1,2p'
df -h "$ROOT" | tail -n 1

echo "=== deps ==="
# Container restarts reset the image layer, so treat pip itself as missing.
if ! python3 -m pip --version >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip
fi
python3 -m pip install --no-cache-dir -q \
  numpy==1.24.4 scipy==1.10.1 scikit-learn==1.3.2 joblib==1.4.2 huggingface_hub==0.24.7
python3 -c "import numpy, sklearn, joblib, huggingface_hub as h; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'joblib', joblib.__version__, 'hub', h.__version__)"

echo "=== download to local disk ==="
HF_TOKEN="$(tr -d '\r\n' < /root/.hf_token)" \
TAE_MODEL="$MODEL" TAE_ROOT="$ROOT" TAE_REPO="$REPO" TAE_REV="$REV" \
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

model, root = os.environ["TAE_MODEL"], Path(os.environ["TAE_ROOT"])
repo, rev, token = os.environ["TAE_REPO"], os.environ["TAE_REV"], os.environ["HF_TOKEN"]
targets = [
    (f"activations/{model}/activations_all_layers.npz", root / "v3-activations"),
    ("rendered_items_v3_full320_8-17-2026.jsonl", root / "v3-activations"),
    (f"results/layer_sweep_full320-v3_{model}.json", root),
]
for filename, local_dir in targets:
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo, repo_type="dataset", revision=rev,
        filename=filename, local_dir=str(local_dir), token=token,
    )
    print(f"{filename} -> {path} ({os.path.getsize(path)} bytes)", flush=True)
PY

echo "=== hash gate ==="
ACT="$ROOT/v3-activations/activations/$MODEL/activations_all_layers.npz"
ITEMS="$ROOT/v3-activations/rendered_items_v3_full320_8-17-2026.jsonl"
echo "$ACT_SHA  $ACT" | sha256sum -c -
echo "$ITEMS_SHA  $ITEMS" | sha256sum -c -
test -s "$ROOT/results/full320_v3_erasure_full_layer_sweep.json"
test -s "$ROOT/results/layer_sweep_full320-v3_$MODEL.json"

echo "=== one-layer benchmark ==="
mkdir -p "$ROOT/results" "$ROOT/runlogs"
cd "$ROOT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 scripts/full320_v3_emergency_cperp_sweep.py \
  --model "$MODEL" \
  --npz "$ACT" \
  --items "$ITEMS" \
  --shard-dir "$ROOT/shards_$MODEL" \
  --out "$ROOT/results/emergency_cperp_scores_$MODEL.npz" \
  --partial-dir "$ROOT/results/emergency_cperp_scores_$MODEL.npz.partial" \
  --timing-out "$ROOT/results/emergency_cperp_benchmark_$MODEL.json" \
  --layers 0 --benchmark --n-jobs 1
echo "=== setup complete for $MODEL ==="
