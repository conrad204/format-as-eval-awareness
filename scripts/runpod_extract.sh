#!/bin/bash
# Pod-side extraction driver for the full320-v3 freeze.
#
#   bash scripts/runpod_extract.sh llama31_8b      # 1x A100 40GB is enough
#   bash scripts/runpod_extract.sh llama31_70b     # bf16 ~140GB+, 2x A100-80GB / 2x H100
#   bash scripts/runpod_extract.sh llama33_70b     # same as above
#
# Expects HF_TOKEN in the environment (or repo-root .env): the token must have
# access to the gated meta-llama models AND the private format-EA dataset org.
# Run from the repo root. Model revisions are pinned below — keep them in sync
# with README.md; the extraction config.json records what was actually used.
set -euo pipefail

FREEZE=full320-v3
ITEMS=data/$FREEZE/rendered_items_v3_full320_8-17-2026.jsonl
MANIFEST=provenance/$FREEZE/render_manifest_v3_full320.json

case "${1:-}" in
  llama31_8b)
    MODEL=meta-llama/Llama-3.1-8B-Instruct
    REVISION=0e9e39f249a16976918f6564b8830bc894c89659
    BATCH=8
    EXTRA=() ;;
  llama31_70b)
    MODEL=meta-llama/Llama-3.1-70B-Instruct
    REVISION=1605565b47bb9346c5515c34102e054115b4f98b
    BATCH=16
    EXTRA=(--device-map auto) ;;
  llama33_70b)
    MODEL=meta-llama/Llama-3.3-70B-Instruct
    REVISION=6f6073b423013f6a7d4d9f39144961bfbfbc386b
    # 8 (not 16): a single H200 leaves only ~8GiB after weights, and batch
    # scales the KV/activation transient; batch never changes values.
    BATCH=8
    EXTRA=(--device-map auto) ;;
  *)
    echo "usage: $0 {llama31_8b|llama31_70b|llama33_70b}" >&2; exit 2 ;;
esac
KEY=$1
OUT=activations/$FREEZE/$KEY

if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?set HF_TOKEN (gated meta-llama + private format-EA access)}"

pip install -q -e ".[extract,analysis]"

# Materialize the freeze at its pinned HF revision, then byte-verify it.
if [ ! -f "$ITEMS" ]; then
  hf download format-EA/$FREEZE --repo-type dataset \
    --revision "$(python3 -c "import json;print(json.load(open('provenance/$FREEZE/hf_snapshot.json'))['revision'])")" \
    --local-dir data/$FREEZE
fi
eval-format-mvp verify-manifest-v3 --manifest "$MANIFEST"

# Batch 32 OOMed on 2x H100 (weights leave ~4-8GB headroom per GPU and the
# default allocator fragments it); expandable_segments only changes allocator
# behavior, never values.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/extract_activations.py \
  --items "$ITEMS" \
  --out-dir "$OUT" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --all-layers --batch-size "$BATCH" "${EXTRA[@]}"

# Record the hardware alongside the extraction config: bf16 forward passes
# are not bit-identical across GPU architectures, so each model's full item
# set must come from a single run on one machine (this script guarantees
# that); which architecture it was is provenance worth keeping.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu_info.txt" || true
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" >> "$OUT/gpu_info.txt" || true

echo "== extraction config =="
cat "$OUT/config.json"
echo
echo "== next steps =="
echo "sweep (on pod):    python scripts/sweep_full.py $ITEMS $OUT/activations_all_layers.npz results/layer_sweep_${FREEZE}_${KEY}.json"
echo "archive npz to HF: hf upload format-EA/$FREEZE $OUT activations/$KEY --repo-type dataset"
