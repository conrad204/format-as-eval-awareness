#!/bin/bash
# Run both 70B extractions back to back on one pod (2x H100 / 2x A100-80GB).
# The first model's HF upload runs in the background so the GPUs start on the
# second model instead of idling; the script only exits 0 once both models are
# extracted AND both uploads have succeeded — don't stop the pod before then.
#
#   bash scripts/runpod_queue_70b.sh
#
# Same requirements as runpod_extract.sh: run from the repo root, HF_TOKEN in
# the environment or repo-root .env (gated meta-llama + private format-EA).
set -euo pipefail

FREEZE=full320-v3

if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?set HF_TOKEN (gated meta-llama + private format-EA access)}"

bash scripts/runpod_extract.sh llama31_70b

hf upload format-EA/$FREEZE activations/$FREEZE/llama31_70b activations/llama31_70b \
  --repo-type dataset > upload_llama31_70b.log 2>&1 &
UPLOAD1=$!

bash scripts/runpod_extract.sh llama33_70b

hf upload format-EA/$FREEZE activations/$FREEZE/llama33_70b activations/llama33_70b \
  --repo-type dataset

if ! wait "$UPLOAD1"; then
  echo "error: background upload of llama31_70b failed — see upload_llama31_70b.log" >&2
  exit 1
fi
echo "both 70B extractions uploaded to hf.co/datasets/format-EA/$FREEZE"
