#!/usr/bin/env bash
# Fan the within-Q_B rotation sweep across many small CPU boxes, two layers each.
#
#   ./launch_rotation_fleet.sh <model> <host> <port:layer,layer> [<port:layer,layer> ...]
#
# Each box: receives the scripts, gets the HF token, installs pinned deps,
# downloads and hash-verifies the pinned activation bank, expands ONLY its own
# layer shards, then computes its layers and publishes atomic per-layer
# checkpoints. Checkpoints are later merged and assembled on one box.
#
# HF_TOKEN must already be exported (source .env.local first).
set -euo pipefail

MODEL="${1:?usage: launch_rotation_fleet.sh <model> <host> <port:layers> ...}"
HOST="${2:?missing host}"
shift 2
: "${HF_TOKEN:?export HF_TOKEN first}"

KEY="${TAE_SSH_KEY:-langea}"
PAYLOAD=$(mktemp)
tar -cf "$PAYLOAD" \
  scripts/full320_v3_emergency_cperp_sweep.py \
  scripts/full320_v3_factorial_residual_sweep.py \
  scripts/format_erasure_development.py \
  scripts/emergency_pod_setup.sh \
  scripts/full320_v3_qb_rotation_sweep.py \
  results/full320_v3_erasure_full_layer_sweep.json

launch_one() {
  local port="${1%%:*}"
  local layers="${1#*:}"
  local layer_args="${layers//,/ }"
  local ssh_cmd="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=25 -p $port root@$HOST"

  $ssh_cmd "mkdir -p /root/tae" < /dev/null
  $ssh_cmd "tar -xf - -C /root/tae" < "$PAYLOAD"
  printf '%s\n' "$HF_TOKEN" | $ssh_cmd "cat > /root/.hf_token && chmod 600 /root/.hf_token"
  local sweep="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/full320_v3_qb_rotation_sweep.py --model $MODEL --npz /root/tae/v3-activations/activations/$MODEL/activations_all_layers.npz --items /root/tae/v3-activations/rendered_items_v3_full320_8-17-2026.jsonl --shard-dir /root/tae/shards_$MODEL --out /root/tae/results/qb_rotation_scores_$MODEL.npz --partial-dir /root/tae/results/qb_rotation_scores_$MODEL.npz.partial --timing-out /root/tae/results/qb_rotation_timing_$MODEL.json --resume --n-jobs 2 --layers $layer_args"
  $ssh_cmd -n "cd /root/tae && mkdir -p runlogs && nohup bash -c 'TAE_SKIP_BENCHMARK=1 bash scripts/emergency_pod_setup.sh $MODEL && $sweep' > runlogs/rot_$port.log 2>&1 < /dev/null & sleep 1; echo launched:$port:$layers"
}

for spec in "$@"; do
  launch_one "$spec" &
done
wait
rm -f "$PAYLOAD"
echo "fleet launched: $*"
