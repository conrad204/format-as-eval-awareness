#!/usr/bin/env bash
# Report per-box progress for the sharded within-Q_B rotation fleet.
#
#   ./rotation_fleet_status.sh <model> <host:port> [<host:port> ...]
#
# Prints one line per box: which layer checkpoints it has published, plus the
# last log line, and a total across the fleet.
set -uo pipefail

MODEL="${1:?usage: rotation_fleet_status.sh <model> <host:port> ...}"
shift
KEY="${TAE_SSH_KEY:-langea}"
PARTIAL="/root/tae/results/qb_rotation_scores_${MODEL}.npz.partial"
total=0

for target in "$@"; do
  host="${target%%:*}"
  port="${target##*:}"
  out=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 \
    -p "$port" "root@$host" \
    "ls $PARTIAL/layer_*.npz 2>/dev/null | sed 's#.*layer_##; s#\.npz##' | tr '\n' ',' ; echo -n '|'; tail -n 1 /root/tae/runlogs/rot_$port.log 2>/dev/null | tr -d '\r'" 2>/dev/null)
  layers="${out%%|*}"
  last="${out#*|}"
  count=$(printf '%s' "$layers" | tr -cd ',' | wc -c)
  total=$((total + count))
  printf '%-24s layers=[%s] %s\n' "$host:$port" "${layers%,}" "${last:0:70}"
done
echo "fleet checkpoints published: $total"
