#!/usr/bin/env bash
# Pull per-layer rotation checkpoints off every fleet box into one local directory.
#
#   ./collect_rotation_partials.sh <model> <local_partial_dir> <host:port> [<host:port> ...]
#
# Checkpoints are immutable per layer, so pulling the same layer twice is a
# no-op and boxes that duplicated a layer are harmless.
set -uo pipefail

MODEL="${1:?usage: collect_rotation_partials.sh <model> <dir> <host:port> ...}"
DEST="${2:?missing local partial dir}"
shift 2
KEY="${TAE_SSH_KEY:-langea}"
REMOTE="/root/tae/results/qb_rotation_scores_${MODEL}.npz.partial"

mkdir -p "$DEST"
for target in "$@"; do
  host="${target%%:*}"
  port="${target##*:}"
  names=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 \
    -p "$port" "root@$host" "cd $REMOTE 2>/dev/null && ls layer_*.npz 2>/dev/null" 2>/dev/null | tr -d '\r')
  for name in $names; do
    if [ -s "$DEST/$name" ]; then
      continue
    fi
    ssh -i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 \
      -p "$port" "root@$host" "cat $REMOTE/$name" > "$DEST/$name.part" 2>/dev/null
    if [ -s "$DEST/$name.part" ]; then
      mv "$DEST/$name.part" "$DEST/$name"
      echo "pulled $host:$port $name"
    else
      rm -f "$DEST/$name.part"
      echo "FAILED $host:$port $name" >&2
    fi
  done
done
echo "local checkpoints: $(ls "$DEST"/layer_*.npz 2>/dev/null | wc -l)"
