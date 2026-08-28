#!/bin/bash
#SBATCH --job-name=tae_confirm_gen
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:3
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_confirm_gen-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_confirm_gen-%j.err
set -euo pipefail
export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/jppatton/langea/huggingface/transformers
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN_DIR=${RUN_DIR:?sbatch --export=RUN_DIR=... is required}
SOURCE_DIR=${SOURCE_DIR:-/scratch/jppatton/langea/confirmation_work/tae_2026}
MODEL=${MODEL:-stockmark/Stockmark-2-100B-Instruct}
MODEL_REVISION=${MODEL_REVISION:-98e959a472d0fcaf0c8b6576af19fe3cba7ca4ab}
mkdir -p /scratch/jppatton/langea/cache /scratch/jppatton/langea/tmp /scratch/jppatton/langea/logs
if [ -e "$RUN_DIR/.run_started" ]; then echo "immutable run already started: $RUN_DIR" >&2; exit 2; fi
mkdir -p "$RUN_DIR"
touch "$RUN_DIR/.run_started"
/bighome/jppatton/langea/runtime/vllm/bin/python /scratch/jppatton/langea/deploy/scripts/serve_stage3_introspection.py \
  --model "$MODEL" --revision "$MODEL_REVISION" \
  --host 127.0.0.1 --port 8123 --dtype bfloat16 --gpu-memory-gib 72 > "$RUN_DIR/server.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for attempt in $(seq 1 180); do
  if /bighome/jppatton/langea/runtime/vllm/bin/python -c 'import json,urllib.request; json.load(urllib.request.urlopen("http://127.0.0.1:8123/health", timeout=2))' >/dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then echo 'generation server did not become healthy' >&2; exit 3; fi
KIND=${KIND:-both}
START_BATCH=${START_BATCH:-0}
BATCHES=${BATCHES:-}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0.2}
SEED=${SEED:-20260817}
FRAME_SEED=${FRAME_SEED:-20260827}
if [ "$KIND" = "payloads" ] || [ "$KIND" = "both" ]; then
  /bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/generate_confirmation_candidates.py" \
    --url http://127.0.0.1:8123/v1/chat/completions --model "$MODEL" --model-revision "$MODEL_REVISION" \
    --kind payloads --batches "${BATCHES:-10}" --start-batch "$START_BATCH" --seed "$SEED" --temperature "$TEMPERATURE" --max-tokens "$MAX_TOKENS" --out-dir "$RUN_DIR"
fi
if [ "$KIND" = "frames" ] || [ "$KIND" = "both" ]; then
  /bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/generate_confirmation_candidates.py" \
    --url http://127.0.0.1:8123/v1/chat/completions --model "$MODEL" --model-revision "$MODEL_REVISION" \
    --kind frames --batches "${BATCHES:-5}" --start-batch "$START_BATCH" --seed "$FRAME_SEED" --temperature "$TEMPERATURE" --max-tokens "$MAX_TOKENS" --out-dir "$RUN_DIR"
fi
