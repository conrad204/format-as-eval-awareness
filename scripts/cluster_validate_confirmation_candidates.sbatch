#!/bin/bash
#SBATCH --job-name=tae_confirm_val
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:2
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_confirm_val-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_confirm_val-%j.err
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
MODEL=${MODEL:-tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4}
MODEL_REVISION=${MODEL_REVISION:-66ff64690b07545b2807293b0e24eb1e17c88da4}
if [ ! -s "$RUN_DIR/payloads_candidates.json" ] || [ ! -s "$RUN_DIR/frames_candidates.json" ]; then echo 'candidate inputs missing' >&2; exit 2; fi
if [ -e "$RUN_DIR/payloads_semantic_validation.jsonl" ] || [ -e "$RUN_DIR/frames_semantic_validation.jsonl" ]; then echo 'validator outputs already exist; refusing overwrite' >&2; exit 2; fi
/bighome/jppatton/langea/runtime/vllm/bin/python /scratch/jppatton/langea/deploy/scripts/serve_stage3_introspection.py \
  --model "$MODEL" --revision "$MODEL_REVISION" --host 127.0.0.1 --port 8124 --dtype bfloat16 --gpu-memory-gib 72 > "$RUN_DIR/server_validator.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for attempt in $(seq 1 180); do
  if /bighome/jppatton/langea/runtime/vllm/bin/python -c 'import json,urllib.request; json.load(urllib.request.urlopen("http://127.0.0.1:8124/health", timeout=2))' >/dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then echo 'validator server did not become healthy' >&2; exit 3; fi
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/validate_confirmation_candidates.py" \
  --url http://127.0.0.1:8124/v1/chat/completions --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --kind payloads --input "$RUN_DIR/payloads_candidates.json" --seed 20260901 --out "$RUN_DIR/payloads_semantic_validation.jsonl"
/bighome/jppatton/langea/runtime/vllm/bin/python "$SOURCE_DIR/validate_confirmation_candidates.py" \
  --url http://127.0.0.1:8124/v1/chat/completions --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --kind frames --input "$RUN_DIR/frames_candidates.json" --seed 20261001 --out "$RUN_DIR/frames_semantic_validation.jsonl"
