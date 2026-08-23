#!/bin/bash
#SBATCH --job-name=tae_factorial_causal
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=sdibbo_grp
#SBATCH --gres=gpu:h100-80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_factorial_causal-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_factorial_causal-%j.err
set -euo pipefail

: "${SOURCE_ROOT:?set SOURCE_ROOT under /scratch/jppatton/langea}"
: "${MODEL_ID:?set MODEL_ID to the exact configured model id}"
: "${OUT_DIR:?set OUT_DIR under /scratch/jppatton/langea}"
SOURCE_ROOT="$(realpath -e "${SOURCE_ROOT}")"
OUT_DIR="$(realpath -m "${OUT_DIR}")"
case "${SOURCE_ROOT}:${OUT_DIR}" in
  /scratch/jppatton/langea/*:/scratch/jppatton/langea/*) ;;
  *) echo "resolved SOURCE_ROOT and OUT_DIR must stay under /scratch/jppatton/langea" >&2; exit 2 ;;
esac
mkdir -p "${OUT_DIR}"

export XDG_CACHE_HOME=/scratch/jppatton/langea/cache
export HF_HOME=/scratch/jppatton/langea/huggingface
export HF_HUB_CACHE=/scratch/jppatton/langea/huggingface/hub
export TMPDIR=/scratch/jppatton/langea/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Deterministic cuBLAS reduction order (paired with torch.use_deterministic_algorithms
# in run_pilot80_factorial_causal.py); set at the process level so it predates
# CUDA context init regardless of Python import order.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${OUT_DIR}/plan" "${OUT_DIR}/raw"
PYTHON=/bighome/jppatton/langea/runtime/vllm/bin/python
CONFIG="${CONFIG_PATH:-${SOURCE_ROOT}/configs/pilot80_factorial_causal_mediation_v1.json}"
PLAN_JSON="${OUT_DIR}/plan/causal_plan.json"
PLAN_NPZ="${OUT_DIR}/plan/causal_plan.npz"

if [[ -e "${PLAN_JSON}" || -e "${PLAN_NPZ}" ]]; then
  if [[ ! -f "${PLAN_JSON}" || ! -f "${PLAN_NPZ}" ]]; then
    echo "partial immutable plan exists; refusing to rebuild" >&2
    exit 2
  fi
else
  "${PYTHON}" "${SOURCE_ROOT}/scripts/build_pilot80_factorial_causal_plan.py" \
    --config "${CONFIG}" \
    --model-id "${MODEL_ID}" \
    --out-dir "${OUT_DIR}/plan"
fi

"${PYTHON}" "${SOURCE_ROOT}/scripts/run_pilot80_factorial_causal.py" \
  --config "${CONFIG}" \
  --plan-json "${PLAN_JSON}" \
  --plan-npz "${PLAN_NPZ}" \
  --model-id "${MODEL_ID}" \
  --out-dir "${OUT_DIR}/raw" \
  --batch-size "${BATCH_SIZE:-8}" \
  --resume
