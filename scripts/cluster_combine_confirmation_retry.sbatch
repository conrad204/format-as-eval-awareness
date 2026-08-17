#!/bin/bash
#SBATCH --job-name=tae_confirm_combine
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --account=sdibbo_grp
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/jppatton/langea/logs/tae_confirm_combine-%j.out
#SBATCH --error=/scratch/jppatton/langea/logs/tae_confirm_combine-%j.err
set -euo pipefail
RUN_DIR=${RUN_DIR:?}
RETRY_DIR=${RETRY_DIR:?}
OUT_DIR=${OUT_DIR:?}
if [ -e "$OUT_DIR/.assembly_started" ]; then echo "immutable combined run already exists" >&2; exit 2; fi
mkdir -p "$OUT_DIR"
touch "$OUT_DIR/.assembly_started"
/bighome/jppatton/langea/runtime/vllm/bin/python - "$RUN_DIR" "$RETRY_DIR" "$OUT_DIR" <<'PY'
import json, shutil, sys
from pathlib import Path
run, retry, out = map(Path, sys.argv[1:])
original = json.loads((run / "payloads_candidates.json").read_text())
replacement = json.loads((retry / "payloads_candidates.json").read_text())
frames = json.loads((run / "frames_candidates.json").read_text())
if len(original) != 72 or len(replacement) != 8 or len(frames) != 50:
    raise SystemExit(f"unexpected candidate counts: original={len(original)} retry={len(replacement)} frames={len(frames)}")
combined = original[:56] + replacement + original[56:]
if len(combined) != 80:
    raise SystemExit("combined payload count is not 80")
(out / "payloads_candidates.json").write_text(json.dumps(combined, indent=2) + "\n")
(out / "frames_candidates.json").write_text(json.dumps(frames, indent=2) + "\n")
for source, name in ((run / "payloads_generation_log.jsonl", "payloads_generation_log_original.jsonl"),
                     (retry / "payloads_generation_log.jsonl", "payloads_generation_log_retry.jsonl"),
                     (run / "frames_generation_log.jsonl", "frames_generation_log.jsonl"),
                     (run / "payloads_semantic_validation.jsonl", "payloads_semantic_validation_original.jsonl"),
                     (run / "frames_semantic_validation.jsonl", "frames_semantic_validation.jsonl")):
    shutil.copy2(source, out / name)
(out / "combination_provenance.json").write_text(json.dumps({
    "status": "combined_pre_activation_candidates",
    "original_run": str(run),
    "retry_run": str(retry),
    "original_payload_count": 72,
    "retry_payload_count": 8,
    "combined_payload_count": 80,
    "frames_count": 50,
    "insertion_index": 56,
    "reason": "Restore frozen 80-payload bank after one pre-outcome wrong-count generation batch; preserve both raw logs.",
}, indent=2) + "\n")
PY
