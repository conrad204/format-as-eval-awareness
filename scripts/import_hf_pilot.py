"""Stream an exact Hugging Face dataset revision into approved cluster storage.

The HF token is read only from the local process environment and is never
placed in a command line, log, manifest, or artifact. Files are streamed over
SSH directly to the approved remote destination, avoiding a second local copy
of multi-gigabyte activation archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FILES = (
    ".gitattributes",
    "activations/llama31_8b_pilot80/activations.npz",
    "activations/llama31_8b_pilot80/activations_all_layers.npz",
    "activations/llama31_8b_pilot80/config.json",
    "activations/llama31_8b_pilot80/meta.jsonl",
    "activations/llama31_70b_pilot80/activations.npz",
    "activations/llama31_70b_pilot80/activations_all_layers.npz",
    "activations/llama31_70b_pilot80/config.json",
    "activations/llama31_70b_pilot80/meta.jsonl",
    "activations/llama33_70b_pilot80/activations.npz",
    "activations/llama33_70b_pilot80/activations_all_layers.npz",
)


def stream_file(url: str, token: str, host: str, remote_path: str) -> dict:
    head = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="HEAD")
    with urllib.request.urlopen(head, timeout=120) as response:
        expected = int(response.headers["Content-Length"])
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        digest = hashlib.sha256()
        byte_count = 0
        command = f"mkdir -p $(dirname '{remote_path}') && cat > '{remote_path}'"
        process = subprocess.Popen(["ssh", host, command], stdin=subprocess.PIPE)
        assert process.stdin is not None
        try:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                process.stdin.write(chunk)
            process.stdin.close()
            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise
        if return_code != 0:
            raise RuntimeError(f"remote write failed for {remote_path}: ssh exit {return_code}")
    if byte_count != expected:
        raise RuntimeError(f"download length mismatch for {remote_path}: expected {expected}, got {byte_count}")
    return {"bytes": byte_count, "sha256": digest.hexdigest(), "content_length": expected}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="format-EA/format_EA_activations_pilot_run")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--host", default="uahpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set in the local process environment")
    if not args.remote_root.startswith(("/home/jppatton/langea/", "/bighome/jppatton/langea/", "/scratch/jppatton/langea/")):
        raise SystemExit("remote root is outside the approved jppatton/langea roots")
    records = {}
    for filename in FILES:
        url = f"https://huggingface.co/datasets/{args.repo}/resolve/{args.revision}/{filename}"
        remote_path = f"{args.remote_root.rstrip('/')}/{filename}"
        print(json.dumps({"status": "downloading", "file": filename}), flush=True)
        records[filename] = stream_file(url, token, args.host, remote_path)
        print(json.dumps({"status": "written", "file": filename, **records[filename]}), flush=True)
    manifest = {
        "schema_version": 1,
        "dataset": args.repo,
        "revision": args.revision,
        "remote_root": args.remote_root,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "activation_contract": {
            "n_items": 1760,
            "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
            "position": "last_prompt_token_after_chat_template_before_generation",
            "stored_dtype": "float32",
            "inference_dtype": "bfloat16",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_json = json.dumps(manifest, sort_keys=True) + "\n"
    command = f"mkdir -p $(dirname '{args.remote_root}/provenance/pilot_activation_dataset_import.json') && cat > '{args.remote_root}/provenance/pilot_activation_dataset_import.json'"
    completed = subprocess.run(["ssh", args.host, command], input=manifest_json.encode(), check=False)
    if completed.returncode != 0:
        raise RuntimeError("remote provenance write failed")
    print(json.dumps({"status": "complete", "files": len(records), "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
