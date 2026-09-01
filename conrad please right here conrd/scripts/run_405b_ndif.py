"""Run a frozen hidden-channel bank on Llama-3.1-405B-Instruct through NDIF.

Needs a machine that can reach ndif.dev. It could not be reached from the project
workstation or the uahpc login node (TCP 443 timeout / http 000), which is why this
is a handoff script rather than a completed run.

Read-out: log odds of the forced "A" versus "B" continuation at the final prompt
position, matching the local BF16 runs exactly. Optionally also saves final-token
activations at chosen layers so the same cross-fitted probe can be applied.

    pip install nnsight
    export NDIF_API_KEY=...            # or put it in .env.local
    python run_405b_ndif.py --items ../banks/latent_recognition.jsonl \
        --out ../out/latent_recognition_llama31_405b.json

Add --layers 0,20,40,60,79 to also collect activations for the probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

MODEL_ID = "meta-llama/Meta-Llama-3.1-405B-Instruct"


def load_env() -> None:
    for candidate in (Path(".env.local"), Path("../.env.local"), Path("../../.env.local")):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*(?:export\s+)?([A-Z_]+)\s*=\s*(.+)", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="", help="comma-separated layer indices for activations")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on the first N rows")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    load_env()
    key = os.environ.get("NDIF_API_KEY")
    if not key:
        raise SystemExit("set NDIF_API_KEY (env or .env.local)")

    import torch
    from nnsight import CONFIG, LanguageModel

    CONFIG.API.APIKEY = key
    model = LanguageModel(MODEL_ID)
    tok = model.tokenizer
    n_layers = int(model.config.num_hidden_layers)
    letters = {}
    for letter in ("A", "B"):
        ids = tok.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"{letter!r} is not a single token under this tokenizer: {ids}")
        letters[letter] = ids[0]
    want_layers = [int(x) for x in args.layers.split(",") if x.strip()]
    for layer in want_layers:
        if not 0 <= layer < n_layers:
            raise SystemExit(f"layer {layer} outside 0..{n_layers - 1}")

    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{MODEL_ID}: {n_layers} layers, {len(rows)} prompts, batch {args.batch_size}", flush=True)

    outputs, acts = [], {}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
                   for r in batch]
        for attempt in range(args.retries):
            try:
                with model.session(remote=True):
                    saved_logits, saved_hidden = [], {layer: [] for layer in want_layers}
                    for prompt in prompts:
                        with model.trace(prompt):
                            saved_logits.append(model.lm_head.output[0, -1, :].save())
                            for layer in want_layers:
                                block = model.model.layers[layer].output
                                tensor = block[0] if isinstance(block, tuple) else block
                                saved_hidden[layer].append(tensor[0, -1, :].save())
                break
            except Exception as exc:  # transient NDIF queueing/backends
                if attempt == args.retries - 1:
                    raise
                wait = 15 * (attempt + 1)
                print(f"  retry {attempt + 1} after {type(exc).__name__}: {str(exc)[:120]} "
                      f"(sleeping {wait}s)", flush=True)
                time.sleep(wait)

        for row, logits in zip(batch, saved_logits):
            vector = logits.float()
            a, b = float(vector[letters["A"]]), float(vector[letters["B"]])
            target = row.get("target_letter")
            margin = (a - b) if target == "A" else (b - a)
            outputs.append({
                "kind": row["kind"], "item_id": row["item_id"], "unit_id": row["unit_id"],
                "episode_family": row["episode_family"],
                "question_family": row.get("question_family"),
                "condition": row["condition"], "query_status": row["query_status"],
                "target_letter": target, "logit_a": a, "logit_b": b,
                "target_logit_margin": margin,
                "p_target": float(torch.softmax(torch.tensor([a, b]), 0)[0 if target == "A" else 1]),
            })
        for layer in want_layers:
            for row, hidden in zip(batch, saved_hidden[layer]):
                acts.setdefault(layer, {})[row["item_id"]] = hidden.float().tolist()
        print(f"  {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    payload = {
        "schema_version": 1, "model_key": "llama31_405b", "model": MODEL_ID,
        "precision": "ndif-hosted", "load_in_8bit": False,
        "items": str(args.items),
        "items_sha256": hashlib.sha256(args.items.read_bytes()).hexdigest(),
        "n_items": len(outputs), "n_layers": n_layers,
        "candidate_token_ids": letters, "rows": outputs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)
    if acts:
        import numpy as np
        ids = [r["item_id"] for r in outputs]
        stack = np.stack([[acts[layer][i] for layer in want_layers] for i in ids]).astype(np.float16)
        npz = args.out.with_suffix(".activations.npz")
        np.savez_compressed(
            npz, X=stack, item_ids=np.asarray(ids), layers=np.asarray(want_layers),
            episode_family=np.asarray([r["episode_family"] for r in outputs]),
            condition=np.asarray([r["condition"] for r in outputs]),
            query_status=np.asarray([r["query_status"] for r in outputs]),
            query_token=np.asarray([r["condition"].split("_")[-1] for r in outputs]),
        )
        print("wrote", npz, stack.shape)


if __name__ == "__main__":
    main()
