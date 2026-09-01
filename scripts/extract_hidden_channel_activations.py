"""Extract last-prompt-token activations for the frozen hidden-channel prompts.

Only the six mapping-relevant action conditions are extracted. Status is
orthogonal to the visible channel letter by construction (map0_K is closed with
letter K, map1_K is active with the same letter), so a probe that reads the letter
alone cannot decode status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONDITIONS = ("map0_K", "map0_M", "map1_K", "map1_M", "scrambled_K", "scrambled_M")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-cpu-memory", default=None)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256(args.items) != manifest["items_sha256"]:
        raise RuntimeError("hidden-channel bank hash mismatch")
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line]
    subset = [r for r in rows if r["kind"] == "action" and r["condition"] in CONDITIONS]
    if len(subset) != 32 * 16 * len(CONDITIONS):
        raise RuntimeError(f"unexpected extraction subset size {len(subset)}")

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    max_memory = None
    if args.max_cpu_memory:
        max_memory = {i: "74GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = args.max_cpu_memory
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        local_files_only=True, low_cpu_mem_usage=True, max_memory=max_memory)
    model.eval()
    device = model.get_input_embeddings().weight.device
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # `output_hidden_states=True` materialises every layer for every position, which
    # is ~7 GB per batch of 8 at 80 layers x 8192 hidden and OOMs an 80 GB card once
    # offloaded weights are resident. Forward hooks keep only the final position.
    store = np.empty((len(subset), n_layers, hidden), dtype=np.float16)
    blocks = model.model.layers
    if len(blocks) != n_layers:
        raise RuntimeError(f"expected {n_layers} decoder blocks, found {len(blocks)}")
    captured: dict[int, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[index] = tensor[:, -1, :].detach().to("cpu", torch.float32)
        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(blocks)]
    try:
        for start in range(0, len(subset), args.batch_size):
            batch = subset[start:start + args.batch_size]
            texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            captured.clear()
            with torch.inference_mode():
                model(**enc, use_cache=False)
            if len(captured) != n_layers:
                raise RuntimeError(f"captured {len(captured)} of {n_layers} layers")
            stack = torch.stack([captured[i] for i in range(n_layers)], dim=1)
            store[start:start + len(batch)] = stack.numpy().astype(np.float16)
            print(f"extracted {min(start + args.batch_size, len(subset))}/{len(subset)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz = args.out_dir / f"hidden_channel_activations_{args.model_key}.npz"
    np.savez_compressed(
        npz, X=store,
        item_ids=np.asarray([r["item_id"] for r in subset]),
        episode_family=np.asarray([r["episode_family"] for r in subset]),
        arithmetic_item=np.asarray([r["arithmetic_item"] for r in subset]),
        condition=np.asarray([r["condition"] for r in subset]),
        query_token=np.asarray([r["query_token"] for r in subset]),
        query_status=np.asarray([r["query_status"] for r in subset]),
    )
    meta = {
        "model_key": args.model_key, "model": args.model, "precision": "bfloat16",
        "items_sha256": manifest["items_sha256"], "n_rows": len(subset),
        "n_layers": n_layers, "hidden_size": hidden, "dtype_stored": "float16",
        "position": "last_prompt_token_after_chat_template_before_generation",
        "layer_semantics": "transformer_block_outputs_1_to_n_excluding_embedding_stream",
        "conditions": list(CONDITIONS), "activations_sha256": sha256(npz),
    }
    (args.out_dir / f"hidden_channel_activations_{args.model_key}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
