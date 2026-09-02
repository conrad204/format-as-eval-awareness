"""Collect last-prompt-token activations for a frozen hidden-channel bank on 405B.

Writes exactly the format the project's analysis scripts read, so nothing needs
adapting afterwards:

    <out-dir>/hidden_channel_activations_<tag>.npz        X, item_ids, episode_family,
                                                          arithmetic_item, condition,
                                                          query_token, query_status, layers
    <out-dir>/hidden_channel_activations_<tag>.meta.json  hashes, layer list, semantics

Two routes, same output:

    --route local   transformers + forward hooks, needs the weights on disk
    --route ndif    nnsight against NDIF, needs NDIF_API_KEY and network access

Only `action` rows are used. The probed position is the final prompt token after
the chat template and before any generation, and the stored layer values are
transformer-block outputs (zero-based, before the final model norm) -- identical
semantics to the 8B and 70B extractions this will be compared against.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-405B-Instruct"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(items: Path, conditions: tuple[str, ...]) -> list[dict]:
    rows = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines() if line.strip()]
    subset = [r for r in rows if r["kind"] == "action" and r["condition"] in conditions]
    if not subset:
        raise SystemExit(f"no action rows with conditions {conditions} in {items}")
    families = {r["episode_family"] for r in subset}
    arith = {r.get("arithmetic_item") for r in subset}
    expected = len(families) * len(arith) * len(conditions)
    if len(subset) != expected:
        raise SystemExit(f"crossing incomplete: {len(subset)} rows, expected {expected}")
    print(f"{len(subset)} action rows | {len(families)} families x {len(arith)} items "
          f"x {len(conditions)} conditions", flush=True)
    return subset


def write_output(out_dir: Path, tag: str, rows: list[dict], layers: list[int], X: np.ndarray,
                 model_id: str, items: Path, n_layers_total: int, hidden: int, route: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"hidden_channel_activations_{tag}.npz"
    np.savez_compressed(
        npz,
        X=X.astype(np.float16),
        item_ids=np.asarray([f"{r['episode_family']}::{r['arithmetic_item']}::{r['condition']}"
                             for r in rows]),
        episode_family=np.asarray([r["episode_family"] for r in rows]),
        arithmetic_item=np.asarray([r["arithmetic_item"] for r in rows]),
        condition=np.asarray([r["condition"] for r in rows]),
        query_token=np.asarray([r.get("query_token", "") for r in rows]),
        query_status=np.asarray([r["query_status"] for r in rows]),
        layers=np.asarray(layers),
    )
    meta = {
        "model_key": tag, "model": model_id, "route": route,
        "precision": "bfloat16 forward, float16 stored",
        "items": items.name, "items_sha256": sha256(items),
        "n_rows": int(X.shape[0]), "n_layers": n_layers_total,
        "stored_layers": layers, "hidden_size": hidden,
        "dtype_stored": "float16",
        "position": "last_prompt_token_after_chat_template_before_generation",
        "layer_semantics": "transformer_block_outputs_1_to_n_excluding_embedding_stream",
        "conditions": sorted({r["condition"] for r in rows}),
        "activations_sha256": sha256(npz),
    }
    (out_dir / f"hidden_channel_activations_{tag}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("wrote", npz, X.shape, flush=True)


def run_local(rows: list[dict], model_path: str, layers_arg: str, batch_size: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    layers = resolve_layers(layers_arg, n_layers)
    device = model.get_input_embeddings().weight.device

    store = np.empty((len(rows), len(layers), hidden), dtype=np.float16)
    captured: dict[int, "torch.Tensor"] = {}

    def make_hook(index: int):
        def hook(_m, _i, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[index] = tensor[:, -1, :].detach().to(torch.float32).cpu()
        return hook

    handles = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in layers]
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [tok.apply_chat_template(r["messages"], tokenize=False,
                                             add_generation_prompt=True) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            captured.clear()
            with torch.no_grad():
                model(**enc)
            if len(captured) != len(layers):
                raise RuntimeError(f"captured {len(captured)} of {len(layers)} layers")
            stack = np.stack([captured[L].numpy() for L in layers], axis=1)
            store[start:start + len(batch)] = stack.astype(np.float16)
            print(f"extracted {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return store, layers, n_layers, hidden, model_path


def run_ndif(rows: list[dict], model_id: str, layers_arg: str, batch_size: int):
    import time
    import torch
    from nnsight import CONFIG, LanguageModel

    key = os.environ.get("NDIF_API_KEY")
    if not key:
        raise SystemExit("set NDIF_API_KEY")
    CONFIG.API.APIKEY = key
    model = LanguageModel(model_id)
    n_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    layers = resolve_layers(layers_arg, n_layers)
    store = np.empty((len(rows), len(layers), hidden), dtype=np.float16)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [model.tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                                     add_generation_prompt=True) for r in batch]
        for attempt in range(4):
            try:
                # nnsight 0.7 runs the session body remotely; per-tensor .save()s
                # collected in a plain dict never reach this frame (UnboundLocalError
                # on Python <3.13). Only a session-level container created via
                # list().save() survives the round trip, so results are appended to
                # it in trace order: one (len(layers), hidden) stack per prompt.
                with model.session(remote=True):
                    acts = list().save()
                    for text in texts:
                        with model.trace(text):
                            outs = [model.model.layers[L].output for L in layers]
                            h = torch.stack(
                                [(o[0] if isinstance(o, (tuple, list)) else o)[0, -1, :].detach().cpu()
                                 for o in outs])
                            acts.append(h)
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                wait = 15 * (attempt + 1)
                print(f"  retry {attempt + 1} after {type(exc).__name__}: "
                      f"{str(exc)[:120]} (sleeping {wait}s)", flush=True)
                time.sleep(wait)
        if len(acts) != len(batch):
            raise RuntimeError(f"session returned {len(acts)} traces for a batch of {len(batch)}")
        for j in range(len(batch)):
            store[start + j] = np.asarray(acts[j].float()).astype(np.float16)
        print(f"extracted {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return store, layers, n_layers, hidden, model_id


def resolve_layers(layers_arg: str, n_layers: int) -> list[int]:
    """`auto` = the pre-declared primary depth plus a coarse trajectory.

    The primary layer matches the relative depth already fixed for the smaller
    checkpoints (L18/80 = 0.225 for 70B, L8/32 = 0.25 for 8B), so it is chosen
    before any 405B outcome is seen. The trajectory layers are descriptive only.
    """
    if layers_arg == "auto":
        primary = round(0.225 * n_layers)
        traj = sorted(set(list(range(0, n_layers, max(1, n_layers // 20))) + [n_layers - 1, primary]))
        print(f"primary layer L{primary} of {n_layers} (relative depth 0.225); "
              f"storing {len(traj)} layers", flush=True)
        return traj
    layers = sorted({int(x) for x in layers_arg.split(",") if x.strip()})
    for L in layers:
        if not 0 <= L < n_layers:
            raise SystemExit(f"layer {L} outside 0..{n_layers - 1}")
    return layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("../out/acts"))
    ap.add_argument("--tag", required=True, help="e.g. llama31_405b or llama31_405b_stated")
    ap.add_argument("--conditions", required=True,
                    help="map0_K,map0_M,map1_K,map1_M for the latent bank; "
                         "explicit_closed,explicit_active for the stated bank")
    ap.add_argument("--route", choices=("local", "ndif"), required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID,
                    help="local weights path, or the NDIF model id")
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the bank and the layer plan without loading the model")
    args = ap.parse_args()

    conditions = tuple(c for c in args.conditions.split(",") if c)
    rows = load_rows(args.items, conditions)
    if args.dry_run:
        for n in (126, 80, 32):
            print(f"  if the model has {n} layers -> {resolve_layers(args.layers, n)}")
        print("dry run OK")
        return
    runner = run_local if args.route == "local" else run_ndif
    X, layers, n_layers, hidden, model_id = runner(rows, args.model, args.layers, args.batch_size)
    write_output(args.out_dir, args.tag, rows, layers, X, model_id, args.items,
                 n_layers, hidden, args.route)


if __name__ == "__main__":
    main()
