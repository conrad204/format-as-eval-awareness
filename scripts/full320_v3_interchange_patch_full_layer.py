#!/usr/bin/env python3
"""Real, hooked, full-layer interchange-intervention (activation patching) test
for whether the matched-format subspace Q_format(r) is a functionally causal
separation between purpose and format, not just a geometric one.

Development-tier script (no frozen pilot80-plan/protocol ceremony -- Q_format(r)
and the purpose/format readout probes are refit on the fly per held-out-family
fold, same convention as scripts/full320_v3_erasure_dose_response_bootstrap.py
and scripts/full320_v3_geometry_full_layer_sweep.py). Real forward passes on
real bf16 model weights via register_forward_hook, same low-level mechanism as
scripts/run_full320_orthc_transfer.py (imported from
scripts/run_pilot80_factorial_causal.py as `base`), just with a different hook
body: a subspace-projection SWAP using a REAL matched counterpart's content,
instead of an additive synthetic steering vector.

Method
------
The 2x2 factorial design gives, for free, matched (payload_block_id,
purpose_pair_id, intended_purpose) pairs that differ ONLY in format
(benchmark vs. casual). For a held-out test row i and its matched counterpart
j (same purpose, opposite format), at transformer layer L:

  h_i' = h_i - Q Q^T h_i + Q Q^T h_j        (real_swap: Q = Q_format(r))
  h_i' = h_i - R R^T h_i + R R^T h_j        (random_control: R = random rank-r
                                              orthonormal basis, same real h_j)

Both conditions swap the SAME real content (j's actual activation) through the
SAME rank-r operation -- the only thing that differs is which r-dimensional
subspace is swapped. This isolates "is it specifically Q_format(r) that
matters, or would swapping any r-dim slice of real content produce a similar
effect."

The patched residual is injected via register_forward_hook at layer L and the
network runs FORWARD FOR REAL from L to the final layer (register_forward_hook
on the target layer captures the resulting hidden state) -- this is a real
causal intervention on the model's own computation, not a post-hoc linear
readout on a frozen vector.

Read out at the final layer with purpose/format probes (StandardScaler +
LogisticRegression, PROBE hyperparameters from format_erasure_development.py)
fit fresh on the TRAIN split of a leave-one-purpose-family-out fold. Baseline
(unpatched) scores are computed directly from the pre-extracted static
activation for row i (no redundant unpatched forward pass needed -- verified
against a live unhooked forward pass for a handful of rows at the start of
each run as a cheap end-to-end reproduction sanity check).

Two predictions, both required for a mechanism claim (see
scripts/full320_v3_interchange_patch_bootstrap.py for the corresponding
correlation-aware analysis):
  H1 (purpose is protected): |purpose_shift| under real_swap < under
     random_control (swapping the identified subspace disrupts the purpose
     readout LESS than swapping a generic same-rank slice of the same real
     content).
  H2 (format is carried): format interchange-intervention accuracy (does the
     format readout move, in sign, toward the counterpart's own baseline
     format score) is HIGHER under real_swap than under random_control.

Usage:
  python scripts/full320_v3_interchange_patch_full_layer.py \\
      --model llama31_8b --rank 16 \\
      --out results/full320_v3_interchange_patch_llama31_8b.jsonl
  python scripts/full320_v3_interchange_patch_full_layer.py \\
      --model llama31_8b --rank 16 --layers 0,15,31 --max-pairs-per-fold 4 \\
      --out /tmp/smoke.jsonl   # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_erasure_development import PROBE, factorial_vectors, groups, orth_basis  # noqa: E402
from run_pilot80_factorial_causal import (  # noqa: E402
    atomic_json, cosine, load_rows, output_hidden, pad_left, replace_hidden, resolve_layers, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "v3-activations" / "rendered_items_v3_full320_8-17-2026.jsonl"

MODELS = {
    "llama31_8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "checkpoint": "/scratch/jppatton/langea/confirmation_work/llama31_8b_model_0e9e39f2",
        "tokenizer": "/scratch/jppatton/langea/confirmation_work/llama31_8b_tokenizer_0e9e39f2",
        "npz": ROOT / "v3-activations/activations/llama31_8b/activations_all_layers.npz",
        "n_layers": 32, "hidden": 4096,
    },
    "llama31_70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "checkpoint": "/scratch/jppatton/langea/huggingface/transformers/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b",
        "tokenizer": "/scratch/jppatton/langea/huggingface/transformers/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b",
        "npz": ROOT / "v3-activations/activations/llama31_70b/activations_all_layers.npz",
        "n_layers": 80, "hidden": 8192,
    },
    "llama33_70b": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "checkpoint": "/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "tokenizer": "/scratch/jppatton/langea/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "npz": ROOT / "v3-activations/activations/llama33_70b/activations_all_layers.npz",
        "n_layers": 80, "hidden": 8192,
    },
}
DEFAULT_RANK = 16
RANDOM_SUBSPACE_SEED_BASE = 20260829


def fit_probe_frozen(X: np.ndarray, y: np.ndarray) -> dict:
    sc = StandardScaler().fit(X)
    m = LogisticRegression(**PROBE).fit(sc.transform(X), y)
    return {"mean": sc.mean_, "scale": sc.scale_, "coef": m.coef_[0], "intercept": float(m.intercept_[0])}


def score_frozen(probe: dict, x: np.ndarray) -> float:
    return float(np.dot((x - probe["mean"]) / probe["scale"], probe["coef"]) + probe["intercept"])


def random_orth_basis(d: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, rank)))
    return q[:, :rank]


def matched_pairs(rows: list[dict]) -> list[dict]:
    """All (source_row_index, target_row_index) pairs sharing (payload_block_id,
    purpose_pair_id, intended_purpose) and differing only in format, both
    directions (benchmark->casual and casual->benchmark)."""
    stated_mask = np.array([r["render_slot"] == "stated" for r in rows])
    gs = groups(rows, stated_mask, np.ones(len(rows), bool))
    pairs = []
    for _, eb, ec, db, dc in gs:
        eb, ec, db, dc = int(eb), int(ec), int(db), int(dc)
        pairs.append({"src": eb, "tgt": ec, "direction": "b2c", "intended_purpose": "evaluation"})
        pairs.append({"src": ec, "tgt": eb, "direction": "c2b", "intended_purpose": "evaluation"})
        pairs.append({"src": db, "tgt": dc, "direction": "b2c", "intended_purpose": "deployment"})
        pairs.append({"src": dc, "tgt": db, "direction": "c2b", "intended_purpose": "deployment"})
    return pairs


def sanity_check_reproduction(model, layers, tokenizer, rows, encoded, device, stored_final, target_layer, torch):
    """One unhooked forward pass per a handful of rows: confirms the checkpoint,
    revision, dtype, and chat-template convention reproduce the pre-extracted
    static activations before spending any GPU time on the sweep.

    Tolerance is scale-relative, not a fixed absolute threshold: this is a
    cross-run (different script, session, possibly different accelerate
    device placement) reproduction check, not the frozen-protocol pipeline's
    tighter within-run bit-reproducibility gate (whose fixed 0.125 absolute /
    0.999 cosine bar assumes a single deterministic run reproducing itself).
    bf16 accumulates real reduction-order noise across 32-80 layers even
    across two otherwise-identical forward passes; on activations with L2
    norm ~40 this can reach an absolute max-elementwise difference of ~0.2
    while cosine similarity still sits at 0.9998 -- cosine is the metric that
    actually discriminates "same model/checkpoint" from a real mismatch
    (which drops cosine dramatically, not marginally).
    """
    sample = list(encoded)[:4]
    capture = {}

    def hook(_m, _i, output):
        capture["h"] = output_hidden(output)[:, -1, :].detach().float().cpu().numpy()
        return output

    handle = layers[target_layer].register_forward_hook(hook)
    try:
        for index in sample:
            input_ids, attention_mask = pad_left([encoded[index]], tokenizer.pad_token_id, torch, device)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            actual = capture["h"][0]
            expected = stored_final[index]
            rel_max_abs = float(np.max(np.abs(actual - expected)) / max(np.linalg.norm(expected), 1e-6))
            cos = cosine(actual, expected)
            if rel_max_abs > 0.02 or cos < 0.995:
                raise RuntimeError(f"reproduction sanity check failed row={index} rel_max_abs={rel_max_abs} cosine={cos}")
    finally:
        handle.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--layers", default=None, help="comma-separated layer indices; default = all layers")
    ap.add_argument("--max-pairs-per-fold", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--conditions", default="real_swap,random_control",
                     help="comma-separated subset of real_swap,random_control,matched_control. "
                          "matched_control: fresh random-direction ADDITIVE perturbation, magnitude-"
                          "matched per-row to that row's real_swap delta norm (computed offline from "
                          "static activations, no extra GPU cost) -- isolates whether real_swap's "
                          "purpose disruption is specific to Q_format(r) or just perturbation size.")
    args = ap.parse_args()
    conditions = args.conditions.split(",")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    cfg = MODELS[args.model]
    n_layers, d = cfg["n_layers"], cfg["hidden"]
    target_layer = n_layers - 1
    rank = args.rank
    layer_list = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_layers))

    rows_all = load_rows(ITEMS_PATH)
    with np.load(cfg["npz"], allow_pickle=False) as arc:
        item_ids_all = arc["item_ids"].astype(str).tolist()
        assert item_ids_all == [r["item_id"] for r in rows_all], "activation/item order mismatch"
        X_all_raw = arc["X"]  # (n_items, n_layers, hidden), float32

    # Drop unlabeled no_cue rows (render_slot != "stated") before building fam/labels/
    # folds/probe-training data -- these have intended_purpose="none" and would
    # otherwise silently collapse into the "deployment" class and contaminate every
    # held-out-family probe fit (matched_pairs() already restricts pairs to stated
    # rows internally, but fam/y_purpose/y_format/folds must be filtered too).
    keep_mask = np.array([r["render_slot"] == "stated" for r in rows_all])
    rows = [r for r, keep in zip(rows_all, keep_mask) if keep]
    X_all = X_all_raw[keep_mask]

    fam = np.array([r["purpose_family_id"] for r in rows])
    y_purpose = np.array([int(r["intended_purpose"] == "evaluation") for r in rows])
    y_format = np.array([int(r["format"] == "benchmark") for r in rows])
    folds = sorted(set(fam))
    pairs = matched_pairs(rows)
    print(f"model={args.model} n_layers={n_layers} rank={rank} n_pairs={len(pairs)} n_folds={len(folds)}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["layer"], r["fold"], r["src_row"], r["tgt_row"], r["condition"]))
    elif out_path.exists() and not args.resume:
        raise FileExistsError(f"{out_path} exists; pass --resume or remove it")

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["checkpoint"], local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, attn_implementation="eager",
    )
    model.eval()
    layers = resolve_layers(model)
    assert len(layers) == n_layers
    device = model.get_input_embeddings().weight.device

    all_needed_rows = sorted({p["src"] for p in pairs} | {p["tgt"] for p in pairs})
    encoded = {}
    for index in all_needed_rows:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": rows[index]["text"]}], tokenize=True,
            add_generation_prompt=True, add_special_tokens=False,
        )
        encoded[index] = [int(t) for t in (tokens.tolist() if hasattr(tokens, "tolist") else tokens)]

    stored_final = X_all[:, target_layer, :].astype(np.float32)
    print("running reproduction sanity check...", file=sys.stderr, flush=True)
    sanity_check_reproduction(model, layers, tokenizer, rows, encoded, device, stored_final, target_layer, torch)
    print("sanity check passed", file=sys.stderr, flush=True)

    started = time.time()
    n_written = 0
    for layer in layer_list:
        Xl = X_all[:, layer, :].astype(np.float64)
        for fold in folds:
            train_mask = fam != fold
            test_fold_pairs = [p for p in pairs if fam[p["src"]] == fold]
            if args.max_pairs_per_fold is not None:
                test_fold_pairs = test_fold_pairs[: args.max_pairs_per_fold]
            if not test_fold_pairs:
                continue
            todo = [p for p in test_fold_pairs
                    if any((layer, fold, p["src"], p["tgt"], c) not in done_keys for c in conditions)]
            if not todo:
                continue

            _, B_train, _, _ = factorial_vectors(Xl, rows, train_mask)
            Q = orth_basis(B_train, rank)
            R = random_orth_basis(d, Q.shape[1], seed=RANDOM_SUBSPACE_SEED_BASE + layer * 1009 + fold_hash(fold))
            purpose_probe = fit_probe_frozen(X_all[train_mask, target_layer, :].astype(np.float64), y_purpose[train_mask])
            format_probe = fit_probe_frozen(X_all[train_mask, target_layer, :].astype(np.float64), y_format[train_mask])

            offset = 0
            while offset < len(todo):
                batch = todo[offset: offset + args.batch_size]
                batch_offset = offset
                offset += len(batch)
                src_h_np = np.stack([X_all[p["src"], layer, :] for p in batch]).astype(np.float64)
                tgt_h_np = np.stack([X_all[p["tgt"], layer, :] for p in batch]).astype(np.float64)
                real_delta_np = (tgt_h_np - src_h_np) @ Q @ Q.T
                for condition in conditions:
                    if condition == "real_swap":
                        delta_np = real_delta_np
                    elif condition == "random_control":
                        delta_np = (tgt_h_np - src_h_np) @ R @ R.T
                    elif condition == "matched_control":
                        # Fresh random ADDITIVE direction per row, magnitude-matched to
                        # this row's own real_swap delta norm (computed offline from
                        # static activations -- no extra GPU cost). Isolates whether
                        # real_swap's purpose disruption is specific to Q_format(r) or
                        # just a function of perturbation size.
                        real_delta_norm = np.linalg.norm(real_delta_np, axis=-1, keepdims=True)
                        rng_row = np.random.default_rng(
                            RANDOM_SUBSPACE_SEED_BASE + layer * 100003 + fold_hash(fold) * 131 + batch_offset)
                        u = rng_row.standard_normal(real_delta_np.shape)
                        u /= np.linalg.norm(u, axis=-1, keepdims=True)
                        delta_np = u * real_delta_norm
                    else:
                        raise ValueError(f"unknown condition {condition}")
                    delta_t = torch.as_tensor(delta_np.astype(np.float32), device=device)
                    src_indices = [p["src"] for p in batch]
                    input_ids, attention_mask = pad_left([encoded[i] for i in src_indices], tokenizer.pad_token_id, torch, device)
                    capture: dict[str, Any] = {"h": None}

                    def source_hook(_m, _i, output, _delta=delta_t):
                        hidden = output_hidden(output)
                        modified = hidden.clone()
                        last = modified[:, -1, :].to(torch.float32)
                        modified[:, -1, :] = (last + _delta.to(last.device)).to(hidden.dtype)
                        return replace_hidden(output, modified)

                    def target_hook(_m, _i, output):
                        capture["h"] = output_hidden(output)[:, -1, :].detach().float().cpu().numpy()
                        return output

                    src_handle = layers[layer].register_forward_hook(source_hook)
                    tgt_handle = layers[target_layer].register_forward_hook(target_hook)
                    try:
                        with torch.inference_mode():
                            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                    finally:
                        src_handle.remove()
                        tgt_handle.remove()

                    records = []
                    for bi, p in enumerate(batch):
                        i, j = p["src"], p["tgt"]
                        patched = capture["h"][bi]
                        if not np.isfinite(patched).all():
                            raise ValueError(f"nonfinite patched activation layer={layer} fold={fold} src={i}")
                        purpose_patched = score_frozen(purpose_probe, patched)
                        format_patched = score_frozen(format_probe, patched)
                        purpose_base_i = score_frozen(purpose_probe, stored_final[i])
                        format_base_i = score_frozen(format_probe, stored_final[i])
                        format_base_j = score_frozen(format_probe, stored_final[j])
                        records.append({
                            "model": args.model, "layer": layer, "target_layer": target_layer,
                            "rank": rank, "fold": fold, "condition": condition,
                            "direction": p["direction"], "intended_purpose": p["intended_purpose"],
                            "src_row": i, "tgt_row": j,
                            "src_item_id": rows[i]["item_id"], "tgt_item_id": rows[j]["item_id"],
                            "payload_block_id": rows[i]["payload_block_id"],
                            "purpose_score_patched": purpose_patched, "purpose_score_baseline": purpose_base_i,
                            "format_score_patched": format_patched, "format_score_baseline_src": format_base_i,
                            "format_score_baseline_tgt": format_base_j,
                        })
                    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
                        for r in records:
                            fh.write(json.dumps(r) + "\n")
                    n_written += len(records)
            print(f"  layer={layer} fold={fold} written={n_written} elapsed={time.time()-started:.0f}s", file=sys.stderr, flush=True)

    manifest_path = out_path.with_suffix(".manifest.json")
    atomic_json(manifest_path, {
        "schema_version": 1, "mode": "development", "model": args.model, "model_id": cfg["model_id"],
        "n_layers": n_layers, "rank": rank, "layers_run": layer_list,
        "activation_sha256": sha256_file(cfg["npz"]), "items_sha256": sha256_file(ITEMS_PATH),
        "script_sha256": sha256_file(Path(__file__)), "n_records": n_written,
        "elapsed_seconds": time.time() - started,
    })
    print(f"wrote {out_path} ({n_written} records) and {manifest_path}")


def fold_hash(fold: str) -> int:
    return abs(hash(fold)) % 9973


if __name__ == "__main__":
    main()
