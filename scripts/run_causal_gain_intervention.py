"""Run a small development-only residual intervention and J first-order check."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

LAYERS = (4, 12, 15, 24)
CONDITIONS = ("full_purpose", "top_j_99", "low_gain_99", "format", "random")
ALPHAS = (0.25, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--normalization", choices=("equal_residual", "equal_predicted_gain"), default="equal_residual")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["render_slot"] == "no_cue"]
    rows = sorted(rows, key=lambda r: r["item_id"])[: args.max_items]
    vectors_np = np.load(args.vectors)
    layers = [int(x) for x in vectors_np["layers"]]
    names = [str(x) for x in vectors_np["names"]]
    vectors = vectors_np["vectors"]
    lens_data = torch.load(args.lens, map_location="cpu", weights_only=False)
    jacobians = {int(k): torch.as_tensor(v, dtype=torch.float32) for k, v in lens_data["J"].items()}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    model.config.use_cache = False
    device = next(model.parameters()).device
    eval_ids = tokenizer.encode(" evaluation", add_special_tokens=False)
    deploy_ids = tokenizer.encode(" deployment", add_special_tokens=False)
    if len(eval_ids) != 1 or len(deploy_ids) != 1:
        raise ValueError(f"choice labels must be single tokens: {eval_ids}, {deploy_ids}")
    eval_id, deploy_id = eval_ids[0], deploy_ids[0]
    records = []
    for row in rows:
        encoded = tokenizer(row["text"], return_tensors="pt", add_special_tokens=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        baseline_target = {}
        baseline_margin = None
        with torch.no_grad():
            def capture_target(module, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                baseline_target["h"] = value[:, -1, :].detach().float()
            h = model.model.layers[31].register_forward_hook(capture_target)
            out = model(**encoded)
            h.remove()
            logits = out.logits[0, -1].float()
            baseline_margin = float((logits[eval_id] - logits[deploy_id]).item())
        for layer_index, layer in enumerate(layers):
            for condition_index, condition in enumerate(names):
                raw = torch.as_tensor(vectors[layer_index, condition_index], dtype=torch.float32, device=device)
                raw = raw / torch.linalg.vector_norm(raw)
                gain = float(torch.linalg.vector_norm(jacobians[layer].to(device) @ raw).item())
                full_raw = torch.as_tensor(vectors[layer_index, 0], dtype=torch.float32, device=device)
                full_raw = full_raw / torch.linalg.vector_norm(full_raw)
                full_gain = float(torch.linalg.vector_norm(jacobians[layer].to(device) @ full_raw).item())
                for alpha in ALPHAS:
                    scale = alpha if args.normalization == "equal_residual" else alpha * full_gain / gain
                    delta = raw * scale
                    captured = {}
                    def intervene(module, inputs, output, delta=delta):
                        value = output[0] if isinstance(output, tuple) else output
                        value = value.clone()
                        value[:, -1, :] = value[:, -1, :] + delta.to(value.dtype)
                        return (value,) + output[1:] if isinstance(output, tuple) else value
                    def capture(module, inputs, output):
                        value = output[0] if isinstance(output, tuple) else output
                        captured["h"] = value[:, -1, :].detach().float()
                    hs = model.model.layers[layer].register_forward_hook(intervene)
                    ht = model.model.layers[31].register_forward_hook(capture)
                    with torch.no_grad():
                        out_i = model(**encoded)
                    hs.remove(); ht.remove()
                    li = out_i.logits[0, -1].float()
                    actual_margin_delta = float((li[eval_id] - li[deploy_id]).item() - baseline_margin)
                    predicted_h = jacobians[layer] @ delta.detach().cpu()
                    actual_h = captured["h"][0].cpu() - baseline_target["h"][0].cpu()
                    pred_norm = float(torch.linalg.vector_norm(predicted_h).item())
                    actual_norm = float(torch.linalg.vector_norm(actual_h).item())
                    records.append({"item_id": row["item_id"], "payload_block_id": row["payload_block_id"], "layer": layer, "condition": condition, "alpha": alpha, "baseline_margin": baseline_margin, "margin_delta": actual_margin_delta, "predicted_target_norm": pred_norm, "actual_target_norm": actual_norm, "prediction_cosine": float(torch.dot(predicted_h, actual_h).item() / (pred_norm * actual_norm)) if pred_norm and actual_norm else None, "norm_ratio": actual_norm / pred_norm if pred_norm else None, "first_order_gain": gain, "full_gain_reference": full_gain})
        print(f"completed {row['item_id']}", flush=True)
    out = {"schema_version": 1, "status": "development_only", "normalization": args.normalization, "model": args.model, "revision": args.revision, "layers": layers, "conditions": names, "alphas": ALPHAS, "n_items": len(rows), "items": str(args.items), "vectors": str(args.vectors), "lens": str(args.lens), "choice_tokens": {"evaluation": eval_id, "deployment": deploy_id}, "records": records, "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_records": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
