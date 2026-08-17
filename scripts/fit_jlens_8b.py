"""Fit and validate the pinned Jacobian lens on frozen generic text."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tensor_stats(tensor):
    import torch
    finite = bool(torch.isfinite(tensor).all().item())
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": finite,
            "min": float(tensor.min().item()), "max": float(tensor.max().item()),
            "max_abs": float(tensor.abs().max().item()), "frobenius": float(torch.linalg.vector_norm(tensor).item()),
            "nonfinite": int((~torch.isfinite(tensor)).sum().item())}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlens-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--source-layers", type=int, nargs="+", required=True)
    ap.add_argument("--target-layer", type=int, required=True)
    ap.add_argument("--dim-batch", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--heldout-text", default=("A harmless held-out paragraph about a garden path and a blue cup. "
                                                "It contains enough ordinary text to exercise the fitted lens on "
                                                "a valid sequence beyond the skipped prefix."))
    args = ap.parse_args()
    sys.path.insert(0, str(args.jlens_dir))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.fitting import fit
    from jlens.hf import from_hf
    from jlens.lens import JacobianLens
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lens_model = from_hf(model, tokenizer, compile=False, force_bos=True)
    all_records = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    records = all_records[:args.max_prompts] if args.max_prompts is not None else all_records
    prompts = [r["text"] for r in records]
    if not prompts:
        raise ValueError("no prompts")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "fit_checkpoint.pt"
    lens = fit(lens_model, prompts, source_layers=args.source_layers,
               target_layer=args.target_layer, dim_batch=args.dim_batch,
               max_seq_len=args.max_seq_len, skip_first=args.skip_first,
               checkpoint_path=str(checkpoint), checkpoint_every=5, resume=True)
    if lens.n_prompts <= 0 or lens.n_prompts > len(prompts):
        raise RuntimeError(f"invalid fitted prompt count {lens.n_prompts}")
    pre_stats = {str(layer): tensor_stats(J) for layer, J in lens.jacobians.items()}
    if not all(x["finite"] and x["frobenius"] > 0 for x in pre_stats.values()):
        raise RuntimeError("nonfinite or zero Jacobian before save")
    lens_path = args.out_dir / "lens.pt"
    lens.save(str(lens_path), dtype=torch.float32)
    raw = torch.load(lens_path, map_location="cpu", weights_only=True)
    raw_stats = {str(layer): tensor_stats(J) for layer, J in raw["J"].items()}
    if not all(x["dtype"] == "torch.float32" and x["finite"] for x in raw_stats.values()):
        raise RuntimeError("saved lens is not finite FP32")
    reloaded = JacobianLens.load(str(lens_path))
    reload_stats = {str(layer): tensor_stats(J) for layer, J in reloaded.jacobians.items()}
    if not all(x["finite"] for x in reload_stats.values()):
        raise RuntimeError("reloaded lens is nonfinite")
    layers_for_apply = [args.source_layers[0]]
    lens_logits, model_logits, input_ids = reloaded.apply(
        lens_model, args.heldout_text, layers=layers_for_apply, positions=[-1],
        max_seq_len=args.max_seq_len, use_jacobian=True)
    apply_ok = bool(torch.isfinite(lens_logits[layers_for_apply[0]]).all().item() and
                    torch.isfinite(model_logits).all().item() and input_ids.shape[1] > args.skip_first + 1)
    if not apply_ok:
        raise RuntimeError("lens apply validation failed")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"
    manifest = {
        "schema_version": 2, "status": "complete", "implementation": "anthropics/jacobian-lens",
        "jlens_dir": str(args.jlens_dir), "model": args.model, "tokenizer": args.tokenizer,
        "revision": args.revision, "source_layers": args.source_layers, "target_layer": args.target_layer,
        "dim_batch": args.dim_batch, "max_seq_len": args.max_seq_len, "skip_first": args.skip_first,
        "seed": args.seed, "n_input_records": len(records), "n_prompts_fitted": lens.n_prompts,
        "input_sha256": sha256(args.input), "record_ids": [r.get("id") for r in records],
        "lens_sha256": sha256(lens_path), "checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else None,
        "pre_save_stats": pre_stats, "saved_raw_stats": raw_stats, "reload_stats": reload_stats,
        "reload_exact": all(torch.equal(lens.jacobians[k], reloaded.jacobians[k]) for k in lens.jacobians),
        "apply_finite": apply_ok, "environment": {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "git_commit": git_commit,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))

if __name__ == "__main__":
    main()
