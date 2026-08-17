"""Fit the pinned Jacobian lens on the frozen generic corpus."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlens-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--source-layers", type=int, nargs="+", required=True)
    ap.add_argument("--target-layer", type=int, required=True)
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.jlens_dir))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.fitting import fit
    from jlens.hf import from_hf
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                               local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    lens_model = from_hf(model, tokenizer, compile=False, force_bos=True)
    records = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    prompts = [r["text"] for r in records]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "fit_checkpoint.pt"
    lens = fit(lens_model, prompts, source_layers=args.source_layers,
               target_layer=args.target_layer, dim_batch=args.dim_batch,
               max_seq_len=args.max_seq_len, skip_first=args.skip_first,
               checkpoint_path=str(checkpoint), checkpoint_every=25, resume=True)
    lens_path = args.out_dir / "jacobian_lens.pt"
    lens.save(str(lens_path))
    manifest = {"schema_version": 1, "implementation": "anthropics/jacobian-lens",
                "jlens_dir": str(args.jlens_dir), "model": args.model,
                "revision": args.revision, "source_layers": args.source_layers,
                "target_layer": args.target_layer, "dim_batch": args.dim_batch,
                "max_seq_len": args.max_seq_len, "skip_first": args.skip_first,
                "seed": args.seed, "rows": len(records), "input_sha256": sha256(args.input),
                "lens_sha256": sha256(lens_path), "fit_status": "complete"}
    (args.out_dir / "fit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))

if __name__ == "__main__":
    main()
