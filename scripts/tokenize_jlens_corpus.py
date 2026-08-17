"""Tokenize the frozen generic J-lens corpus with the cached Llama tokenizer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--max-seq-len", type=int, default=128)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True,
                                        use_fast=False, add_special_tokens=False)
    records = [json.loads(x) for x in args.raw.read_text(encoding="utf-8").splitlines() if x.strip()]
    out_records = []
    for r in records:
        ids = tok.encode(r["text"], add_special_tokens=False, truncation=True, max_length=args.max_seq_len)
        if len(ids) <= 17:
            raise ValueError(f"tokenized sequence too short after skip_first: {r['id']}")
        out_records.append({"dataset_row_idx": r["dataset_row_idx"], "id": r["id"], "input_ids": ids,
                            "n_tokens": len(ids)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in out_records), encoding="utf-8")
    manifest = {"schema_version": 1, "raw_sha256": sha256(args.raw), "tokenized_sha256": sha256(args.out),
                "model": args.model, "revision": args.revision, "tokenizer": str(tok.name_or_path),
                "vocab_size": int(tok.vocab_size), "rows": len(out_records), "max_seq_len": args.max_seq_len,
                "skip_first": 16, "min_tokens": min(x["n_tokens"] for x in out_records)}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))

if __name__ == "__main__":
    main()
