"""Upload the local-only extracted activation caches to the format-EA HF org.

These tensors are gitignored (individually over GitHub's 100 MB hard limit) but
are NOT regenerable without re-running GPU extraction, so they belong on the Hub
next to format-EA/full320-v3. Layout mirrors the existing pilot80-v2 /
full320-v3 convention: activations/<model>/<file>.

Repos are created PRIVATE with license: other, viewer: false, matching the other
format-EA datasets. Item banks stay in git (they are small); this script records
their sha256 so the HF copy is bound to the exact bank.

Credentials: HF_TOKEN is read from the process environment or .env.local and is
never printed or written to any artifact.

Usage:
  python scripts/upload_activations_to_hf.py --dataset binding_v1 [--dry-run]
  python scripts/upload_activations_to_hf.py --dataset all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "binding_v1": {
        "repo_id": "format-EA/binding-v1",
        "items": ROOT / "results/binding_v1_frozen/items.jsonl",
        "source_dir": ROOT / "results/binding_v1_activations",
        "files": {
            "binding_v1_8b_all_layers.npz": "activations/llama31_8b/binding_v1_8b_all_layers.npz",
            "binding_v1_8b_config.json": "activations/llama31_8b/config.json",
            "binding_v1_llama31_70b_l18.npz": "activations/llama31_70b/binding_v1_llama31_70b_l18.npz",
            "binding_v1_llama33_70b_l18.npz": "activations/llama33_70b/binding_v1_llama33_70b_l18.npz",
        },
        "title": "binding-v1 — factorial relational-binding activations",
        "blurb": (
            "Frozen activations for the binding_v1 factorial relational-binding experiment "
            "(4,096 items, 8 families). Llama-3.1-8B all 32 layers; Llama-3.1-70B and "
            "Llama-3.3-70B at layer 18."
        ),
    },
    "implicit_v2": {
        "repo_id": "format-EA/implicit-v2",
        "items": ROOT / "results/implicit_v2_frozen_r3/items.jsonl",
        "source_dir": ROOT / "results/implicit_v2_activations",
        "files": {
            "implicit_v2_8b_all_layers.npz": "activations/llama31_8b/implicit_v2_8b_all_layers.npz",
            "implicit_v2_llama31_70b_l18.npz": "activations/llama31_70b/implicit_v2_llama31_70b_l18.npz",
            "implicit_v2_llama33_70b_l18.npz": "activations/llama33_70b/implicit_v2_llama33_70b_l18.npz",
        },
        "title": "implicit-v2 — implicit-purpose scenario activations",
        "blurb": (
            "Frozen activations for the implicit_v2 implicit-purpose scenario battery "
            "(3,840 items). Llama-3.1-8B all 32 layers; Llama-3.1-70B and Llama-3.3-70B "
            "at layer 18."
        ),
    },
}

GITHUB = "https://github.com/conrad204/format-as-eval-awareness"


def load_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env = ROOT / ".env.local"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?HF_TOKEN\s*=\s*(.+)", line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in environment or .env.local")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def readme(cfg: dict, items_sha: str, n_items: int, shapes: dict) -> str:
    rows = "\n".join(f"| `{dst}` | {shapes[src]} |" for src, dst in cfg["files"].items() if src in shapes)
    return f"""---
license: other
viewer: false
---

# {cfg['title']}

{cfg['blurb']}

Companion to `format-EA/full320-v3`. Analysis code, item bank, and results live in
the project git repository: {GITHUB}

## Files

| path | contents |
|---|---|
{rows}

Each `.npz` holds `X` (float32) and `item_ids` (str). All-layer tensors are
`(n_items, n_layers, hidden)`; single-layer tensors are `(n_items, hidden)`.
`item_ids` is row-aligned with `X` and matches the item bank exactly.

## Item bank binding

- {n_items} items; `items.jsonl` sha256 `{items_sha}`
- The bank is tracked in git, not duplicated here:
  `{cfg['items'].relative_to(ROOT).as_posix()}`
- Join on `item_ids`; do not assume file order matches the bank's line order.

## Extraction semantics

Taken from the recorded extraction config: activations are the zero-based
transformer block output **before** the final model norm, read at the **last
prompt token** after `tokenizer.apply_chat_template(..., add_generation_prompt=True)`,
with `add_special_tokens=False`. See `activations/llama31_8b/config.json` where present.

## Scope

Development evidence for a workshop-scale mechanistic-interpretability project, not a
general-purpose benchmark. Activations are model-internal states for specific frozen
prompts; they carry no evaluation/deployment ground truth beyond the labels in the item
bank. Private by default, matching the other `format-EA` datasets.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=[*DATASETS, "all"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = load_token()
    api = HfApi(token=token)
    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    provenance = {}

    for name in targets:
        cfg = DATASETS[name]
        src = cfg["source_dir"]
        missing = [f for f in cfg["files"] if not (src / f).exists()]
        if missing:
            raise SystemExit(f"{name}: missing local files {missing}")
        items = cfg["items"]
        if not items.exists():
            raise SystemExit(f"{name}: missing item bank {items}")

        import numpy as np
        n_items = sum(1 for line in items.open(encoding="utf-8") if line.strip())
        shapes, hashes = {}, {}
        for f in cfg["files"]:
            p = src / f
            hashes[f] = sha256(p)
            if p.suffix == ".npz":
                with np.load(p, mmap_mode="r") as z:
                    shapes[f] = f"`X` {tuple(z['X'].shape)} float32, {len(z['item_ids'])} ids"
            else:
                shapes[f] = "extraction config"

        print(f"=== {name} -> {cfg['repo_id']} ({len(cfg['files'])} files, "
              f"{sum((src / f).stat().st_size for f in cfg['files']) / 2**30:.2f} GiB)")
        for f, dst in cfg["files"].items():
            print(f"   {f} -> {dst}  sha256={hashes[f][:16]}")
        provenance[name] = {
            "repo_id": cfg["repo_id"], "repo_type": "dataset", "private": True,
            "items_path": items.relative_to(ROOT).as_posix(),
            "items_sha256": sha256(items), "n_items": n_items,
            "files": {dst: {"local": (src / s).relative_to(ROOT).as_posix(), "sha256": hashes[s]}
                      for s, dst in cfg["files"].items()},
        }
        if args.dry_run:
            continue

        api.create_repo(cfg["repo_id"], repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=readme(cfg, provenance[name]["items_sha256"], n_items, shapes).encode("utf-8"),
                        path_in_repo="README.md", repo_id=cfg["repo_id"], repo_type="dataset")
        for f, dst in cfg["files"].items():
            print(f"   uploading {dst} ...", flush=True)
            api.upload_file(path_or_fileobj=str(src / f), path_in_repo=dst,
                            repo_id=cfg["repo_id"], repo_type="dataset")
        print(f"   done: https://huggingface.co/datasets/{cfg['repo_id']}", flush=True)

    out = ROOT / "provenance" / "hf_activation_uploads.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(provenance)
    out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
