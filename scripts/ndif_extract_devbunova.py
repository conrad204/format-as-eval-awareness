"""Chunked NDIF extraction driver: Devbunova 2x2 items -> Llama 3.1 {8B,70B,405B} -> HF.

Generalization of scripts/ndif_extract_405b.py for the chat-template-vs-raw
replication study (see external/eval-awareness-format/VENDORED.md). Streams
last-prompt-token residuals of every transformer block (before the final model
norm — same semantics as scripts/extract_activations.py) from NDIF straight
into an HF dataset repo. Activations exist only in RAM on the driver machine.
Resumable: chunks already present under this run's repo prefix are skipped.

Two tokenization regimes, mirroring src/eval_format_mvp/extract.py exactly:
  template (default)   chat template rendered to text, add_generation_prompt=True,
                       add_special_tokens=False; probed token = constant
                       assistant header.
  raw (--no-chat-template)   bare prompt text, add_special_tokens=True (BOS
                       only) — Devbunova's regime, token-identical to her
                       tokenizer(batch, padding=True) + attention_mask.sum()-1
                       indexing; probed token = final content token.
Both use left padding and prompts are never truncated. Compute on NDIF is
bf16; stored as raw bf16 bits in uint16 (fp32 = (u16 << 16).view(float32)).

Repo layout: <model_key>/<regime>/chunks/chunk_XXXX.npz — one repo holds all
six model x regime runs. The last chunk is partial (8479 = 132*64 + 31).

NDIF etiquette (as in the 405B driver): one request in flight, one session per
chunk, only last-token slices and argmax token ids .save()'d, exponential
backoff, fail fast on server-side RemoteException.

Run from repo root (keys from env or .env: NDIF_API_KEY, HF_TOKEN):
  python scripts/ndif_extract_devbunova.py --model 8b --no-chat-template
  python scripts/ndif_extract_devbunova.py --model 8b
  python scripts/ndif_extract_devbunova.py --model 8b --dry-run          # no network
  python scripts/ndif_extract_devbunova.py --model 8b --limit-chunks 1   # smoke test
  python scripts/ndif_extract_devbunova.py --model 8b --verify-chunk 0   # RAM-only QC
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# model_key -> (hf model id, tokenizer revision, n_layers, hidden, sub_batch).
# Tokenizer revisions match the pins used for the full320-v3 extractions
# (8B: extraction_config_8b; 70B: pinned by commit 5a1c914). The NDIF-side
# model revision is whatever the pinned deployment runs (as for the 405B
# full320-v3 pass, recorded there as "unpinned_ndif_deployment").
# sub_batch: per-trace batch size. The 8B deployment sits on a shared 44GB GPU
# with a ~17.7GiB process cap; bf16 weights leave <2GiB headroom and the full-
# sequence logits scale with batch x seq x 128k vocab, so 8B OOMs at 16 and
# needs 4. The multi-GPU 70B/405B deployments take 16 fine.
MODELS = {
    "8b": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", 32, 4096, 4),
    "70b": ("meta-llama/Llama-3.1-70B-Instruct", "1605565b47bb9346c5515c34102e054115b4f98b", 80, 8192, 16),
    "405b": ("meta-llama/Llama-3.1-405B-Instruct", None, 126, 16384, 16),
}

HF_REPO = "Conradf/devbunova-2x2-activations"
ITEMS_DEFAULT = "data/devbunova-2x2/items_devbunova_2x2.jsonl"
# output_sha256 from provenance/devbunova-2x2/adapter_manifest.json
ITEMS_SHA256 = "bd4b7151cb0a2e4608674934c0f8d2f94efb4c886fd06ea53333d242beb74f26"
CHUNK_SIZE = 64
MAX_LENGTH = 2048
MAX_RETRIES = 5
BACKOFF_S = 60


def load_keys() -> tuple[str, str]:
    env = dict(os.environ)
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env.setdefault(k.strip(), v.strip().strip("'\""))
    ndif, hf = env.get("NDIF_API_KEY"), env.get("HF_TOKEN")
    if not ndif or not hf:
        sys.exit("NDIF_API_KEY and HF_TOKEN required (env or .env at repo root)")
    os.environ.setdefault("HF_TOKEN", hf)
    return ndif, hf


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bf16_bits_to_fp32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def chunk_name(prefix: str, index: int) -> str:
    return f"{prefix}/chunks/chunk_{index:04d}.npz"


def build_prompts(tokenizer, rows, chat_template: bool):
    """Mirrors src/eval_format_mvp/extract.py — keep the two in sync."""
    if chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(row["text"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in rows
        ]
    else:
        prompts = [str(row["text"]) for row in rows]
    add_special_tokens = not chat_template
    unpadded = tokenizer(
        prompts, add_special_tokens=add_special_tokens, padding=False, truncation=False
    )["input_ids"]
    longest = max(len(ids) for ids in unpadded)
    if longest > MAX_LENGTH:
        sys.exit(
            f"Longest prompt has {longest} tokens > {MAX_LENGTH}; prompts are never truncated."
        )
    return prompts, [len(ids) for ids in unpadded], add_special_tokens


def qc_report(x_fp32: np.ndarray, top_ids: np.ndarray, tokenizer, rows, n_layers: int) -> None:
    if not np.isfinite(x_fp32).all():
        sys.exit("QC FAIL: non-finite values in activations")
    norms = np.linalg.norm(x_fp32, axis=2).mean(axis=0)  # mean norm per layer
    picks = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print("QC mean |h| by layer: " + "  ".join(f"L{i}={norms[i]:.1f}" for i in picks))
    for j in range(min(3, len(rows))):
        tok = tokenizer.decode(int(top_ids[j]))
        print(f"QC top-1 next token for {rows[j]['item_id']}: {tok!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--no-chat-template", action="store_true",
                    help="raw tokenization (Devbunova regime); default is chat template")
    ap.add_argument("--items", default=ITEMS_DEFAULT)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--limit-chunks", type=int, default=None)
    ap.add_argument("--verify-chunk", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="verify items + tokenize + print the chunk plan; no network beyond tokenizer cache")
    args = ap.parse_args()

    model_id, tok_revision, n_layers, hidden, sub_batch = MODELS[args.model]
    regime = "raw" if args.no_chat_template else "template"
    prefix = f"{args.model}/{regime}"

    items_path = Path(args.items)
    actual = sha256_file(items_path)
    if actual != ITEMS_SHA256:
        sys.exit(f"items sha256 mismatch: {actual} != {ITEMS_SHA256}")
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    n_chunks = (len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE

    if args.dry_run:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        ndif_key = hf_token = None
    else:
        ndif_key, hf_token = load_keys()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tok_revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, token_lengths, add_special_tokens = build_prompts(
        tokenizer, rows, chat_template=not args.no_chat_template
    )
    last_size = len(rows) - (n_chunks - 1) * CHUNK_SIZE
    print(f"{model_id} [{regime}] -> {args.repo}/{prefix}", flush=True)
    print(f"{len(rows)} items, {n_chunks} chunks of {CHUNK_SIZE} (last={last_size}), "
          f"max prompt tokens {max(token_lengths)}", flush=True)

    if args.dry_run:
        import torch

        enc = tokenizer(prompts[:4], add_special_tokens=add_special_tokens,
                        return_tensors="pt", padding=True, truncation=False)
        assert bool(torch.all(enc["attention_mask"][:, -1] == 1))
        lasts = enc["input_ids"][:, -1].tolist()
        print("first-4 probed last tokens:", [repr(tokenizer.decode([t])) for t in lasts])
        print("DRY RUN OK")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    if args.verify_chunk is not None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=hf_token)
        with fs.open(f"datasets/{args.repo}/{chunk_name(prefix, args.verify_chunk)}", "rb") as f:
            data = np.load(io.BytesIO(f.read()))
        assert str(data["model_id"]) == model_id and str(data["regime"]) == regime
        start = int(data["row_start"])
        sub = rows[start : start + data["X_bf16_u16"].shape[0]]
        assert [str(x) for x in data["item_ids"]] == [r["item_id"] for r in sub]
        assert data["X_bf16_u16"].shape[1:] == (n_layers, hidden)
        qc_report(bf16_bits_to_fp32(data["X_bf16_u16"]), data["top1_token_ids"], tokenizer, sub, n_layers)
        print(f"VERIFY OK: chunk {args.verify_chunk}, shape {data['X_bf16_u16'].shape}")
        return

    import torch
    from nnsight import CONFIG, LanguageModel

    CONFIG.API.APIKEY = ndif_key
    model = LanguageModel(model_id)
    assert int(model.config.num_hidden_layers) == n_layers
    assert int(model.config.hidden_size) == hidden

    done = {
        f
        for f in api.list_repo_files(args.repo, repo_type="dataset")
        if f.startswith(prefix + "/chunks/")
    }
    todo = [c for c in range(n_chunks) if chunk_name(prefix, c) not in done]
    if args.limit_chunks is not None:
        todo = todo[: args.limit_chunks]
    print(f"{len(done)} chunks already uploaded, {len(todo)} to run", flush=True)

    for c in todo:
        start = c * CHUNK_SIZE
        sub_rows = rows[start : start + CHUNK_SIZE]
        encodings = []
        for s in range(0, len(sub_rows), sub_batch):
            enc = tokenizer(
                prompts[start + s : start + s + sub_batch],
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            if not bool(torch.all(enc["attention_mask"][:, -1] == 1)):
                sys.exit("Last position must be a real prompt token, not padding")
            encodings.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                with model.session(remote=True):
                    acts = list().save()
                    tops = list().save()
                    for enc in encodings:
                        with model.trace(enc):
                            # transformers >=5 decoder layers return the hidden-states
                            # tensor directly; older versions return a tuple. This
                            # comprehension executes on the NDIF worker, so it must
                            # handle whichever version the deployment runs. Layer
                            # outputs may be sharded across the deployment's GPUs,
                            # so each last-token slice moves to CPU before the stack.
                            outs = [model.model.layers[i].output for i in range(n_layers)]
                            h = torch.stack(
                                [
                                    (o[0] if isinstance(o, (tuple, list)) else o)[:, -1, :].detach().cpu()
                                    for o in outs
                                ],
                                dim=1,
                            )
                            acts.append(h)
                            tops.append(model.output.logits[:, -1, :].argmax(dim=-1).detach().cpu())
                break
            except Exception as exc:  # noqa: BLE001 — network/queue errors; back off, stay serial
                if type(exc).__name__ == "RemoteException":
                    # Our trace code failed server-side; deterministic, so
                    # retrying only burns NDIF compute. Fail immediately.
                    raise
                wait = BACKOFF_S * (2**attempt)
                print(f"chunk {c} attempt {attempt + 1} failed ({exc}); retrying in {wait}s", flush=True)
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(wait)

        x = torch.cat([t for t in acts], dim=0).contiguous()
        assert x.dtype == torch.bfloat16, f"expected bf16 from NDIF, got {x.dtype}"
        assert x.shape == (len(sub_rows), n_layers, hidden)
        x_u16 = x.view(torch.int16).numpy().view(np.uint16)
        top_ids = torch.cat([t for t in tops], dim=0).numpy()
        qc_report(bf16_bits_to_fp32(x_u16), top_ids, tokenizer, sub_rows, n_layers)

        buf = io.BytesIO()
        np.savez(
            buf,
            X_bf16_u16=x_u16,
            item_ids=np.asarray([str(r["item_id"]) for r in sub_rows], dtype=str),
            layers=np.arange(n_layers, dtype=np.int16),
            token_lengths=np.asarray(token_lengths[start : start + len(sub_rows)], dtype=np.int32),
            top1_token_ids=top_ids,
            row_start=np.int64(start),
            model_id=np.str_(model_id),
            regime=np.str_(regime),
        )
        buf.seek(0)
        for attempt in range(MAX_RETRIES):
            try:
                api.upload_file(
                    path_or_fileobj=buf,
                    path_in_repo=chunk_name(prefix, c),
                    repo_id=args.repo,
                    repo_type="dataset",
                )
                break
            except Exception as exc:  # noqa: BLE001
                wait = BACKOFF_S * (2**attempt)
                print(f"chunk {c} upload attempt {attempt + 1} failed ({exc}); retrying in {wait}s", flush=True)
                if attempt == MAX_RETRIES - 1:
                    raise
                buf.seek(0)
                time.sleep(wait)
        print(f"chunk {c + 1}/{n_chunks} done in {time.time() - t0:.0f}s "
              f"(items {start}-{start + len(sub_rows) - 1})", flush=True)

    print(f"EXTRACTION PASS DONE: {model_id} [{regime}]")


if __name__ == "__main__":
    main()
