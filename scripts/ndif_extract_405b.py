"""Chunked NDIF extraction driver: full320-v3 -> Llama-3.1-405B-Instruct -> HF.

Streams last-prompt-token residuals of every transformer block (before the
final model norm — same semantics as scripts/extract_activations.py) from
NDIF straight into the HF dataset repo. Activations exist only in RAM on the
driver machine; nothing is written to its disk. Resumable: chunks already
present in the repo are skipped, so the script can be re-run after any crash.

Tokenization mirrors src/eval_format_mvp/extract.py exactly: chat template
rendered to text with add_generation_prompt=True, add_special_tokens=False,
left padding, prompts never truncated. Compute on NDIF is bf16; results are
downloaded as bf16 and stored as raw bf16 bits in uint16 (lossless; fp32 =
(u16 << 16).view(float32)).

NDIF etiquette baked in: one request in flight at a time, a single
session per chunk (4 traces of 16 items, minutes per request — far under the
1-hour kill), only last-token slices and argmax token ids are .save()'d, and
failures back off exponentially instead of retrying hot.

Run from repo root (keys from env or .env: NDIF_API_KEY, HF_TOKEN):
  python scripts/ndif_extract_405b.py \
    --items data/full320-v3/rendered_items_v3_full320_8-17-2026.jsonl
  python scripts/ndif_extract_405b.py --limit-chunks 1        # smoke test
  python scripts/ndif_extract_405b.py --verify-chunk 0        # RAM-only QC
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

MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
HF_REPO = "Conradf/v3-llama-3.1-405B-instruct"
ITEMS_SHA256 = "5528819d74e696d1a595062047a829836dfde1d2a45287b06a5ac1ba5d1b6c67"
N_LAYERS = 126
HIDDEN = 16384
CHUNK_SIZE = 64
SUB_BATCH = 16
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


def chunk_name(index: int) -> str:
    return f"chunks/chunk_{index:04d}.npz"


def build_prompts(tokenizer, rows):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["text"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    unpadded = tokenizer(
        prompts, add_special_tokens=False, padding=False, truncation=False
    )["input_ids"]
    longest = max(len(ids) for ids in unpadded)
    if longest > MAX_LENGTH:
        sys.exit(
            f"Longest prompt has {longest} tokens > {MAX_LENGTH}; prompts are never truncated."
        )
    return prompts, [len(ids) for ids in unpadded]


def qc_report(x_fp32: np.ndarray, top_ids: np.ndarray, tokenizer, rows) -> None:
    if not np.isfinite(x_fp32).all():
        sys.exit("QC FAIL: non-finite values in activations")
    norms = np.linalg.norm(x_fp32, axis=2).mean(axis=0)  # mean norm per layer
    picks = [0, N_LAYERS // 4, N_LAYERS // 2, 3 * N_LAYERS // 4, N_LAYERS - 1]
    print("QC mean |h| by layer: " + "  ".join(f"L{i}={norms[i]:.1f}" for i in picks))
    for j in range(min(3, len(rows))):
        tok = tokenizer.decode(int(top_ids[j]))
        print(f"QC top-1 next token for {rows[j]['item_id']}: {tok!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/full320-v3/rendered_items_v3_full320_8-17-2026.jsonl")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--limit-chunks", type=int, default=None)
    ap.add_argument("--verify-chunk", type=int, default=None)
    args = ap.parse_args()

    ndif_key, hf_token = load_keys()
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    api = HfApi(token=hf_token)
    items_path = Path(args.items)
    actual = sha256_file(items_path)
    if actual != ITEMS_SHA256:
        sys.exit(f"items sha256 mismatch: {actual} != {ITEMS_SHA256}")
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    assert len(rows) % CHUNK_SIZE == 0, "chunk size must divide n_items"
    n_chunks = len(rows) // CHUNK_SIZE

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, token_lengths = build_prompts(tokenizer, rows)
    print(f"{len(rows)} items, {n_chunks} chunks of {CHUNK_SIZE}, "
          f"max prompt tokens {max(token_lengths)}", flush=True)

    if args.verify_chunk is not None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=hf_token)
        with fs.open(f"datasets/{args.repo}/{chunk_name(args.verify_chunk)}", "rb") as f:
            data = np.load(io.BytesIO(f.read()))
        start = int(data["row_start"])
        sub = rows[start : start + data["X_bf16_u16"].shape[0]]
        assert [str(x) for x in data["item_ids"]] == [r["item_id"] for r in sub]
        assert data["X_bf16_u16"].shape[1:] == (N_LAYERS, HIDDEN)
        qc_report(bf16_bits_to_fp32(data["X_bf16_u16"]), data["top1_token_ids"], tokenizer, sub)
        print(f"VERIFY OK: chunk {args.verify_chunk}, shape {data['X_bf16_u16'].shape}")
        return

    import torch
    from nnsight import CONFIG, LanguageModel

    CONFIG.API.APIKEY = ndif_key
    model = LanguageModel(MODEL_ID)
    assert int(model.config.num_hidden_layers) == N_LAYERS
    assert int(model.config.hidden_size) == HIDDEN

    done = {f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.startswith("chunks/")}
    todo = [c for c in range(n_chunks) if chunk_name(c) not in done]
    if args.limit_chunks is not None:
        todo = todo[: args.limit_chunks]
    print(f"{len(done)} chunks already uploaded, {len(todo)} to run", flush=True)

    for c in todo:
        start = c * CHUNK_SIZE
        sub_rows = rows[start : start + CHUNK_SIZE]
        encodings = []
        for s in range(0, CHUNK_SIZE, SUB_BATCH):
            enc = tokenizer(
                prompts[start + s : start + s + SUB_BATCH],
                add_special_tokens=False,
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
                            # handle whichever version the deployment runs.
                            # Layer outputs are spread across the deployment's GPUs
                            # (405B is sharded), so each last-token slice moves to CPU
                            # before the stack.
                            outs = [model.model.layers[i].output for i in range(N_LAYERS)]
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
        assert x.shape == (CHUNK_SIZE, N_LAYERS, HIDDEN)
        x_u16 = x.view(torch.int16).numpy().view(np.uint16)
        top_ids = torch.cat([t for t in tops], dim=0).numpy()
        qc_report(bf16_bits_to_fp32(x_u16), top_ids, tokenizer, sub_rows)

        buf = io.BytesIO()
        np.savez(
            buf,
            X_bf16_u16=x_u16,
            item_ids=np.asarray([str(r["item_id"]) for r in sub_rows], dtype=str),
            layers=np.arange(N_LAYERS, dtype=np.int16),
            token_lengths=np.asarray(token_lengths[start : start + CHUNK_SIZE], dtype=np.int32),
            top1_token_ids=top_ids,
            row_start=np.int64(start),
        )
        buf.seek(0)
        for attempt in range(MAX_RETRIES):
            try:
                api.upload_file(
                    path_or_fileobj=buf,
                    path_in_repo=chunk_name(c),
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
              f"(items {start}-{start + CHUNK_SIZE - 1})", flush=True)

    print("EXTRACTION PASS DONE")


if __name__ == "__main__":
    main()
