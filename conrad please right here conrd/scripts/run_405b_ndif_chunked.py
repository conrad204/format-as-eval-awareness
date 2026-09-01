"""Chunked NDIF runner: hidden-channel banks -> Llama-3.1-405B-Instruct.

Rewrite of run_405b_ndif.py following the proven full320-v3 NDIF pipeline
(scripts/ndif_extract_405b.py + notes/8-19-26-ndif-runpod-guide.md):

- Pre-tokenized inputs passed to model.trace() as {input_ids, attention_mask}
  dicts (chat template rendered to text, add_special_tokens=False, left
  padding, hard assert the last position is a real token). The original
  handoff script passed templated *text*, which makes nnsight re-tokenize
  with a duplicate BOS and breaks parity with the local BF16 runs.
- One NDIF session per chunk of 64 items (4 traces x 16), minutes per
  request, one request in flight at a time.
- Only reductions are .save()'d: the two letter logits, the top-1 token id
  (QC), and optionally last-token hidden states at a few layers.
- RemoteException (our code failed server-side) fails fast; transient
  errors back off exponentially.
- Durable per chunk: each chunk is written to <out>.parts/chunk_XXXX.npz
  before the next starts; re-running skips completed chunks. The final
  merge emits the same JSON schema as run_405b_ndif.py so
  analyze_recognition_bank.py works unchanged, plus <out>.activations.npz
  when --layers is given.

    python run_405b_ndif_chunked.py --items ../banks/latent_recognition.jsonl \
        --out ../out/latent_recognition_llama31_405b.json --limit 8   # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
N_LAYERS = 126
HIDDEN = 16384
CHUNK_SIZE = 64
SUB_BATCH = 16
MAX_LENGTH = 1024
MAX_RETRIES = 5
BACKOFF_S = 60

BANK_SHA256 = {
    "latent_recognition.jsonl": "e2467ceabb9435931b59f8a7c7a5c23bdebb31d13a0a970f0560e747c69820e6",
    "arbitrary_control.jsonl": "3d069ab3d346ed0a073fb44cd276efeb33697408faa59cf9f1bd2a6ccaef51c4",
    "cot_reasoning.jsonl": "068a241a9d3053c7a15c29035055e4e90da055e5cea4f9c8471fe840742095d2",
    "explicit_ceiling.jsonl": "1212c018382423676b6b12b31ac0eaeb105abf60a5e9ae079bff35e7dbd457b8",
}


def load_env() -> None:
    for candidate in (Path(".env"), Path("../.env"), Path("../../.env"),
                      Path(".env.local"), Path("../.env.local"), Path("../../.env.local")):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*(?:export\s+)?([A-Z_]+)\s*=\s*(.+)", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="", help="comma-separated layer indices for activations")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on the first N rows")
    args = ap.parse_args()

    load_env()
    key = os.environ.get("NDIF_API_KEY")
    if not key:
        sys.exit("set NDIF_API_KEY (env, .env, or .env.local)")

    expected = BANK_SHA256.get(args.items.name)
    if expected:
        actual = sha256_file(args.items)
        if actual != expected:
            sys.exit(f"{args.items.name} sha256 mismatch: {actual} != {expected}")

    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    letters = {}
    for letter in ("A", "B"):
        ids = tok.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            sys.exit(f"{letter!r} is not a single token under this tokenizer: {ids}")
        letters[letter] = ids[0]
    letter_ids = [letters["A"], letters["B"]]

    want_layers = [int(x) for x in args.layers.split(",") if x.strip()]
    for layer in want_layers:
        if not 0 <= layer < N_LAYERS:
            sys.exit(f"layer {layer} outside 0..{N_LAYERS - 1}")

    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]

    prompts = [
        tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        for r in rows
    ]
    unpadded = tok(prompts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    longest = max(len(ids) for ids in unpadded)
    if longest > MAX_LENGTH:
        sys.exit(f"Longest prompt has {longest} tokens > {MAX_LENGTH}; prompts are never truncated.")

    n_chunks = (len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE
    parts_dir = Path(str(args.out) + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    print(f"{MODEL_ID}: {len(rows)} prompts, {n_chunks} chunks of {CHUNK_SIZE}, "
          f"max tokens {longest}, layers {want_layers or 'none'}", flush=True)

    from nnsight import CONFIG, LanguageModel

    CONFIG.API.APIKEY = key
    model = LanguageModel(MODEL_ID, tokenizer=tok)
    assert int(model.config.num_hidden_layers) == N_LAYERS

    for c in range(n_chunks):
        part_path = parts_dir / f"chunk_{c:04d}.npz"
        if part_path.exists():
            continue
        start = c * CHUNK_SIZE
        sub_rows = rows[start : start + CHUNK_SIZE]
        encodings = []
        for s in range(0, len(sub_rows), SUB_BATCH):
            enc = tok(
                prompts[start + s : start + s + len(sub_rows[s : s + SUB_BATCH])],
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
                    letter_logits = list().save()
                    tops = list().save()
                    hiddens = list().save()
                    for enc in encodings:
                        with model.trace(enc):
                            last = model.output.logits[:, -1, :]
                            letter_logits.append(last[:, letter_ids].detach().cpu())
                            tops.append(last.argmax(dim=-1).detach().cpu())
                            if want_layers:
                                # transformers >=5 layers return the tensor directly,
                                # older return a tuple; 405B is sharded across GPUs so
                                # each slice moves to CPU before the stack.
                                outs = [model.model.layers[i].output for i in want_layers]
                                h = torch.stack(
                                    [
                                        (o[0] if isinstance(o, (tuple, list)) else o)[:, -1, :].detach().cpu()
                                        for o in outs
                                    ],
                                    dim=1,
                                )
                                hiddens.append(h)
                break
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ == "RemoteException":
                    raise  # our trace code failed server-side; retrying burns NDIF compute
                wait = BACKOFF_S * (2 ** attempt)
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"chunk {c} attempt {attempt + 1} failed ({type(exc).__name__}: "
                      f"{str(exc)[:120]}); retrying in {wait}s", flush=True)
                time.sleep(wait)

        ll = torch.cat(list(letter_logits), dim=0).float().numpy()
        top_ids = torch.cat(list(tops), dim=0).numpy()
        assert ll.shape == (len(sub_rows), 2) and np.isfinite(ll).all()
        payload = {
            "letter_logits": ll,
            "top1_token_ids": top_ids,
            "item_ids": np.asarray([r["item_id"] for r in sub_rows], dtype=str),
            "row_start": np.int64(start),
        }
        if want_layers:
            h = torch.cat(list(hiddens), dim=0).float().numpy()
            assert h.shape == (len(sub_rows), len(want_layers), HIDDEN) and np.isfinite(h).all()
            payload["hidden"] = h.astype(np.float32)
            payload["layers"] = np.asarray(want_layers, dtype=np.int16)
        tmp = part_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, part_path)
        top_dec = [tok.decode(int(t)) for t in top_ids[:4]]
        print(f"chunk {c + 1}/{n_chunks} done in {time.time() - t0:.0f}s; "
              f"QC top-1 next tokens {top_dec}", flush=True)

    # merge
    outputs, acts = [], []
    for c in range(n_chunks):
        data = np.load(parts_dir / f"chunk_{c:04d}.npz", allow_pickle=False)
        start = int(data["row_start"])
        sub_rows = rows[start : start + data["letter_logits"].shape[0]]
        assert [str(x) for x in data["item_ids"]] == [r["item_id"] for r in sub_rows]
        for j, row in enumerate(sub_rows):
            a, b = float(data["letter_logits"][j, 0]), float(data["letter_logits"][j, 1])
            target = row.get("target_letter")
            margin = (a - b) if target == "A" else (b - a)
            outputs.append({
                "kind": row["kind"], "item_id": row["item_id"], "unit_id": row["unit_id"],
                "episode_family": row["episode_family"],
                "question_family": row.get("question_family"),
                "condition": row["condition"], "query_status": row["query_status"],
                "target_letter": target, "logit_a": a, "logit_b": b,
                "target_logit_margin": margin,
                "p_target": float(np.exp(margin) / (1.0 + np.exp(margin))),
            })
        if "hidden" in data:
            acts.append(data["hidden"])

    payload = {
        "schema_version": 1, "model_key": "llama31_405b", "model": MODEL_ID,
        "precision": "ndif-hosted-bf16", "load_in_8bit": False,
        "items": str(args.items),
        "items_sha256": sha256_file(args.items),
        "n_items": len(outputs), "n_layers": N_LAYERS,
        "candidate_token_ids": letters, "rows": outputs,
        "runner": "run_405b_ndif_chunked.py",
        "tokenization": "chat_template + add_special_tokens=False, left pad, pre-tokenized ids",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out, f"({len(outputs)} rows)")

    if acts:
        stack = np.concatenate(acts, axis=0)
        npz = args.out.with_suffix(".activations.npz")
        np.savez_compressed(
            npz, X=stack,
            item_ids=np.asarray([r["item_id"] for r in outputs]),
            layers=np.asarray(want_layers),
            episode_family=np.asarray([r["episode_family"] for r in outputs]),
            condition=np.asarray([r["condition"] for r in outputs]),
            query_status=np.asarray([r["query_status"] for r in outputs]),
            query_token=np.asarray([r["condition"].split("_")[-1] for r in outputs]),
        )
        print("wrote", npz, stack.shape)


if __name__ == "__main__":
    main()
