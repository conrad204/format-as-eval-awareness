"""Repack NDIF chunk shards into per-layer files on HF (run on a pod).

Reads chunks/chunk_XXXX.npz from the HF dataset repo (streamed into RAM one at
a time), scatters rows into 126 per-layer uint16 arrays, verifies item order
against the frozen jsonl, then uploads layers/layer_XXX.npz plus config.json
provenance. Chunks are left in place unless --delete-chunks is passed after a
clean verify.

Two storage strategies:
  --work-dir DIR         memmaps on disk (~48GB volume, single pass over chunks)
  --ram-group-size N     in-RAM arrays for N layers at a time, streaming the
                         full chunk set once per group — for pods with small
                         disks (needs ~N*377MB RAM, ~400MB scratch disk, and
                         downloads the chunk set ceil(126/N) times)

  python scripts/ndif_repack_405b.py --items <jsonl> --ram-group-size 30
"""

import argparse
import hashlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ndif_extract_405b import (  # noqa: E402
    CHUNK_SIZE,
    HF_REPO,
    HIDDEN,
    ITEMS_SHA256,
    MODEL_ID,
    N_LAYERS,
    SUB_BATCH,
    bf16_bits_to_fp32,
    chunk_name,
    load_keys,
    sha256_file,
)


def layer_name(index: int) -> str:
    return f"layers/layer_{index:03d}.npz"


def fetch_chunk(fs, repo: str, c: int, retries: int = 4, chunks_dir: Path | None = None):
    # A pre-downloaded local mirror (hf download --include "chunks/*") is far
    # more robust than streaming reads: one flaky read mid-scatter loses the
    # whole in-RAM state, while the mirror download is resumable.
    if chunks_dir is not None:
        return np.load(chunks_dir / f"chunk_{c:04d}.npz")
    for attempt in range(retries):
        try:
            with fs.open(f"datasets/{repo}/{chunk_name(c)}", "rb") as f:
                return np.load(io.BytesIO(f.read()))
        except Exception as exc:  # noqa: BLE001 — transient network; back off
            if attempt == retries - 1:
                raise
            print(f"chunk {c} fetch attempt {attempt + 1} failed ({exc}); retrying", flush=True)
            time.sleep(30 * (attempt + 1))


def upload_layer(api, repo: str, L: int, arr: np.ndarray, ids_arr: np.ndarray) -> str:
    """Write one layer npz to scratch disk, upload it, return its sha256."""
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        np.savez(tmp, X_bf16_u16=arr, item_ids=ids_arr, layer=np.int16(L))
        tmp.flush()
        sha = sha256_file(Path(tmp.name))
        for attempt in range(4):
            try:
                api.upload_file(
                    path_or_fileobj=tmp.name,
                    path_in_repo=layer_name(L),
                    repo_id=repo,
                    repo_type="dataset",
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"layer {L} upload attempt {attempt + 1} failed ({exc}); retrying", flush=True)
                time.sleep(30 * (attempt + 1))
    return sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/full320-v3/rendered_items_v3_full320_8-17-2026.jsonl")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--ram-group-size", type=int, default=0)
    ap.add_argument("--chunks-dir", default=None,
                    help="local dir with chunk_XXXX.npz (from hf download); skips streaming fetches")
    ap.add_argument("--delete-chunks", action="store_true")
    args = ap.parse_args()
    if bool(args.work_dir) == bool(args.ram_group_size):
        sys.exit("pass exactly one of --work-dir or --ram-group-size")

    _, hf_token = load_keys()
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi(token=hf_token)
    fs = HfFileSystem(token=hf_token)

    items_path = Path(args.items)
    if sha256_file(items_path) != ITEMS_SHA256:
        sys.exit("items sha256 mismatch")
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    n_items = len(rows)
    n_chunks = n_items // CHUNK_SIZE
    ids_expected = [r["item_id"] for r in rows]
    ids_arr = np.asarray(ids_expected, dtype=str)

    have = set(api.list_repo_files(args.repo, repo_type="dataset"))
    missing = [c for c in range(n_chunks) if chunk_name(c) not in have]
    if missing:
        sys.exit(f"{len(missing)} chunks missing, e.g. {missing[:5]} — finish extraction first")
    layers_done = {f for f in have if f.startswith("layers/")}

    token_lengths = np.zeros(n_items, dtype=np.int32)
    max_prompt_tokens = 0
    layer_shas: dict[str, str] = {}

    if args.ram_group_size:
        G = args.ram_group_size
        groups = [list(range(s, min(s + G, N_LAYERS))) for s in range(0, N_LAYERS, G)]
        todo = [(gi, grp) for gi, grp in enumerate(groups)
                if not all(layer_name(L) in layers_done for L in grp)]
        if not todo:
            if "config.json" in have:
                print("all layer files and config.json already uploaded; nothing to do")
                return
            sys.exit("all layers uploaded but config.json missing — rerun after deleting one layer file")
        first_gi = todo[0][0]
        for gi, grp in todo:
            arrs = {L: np.empty((n_items, HIDDEN), dtype=np.uint16) for L in grp}
            for c in range(n_chunks):
                data = fetch_chunk(fs, args.repo, c, chunks_dir=Path(args.chunks_dir) if args.chunks_dir else None)
                start = int(data["row_start"])
                assert start == c * CHUNK_SIZE
                x = data["X_bf16_u16"]
                assert x.shape == (CHUNK_SIZE, N_LAYERS, HIDDEN)
                for L in grp:
                    arrs[L][start : start + CHUNK_SIZE] = x[:, L, :]
                if gi == first_gi:
                    assert [str(i) for i in data["item_ids"]] == ids_expected[start : start + CHUNK_SIZE], (
                        f"item order mismatch in chunk {c}"
                    )
                    token_lengths[start : start + CHUNK_SIZE] = data["token_lengths"]
                if (c + 1) % 30 == 0:
                    print(f"group {gi + 1}/{len(groups)}: {c + 1}/{n_chunks} chunks scattered", flush=True)
            for L in grp:
                sample = bf16_bits_to_fp32(arrs[L][:64])
                assert np.isfinite(sample).all(), f"non-finite values in layer {L}"
                layer_shas[layer_name(L)] = upload_layer(api, args.repo, L, arrs[L], ids_arr)
                del arrs[L]
                print(f"uploaded {layer_name(L)}", flush=True)
            max_prompt_tokens = max(max_prompt_tokens, int(token_lengths.max()))
    else:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        mm = {
            L: np.lib.format.open_memmap(
                work / f"layer_{L:03d}.npy", mode="w+", dtype=np.uint16, shape=(n_items, HIDDEN)
            )
            for L in range(N_LAYERS)
        }
        for c in range(n_chunks):
            data = fetch_chunk(fs, args.repo, c, chunks_dir=Path(args.chunks_dir) if args.chunks_dir else None)
            start = int(data["row_start"])
            assert start == c * CHUNK_SIZE
            x = data["X_bf16_u16"]
            assert x.shape == (CHUNK_SIZE, N_LAYERS, HIDDEN)
            for L in range(N_LAYERS):
                mm[L][start : start + CHUNK_SIZE] = x[:, L, :]
            assert [str(i) for i in data["item_ids"]] == ids_expected[start : start + CHUNK_SIZE]
            token_lengths[start : start + CHUNK_SIZE] = data["token_lengths"]
            print(f"scattered chunk {c + 1}/{n_chunks}", flush=True)
        for L in range(N_LAYERS):
            mm[L].flush()
            sample = bf16_bits_to_fp32(mm[L][:64])
            assert np.isfinite(sample).all(), f"non-finite values in layer {L}"
            layer_shas[layer_name(L)] = upload_layer(api, args.repo, L, np.asarray(mm[L]), ids_arr)
            print(f"uploaded {layer_name(L)}", flush=True)
        max_prompt_tokens = int(token_lengths.max())

    # Crash-resume can skip groups uploaded in an earlier run; recover their
    # sha256s from HF's LFS metadata so config.json stays complete.
    missing_shas = [layer_name(L) for L in range(N_LAYERS) if layer_name(L) not in layer_shas]
    if missing_shas:
        info = api.repo_info(args.repo, repo_type="dataset", files_metadata=True)
        by_name = {s.rfilename: s for s in info.siblings}
        for name in missing_shas:
            layer_shas[name] = by_name[name].lfs.sha256 if by_name[name].lfs else "unknown"
    layer_shas = dict(sorted(layer_shas.items()))

    import nnsight

    config = {
        "schema_version": 1,
        "model": MODEL_ID,
        "model_revision": "unpinned_ndif_deployment",
        "extraction_backend": "NDIF remote via nnsight",
        "nnsight_version": nnsight.__version__,
        "n_transformer_layers": N_LAYERS,
        "layers": list(range(N_LAYERS)),
        "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
        "position": "last_prompt_token_after_chat_template_before_generation",
        "chat_template": "tokenizer.apply_chat_template(user_message, add_generation_prompt=True)",
        "add_special_tokens": False,
        "padding_side": "left",
        "chunk_size": CHUNK_SIZE,
        "sub_batch": SUB_BATCH,
        "dtype_compute": "bfloat16 (NDIF deployment)",
        "dtype_stored": "bfloat16_bits_as_uint16",
        "reconstruction": "fp32 = (X_bf16_u16.astype(np.uint32) << 16).view(np.float32)",
        "n_items": n_items,
        "hidden_size": HIDDEN,
        "max_prompt_tokens": max_prompt_tokens,
        "items_path": str(items_path),
        "items_sha256": ITEMS_SHA256,
        "layer_file_sha256": layer_shas,
    }
    api.upload_file(
        path_or_fileobj=json.dumps(config, indent=1).encode(),
        path_in_repo="config.json",
        repo_id=args.repo,
        repo_type="dataset",
    )
    Path("extraction_config_405b_ndif.json").write_text(json.dumps(config, indent=1))
    print("config.json uploaded; local copy ./extraction_config_405b_ndif.json for provenance/")

    if args.delete_chunks:
        api.delete_folder(path_in_repo="chunks", repo_id=args.repo, repo_type="dataset")
        print("chunks/ deleted from repo")
    print("REPACK DONE")


if __name__ == "__main__":
    main()
