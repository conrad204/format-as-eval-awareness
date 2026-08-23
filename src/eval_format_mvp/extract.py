"""Extract one fixed last-token residual-stream activation per rendered prompt."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .io import MVPError, load_jsonl, sha256_file, write_json, write_jsonl

# Must be set before any CUDA context is created (first torch.cuda.* call
# anywhere in the process), so this runs at import time rather than inside
# extract_activations. Required for torch.use_deterministic_algorithms to
# make cuBLAS reduction order reproducible across runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _write_npz_deflate1(path: Path, arrays: dict[str, Any]) -> None:
    """np.load-compatible npz at deflate level 1: numpy's savez_compressed is
    pinned to single-threaded zlib level 6, which takes ~an hour on the 70B
    all-layers array while the pod's GPUs sit idle. Level 1 is ~10x faster for
    a few percent more bytes; the stored values are identical either way."""
    import zipfile

    import numpy as np
    from numpy.lib import format as npformat

    with zipfile.ZipFile(
        path, "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=1
    ) as zf:
        for name, arr in arrays.items():
            with zf.open(name + ".npy", "w", force_zip64=True) as handle:
                npformat.write_array(handle, np.asanyarray(arr), allow_pickle=False)


def extract_activations(
    *,
    items_path: Path,
    output_dir: Path,
    model_name: str,
    model_revision: str | None,
    layer: int,
    batch_size: int,
    max_length: int,
    device_map: str | None = None,
    all_layers: bool = False,
) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise MVPError(
            "Activation extraction requires the 'extract' optional dependencies"
        ) from exc

    # Pin deterministic GPU execution before any CUDA op runs (device queries
    # below count): otherwise non-deterministic cuBLAS/cuDNN reduction order
    # under BF16 accumulates into late-layer activation drift across runs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    rows = load_jsonl(items_path)
    if not rows:
        raise MVPError("Rendered item file is empty")
    if batch_size < 1 or max_length < 1:
        raise MVPError("batch_size and max_length must be positive")
    # Fail on an unwritable/occupied output path now, not after the GPU hours:
    # a stale file or dangling symlink at output_dir raises FileExistsError here.
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    load_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        # Pinned rather than left on transformers' auto-select, which can
        # choose SDPA or flash-attention kernels with different, run-to-run
        # non-deterministic reduction order.
        "attn_implementation": "eager",
    }
    # Weights larger than one accelerator are sharded by accelerate. Per-block
    # forward hooks still fire on whichever device owns that block.
    if device_map:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if not device_map:
        model = model.to(device)
    model.eval()
    input_device = model.device
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise MVPError("Expected a causal-LM exposing transformer blocks at .model.layers")
    n_layers = int(model.config.num_hidden_layers)
    if not 0 <= layer < n_layers:
        raise MVPError(f"layer must be in [0, {n_layers - 1}], got {layer}")

    # The chat template is rendered to text first, so it already carries the
    # model's control tokens; add_special_tokens must stay False or the
    # tokenizer prepends a duplicate begin-of-text.
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
    if longest > max_length:
        raise MVPError(
            f"Longest prompt has {longest} tokens, exceeding max_length={max_length}. "
            "Raise --max-length intentionally; prompts are never truncated."
        )

    # A forward hook on each block reads its output residual stream directly,
    # before the final model norm that hidden_states applies at the last layer.
    target_layers = tuple(range(n_layers)) if all_layers else (layer,)
    captured: dict[int, Any] = {}

    def _make_hook(index: int):
        def capture_last_token(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured[index] = hidden[:, -1, :].detach()

        return capture_last_token

    handles = [
        model.model.layers[index].register_forward_hook(_make_hook(index))
        for index in target_layers
    ]
    batches: dict[int, list[Any]] = {index: [] for index in target_layers}
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                captured.clear()
                encoded = tokenizer(
                    prompts[start : start + batch_size],
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)):
                    raise MVPError("Last position must be a real prompt token, not padding")
                encoded = {
                    key: value.to(input_device) for key, value in encoded.items()
                }
                model.model(**encoded, use_cache=False, return_dict=True)
                for index in target_layers:
                    if index not in captured:
                        raise MVPError(f"Forward hook did not fire for layer {index}")
                    batches[index].append(captured[index].float().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()

    # Everything below is CPU-side; release the weights first so the GPUs are
    # not held (billed but idle) through the numpy assemble/compress phase.
    del captured
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_layer = {
        index: np.concatenate(values, axis=0) for index, values in batches.items()
    }
    matrix = per_layer[layer]
    item_ids = np.asarray([str(row["item_id"]) for row in rows], dtype=str)
    token_lengths = np.asarray([len(ids) for ids in unpadded], dtype=np.int32)
    activation_path = output_dir / "activations.npz"
    _write_npz_deflate1(activation_path, {"X": matrix, "item_ids": item_ids})
    all_layers_path = output_dir / "activations_all_layers.npz"
    if all_layers:
        _write_npz_deflate1(
            all_layers_path,
            {
                "X": np.stack([per_layer[index] for index in target_layers], axis=1),
                "item_ids": item_ids,
                "layers": np.asarray(target_layers, dtype=np.int16),
                "token_lengths": token_lengths,
            },
        )
    write_jsonl(output_dir / "meta.jsonl", rows)
    config = {
        "schema_version": 1,
        "model": model_name,
        "model_revision": model_revision or "default_resolved_by_hub",
        "n_transformer_layers": n_layers,
        "layer": layer,
        "primary_layer": layer,
        "layers": list(target_layers),
        "device_map": device_map or "single_device",
        "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
        "position": "last_prompt_token_after_chat_template_before_generation",
        "chat_template": "tokenizer.apply_chat_template(user_message, add_generation_prompt=True)",
        "add_special_tokens": False,
        "npz_compression": "zip_deflate_level1",
        "max_length": max_length,
        "max_prompt_tokens": longest,
        "dtype_stored": "float32",
        "n_items": len(rows),
        "hidden_size": int(matrix.shape[1]),
        "items_path": str(items_path),
        "items_sha256": sha256_file(items_path),
        "activations_sha256": sha256_file(activation_path),
        "activations_all_layers_sha256": (
            sha256_file(all_layers_path) if all_layers else None
        ),
    }
    write_json(output_dir / "config.json", config)
    return config

