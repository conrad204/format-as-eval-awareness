"""Tokenizer-derived metadata as a sidecar pass (validator plan section 10).

`prompt_token_length` and `terminal_token_id` are first-order covariates for
last-token probes, but computing them inside render/validate would inject a
transformers dependency and cross-machine variance into a pure deterministic
loop. This pass therefore runs separately: it reads a rendered JSONL, mirrors
extract.py's prompt construction exactly (chat template rendered to text with
add_generation_prompt=True, then tokenized with add_special_tokens=False),
and writes a sidecar JSONL keyed by item_id plus a manifest pinning the
tokenizer and hashing input and output.

The terminal token recorded here is the last token of the full templated
prompt — the position extract.py actually probes (after the assistant
header), which is why both the content-terminal and prompt-terminal ids are
recorded: their divergence is exactly the section-7.4 question the plan
wants settled empirically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import MVPError, load_jsonl, sha256_file, write_json, write_jsonl


def annotate_tokens(
    *,
    items_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_name: str,
    revision: str | None = None,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise MVPError(
            "annotate-tokens needs transformers (pip install -e '.[extract]')"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    rows = load_jsonl(items_path)
    if not rows:
        raise MVPError(f"No rows in {items_path}")
    sidecar: list[dict[str, Any]] = []
    for row in rows:
        content = str(row["text"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, truncation=False
        )["input_ids"]
        content_ids = tokenizer(
            content, add_special_tokens=False, truncation=False
        )["input_ids"]
        sidecar.append(
            {
                "item_id": row["item_id"],
                "prompt_token_length": len(prompt_ids),
                "prompt_terminal_token_id": int(prompt_ids[-1]),
                "prompt_terminal_token_text": tokenizer.decode([prompt_ids[-1]]),
                "content_terminal_token_id": int(content_ids[-1]),
                "content_terminal_token_text": tokenizer.decode([content_ids[-1]]),
            }
        )
    write_jsonl(output_path, sidecar)
    manifest = {
        "manifest_kind": "token_annotation_v1",
        "tokenizer": {"model": model_name, "revision": revision},
        "probe_position_note": (
            "extract.py probes the last token of the templated prompt "
            "(assistant header), not the last content token; both terminal "
            "ids are recorded per item so their divergence is measurable."
        ),
        "prompt_construction": (
            "apply_chat_template(user_message, tokenize=False, "
            "add_generation_prompt=True); add_special_tokens=False"
        ),
        "inputs": {"items": {"path": str(items_path), "sha256": sha256_file(items_path)}},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "items": len(sidecar),
    }
    write_json(manifest_path, manifest)
    return manifest
