"""Adapt Devbunova's eval-awareness-2x2 parquets into our rendered-items jsonl shape.

Bridges the vendored third-party dataset (external/eval-awareness-2x2/, verified
byte-identical to hf.co/datasets/viliana-dev/eval-awareness-2x2 @ a50e4c98 — see
provenance/devbunova-2x2/hf_snapshot.json) into the jsonl schema that
scripts/extract_activations.py and the NDIF drivers consume: one row per item
with `item_id` and `text`, plus her Context/Format/Genre taxonomy as metadata.

The prompt strings are taken verbatim from the parquets — her probe scripts and
her HF upload script build them with the same `format_bench_eval_prompt`
function (external/eval-awareness-format/scripts/{evaluate_probe_cross,
upload_to_hf}.py), which this adapter re-verifies byte-for-byte against the
`formatted_prompt` column before writing anything.

Besides the six cells as-is, the three benchmark-format cells are emitted a
second time with the trailing "\\nAnswer:"/"\\nSolution:" scaffold stripped
(variant "stripped", per-item `stripped_changed` flag): scoring the confounded
probe on both variants measures how much of the format trap the dangling
scaffold token alone carries. 41/835 bench_deploy_rewritten and 49/835
bench_deploy_formal rows have no scaffold and pass through unchanged.

Outputs (deterministic; rerunning must reproduce the manifest sha256):
  data/devbunova-2x2/items_devbunova_2x2.jsonl   (gitignored, regenerable)
  provenance/devbunova-2x2/adapter_manifest.json (tracked anchor)

Run from the repo root:  python scripts/adapt_devbunova.py
Requires pyarrow:        pip install -e ".[external]"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_format_mvp.io import sha256_file, write_json, write_jsonl  # noqa: E402

EXTERNAL = ROOT / "external" / "eval-awareness-2x2"
SNAPSHOT = ROOT / "provenance" / "devbunova-2x2" / "hf_snapshot.json"
OUT_DIR = ROOT / "data" / "devbunova-2x2"
OUT_JSONL = OUT_DIR / "items_devbunova_2x2.jsonl"
MANIFEST = ROOT / "provenance" / "devbunova-2x2" / "adapter_manifest.json"

# (cell, prompt_column, context, format, genre, source)  — her taxonomy:
# Context = eval|deploy, Format = bench|casual (surface structure),
# Genre = linguistic register. Source column exists only in bench_eval.
CELLS = [
    ("bench_eval", "formatted_prompt", "eval", "bench", "benchmark", None),
    ("bench_deploy_rewritten", "rewritten", "deploy", "bench", "casual", "sharegpt_deepseek_mcq_rewrite"),
    ("casual_deploy_matched", "first_turn", "deploy", "casual", "casual", "sharegpt"),
    ("casual_eval_mutual", "first_turn", "eval", "casual", "casual", "mutual"),
    ("bench_deploy_formal", "rewritten", "deploy", "bench", "formal", "sharegpt_deepseek_mcq_formal_rewrite"),
    ("casual_deploy_formal", "rewritten", "deploy", "casual", "formal", "sharegpt_deepseek_formal_rewrite"),
]
STRIPPED_CELLS = ("bench_eval", "bench_deploy_rewritten", "bench_deploy_formal")
STRIP_PATTERN = re.compile(r"\n+(Answer|Solution):\s*$")


def format_bench_eval_prompt(question: str, choices, source: str) -> str:
    """Byte-exact port of her format_bench_eval_prompt (evaluate_probe_cross.py:53)."""
    if source == "mmlu":
        return f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
    if source == "arc":
        if isinstance(choices, dict):
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choice_str = ", ".join(f"{l}) {t}" for l, t in zip(labels, texts))
        else:
            choice_str = str(choices)
        return f"Question: {question}\nOptions: {choice_str}\nAnswer:"
    if source == "gsm8k":
        return f"Problem: {question}\nSolution:"
    raise ValueError(f"unknown bench_eval source: {source!r}")


def load_cell(cell: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow required: pip install -e '.[external]'") from exc
    path = EXTERNAL / cell / "train.parquet"
    snapshot = json.loads(SNAPSHOT.read_text())
    pinned = next(
        entry for entry in snapshot["files"].values()
        if entry.get("local") == str(path.relative_to(ROOT))
    )
    actual = sha256_file(path)
    if actual != pinned["sha256"]:
        raise SystemExit(f"{path}: sha256 {actual} != pinned {pinned['sha256']}")
    return pq.read_table(path).to_pylist()


def verify_bench_eval_formatting(rows) -> None:
    """The parquet formatted_prompt must be reproducible by her format function."""
    mismatches = 0
    for row in rows:
        choices = json.loads(row["choices"]) if row["choices"] else []
        expect = format_bench_eval_prompt(row["question"], choices, row["source"])
        if expect != row["formatted_prompt"]:
            mismatches += 1
    if mismatches:
        raise SystemExit(
            f"bench_eval: {mismatches}/{len(rows)} formatted_prompt rows do not "
            "match her format function — parquet is not what her probes saw"
        )


def make_row(cell, variant, index, text, meta, source, source_id, stripped_changed=None):
    row = {
        "item_id": f"dev2x2::{cell}::{variant}::{index:04d}",
        "text": text,
        "cell": cell,
        "variant": variant,
        "context": meta[0],
        "format": meta[1],
        "genre": meta[2],
        "source": source,
        "source_id": source_id,
        "orig_index": index,
    }
    if stripped_changed is not None:
        row["stripped_changed"] = stripped_changed
    return row


def main() -> None:
    out_rows = []
    counts = {}
    cache = {}
    for cell, column, context, fmt, genre, fixed_source in CELLS:
        rows = load_cell(cell)
        if cell == "bench_eval":
            verify_bench_eval_formatting(rows)
        cache[cell] = (rows, column, (context, fmt, genre), fixed_source)
        for index, row in enumerate(rows):
            text = str(row[column])
            if not text.strip():
                raise SystemExit(f"{cell}[{index}]: empty prompt")
            source = fixed_source or str(row.get("source", "unknown"))
            out_rows.append(
                make_row(cell, "asis", index, text, (context, fmt, genre), source, str(row["id"]))
            )
        counts[cell] = {"asis": len(rows)}

    for cell in STRIPPED_CELLS:
        rows, column, meta, fixed_source = cache[cell]
        changed = 0
        for index, row in enumerate(rows):
            text = str(row[column])
            stripped = STRIP_PATTERN.sub("", text)
            was_changed = stripped != text
            changed += was_changed
            out_rows.append(
                make_row(cell, "stripped", index, stripped, meta,
                         fixed_source or str(row.get("source", "unknown")),
                         str(row["id"]), stripped_changed=was_changed)
            )
        counts[cell]["stripped"] = len(rows)
        counts[cell]["stripped_changed"] = changed

    ids = [row["item_id"] for row in out_rows]
    assert len(ids) == len(set(ids)), "item_id collision"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_JSONL, out_rows)

    snapshot = json.loads(SNAPSHOT.read_text())
    manifest = {
        "schema_version": 1,
        "generated_by": "scripts/adapt_devbunova.py",
        "source_hf_repo": snapshot["repo_id"],
        "source_hf_revision": snapshot["revision"],
        "source_parquets": {
            cell: sha256_file(EXTERNAL / cell / "train.parquet")
            for cell, *_ in CELLS
        },
        "strip_pattern": STRIP_PATTERN.pattern,
        "bench_eval_format_check": "all rows reproduced by format_bench_eval_prompt port",
        "cells": counts,
        "n_items": len(out_rows),
        "output": str(OUT_JSONL.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT_JSONL),
        "notes": [
            "Prompt strings are verbatim from the parquets (her published, sampled configs).",
            "Her paper run drew random.sample from the *_full pools with global seed 42; the published n=1076/835 sampled configs are the citable reference but item-set identity with her exact run is not guaranteed.",
            "Her `id` column is non-unique (e.g. 692 duplicate ids in bench_eval); kept as source_id, item_id is index-based.",
            "Duplicate prompt texts exist in her data (96 in casual_eval_mutual, 16 in casual_deploy_matched, ...) and are preserved as-is.",
            "stripped variant: trailing \\n+(Answer|Solution):\\s* removed; rows without the scaffold pass through unchanged (stripped_changed=false).",
        ],
    }
    write_json(MANIFEST, manifest)
    print(f"wrote {len(out_rows)} items -> {OUT_JSONL}")
    for cell, c in counts.items():
        print(f"  {cell}: {c}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
