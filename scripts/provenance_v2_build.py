"""Build provenance_v2 from evaluation corpora and real ChatGPT logs.

Positive class (evaluation provenance): MMLU and ARC-Challenge test items.
Negative class (deployment-log provenance): first user turns from public WildChat
ChatGPT interaction logs. This is the first bank in this project where neither
class is assigned by us.

Every underlying text is rendered under one benchmark and one casual scaffold,
so provenance and the manipulated response-format carrier are exactly crossed.
The two classes are nearest-neighbor matched on pre-scaffold character length.
MMLU and ARC contribute 250 texts each; WildChat contributes 500.

Limits: one deployment-log source; natural topic/lexical content remains
inseparable from provenance; frozen-activation probing is not behavior; corpus
origin is not proof that the model represents its own evaluation status.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from binding_v1_build import (
    BANNED_LABEL_WORDS,
    BANNED_OLD_CUE_VOCAB,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
    tokens,
)
from provenance_v1_build import BENCH_SCAFFOLDS, CASUAL_SCAFFOLDS, clean

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260905
N_EVAL_PER_SOURCE = 250
N_DEPLOY = 500
WILDCHAT_FILE = "data/train-00000-of-00014.parquet"
SOURCES = {
    "eval_corpus_origin": [
        ("cais/mmlu", "all/test-00000-of-00001.parquet", "question"),
        ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet", "question"),
    ],
    "deployment_log_origin": [
        ("allenai/WildChat-1M", WILDCHAT_FILE, "first_user_turn"),
    ],
}


def token_from_env() -> str:
    token = os.environ.get("HF_TOKEN")
    env = ROOT / ".env.local"
    if not token and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?HF_TOKEN\s*=\s*(.+)", line)
            if m:
                token = m.group(1).strip().strip('"').strip("'")
    if not token:
        raise RuntimeError("HF_TOKEN not found")
    return token


def load_eval(repo: str, filename: str, field: str, token: str) -> list[dict]:
    path = hf_hub_download(repo, filename, repo_type="dataset", token=token)
    values = pq.read_table(path, columns=[field]).column(field).to_pylist()
    return [{"source_dataset": repo, "source_file": filename,
             "source_row": i, "prompt": value}
            for i, value in enumerate(values)]


def load_wildchat(token: str) -> list[dict]:
    path = hf_hub_download(
        "allenai/WildChat-1M", WILDCHAT_FILE, repo_type="dataset", token=token
    )
    table = pq.read_table(
        path,
        columns=["conversation_hash", "conversation", "language", "redacted", "toxic"],
    ).to_pydict()
    rows = []
    for i, (conversation_hash, conversation, language, redacted, toxic) in enumerate(
        zip(
            table["conversation_hash"], table["conversation"], table["language"],
            table["redacted"], table["toxic"],
        )
    ):
        if language != "English" or redacted or toxic or not conversation:
            continue
        first = conversation[0]
        if first.get("role") != "user" or first.get("redacted") or first.get("toxic"):
            continue
        rows.append({
            "source_dataset": "allenai/WildChat-1M",
            "source_file": WILDCHAT_FILE,
            "source_row": i,
            "source_conversation_hash": conversation_hash,
            "prompt": first.get("content", ""),
        })
    return rows


def eligible(rows: list[dict], banned: set[str]) -> tuple[list[dict], dict]:
    kept, shape, vocabulary = [], 0, 0
    for row in rows:
        prompt = clean(row["prompt"])
        if prompt is None:
            shape += 1
        elif tokens(prompt) & banned:
            vocabulary += 1
        else:
            kept.append({**row, "prompt": prompt})
    return kept, {
        "raw_rows": len(rows),
        "dropped_shape_or_length": shape,
        "dropped_banned_vocabulary": vocabulary,
        "eligible": len(kept),
    }


def nearest_without_replacement(target_lengths: list[int], pool: list[dict]) -> list[dict]:
    ordered = sorted((len(r["prompt"]), i, r) for i, r in enumerate(pool))
    keys = [(length, i) for length, i, _ in ordered]
    chosen = []
    for target in sorted(target_lengths):
        pos = bisect.bisect_left(keys, (target, -1))
        candidates = [p for p in (pos - 1, pos) if 0 <= p < len(keys)]
        best = min(candidates, key=lambda p: (abs(keys[p][0] - target), keys[p]))
        chosen.append(ordered.pop(best)[2])
        keys.pop(best)
    return chosen


def build(out_dir: Path, token: str) -> None:
    rng = np.random.default_rng(SEED)
    banned = BANNED_LABEL_WORDS | BANNED_OLD_CUE_VOCAB
    stats = {}
    eval_selected = []
    for repo, filename, field in SOURCES["eval_corpus_origin"]:
        pool, report = eligible(load_eval(repo, filename, field, token), banned)
        stats[repo] = report
        if len(pool) < N_EVAL_PER_SOURCE:
            raise RuntimeError(f"{repo}: only {len(pool)} eligible rows")
        idx = rng.choice(len(pool), N_EVAL_PER_SOURCE, replace=False)
        eval_selected.extend(pool[int(i)] for i in sorted(idx))
    deploy_pool, report = eligible(load_wildchat(token), banned)
    stats["allenai/WildChat-1M"] = report
    deploy_selected = nearest_without_replacement(
        [len(r["prompt"]) for r in eval_selected], deploy_pool
    )
    if len(deploy_selected) != N_DEPLOY:
        raise RuntimeError("deployment matching failed")

    selected = {
        "eval_corpus_origin": eval_selected,
        "deployment_log_origin": deploy_selected,
    }
    bench_ids, casual_ids = sorted(BENCH_SCAFFOLDS), sorted(CASUAL_SCAFFOLDS)
    rows = []
    for provenance in sorted(selected):
        ordered = sorted(
            selected[provenance],
            key=lambda r: (r["source_dataset"], r["source_row"]),
        )
        for j, item in enumerate(ordered):
            source_key = item.get("source_conversation_hash", str(item["source_row"]))
            pair_id = (
                f"{provenance}::{item['source_dataset'].replace('/', '_')}::{source_key}"
            )
            for fmt, template_id in (
                ("benchmark", bench_ids[j % len(bench_ids)]),
                ("casual", casual_ids[j % len(casual_ids)]),
            ):
                scaffold = (
                    BENCH_SCAFFOLDS if fmt == "benchmark" else CASUAL_SCAFFOLDS
                )[template_id]
                text = scaffold.format(q=item["prompt"])
                rows.append({
                    "item_id": f"provenance_v2::{pair_id}::{fmt}::{template_id}",
                    "text": text,
                    "provenance": provenance,
                    "label": int(provenance == "eval_corpus_origin"),
                    "format": fmt,
                    "template_id": template_id,
                    "pair_id": pair_id,
                    "source_dataset": item["source_dataset"],
                    "source_file": item["source_file"],
                    "source_row": item["source_row"],
                    "prompt_chars": len(item["prompt"]),
                    "render_slot": "provenance",
                    "cue_style": "none",
                    "cue_text": None,
                    "intended_purpose": None,
                    "rendered_text_sha256": sha256_bytes(text.encode("utf-8")),
                })

    cross = collections.Counter((r["provenance"], r["format"]) for r in rows)
    if cross != {
        ("eval_corpus_origin", "benchmark"): 500,
        ("eval_corpus_origin", "casual"): 500,
        ("deployment_log_origin", "benchmark"): 500,
        ("deployment_log_origin", "casual"): 500,
    }:
        raise RuntimeError(f"unbalanced provenance x format: {cross}")
    rendered = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
    )
    api = HfApi(token=token)
    revisions = {}
    for specs in SOURCES.values():
        for repo, _, _ in specs:
            info = api.dataset_info(repo)
            revisions[repo] = {
                "revision": info.sha,
                "license": (info.card_data or {}).get("license") if info.card_data else None,
            }
    eval_lengths = np.array([len(r["prompt"]) for r in eval_selected])
    deploy_lengths = np.array([len(r["prompt"]) for r in deploy_selected])
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ACTIVATIONS",
        "experiment": "provenance_v2",
        "seed": SEED,
        "label_definition": "label = 1 iff text originates in MMLU/ARC evaluation corpora; "
                            "label = 0 iff text is a first user turn from a real WildChat "
                            "ChatGPT interaction log",
        "design": "corpus provenance crossed exactly with manipulated benchmark/casual "
                  "scaffold; every underlying text appears once per format",
        "independent_inference_unit": "complete underlying-text pair",
        "source_robustness_units": ["cais/mmlu", "allenai/ai2_arc"],
        "known_limitations": [
            "only one deployment-log source (WildChat)",
            "natural topic and lexical content are inseparable from provenance",
            "corpus origin is not proof that the model represents its own evaluation status",
            "frozen-activation probe scoring is not a behavioral or causal intervention",
        ],
        "n_rows": len(rows),
        "n_texts_per_class": 500,
        "sources": {p: [{"repo": r, "file": f, "field": c} for r, f, c in specs]
                    for p, specs in SOURCES.items()},
        "source_revisions": revisions,
        "source_selection_stats": stats,
        "length_matching": {
            "method": "nearest-neighbor without replacement on pre-scaffold character length",
            "eval_mean": float(eval_lengths.mean()),
            "deployment_mean": float(deploy_lengths.mean()),
            "mean_difference": float(eval_lengths.mean() - deploy_lengths.mean()),
            "max_absolute_pair_difference": int(
                np.max(np.abs(np.sort(eval_lengths) - np.sort(deploy_lengths)))
            ),
        },
        "benchmark_scaffolds": BENCH_SCAFFOLDS,
        "casual_scaffolds": CASUAL_SCAFFOLDS,
        "items_sha256": sha256_bytes(rendered.encode("utf-8")),
    }
    manifest["manifest_sha256_of_spec"] = canonical_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256_of_spec"}
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    items = out_dir / "items.jsonl"
    if items.exists():
        if sha256_file(items) != manifest["items_sha256"]:
            raise RuntimeError(f"existing non-identical freeze at {items}")
        print("identical freeze already present")
        return
    items.write_bytes(rendered.encode("utf-8"))
    (out_dir / "freeze_manifest.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    )
    (out_dir / "FROZEN.sha256").write_bytes(
        (f"items.jsonl {manifest['items_sha256']}\n"
         f"spec {manifest['manifest_sha256_of_spec']}\n").encode("utf-8")
    )
    print(f"wrote {len(rows)} rows to {items}")
    print(f"items sha256 {manifest['items_sha256']}")
    print(f"length match {manifest['length_matching']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "provenance_v2_frozen")
    args = ap.parse_args()
    build(args.out_dir, token_from_env())


if __name__ == "__main__":
    main()
