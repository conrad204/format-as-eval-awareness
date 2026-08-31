"""Build provenance_v3: evaluation corpora vs TWO independent deployment logs.

provenance_v2 had a single deployment-log source, so "does this generalize
across log corpora" was untestable. v3 adds a second, independent log corpus:

  eval_corpus_origin      MMLU (250) + ARC-Challenge (250)
  deployment_log_origin   WildChat-1M (250) + LMSYS-Chat-1M (250)

That makes a genuine 2 x 2 source-holdout grid possible, and it makes
leave-one-deployment-corpus-out a real test rather than a relabeling.

THE TWO LOG CORPORA ARE NOT INTERCHANGEABLE, and this is deliberate:

  * WildChat is free general-purpose ChatGPT access -- ordinary use.
  * LMSYS-Chat-1M is Chatbot Arena traffic, where users are deliberately
    comparing and probing models. Arena traffic is arguably itself a mild
    evaluation context.

So the two are not two samples of one "deployment" population. Treating them as
such would hide a real distinction. Instead both are carried separately, and any
consistent WildChat-vs-LMSYS difference in the frozen eval-ness score is
reported as a finding in its own right rather than averaged away. A probe that
truly tracked "am I being evaluated" should, if anything, score arena traffic
above general use.

Length matching: the 500 eval texts are sorted by pre-scaffold character length
and their targets dealt alternately to the two log corpora, so each log corpus is
nearest-neighbor matched to a systematic half of the eval length distribution and
the two log corpora are therefore also matched to each other.

Every text is rendered once under a benchmark scaffold and once under a casual
scaffold, so provenance stays exactly crossed with the manipulated format.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from binding_v1_build import canonical_sha256, sha256_bytes, sha256_file
from provenance_v1_build import BENCH_SCAFFOLDS, CASUAL_SCAFFOLDS
from provenance_v2_build import (
    eligible,
    load_eval,
    load_wildchat,
    nearest_without_replacement,
    token_from_env,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260906
N_EVAL_PER_SOURCE = 250
N_DEPLOY_PER_SOURCE = 250
LMSYS_FILE = "data/train-00000-of-00006-4feeb3f83346a0e9.parquet"
WILDCHAT_FILE = "data/train-00000-of-00014.parquet"

EVAL_SOURCES = [
    ("cais/mmlu", "all/test-00000-of-00001.parquet", "question"),
    ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet", "question"),
]
LOG_SOURCES = ["allenai/WildChat-1M", "lmsys/lmsys-chat-1m"]
# Recorded so the interpretation travels with the data.
LOG_CONTEXT = {
    "allenai/WildChat-1M": "general-purpose free ChatGPT access (ordinary use)",
    "lmsys/lmsys-chat-1m": "Chatbot Arena traffic (users deliberately comparing models; "
                           "arguably a mild evaluation context itself)",
}


def load_lmsys(token: str) -> list[dict]:
    path = hf_hub_download("lmsys/lmsys-chat-1m", LMSYS_FILE, repo_type="dataset", token=token)
    table = pq.read_table(
        path,
        columns=["conversation_id", "conversation", "language", "redacted", "openai_moderation"],
    ).to_pydict()
    rows = []
    for i, (cid, conversation, language, redacted, moderation) in enumerate(
        zip(table["conversation_id"], table["conversation"], table["language"],
            table["redacted"], table["openai_moderation"])
    ):
        if language != "English" or redacted or not conversation:
            continue
        first = conversation[0]
        if first.get("role") != "user":
            continue
        if moderation and moderation[0] and moderation[0].get("flagged"):
            continue
        rows.append({
            "source_dataset": "lmsys/lmsys-chat-1m",
            "source_file": LMSYS_FILE,
            "source_row": i,
            "source_conversation_hash": cid,
            "prompt": first.get("content", ""),
        })
    return rows


def build(out_dir: Path, token: str) -> None:
    rng = np.random.default_rng(SEED)
    from binding_v1_build import BANNED_LABEL_WORDS, BANNED_OLD_CUE_VOCAB
    banned = BANNED_LABEL_WORDS | BANNED_OLD_CUE_VOCAB
    stats: dict[str, dict] = {}

    eval_selected: list[dict] = []
    for repo, filename, field in EVAL_SOURCES:
        pool, report = eligible(load_eval(repo, filename, field, token), banned)
        stats[repo] = report
        if len(pool) < N_EVAL_PER_SOURCE:
            raise RuntimeError(f"{repo}: only {len(pool)} eligible")
        idx = rng.choice(len(pool), N_EVAL_PER_SOURCE, replace=False)
        eval_selected.extend(pool[int(i)] for i in sorted(idx))

    log_pools = {}
    for repo, loader in (("allenai/WildChat-1M", load_wildchat),
                         ("lmsys/lmsys-chat-1m", load_lmsys)):
        pool, report = eligible(loader(token), banned)
        report["context"] = LOG_CONTEXT[repo]
        stats[repo] = report
        log_pools[repo] = pool

    # Deal length targets alternately so each log corpus matches a systematic
    # half of the eval length distribution, and the two halves match each other.
    ordered_lengths = sorted(len(r["prompt"]) for r in eval_selected)
    targets = {LOG_SOURCES[0]: ordered_lengths[0::2], LOG_SOURCES[1]: ordered_lengths[1::2]}
    deploy_selected: list[dict] = []
    for repo in LOG_SOURCES:
        want = targets[repo]
        if len(want) != N_DEPLOY_PER_SOURCE:
            raise RuntimeError(f"{repo}: {len(want)} targets, expected {N_DEPLOY_PER_SOURCE}")
        deploy_selected.extend(nearest_without_replacement(want, log_pools[repo]))

    selected = {"eval_corpus_origin": eval_selected, "deployment_log_origin": deploy_selected}
    bench_ids, casual_ids = sorted(BENCH_SCAFFOLDS), sorted(CASUAL_SCAFFOLDS)
    rows = []
    for provenance in sorted(selected):
        ordered = sorted(selected[provenance],
                         key=lambda r: (r["source_dataset"], r["source_row"]))
        for j, item in enumerate(ordered):
            source_key = item.get("source_conversation_hash", str(item["source_row"]))
            pair_id = f"{provenance}::{item['source_dataset'].replace('/', '_')}::{source_key}"
            for fmt, template_id in (("benchmark", bench_ids[j % len(bench_ids)]),
                                     ("casual", casual_ids[j % len(casual_ids)])):
                scaffold = (BENCH_SCAFFOLDS if fmt == "benchmark" else CASUAL_SCAFFOLDS)[template_id]
                text = scaffold.format(q=item["prompt"])
                rows.append({
                    "item_id": f"provenance_v3::{pair_id}::{fmt}::{template_id}",
                    "text": text,
                    "provenance": provenance,
                    "label": int(provenance == "eval_corpus_origin"),
                    "format": fmt,
                    "template_id": template_id,
                    "pair_id": pair_id,
                    "source_dataset": item["source_dataset"],
                    "source_file": item["source_file"],
                    "source_row": item["source_row"],
                    "source_context": LOG_CONTEXT.get(item["source_dataset"], "evaluation corpus"),
                    "prompt_chars": len(item["prompt"]),
                    "render_slot": "provenance",
                    "cue_style": "none",
                    "cue_text": None,
                    "intended_purpose": None,
                    "rendered_text_sha256": sha256_bytes(text.encode("utf-8")),
                })

    cross = collections.Counter((r["provenance"], r["format"]) for r in rows)
    if set(cross.values()) != {500}:
        raise RuntimeError(f"unbalanced provenance x format: {cross}")
    per_source = collections.Counter((r["source_dataset"], r["format"]) for r in rows)
    if set(per_source.values()) != {250}:
        raise RuntimeError(f"unbalanced source x format: {per_source}")
    if len({r["pair_id"] for r in rows}) != 1000:
        raise RuntimeError("pair_id not unique per text")

    rendered = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    api = HfApi(token=token)
    revisions = {}
    for repo in [s[0] for s in EVAL_SOURCES] + LOG_SOURCES:
        info = api.dataset_info(repo)
        revisions[repo] = {
            "revision": info.sha,
            "license": (info.card_data or {}).get("license") if info.card_data else None,
            "gated": info.gated,
        }
    lengths = {repo: np.array([len(r["prompt"]) for r in deploy_selected
                               if r["source_dataset"] == repo], dtype=float)
               for repo in LOG_SOURCES}
    eval_lengths = np.array([len(r["prompt"]) for r in eval_selected], dtype=float)
    deploy_lengths = np.concatenate([lengths[r] for r in LOG_SOURCES])
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ACTIVATIONS",
        "experiment": "provenance_v3",
        "seed": SEED,
        "label_definition": "label = 1 iff the text originates in an evaluation corpus "
                            "(MMLU/ARC); label = 0 iff it is a first user turn from a real "
                            "deployment log (WildChat or LMSYS-Chat-1M)",
        "design": "corpus provenance x manipulated benchmark/casual scaffold, with TWO "
                  "independent deployment-log corpora so leave-one-log-corpus-out is testable",
        "independent_inference_unit": "complete underlying-text pair",
        "source_robustness_grid": {
            "eval_sources": [s[0] for s in EVAL_SOURCES],
            "deployment_log_sources": LOG_SOURCES,
            "n_source_pairs": len(EVAL_SOURCES) * len(LOG_SOURCES),
        },
        "log_corpus_context": LOG_CONTEXT,
        "log_corpora_are_not_interchangeable": (
            "LMSYS is Chatbot Arena traffic, where users deliberately compare models, so it is "
            "arguably a mild evaluation context; WildChat is general free ChatGPT use. Any "
            "consistent difference between them in the frozen eval-ness score is reported "
            "separately rather than averaged away."
        ),
        "known_limitations": [
            "natural topic and lexical content remain inseparable from provenance; the audit "
            "reports the surface ceiling that bounds every interpretation",
            "corpus origin is not the model's own judgment about being evaluated",
            "frozen-activation probe scoring is not behavior and not a causal intervention",
            "both log corpora are public research releases, not private production traffic",
        ],
        "n_rows": len(rows),
        "n_texts_per_class": 500,
        "n_texts_per_source": 250,
        "sources": {
            "eval_corpus_origin": [{"repo": r, "file": f, "field": c} for r, f, c in EVAL_SOURCES],
            "deployment_log_origin": [{"repo": r, "context": LOG_CONTEXT[r]} for r in LOG_SOURCES],
        },
        "source_revisions": revisions,
        "source_selection_stats": stats,
        "length_matching": {
            "method": "eval lengths sorted and dealt alternately to the two log corpora, then "
                      "nearest-neighbor matched without replacement within each corpus",
            "eval_mean": float(eval_lengths.mean()),
            "deployment_mean": float(deploy_lengths.mean()),
            "mean_difference": float(eval_lengths.mean() - deploy_lengths.mean()),
            "max_absolute_sorted_difference": float(
                np.max(np.abs(np.sort(eval_lengths) - np.sort(deploy_lengths)))
            ),
            "per_log_corpus_mean": {r: float(v.mean()) for r, v in lengths.items()},
            "log_corpus_mean_difference": float(
                lengths[LOG_SOURCES[0]].mean() - lengths[LOG_SOURCES[1]].mean()
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
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    (out_dir / "FROZEN.sha256").write_bytes(
        (f"items.jsonl {manifest['items_sha256']}\n"
         f"spec {manifest['manifest_sha256_of_spec']}\n").encode("utf-8"))
    print(f"wrote {len(rows)} rows to {items}")
    print(f"items sha256 {manifest['items_sha256']}")
    print(f"length matching {json.dumps(manifest['length_matching'], indent=2)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "provenance_v3_frozen")
    build(ap.parse_args().out_dir, token_from_env())


if __name__ == "__main__":
    main()
