"""Build and freeze provenance_v1: real data provenance crossed with format.

Every existing purpose result in this project uses a purpose variable that WE
assign -- a stated cue, an implied context, or a relational binding. This bank
replaces the assigned variable with one we do not control:

    provenance = was this text actually an evaluation item, or actually a human
                 turn addressed to an assistant?

Ground truth is the ORIGIN of the text, not a label we stipulate. The frozen
stated-purpose probes can then be applied without retraining, testing whether a
direction learned from synthetic cues aligns with real eval-vs-deployment
provenance.

THE CENTRAL HAZARD, and how it is handled: eval corpora are benchmark-shaped and
user corpora are free-form, so provenance is naturally correlated with format --
the exact confound this project studies. Therefore provenance is CROSSED with
format, never correlated with it: every selected text is rendered once under a
benchmark scaffold and once under a casual scaffold, so
    provenance _|_ format
holds by construction, and the format factor is ours while the provenance factor
is the corpus's.

REMAINING LIMITATION, stated plainly and measured by the audit: in natural
corpora provenance is inseparable from topic and lexical content. A surface-text
baseline WILL predict provenance well above chance, and that is not a bug in the
bank -- it is a property of real data. The audit reports that ceiling so any
later probe result is read against it. What is cleanly identified here is the
FORMAT manipulation and the paired within-text contrast, not a content-free
notion of provenance.

Sources (all open, non-gated; revisions recorded in the freeze manifest):
  eval_origin       cais/mmlu (test), allenai/ai2_arc (ARC-Challenge test)
  deployment_origin OpenAssistant/oasst1 (train, first-turn human prompts),
                    HuggingFaceH4/no_robots (train_sft, first human turn)

Only the prompt text is used. MCQ options are dropped so the two provenance
sides carry the same shape of content and the only systematic surface
difference is the scaffold we add. Scaffolds are rotated over several benchmark
and several casual variants drawn from this project's own no-cue renderings, so
no single template can drive a later result.

Frozen before activations; an existing non-identical freeze is never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binding_v1_build import (  # noqa: E402
    BANNED_LABEL_WORDS, BANNED_OLD_CUE_VOCAB, canonical_sha256, sha256_bytes, tokens,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260902
N_PER_SIDE = 500
MIN_CHARS, MAX_CHARS = 60, 600

SOURCES = {
    "eval_origin": [
        ("cais/mmlu", "all/test-00000-of-00001.parquet", "question"),
        ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet", "question"),
    ],
    "deployment_origin": [
        ("OpenAssistant/oasst1", "data/train-00000-of-00001-b42a775f407cee45.parquet", "_oasst"),
        ("HuggingFaceH4/no_robots", "data/train_sft-00000-of-00001.parquet", "_messages"),
    ],
}

# Benchmark and casual scaffolds, options-free, in the style of the project's own
# no-cue templates (bench_imperative_v2 / bench_preamble_v2 / bench_sectioned_v2
# and deploy_bare_v2 / deploy_lead_in_v2 / deploy_markdown_v2). {q} is the text.
BENCH_SCAFFOLDS = {
    "prov_bench_imperative": "Answer the following question.\n\nQuestion:\n{q}\n\nGive your final answer.",
    "prov_bench_preamble": "The following is a question. Read it carefully and answer it.\n\nQuestion: {q}\n\nAnswer:",
    "prov_bench_sectioned": "### Question\n{q}\n\n### Answer\n",
    "prov_bench_tagged": "[QUESTION]\n{q}\n[/QUESTION]\n\nProvide the answer below.",
}
CASUAL_SCAFFOLDS = {
    "prov_casual_bare": "{q}",
    "prov_casual_lead_in": "Hey, quick thing I was wondering about: {q}",
    "prov_casual_markdown": "{q}\n\nThanks in advance, whatever detail you can give is fine.",
}

CODEY = re.compile(r"```|def |import |<[a-z]+>|\{\{|\$\$|http[s]?://")
WHITESPACE = re.compile(r"\s+")


def load_column(repo: str, filename: str, column: str, token: str) -> list[str]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, filename, repo_type="dataset", token=token)
    table = pq.read_table(path)
    if column == "_oasst":
        # First-turn human prompts only, English, top-ranked roots.
        cols = table.select(["text", "role", "parent_id", "lang"]).to_pydict()
        return [t for t, r, p, lang in zip(cols["text"], cols["role"], cols["parent_id"], cols["lang"])
                if r == "prompter" and p is None and lang == "en"]
    if column == "_messages":
        out = []
        for msgs in table.column("messages").to_pylist():
            for m in msgs:
                if m.get("role") == "user":
                    out.append(m.get("content", ""))
                    break
        return out
    return table.column(column).to_pylist()


def clean(text: str) -> str | None:
    text = WHITESPACE.sub(" ", (text or "").strip())
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    if CODEY.search(text):
        return None
    if not text.isascii():
        return None
    # Single self-contained request: reject multi-question dumps.
    if text.count("?") > 2:
        return None
    return text


def build(out_dir: Path, token: str) -> None:
    rng = np.random.default_rng(SEED)
    pools: dict[str, list[dict]] = {}
    stats: dict[str, dict] = {}
    banned = BANNED_LABEL_WORDS | BANNED_OLD_CUE_VOCAB
    for provenance, specs in SOURCES.items():
        rows, per_source = [], {}
        for repo, filename, column in specs:
            raw = load_column(repo, filename, column, token)
            kept, dropped_shape, dropped_banned = [], 0, 0
            for i, text in enumerate(raw):
                cleaned = clean(text)
                if cleaned is None:
                    dropped_shape += 1
                    continue
                if tokens(cleaned) & banned:
                    dropped_banned += 1
                    continue
                kept.append({"source_dataset": repo, "source_file": filename,
                             "source_row": i, "prompt": cleaned})
            per_source[repo] = {"raw_rows": len(raw), "dropped_shape_or_length": dropped_shape,
                               "dropped_banned_vocabulary": dropped_banned, "eligible": len(kept)}
            rows.extend(kept)
        pools[provenance] = rows
        stats[provenance] = per_source
        print(f"{provenance}: {len(rows)} eligible from {len(specs)} sources", file=sys.stderr)

    # Length-match the two sides by sampling within shared character-length bins,
    # so any later provenance effect is not a length effect.
    edges = np.linspace(MIN_CHARS, MAX_CHARS, 13)
    binned = {p: collections.defaultdict(list) for p in pools}
    for p, rows in pools.items():
        for r in rows:
            binned[p][int(np.digitize(len(r["prompt"]), edges))].append(r)
    selected: dict[str, list[dict]] = {p: [] for p in pools}
    shared = sorted(set(binned["eval_origin"]) & set(binned["deployment_origin"]))
    quota = {b: min(len(binned["eval_origin"][b]), len(binned["deployment_origin"][b])) for b in shared}
    total_available = sum(quota.values())
    if total_available < N_PER_SIDE:
        raise RuntimeError(f"only {total_available} length-matched pairs available, need {N_PER_SIDE}")
    # Proportional draw from each bin, capped by the smaller side.
    order = sorted(shared, key=lambda b: -quota[b])
    remaining = N_PER_SIDE
    take = {}
    for b in order:
        share = int(round(N_PER_SIDE * quota[b] / total_available))
        take[b] = min(share, quota[b], remaining)
        remaining -= take[b]
    for b in order:
        if remaining <= 0:
            break
        extra = min(quota[b] - take[b], remaining)
        take[b] += extra
        remaining -= extra
    for p in pools:
        for b, k in take.items():
            pool = binned[p][b]
            idx = rng.choice(len(pool), k, replace=False)
            selected[p].extend(pool[i] for i in sorted(idx))
    for p in selected:
        if len(selected[p]) != N_PER_SIDE:
            raise RuntimeError(f"{p}: selected {len(selected[p])}, expected {N_PER_SIDE}")

    # Render each text under one benchmark and one casual scaffold, rotating
    # scaffolds so scaffold is orthogonal to provenance.
    bench_ids = sorted(BENCH_SCAFFOLDS)
    casual_ids = sorted(CASUAL_SCAFFOLDS)
    rows = []
    for provenance in sorted(selected):
        for j, item in enumerate(sorted(selected[provenance], key=lambda r: (r["source_dataset"], r["source_row"]))):
            pair_id = f"{provenance}::{item['source_dataset'].replace('/', '_')}::{item['source_row']}"
            for fmt, scaffold_id in (("benchmark", bench_ids[j % len(bench_ids)]),
                                     ("casual", casual_ids[j % len(casual_ids)])):
                template = (BENCH_SCAFFOLDS if fmt == "benchmark" else CASUAL_SCAFFOLDS)[scaffold_id]
                text = template.format(q=item["prompt"])
                rows.append({
                    "item_id": f"provenance_v1::{pair_id}::{fmt}::{scaffold_id}",
                    "text": text,
                    "provenance": provenance,
                    "label": int(provenance == "eval_origin"),
                    "format": fmt,
                    "template_id": scaffold_id,
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

    expected = 2 * N_PER_SIDE * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, built {len(rows)}")
    # provenance _|_ format, exactly.
    cross = collections.Counter((r["provenance"], r["format"]) for r in rows)
    if len(set(cross.values())) != 1:
        raise RuntimeError(f"provenance x format not balanced: {cross}")
    if len({r["pair_id"] for r in rows}) != 2 * N_PER_SIDE:
        raise RuntimeError("pair_id is not unique per underlying text")

    out_dir.mkdir(parents=True, exist_ok=True)
    items = out_dir / "items.jsonl"
    rendered = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    from huggingface_hub import HfApi
    hf = HfApi(token=token)
    revisions = {}
    for specs in SOURCES.values():
        for repo, _, _ in specs:
            info = hf.dataset_info(repo)
            revisions[repo] = {"revision": info.sha,
                               "license": (info.card_data or {}).get("license") if info.card_data else None}
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ACTIVATIONS",
        "experiment": "provenance_v1",
        "seed": SEED,
        "question": "does a direction fit on SYNTHETIC stated purpose cues align with REAL "
                    "eval-vs-deployment data provenance",
        "label_definition": "label = 1 iff the text actually came from an evaluation corpus",
        "design": "provenance (corpus origin, not assigned) x format (ours) x scaffold; every "
                  "underlying text appears exactly once per format, so provenance is orthogonal "
                  "to format by construction",
        "independent_inference_unit": "source_dataset (and pair_id for paired contrasts)",
        "known_limitation": "in natural corpora provenance is inseparable from topic and lexical "
                            "content; a surface-text baseline predicts provenance well above "
                            "chance and the audit reports that ceiling. The cleanly identified "
                            "factor here is format, plus the paired within-text contrast.",
        "n_rows": len(rows),
        "n_texts_per_provenance": N_PER_SIDE,
        "prompt_char_range": [MIN_CHARS, MAX_CHARS],
        "length_matching": "sampled within shared character-length bins across the two provenance sides",
        "sources": {p: [{"repo": r, "file": f, "field": c} for r, f, c in s] for p, s in SOURCES.items()},
        "source_revisions": revisions,
        "source_selection_stats": stats,
        "benchmark_scaffolds": BENCH_SCAFFOLDS,
        "casual_scaffolds": CASUAL_SCAFFOLDS,
        "exclusions": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
                       "reject_regex": CODEY.pattern, "max_question_marks": 2,
                       "banned_vocabulary": sorted(banned)},
        "items_sha256": sha256_bytes(rendered.encode("utf-8")),
    }
    manifest["manifest_sha256_of_spec"] = canonical_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256_of_spec"})

    if items.exists():
        from binding_v1_build import sha256_file
        if sha256_file(items) != manifest["items_sha256"]:
            raise RuntimeError(f"existing non-identical freeze at {items}; refusing to overwrite")
        print("identical freeze already present; nothing rewritten")
        return
    # Binary writes: text mode would translate \n to \r\n on Windows and break
    # the recorded items_sha256.
    items.write_bytes(rendered.encode("utf-8"))
    (out_dir / "freeze_manifest.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    (out_dir / "FROZEN.sha256").write_bytes(
        (f"items.jsonl {manifest['items_sha256']}\n"
         f"spec {manifest['manifest_sha256_of_spec']}\n").encode("utf-8"))
    print(f"wrote {len(rows)} rows ({N_PER_SIDE} texts per provenance side) to {items}")
    print(f"items sha256 {manifest['items_sha256']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "provenance_v1_frozen")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        env = ROOT / ".env.local"
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?HF_TOKEN\s*=\s*(.+)", line)
            if m:
                token = m.group(1).strip().strip('"').strip("'")
    if not token:
        raise SystemExit("HF_TOKEN not found")
    build(args.out_dir, token)


if __name__ == "__main__":
    main()
