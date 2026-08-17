# Purpose-probe plot pipeline

Minimal, self-contained chain that produces the eight plots in `plots/`:

```
rendered_items_v1_8-14-2026.jsonl               288 frozen prompts (the 2x2 purpose x format dataset;
                                                archived at hf.co/datasets/format-EA/v1-8-14-2026)
        |  scripts/extract_activations.py        [GPU, RunPod]
        v
activations/<freeze>/<model>/activations_all_layers.npz   (not checked in; regenerable — pilot80-v2 mirrored at hf.co/datasets/format-EA/pilot80-v2)
        |  scripts/sweep_full.py                 [CPU, sklearn]
        v
results/layer_sweep_llama31_8b.json / layer_sweep_llama33_70b.json   (checked in)
        |  scripts/make_plots.py, gap_plots.py, make_plot_06a.py     [CPU, matplotlib]
        v
plots/*.png
```

Because the two layer-sweep JSONs are checked in, **the plots regenerate on any
machine with no GPU** — the extraction + sweep steps only need to be rerun to
change the dataset or the models.

## Plots-only (no GPU)

```bash
pip install -e ".[analysis]"
python scripts/make_plots.py       # 01-05
python scripts/gap_plots.py        # 06, 07
python scripts/make_plot_06a.py    # 06a
```

## Full rerun on RunPod

Models are gated on Hugging Face (`huggingface-cli login` with an approved
token). Exact revisions used for the checked-in results are pinned below and in
the v1 HF repo's `extraction_config_{8b,70b}.json`.

```bash
pip install -e ".[extract,analysis]"

# v1 freeze is archived on HF (private) — pull it first
huggingface-cli download format-EA/v1-8-14-2026 --repo-type dataset \
  --revision "$(jq -r .revision provenance/v1-8-14-2026/hf_snapshot.json)" \
  --local-dir data/v1-8-14-2026

# Llama-3.1-8B-Instruct — fits on one 40GB GPU
python scripts/extract_activations.py \
  --items data/v1-8-14-2026/rendered_items_v1_8-14-2026.jsonl \
  --out-dir activations/llama31_8b \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --all-layers --batch-size 8

# Llama-3.3-70B-Instruct — bf16 needs ~140GB+, e.g. 2x A100-80GB / 2x H100
python scripts/extract_activations.py \
  --items data/v1-8-14-2026/rendered_items_v1_8-14-2026.jsonl \
  --out-dir activations/llama33_70b \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --all-layers --batch-size 8 --device-map auto

# Layer sweep (run on the pod; the 70B npz is ~750MB, the JSON is ~100KB —
# download only the JSON)
python scripts/sweep_full.py data/v1-8-14-2026/rendered_items_v1_8-14-2026.jsonl \
  activations/llama31_8b/activations_all_layers.npz results/layer_sweep_llama31_8b.json
python scripts/sweep_full.py data/v1-8-14-2026/rendered_items_v1_8-14-2026.jsonl \
  activations/llama33_70b/activations_all_layers.npz results/layer_sweep_llama33_70b.json
```

Then rerun the three plot scripts.

## Determinism

The sweep is deterministic (leave-one-family-out folds, fixed
`random_state=0`). Extraction is bf16 on GPU, so re-extracted activations can
differ in the last bits and nudge AUCs in the third decimal; plots are visually
identical. The checked-in JSONs are the frozen reference; input identity is
`sha256(rendered_items_v1_8-14-2026.jsonl) = ebac61db…` (recorded in the
extraction configs archived with the freeze at
hf.co/datasets/format-EA/v1-8-14-2026).

## Dataset pipeline (self-contained since 2026-08-17)

The full generation → render → validate pipeline lives in this repo (it was
previously in a parent repo). The `eval-format-mvp` CLI (installed by
`pip install -e .`) exposes it; run all commands **from the repo root** —
manifest paths are stored repo-relative.

```bash
pip install -e ".[analysis,dev]"
python -m pytest                    # 62 tests, no network/GPU needed

# verify the frozen chain end to end (recomputes every recorded sha256)
eval-format-mvp verify-manifest-v3  --manifest provenance/full320-v3/render_manifest_v3_full320.json
```

### Data: git anchors, HF bytes

`data/` is gitignored. Frozen datasets live in private HF repos —
`format-EA/<freeze>`, one repo per freeze (`full320-v3` current;
`pilot80-v2`, `v1-8-14-2026` archived) — while git tracks the anchors:
`provenance/<freeze>/` holds each freeze's render manifest (+ extraction
configs) and `hf_snapshot.json`, which pins the exact HF revision and
per-file sha256s. To materialize a freeze locally:

```bash
huggingface-cli download format-EA/full320-v3 --repo-type dataset \
  --revision "$(jq -r .revision provenance/full320-v3/hf_snapshot.json)" \
  --local-dir data/full320-v3
eval-format-mvp verify-manifest-v3 --manifest provenance/full320-v3/render_manifest_v3_full320.json
```

Never trust unpinned HF state: always fetch at the revision recorded in the
tracked snapshot and re-verify against the tracked manifest.

Renders are deterministic: `render-v3 --schedule full` is a pure function of
the payload file, so rendered files are always reproducible from their
payloads at the corresponding commit. Each dataset freeze lives in its own
folder (`data/<freeze>/` + `provenance/<freeze>/`, mirrored 1:1 as HF repo
`format-EA/<freeze>`). The frozen v3 dataset is
`data/full320-v3/rendered_items_v3_full320_8-17-2026.jsonl` (11,520 items),
rendered in one pass from
`data/full320-v3/payloads.grok43gemini31_gpt54.full320_8-17-2026.jsonl`:
320 payload blocks, exactly 10 Grok-4.3 + 10 Gemini-3.1-Pro generated blocks
per topic (generation runs `provenance/full320-v3/generation_runs/`, which
also hold the full160-era runs that produced 160 of its blocks).
It supersedes the retired, generator-imbalanced dev/a80/b80/full160 renders
(retirement rationale in `docs/v2_running_log.md`). Generating *new* payloads
(`generate`) needs API keys (`OPENROUTER_API_KEY`, or gcloud auth for Vertex) —
nothing else in the repo does.

## Layout

- `src/eval_format_mvp/` — the library: payload generation/validation
  (`generation.py`, `openrouter.py`, `vertex.py`), rendering + validators
  (`render.py`, `render_v2.py`, `render_v3.py`, `formats_v2.py`,
  `frames_v2.py`), surface-probe battery (`surface_v3.py`), activation
  extraction (`extract.py`: last-prompt-token residual stream, every block,
  forward hooks before the final norm), and the CLI (`cli.py`).
- `scripts/` — the runnable analysis/plot steps.
- `catalogs/`, `schemas/`, `prompts/`, `configs/` — design catalogs, JSON
  schemas, generation prompt templates, model profiles.
- `tests/` — the full suite (rendering, validators, generation machinery,
  frozen-chain regressions). `tests/data/` holds byte-pinned fixture copies
  of frozen artifacts (v1 payloads/render, pilot80 payloads/render) so the
  regression tests run offline regardless of which freezes remain in `data/`.
- `data/` — gitignored working area, one subfolder per freeze, materialized
  from HF (see "Data: git anchors, HF bytes" above).
- `provenance/` — tracked anchors, one subfolder per freeze forever: render
  manifests (every input/output sha256), `hf_snapshot.json` HF-revision pins,
  extraction configs, and (for the current freeze) generation run records.
