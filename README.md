# Purpose-probe plot pipeline

Minimal, self-contained chain that produces the eight plots in `plots/`:

```
data/rendered_items_v1_8-14-2026.jsonl          288 frozen prompts (the 2x2 purpose x format dataset)
        |  scripts/extract_activations.py        [GPU, RunPod]
        v
activations/<model>/activations_all_layers.npz   (not checked in; regenerable)
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
`provenance/`.

```bash
pip install -e ".[extract,analysis]"

# Llama-3.1-8B-Instruct — fits on one 40GB GPU
python scripts/extract_activations.py \
  --items data/rendered_items_v1_8-14-2026.jsonl \
  --out-dir activations/llama31_8b \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --all-layers --batch-size 8

# Llama-3.3-70B-Instruct — bf16 needs ~140GB+, e.g. 2x A100-80GB / 2x H100
python scripts/extract_activations.py \
  --items data/rendered_items_v1_8-14-2026.jsonl \
  --out-dir activations/llama33_70b \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --all-layers --batch-size 8 --device-map auto

# Layer sweep (run on the pod; the 70B npz is ~750MB, the JSON is ~100KB —
# download only the JSON)
python scripts/sweep_full.py data/rendered_items_v1_8-14-2026.jsonl \
  activations/llama31_8b/activations_all_layers.npz results/layer_sweep_llama31_8b.json
python scripts/sweep_full.py data/rendered_items_v1_8-14-2026.jsonl \
  activations/llama33_70b/activations_all_layers.npz results/layer_sweep_llama33_70b.json
```

Then rerun the three plot scripts.

## Determinism

The sweep is deterministic (leave-one-family-out folds, fixed
`random_state=0`). Extraction is bf16 on GPU, so re-extracted activations can
differ in the last bits and nudge AUCs in the third decimal; plots are visually
identical. The checked-in JSONs are the frozen reference; input identity is
`sha256(data/rendered_items_v1_8-14-2026.jsonl) = ebac61db…` (recorded per run in
`provenance/`).

## Layout

- `src/eval_format_mvp/` — extraction library (last-prompt-token residual
  stream, every block, forward hooks before the final norm). Trimmed from the
  parent repo: the payload-generation / validation / render machinery is gone;
  the dataset is frozen here.
- `scripts/` — the five runnable steps.
- `provenance/` — extraction configs of the runs behind the checked-in
  results (model revisions, hashes, GPU, library versions) and the render
  manifest tying `rendered_items_v1_8-14-2026.jsonl` back to its payload source.
