# Format-as-Evaluation-Awareness: Workshop Repository

This repository is the cleaned snapshot for the Interpretability as a Science workshop submission. It contains the full320-v3 factorial measurement-validity study: stated evaluation/deployment purpose crossed with benchmark/casual format, plus the matched-format-associated subspace, no-cue control, real-forward interchange, geometry/null controls, provenance_v3 specificity audit, and supporting binding_v2 controls.

## Scope

The estimand is **stated response-purpose decodability**, not subjective evaluation awareness. The canonical freeze has 320 purpose-neutral payload blocks, 11,520 rendered rows, 7,680 stated-purpose rows, and 3,840 no-cue rows. Probes use held-purpose-vocabulary-family splits, StandardScaler, and LogisticRegression(C=0.1). Full-layer trajectories and payload/block bootstrap analyses are retained.

Latent-status/hidden-channel experiments, behavioral recognition, 405B work, and their reports are intentionally excluded from this repository. They are in the independent `iclr/` repository.

## Canonical paper

- `paper/main.tex` — source of truth for the workshop submission.
- `paper/appendix.tex` — supplementary figure/evidence source.
- `paper/main.pdf` and `paper/appendix.pdf` — locally compiled/generated submission PDFs.
- `paper/ARTIFACTS.md` — result-to-artifact provenance map.

Compile from `paper/` with the installed LaTeX toolchain, for example:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode appendix.tex
```

## Hugging Face data and activations

Large frozen files are hosted on Hugging Face rather than committed to GitHub:

- [`format-EA/full320-v3`](https://huggingface.co/datasets/format-EA/full320-v3) — frozen payloads, rendered items, manifests, all-layer activations for Llama-3.1-8B, Llama-3.1-70B, and Llama-3.3-70B, and pinned layer-sweep results. Dataset revision used by the paper: `d6c3be7be83c469d9a6a2aa4e797f3a7b538efde`.
- [`format-EA/binding-v1`](https://huggingface.co/datasets/format-EA/binding-v1) — supporting relational-binding activation control.
- [`format-EA/implicit-v2`](https://huggingface.co/datasets/format-EA/implicit-v2) — archived supporting implicit-purpose activation control; not part of the workshop headline.
- [`format-EA/pilot80-v2`](https://huggingface.co/datasets/format-EA/pilot80-v2) and [`format-EA/v1-8-14-2026`](https://huggingface.co/datasets/format-EA/v1-8-14-2026) — separate historical freezes retained as exploratory context.

The local `v3-activations/` directory contains the small item bank, payload, render manifest, and result metadata. Download large tensors with the standalone `hf` CLI, not Git LFS:

```bash
hf download format-EA/full320-v3 --repo-type dataset --local-dir v3-activations
```

Use the activation file paths and hashes recorded in `provenance/full320-v3/` and the submission manifests; do not substitute a different model revision or row order.

## Reproduction

1. Install the environment from `pyproject.toml`/`uv.lock`.
2. Obtain the full320-v3 dataset and activation tensors from the HF dataset above.
3. Run the canonical analysis scripts under `scripts/` using the manifests and fixed model/layer contracts.
4. Run the tests under `tests/`.
5. Regenerate figures with the scripts named in `paper/ARTIFACTS.md`.

All reported conclusions remain observational. Projection from the matched-format-associated subspace is not a causal behavioral intervention.
