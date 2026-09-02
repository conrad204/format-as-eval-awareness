# Format-as-Evaluation-Awareness: Workshop Scientific Repository

This is the cleaned workshop scientific/reproducibility repository for the
stated-purpose measurement-validity and format-confounding study. There is
currently **no canonical manuscript in this repository**. Manuscript work will
be maintained separately.

## Scope

The canonical full320-v3 experiment crosses stated response purpose
(evaluation/deployment) with response format (benchmark/casual), with matched
payloads, held-out purpose-family probes, full-layer trajectories, and
payload/block bootstrap analyses. Supporting analyses cover:

- confounded and controlled probe results;
- cross-format transfer;
- matched-format-associated projections at ranks 1/4/16/64;
- no-cue-derived format-associated subspaces;
- real-forward interchange;
- purpose/format geometry and conditional/null controls;
- provenance-v3 specificity as negative external-validity evidence;
- binding-v1 and binding-v2 relational controls; and
- implicit-v2 as an indirectly worded-context control.

Implicit-v2 describes evaluative structure in its prompts using indirect wording.
It is **not** a latent or inferred-status experiment and is not evidence for
subjective awareness. Its corrected interpretation is recorded in
`docs/implicit_v2_indirect_context_control.md`.

The canonical full320-v3 freeze contains 320 purpose-neutral payload blocks,
11,520 rendered rows, 7,680 stated-purpose rows, and 3,840 no-cue rows.
Conclusions are observational; matched-format-associated projection is not a
behavioral causal intervention.

Hidden-channel latent-status experiments, behavioral recognition/reportability,
405B work, and their reports are intentionally excluded. They live in the
separate `iclr/` repository.

## Data and activation tensors

Large frozen files are hosted on Hugging Face rather than committed to GitHub:

- [`format-EA/full320-v3`](https://huggingface.co/datasets/format-EA/full320-v3)
  — canonical full320 payloads, rendered items, manifests, model activations,
  and pinned layer-sweep results. Revision:
  `d6c3be7be83c469d9a6a2aa4e797f3a7b538efde`.
- [`format-EA/binding-v1`](https://huggingface.co/datasets/format-EA/binding-v1)
  — supporting relational-binding activations.
- [`format-EA/implicit-v2`](https://huggingface.co/datasets/format-EA/implicit-v2)
  — indirectly worded-context activations and metadata.
- [`format-EA/pilot80-v2`](https://huggingface.co/datasets/format-EA/pilot80-v2)
  and [`format-EA/v1-8-14-2026`](https://huggingface.co/datasets/format-EA/v1-8-14-2026)
  — historical supporting freezes.

The local `v3-activations/` directory contains the small full320 item bank,
payload, render manifest, and result metadata. Download large tensors with the
standalone `hf` CLI, not Git LFS:

```bash
hf download format-EA/full320-v3 --repo-type dataset --local-dir v3-activations
```

Use the activation paths and hashes recorded in `provenance/full320-v3/` and
the relevant result manifests. Do not substitute a different model revision or
row order.

## Reproduction

1. Install the environment from `pyproject.toml`/`uv.lock`.
2. Obtain the required full320-v3, binding, and implicit-v2 data from Hugging Face.
3. Run the analysis scripts under `scripts/` using the fixed manifests and model contracts.
4. Run the tests under `tests/`.
5. Consult `docs/ARTIFACTS.md` for analysis-to-artifact and code mappings.
