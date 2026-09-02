# Matched-format projection transfer diagnostic (2026-08-28)

## What this is

A derived, no-new-probes diagnostic joining two already-computed full-layer
sweeps for all 3 models (`Llama-3.1-8B`, `Llama-3.1-70B`, `Llama-3.3-70B`) on
the full320-v3 item bank:

- **raw**: `results/layer_sweep_full320-v3_<model>.json` — per-layer joint
  (`decorrelated`), benchmark→casual (`b2c`), and casual→benchmark (`c2b`)
  purpose-probe AUCs on held-out cue-vocabulary folds, on the untouched
  activations.
- **projected**: `results/full320_v3_erasure_full_layer_sweep.json` — the
  same three AUCs at the same layers, after residualizing out the rank-64
  **matched-format subspace** (the format main-effect/interaction basis `B`
  from the factorial decomposition in `RULES.md` §6, fit per held-out
  cue-vocabulary fold on training rows only, rank 64 — the strongest
  previously-validated erasure point) and retraining the purpose probe on the
  residual. This file was historically named with "erasure"; this diagnostic
  uses "matched-format projection" throughout, per current convention.

Both inputs verified byte/semantically identical to the canonical copies on
`uahpc` (`/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap/results/`
and the pinned `artifacts/pinned_full320/` copies) before use. No activation
extraction or probe fitting was re-run.

Join script: `scripts/full320_v3_matched_format_projection_diagnostic.py`
→ `results/full320_v3_matched_format_projection_diagnostic.json` (+ `.csv`).
Figure script: `scripts/full320_v3_matched_format_projection_plot.py`
→ `matched_format_projection_diagnostic.{png,pdf}` (this directory).

## Definitions

For every layer:

```
raw_gap   = raw_joint_auc  - mean(raw_b2c_auc,  raw_c2b_auc)
proj_gap  = proj_joint_auc - mean(proj_b2c_auc, proj_c2b_auc)
delta_gap = raw_gap - proj_gap                        # gap rescue, ΔG
delta_joint_auc = proj_joint_auc - raw_joint_auc
delta_b2c_auc   = proj_b2c_auc   - raw_b2c_auc
delta_c2b_auc   = proj_c2b_auc   - raw_c2b_auc
rescue_fraction = delta_gap / raw_gap   only where |raw_gap| >= 0.01 AUC,
                  else null + flagged unstable (avoids exploding near G_raw≈0)
```

The **post-purpose-formation region** used for the model-level summaries is
defined transparently from the raw trajectory only: layers with
`raw_joint_auc >= 0.95`. This is not spike/min_gap/final layer-picking; it is
every layer in that band, treated as repeated measurements of one
model-depth curve (not independent replicates) — matching `RULES.md` §10.

## Figure

`matched_format_projection_diagnostic.png` / `.pdf`, 3 panels, all layers, all
3 models, x-axis = relative depth (%):

- **A.** raw gap (solid) and rank-64-projected gap (dashed) vs. depth.
- **B.** gap rescue ΔG = raw_gap − proj_gap vs. depth, zero line.
- **C.** directional rescue ΔAUC(benchmark→casual, solid) and
  ΔAUC(casual→benchmark, dotted) vs. depth, with ΔAUC(joint purpose) shown as
  a faint thin line in the same model color for direct visual comparison of
  scale.

## Headline numbers (post-purpose-formation region, raw_joint_auc ≥ 0.95)

| model | eligible layers | raw gap (median) | proj gap (median) | ΔG (median) | rescue fraction (median, stable only) | mean \|Δjoint AUC\| | mean Δb2c | mean Δc2b | frac layers gap↓ | frac layers both directions↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B  | 17/32 (L4–L20)  | 0.058 | 0.015 | 0.035 | 0.72 | 0.0013 | +0.041 | +0.041 | 1.00 | 1.00 |
| Llama-3.1-70B | 67/80 (L7–L75)  | 0.039 | 0.016 | 0.021 | 0.55 | 0.0006 | +0.031 | +0.022 | 1.00 | 1.00 |
| Llama-3.3-70B | 60/80 (L7–L66)  | 0.050 | 0.014 | 0.033 | 0.74 | 0.0007 | +0.041 | +0.027 | 1.00 | 1.00 |

Full per-layer numbers: `results/full320_v3_matched_format_projection_diagnostic.{json,csv}`.

## Reading

In every layer with formed joint purpose (raw AUC ≥ 0.95), across all 3
models: the rank-64 matched-format projection (i) shrinks the raw gap (100%
of eligible layers, all 3 models), (ii) improves both cross-format
directions simultaneously (100% of eligible layers), and (iii) leaves
joint-format purpose AUC essentially unchanged (mean absolute change
0.0006–0.0013 AUC, two orders of magnitude below the median gap rescue of
0.021–0.035 AUC). Median fraction of the raw gap removed is 0.55–0.74 across
the three models. This is the pattern predicted by "format-associated
structure is largely unnecessary for decoding stated purpose itself, but
substantially responsible for the failure of a purpose readout to transfer
across formats" — see `docs/matched_format_projection_transfer_diagnostic.md`
for the full interpretation, caveats, and what this analysis is not (not a
causal intervention on model behavior; a representational intervention on
frozen activations followed by probe retraining).
