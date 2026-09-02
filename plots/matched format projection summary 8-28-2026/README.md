# Matched-format projection summary figure (2026-08-28)

Main-text companion to the full-depth 3x3 small-multiples figure in
`plots/matched format projection diagnostic 8-28-2026/` (kept unchanged as
appendix/supporting material). This figure is purely derived from
`results/full320_v3_matched_format_projection_diagnostic.json` — no new
probes were fit and no activations were touched to build it.

Script: `scripts/full320_v3_matched_format_projection_summary_plot.py`
Output: `matched_format_projection_summary.{png,pdf}` (this directory).

## What's shown

A single wide bar chart, one group of 3 bars (one per model) for each of:

- **benchmark → casual transfer gain** — mean ΔAUC (projected − raw) for the
  benchmark-trained, casual-tested purpose probe.
- **casual → benchmark transfer gain** — mean ΔAUC (projected − raw) for the
  casual-trained, benchmark-tested purpose probe.
- **both-format purpose decoding, |change|** — mean absolute ΔAUC for the
  probe trained and tested on both formats together.

All three are averaged over each model's **post-purpose-formation region**
(layers where raw both-format purpose AUC ≥ 0.95, defined from the raw depth
trajectory only). Error bars are the min–max range across the layers in that
region — the true depth-trajectory spread, not a symmetric SD/CI. An earlier
draft used ±1 SD, but SD is a poor descriptive statistic here: it is
symmetric and can visually suggest the effect crosses zero even though every
eligible layer, for every model, has strictly positive transfer gain
(confirmed directly from the per-layer table before choosing this
statistic). Min–max avoids that distortion while still showing real
depth-trajectory variability; it is still not an independent-sample CI
(layers within a model are repeated measurements, not independent
replicates; see `RULES.md` §10).

An earlier draft of this figure also had a scatter panel (raw vs. projected
gap, one point per layer) and a separate dumbbell panel; both were dropped
because every layer fell on the same side of the y=x line, so the scatter
panel repeated the same story as the bar chart without adding information —
this bar chart is now the entire figure.

## Numbers used (recomputed from `results/full320_v3_matched_format_projection_diagnostic.json`, not hard-coded)

| model | Δ benchmark→casual (mean, min–max) | Δ casual→benchmark (mean, min–max) | \|Δ both-format purpose\| (mean, min–max) |
|---|---|---|---|
| Llama-3.1-8B  | 0.041, [0.020, 0.063] | 0.041, [0.017, 0.125] | 0.0013, [0.0001, 0.0029] |
| Llama-3.1-70B | 0.031, [0.007, 0.156] | 0.022, [0.006, 0.101] | 0.0006, [0.0000, 0.0026] |
| Llama-3.3-70B | 0.041, [0.005, 0.148] | 0.027, [0.013, 0.114] | 0.0007, [0.0000, 0.0023] |

## Interpretation and caveats

- The core visual contrast: cross-format transfer improves by roughly
  0.02–0.04 AUC on average (and up to ~0.10–0.16 AUC at some individual
  layers), while both-format purpose decoding changes by roughly 0.001 AUC
  everywhere — one to three orders of magnitude smaller at every layer, not
  just on average.
- Every eligible layer, for every model, shows a strictly positive transfer
  gain in both directions — the min–max range never reaches zero, so the
  effect is consistent across the depth trajectory, not driven by a few
  layers.
- This is a **descriptive, across-depth summary**. Layers within a model are
  repeated measurements of one depth trajectory, not independent replicates.
- Supports: **matched-format-associated structure is largely unnecessary for
  linearly decoding stated purpose itself, but contributes substantially to
  the failure of a purpose readout to transfer across prompt formats.**
- This is a **frozen-activation representational intervention followed by
  probe retraining** — not behavioral steering and not a neural-circuit
  causal claim.
- Full per-layer numbers, model summaries, and the exact join/definitions are
  in `results/full320_v3_matched_format_projection_diagnostic.{json,csv}` and
  `docs/matched_format_projection_transfer_diagnostic.md`.
