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
trajectory only). Error bars are ±1 SD across the layers in that region —
descriptive spread across a depth trajectory, not an independent-sample
confidence interval (layers within a model are repeated measurements, not
independent replicates; see `RULES.md` §10).

An earlier draft of this figure also had a scatter panel (raw vs. projected
gap, one point per layer) and a separate dumbbell panel; both were dropped
because every layer fell on the same side of the y=x line, so the scatter
panel repeated the same story as the bar chart without adding information —
this bar chart is now the entire figure.

## Numbers used (recomputed from `results/full320_v3_matched_format_projection_diagnostic.json`, not hard-coded)

| model | mean Δ benchmark→casual | mean Δ casual→benchmark | mean \|Δ both-format purpose\| |
|---|---:|---:|---:|
| Llama-3.1-8B  | 0.041 ± 0.011 | 0.041 ± 0.026 | 0.001 ± 0.001 |
| Llama-3.1-70B | 0.031 ± 0.026 | 0.022 ± 0.017 | 0.001 ± 0.000 |
| Llama-3.3-70B | 0.041 ± 0.025 | 0.027 ± 0.017 | 0.001 ± 0.000 |

(± values are the SD across post-purpose-formation layers used for the error
bars, not standard errors.)

## Interpretation and caveats

- The core visual contrast: cross-format transfer improves by roughly
  0.02–0.04 AUC on average, while both-format purpose decoding changes by
  roughly 0.001 AUC — about one to two orders of magnitude smaller, despite
  real layer-to-layer variability (the transfer-gain error bars are wide
  relative to their means; the purpose-decoding error bars are uniformly
  tiny in absolute terms).
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
