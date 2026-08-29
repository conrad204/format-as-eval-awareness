# Matched-format projection summary figure (2026-08-28)

Main-text companion to the full-depth 3x3 small-multiples figure in
`plots/matched format projection diagnostic 8-28-2026/` (kept unchanged as
appendix/supporting material). This figure is purely derived from
`results/full320_v3_matched_format_projection_diagnostic.json` — no new
probes were fit and no activations were touched to build it.

Script: `scripts/full320_v3_matched_format_projection_summary_plot.py`
Output: `matched_format_projection_summary.{png,pdf}` (this directory).

## What's shown

**Post-purpose-formation region**: layers where the raw both-format purpose
AUC is at least 0.95, defined from the raw depth trajectory only (not
spike/min_gap/final layer-picking). All three panels are restricted to this
region.

- **Panel A** — one faint point per eligible layer, all 3 models: x = raw
  cross-format transfer gap, y = the same gap after rank-64 matched-format
  projection (cross-fitted per held-out cue-vocabulary fold, purpose probe
  retrained on the projected activations). The dashed diagonal is "no
  change"; points below it are layers where projection reduced the gap. One
  bold marker per model shows that model's median (raw, projected) pair.
- **Panel B** — per-model dumbbell: hollow marker = median raw gap, filled
  marker = median projected gap, connected by a line; the percentage label
  is the median fraction of the raw gap removed, recomputed at render time
  from the diagnostic JSON's `rescue_fraction_median_excl_near_zero` field
  (not hand-typed).
- **Panel C** — mean ΔAUC after projection, three bars per model: benchmark→
  casual transfer gain, casual→benchmark transfer gain, and the mean
  absolute change in both-format purpose decoding. The scale contrast is the
  point: transfer gains are ~0.02–0.04 AUC; purpose decoding moves by
  ~0.0006–0.0013 AUC.

## Numbers used (recomputed from `results/full320_v3_matched_format_projection_diagnostic.json`, not hard-coded)

| model | median raw gap | median projected gap | % of median gap removed | mean Δ benchmark→casual | mean Δ casual→benchmark | mean \|Δ both-format purpose\| |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B  | 0.058 | 0.015 | 72% | +0.041 | +0.041 | 0.0013 |
| Llama-3.1-70B | 0.039 | 0.016 | 55% | +0.031 | +0.022 | 0.0006 |
| Llama-3.3-70B | 0.050 | 0.014 | 74% | +0.041 | +0.027 | 0.0007 |

## Definitions (recap)

- **Cross-format transfer gap** = both-format purpose AUC − mean(benchmark→
  casual AUC, casual→benchmark AUC).
- **Matched-format projection** = residualizing out a rank-64 basis of the
  matched-format subspace (the format main-effect/interaction contrast from
  the 2×2 factorial design in `RULES.md` §6), fit per held-out cue-vocabulary
  fold on training rows only, then retraining the purpose probe on the
  residual activations.
- **Post-purpose-formation region** = layers with raw both-format purpose
  AUC ≥ 0.95.

## Interpretation and caveats

- This is a **descriptive, across-depth** figure. Layers within a model are
  repeated measurements of one depth trajectory, not independent
  replicates — Panel A's per-layer points are shown for density/spread, not
  as an inferential sample; the only quantities treated as estimates are the
  per-model medians/means in Panels B and C.
- The pattern supports: **matched-format-associated structure is largely
  unnecessary for linearly decoding stated purpose itself, but contributes
  substantially to the failure of a purpose readout to transfer across
  prompt formats.** Removing that structure closes roughly half to
  three-quarters of the cross-format transfer gap while moving both-format
  purpose decodability by two orders of magnitude less than the transfer
  gain.
- This is a **frozen-activation representational intervention followed by
  probe retraining** — not behavioral steering and not a neural-circuit
  causal claim. It does not establish that the model's actual behavior
  changes, only that the linearly-decodable transfer penalty is
  substantially explained by a removable linear subspace.
- Full per-layer numbers, model summaries, and the exact join/definitions are
  in `results/full320_v3_matched_format_projection_diagnostic.{json,csv}` and
  `docs/matched_format_projection_transfer_diagnostic.md`.
