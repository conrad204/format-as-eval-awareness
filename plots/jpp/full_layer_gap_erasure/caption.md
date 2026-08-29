# Full-layer, normalized-depth matched-format-PCA erasure of the cross-format transfer gap

Replaces the ad hoc "spike/min_gap/final" 3-point depth convention
(`plots/jpp/matched_format_erasure_dose_response/`) with the full continuous
layer trajectory, normalized so the 32-layer 8B and 80-layer 70Bs are directly
comparable.

At every transformer layer, normalized depth `q = layer / (N_layers - 1)`, and
every matched-format-PCA erasure rank `r in {0 (raw), 1, 4, 16, 64}`:

```
g(q, r) = AUC_purpose(q, r) - (AUC_{B->C}(q, r) + AUC_{C->B}(q, r)) / 2
```

held-purpose-family cross-fitted (StandardScaler + LogisticRegression(C=0.1)),
matched-format PCA basis fit only on training folds -- identical convention to
`scripts/full320_v3_erasure_dose_response_bootstrap.py` and
`scripts/full320_v3_erasure_full_layer_sweep.py`, just extended to every layer
and every rank instead of 3 pinned depths / a single rank.

Integrated over the whole normalized-depth axis with the trapezoidal rule:

```
G(r) = int_0^1 g(q, r) dq            fraction_gap_removed(r) = 1 - G(r) / G(0)
```

**Top row**: `g(q, r)` trajectory, one panel per checkpoint, color = erasure
rank (dark = raw, light = rank 64). No layer or rank is highlighted as a
category -- the whole curve is shown for every rank.

**Bottom-left**: `G(r)` vs. rank, all 3 checkpoints, payload-block-bootstrap
95% CI ribbons (N=1000).

**Bottom-right**: fraction of the integrated gap removed vs. rank, same CI
convention.

## Uncertainty

Payload-block bootstrap, N=1000 replicates. Per replicate: ONE resample of the
~320 payload blocks (with replacement) is drawn and reused across **every
layer**, **every rank**, and **all three AUC regimes** (purpose/b2c/c2b); the
entire layerwise gap curve is recomputed from the resampled items and *then*
integrated. No pointwise-CI integration, no independently bootstrapped
layers/AUC components (`scripts/full_layer_gap_erasure_bootstrap.py`).

Paired significance (`scripts/full_layer_gap_erasure_significance.py`, same
block-resample convention, reused cached scores -- no new compute): for all 3
checkpoints, every rank contrast vs. raw, every adjacent-rank step, and every
individual layer show `P(gap reduced) = 1.000` over 1000 replicates (fully
monotone dose-response, layer-universal effect); the residual gap at rank 64
is significantly nonzero (`P(G(64) > 0) = 1.000`, CI excludes 0 for all 3
checkpoints).

## Data provenance

No new transformer forward passes. Reads directly from the frozen v3
activation tensors (`v3-activations/activations/<model>/activations_all_layers.npz`)
via `scripts/full_layer_gap_erasure_sweep.py` (stage 1, run on the uahpc
cluster), reduced by `scripts/full_layer_gap_erasure_bootstrap.py` (stage 2)
into `results/full_layer_gap_erasure_summary.json`, plotted here (stage 3,
this script) with no further activation access or probe refitting.

## Sanity checks (all pass)

- rank 0 exactly reproduces `results/layer_sweep_full320-v3_<model>.json`
  (max abs diff 0.0 at every layer, all 3 checkpoints -- identical fold code
  path).
- rank 64 reproduces `results/full320_v3_erasure_full_layer_sweep.json` to
  max abs diff 2.3e-4 (mean ~1e-5), attributable to independent-run SVD/fold
  floating-point noise, far inside the bootstrap CI widths (~0.005-0.02).
- All 5 ranks at the old spike/min_gap/final layers agree with
  `results/full320_v3_erasure_dose_response_bootstrap.json` to max abs diff
  2.8e-4, and every value falls inside that file's own pinned 95% CI, for all
  3 checkpoints.

## Integrated gap G(r) and fraction removed

| model | r=0 | r=1 | r=4 | r=16 | r=64 | frac. removed @ r=64 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.0793 | 0.0692 | 0.0598 | 0.0438 | 0.0329 | 58.5% [57.1, 59.9] |
| Llama-3.1-70B | 0.0550 | 0.0525 | 0.0448 | 0.0297 | 0.0205 | 62.8% [61.2, 64.4] |
| Llama-3.3-70B | 0.0631 | 0.0605 | 0.0515 | 0.0301 | 0.0219 | 65.3% [63.8, 66.9] |

## Interpretation (no mechanism overclaim)

This is a **full-layer representational erasure analysis of the cross-format
AUC transfer penalty**, not a causal-mechanism claim. The matched-format PCA
subspace removal shrinks the network-wide transfer gap monotonically with
rank, at every layer, for all 3 checkpoints -- but even the strongest tested
erasure (rank 64) leaves 35-42% of the integrated gap intact. A substantial
format-conditioned residual survives full-network erasure at this method's
tested rank ceiling; this does not establish that the removed subspace is the
behaviorally load-bearing cause of cross-format transfer failure, only that it
accounts for a majority-but-incomplete share of the measured AUC penalty.
