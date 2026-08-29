# Interchange-patching causal test: Llama-3.1-8B, rank 16 (2026-08-29)

## Question

Is the matched-format subspace `Q_format(r)` (top-`r` PCA basis of the
factorial format main-effect term `B`) a *functionally* distinguished
direction that the model's own forward computation treats differently from a
generic direction of the same size — or is it merely a geometric correlate
that happens to separate from the purpose-probe direction in frozen
activations? This is the causal-intervention question `RULES.md` §7 Q3 calls
for, applied to the representational object identified by the geometry/
erasure line of work, using **real, hooked forward passes on the real model**
(not frozen-activation projection followed by probe retraining, which is what
the erasure sweeps do).

## Inputs (verified, not recomputed)

- `results/tae_ixpatch_full_llama31_8b.jsonl` — `real_swap` + `random_control`
  conditions, all 32 layers, 135,936 records (job `54161731`, 3:51:24
  elapsed).
- `results/tae_ixpatch_matched_llama31_8b.jsonl` — `matched_control`
  condition, all 32 layers, 67,968 records (job `54201560`, 1:01:41 elapsed).
- `results/tae_ixpatch_full_llama31_8b.manifest.json`,
  `results/tae_ixpatch_matched_llama31_8b.manifest.json`.

Verification performed before analysis:

1. **Activation/items hash match**: both manifests record
   `activation_sha256=e6a71be5...` and `items_sha256=5528819d...`; recomputed
   both hashes locally against
   `v3-activations/activations/llama31_8b/activations_all_layers.npz` and
   `v3-activations/rendered_items_v3_full320_8-17-2026.jsonl` — exact match,
   both runs used the identical, unmodified activation bank.
2. **Record-count sanity**: `scripts/full320_v3_interchange_patch_bootstrap.py`
   asserts every `(layer, condition)` cell has exactly the same row count
   (2,124 — the full matched-pair count) before computing anything; it
   aborts loudly on any mismatch instead of silently computing a CI from a
   truncated sweep. Confirmed: 2,124 rows × 32 layers × 3 conditions,
   consistent.
3. **Reproduction sanity gate** (run automatically inside the GPU driver
   before any sweep layer executes): 4 sampled rows' live unhooked forward
   pass compared against the pre-extracted static activation at the final
   layer, requiring cosine ≥ 0.995 and relative max-elementwise error ≤ 2% —
   passed for both jobs. (Tolerance is scale-relative, not the tighter fixed
   0.125-absolute/0.999-cosine bar used by the frozen-protocol pilot80/full320
   orthc pipelines, which assumes bit-reproducibility within one deterministic
   run; this is a cross-run check that the right checkpoint/tokenizer/
   chat-template convention is loaded, not a bitwise-identity claim.)

## Method

At every layer `L` (0–31) and every held-out purpose-family fold (5-fold,
same convention as `scripts/full320_v3_erasure_dose_response_bootstrap.py`):
`Q = orth_basis(B_train, 16)` fit on the training folds' factorial format
main-effect term. For every matched pair `(i, j)` (same `payload_block_id`,
same `purpose_pair_id`, same `intended_purpose`, opposite `format`, `j` in the
test fold):

```
real_swap:        h_i' = h_i - Q Q^T h_i + Q Q^T h_j
random_control:    h_i' = h_i - R R^T h_i + R R^T h_j     (R: random rank-16 orthonormal basis)
matched_control:    h_i' = h_i + u * ||Q Q^T (h_j - h_i)||  (u: fresh random unit vector per row)
```

`real_swap`/`random_control` swap the *same real content* (`h_j`) through
different subspaces. `matched_control` instead injects a fresh random
*additive* direction, scaled per-row to exactly the norm `real_swap`'s delta
would have injected for that row — computed offline from the static
activations (no extra GPU cost), isolating "is it the specific subspace, or
just how big the perturbation is."

The patch is injected via `register_forward_hook` at layer `L` and the
network runs forward for real to the final layer, where purpose/format
probes (`StandardScaler` + `LogisticRegression`, `C=0.1`, fit fresh per fold
on the training split) read out the patched hidden state. Baseline
(unpatched) scores come directly from the pre-extracted static activation
for row `i` (verified reproducible per the sanity gate above, so this avoids
a redundant unpatched forward pass per row/layer).

Two outcome measures per row:

- **format interchange-intervention accuracy (IIA)**: does
  `sign(format_score_patched − format_score_baseline_i)` match
  `sign(format_score_baseline_j − format_score_baseline_i)` (does the format
  readout move toward the counterpart, in the correct direction).
- **|purpose-readout shift|**: `|purpose_score_patched − purpose_score_baseline_i|`.

**Correlation-aware statistics**: layers are not independent replicates
(residual-stream continuity). Every bootstrap replicate resamples payload
blocks **once** and reuses that same resampled block set to recompute every
per-layer statistic and every control comparison, so the reported intervals
share the true cross-layer correlation instead of treating 32 layers as 32
independent tests. No per-layer significance count is reported anywhere in
this project's causal work, per the same convention already applied to
`plots/jpp/full_layer_geometry_trajectory` and the erasure dose-response
fix. In addition to per-layer CIs, a single joint, family-wise-valid
**worst-layer** statistic is reported per control comparison from the same
shared draws — a genuine whole-depth-range claim, not an aggregate of 32
separate tests.

## Results

`N_BOOT=2000`, `n_blocks=280` (payload blocks appearing in this data), both
control comparisons computed from one shared block-resample per replicate.

| comparison | joint worst-layer purpose-disruption gap (want > 0) | joint worst-layer format-IIA gap (want > 0) |
|---|---|---|
| real_swap vs. random_control (unmatched) | **[−3.27, −2.86]** — fails | **[0.020, 0.066]** — holds |
| real_swap vs. matched_control (norm-matched) | **[−1.17, −0.82]** — fails | **[0.041, 0.087]** — holds |

**H2 (format is causally carried by `Q_format(16)`) holds jointly across the
whole depth range, against both controls**, including the magnitude-matched
one. Point estimates: format-IIA for `real_swap` climbs monotonically from
0.54 (layer 0) to 0.98 (layer 31); `random_control` stays at chance (0.49→0.57);
`matched_control` rises modestly but stays well below `real_swap` throughout
(0.47→0.49 at layer 31, i.e. real_swap ends 49 points higher at matched
magnitude).

**H1 (purpose is protected) fails jointly against both controls** — the
worst-layer statistic requires the effect to hold at *every* layer, and it
does not. But the per-layer trajectory against `matched_control` is not
monotonic, and this matters for interpretation:

| layer | purpose-disruption: real_swap | matched_control | gap (control − real) |
|---:|---:|---:|---:|
| 0 | 0.334 | 0.319 | −0.015 (real slightly worse) |
| 18 (crossover) | 3.165 | ~3.37 | ~+0.20 (flips positive here) |
| 31 | **0.843** | **5.825** | **+4.98** (real dramatically better) |

From layer 18 onward (the back ~44% of the network), `real_swap` is *less*
disruptive to purpose than an equally-sized random push, by a growing and
eventually enormous margin (layer 31: real disrupts purpose 0.84 vs. 5.83 for
matched_control — matched_control is ~7x more disruptive). Through the front
~56% of the network, the relationship is reversed (real_swap disrupts purpose
somewhat more). The whole-depth joint statistic reports the true worst case
(the front half), which is honest but obscures that the back half of the
network shows the opposite, favorable pattern.

## Interpretation

**`Q_format(16)` is a functionally distinguished direction, not just a
correlated one.** Two independent lines of evidence converge:

1. It causally moves the format readout more reliably than a generic
   direction of the same size, at every depth past the first couple of
   layers, and the margin over the matched control grows with depth (H2 holds
   jointly, strongly).
2. At deep layers specifically, pushing along it is markedly *safer* for the
   purpose readout than pushing an equally-sized generic direction — the
   opposite of what a purely magnitude-driven confound would predict.

**But this is not a clean "purpose and format are separately routed"
result.** `real_swap` still substantially disrupts purpose, especially
through the first half of the network, and the offline diagnostic
(`results/full320_v3_qb_c_leakage_diagnostic.json`) shows why: `Q_format(16)`
resolves a real, depth-growing share of the purpose×format interaction term
`C` (from ~5% at layer 3 to ~15% at mid-depth, vs. 0.4% isotropic chance),
and the natural per-purpose format contrast that `real_swap` transplants is
`B+C` (evaluation) or `B−C` (deployment), never a purpose-orthogonal `B`
alone. The early-to-mid-depth purpose disruption is consistent with that
leaked `C` content doing real work; the late-depth reversal is consistent
with `Q_format(16)` becoming comparatively more B-dominated (or the
purpose-relevant computation becoming less dependent on this
particular subspace) by the time the network reaches its output.

## What this does and does not support

- This **is** a real causal intervention on the model's own forward
  computation (`register_forward_hook`, real bf16 weights, real propagation
  from the patch layer to the output) — a step beyond the erasure/geometry
  line of work, which only projects and refits a linear probe on frozen,
  already-computed activations.
- This is **not** a behavioral or generation-level claim. The readout is a
  linear probe on the final-layer hidden state, not validated model output
  (text generation, an answer choice, a logit-level behavioral endpoint).
  Per `RULES.md` §7 Q3's distinction, this result should be described as
  **internal causal mediation**, not **behavioral steering**, until/unless
  paired with an independent behavioral-endpoint replication (e.g. the
  `results/steering/*` pipeline's candidate-logit or generated-text readouts).
- This is a **single rank (16)** and a **single model (8B)**; rank sensitivity
  and cross-model replication (`llama31_70b`, `llama33_70b`) are open,
  in progress as of this writeup.
- The `matched_control` finding does not mean `real_swap` is purpose-neutral
  — it substantially moves the purpose readout at every layer tested (H1
  fails jointly). It means the disruption is not fully explained by
  perturbation size, and at deep layers is markedly smaller than an
  equally-sized generic perturbation would produce.

## Caveats

- `n_blocks=280`, not the full 320 payload blocks — some blocks are excluded
  wherever a complete 2×2 matched group isn't available for a given fold;
  this is the same convention used throughout the erasure/geometry line of
  work, not specific to this test.
- The `matched_control` random direction is a single fresh draw per row (not
  averaged over multiple random directions per row), so its point estimate
  has more per-row variance than `real_swap`'s deterministic subspace swap;
  the block bootstrap CI already accounts for this at the aggregate level,
  but a multi-draw-per-row design would tighten the per-row estimate further
  if pursued later.
- Development-tier evidence per `RULES.md` §5.1: no frozen pre-registered
  protocol, no pilot80-trained/frozen readout — probes are fit fresh per
  fold in this script, matching the rest of this sprint's exploratory
  causal-identification line of work, not the heavier hash-gated
  confirmatory pipeline (`scripts/run_pilot80_factorial_causal.py` and its
  forks).
- Single rank (16) tested; the erasure dose-response line of work shows
  effects are rank-dependent (rank 64 captures substantially more `C` than
  rank 16 per the same offline diagnostic), so this specific magnitude of
  causal effect should not be assumed to generalize to other ranks without
  rerunning.
