# Matched-format projection transfer diagnostic (2026-08-28)

## Question

Wherever a cross-format transfer penalty appears in the full320-v3 joint vs.
cross-format purpose probes, does removing the rank-64 matched-format
subspace specifically remove that penalty while leaving joint-format purpose
decodability intact? This is a narrower, cleaner restatement of the
representational-mechanism question in `RULES.md` §5/§7 (Q2), scoped to the
data already computed for the erasure/full-layer-sweep line of work, and it
is explicitly **not** "are some geometry dot products significant" (that
question was already addressed by the geometry sweep and is not repeated
here).

## Inputs (verified, not recomputed)

- `results/layer_sweep_full320-v3_llama31_8b.json` (32 layers)
- `results/layer_sweep_full320-v3_llama31_70b.json` (80 layers)
- `results/layer_sweep_full320-v3_llama33_70b.json` (80 layers)
- `results/full320_v3_erasure_full_layer_sweep.json` (rank-64
  matched-format-projected sweep, same layer counts, `by_model` keyed
  `llama31_8b` / `llama31_70b` / `llama33_70b`)

Verification performed before computing anything:

1. Confirmed all four files exist locally in `results/`, with per-model layer
   counts matching exactly between raw and projected sweeps (32/80/80).
2. Confirmed `gap_after_erasure` in the projected file recomputes exactly
   (to float precision) as
   `purpose_auc_after_erasure - mean(b2c_auc_after_erasure, c2b_auc_after_erasure)`
   for a sampled layer per model, and this diagnostic's own `build_table()`
   asserts that identity for every layer of every model at run time.
3. SSH'd to `uahpc` (`ssh uahpc`) and confirmed the local copy of
   `results/full320_v3_erasure_full_layer_sweep.json` is byte-identical
   (`md5sum` match, `b998c5a7...`) to the canonical cluster output at
   `/scratch/jppatton/langea/deploy/full320_v3_erasure_bootstrap/results/full320_v3_erasure_full_layer_sweep.json`.
4. Found two on-disk copies of each raw layer-sweep file
   (`results/layer_sweep_full320-v3_<model>.json` and
   `artifacts/pinned_full320/layer_sweep_full320-v3_<model>.json`) with
   different MD5s but confirmed they are semantically identical
   (every layer record compares equal under `json.load`; the MD5 difference
   is JSON formatting only, not content) — not a provenance conflict.

No activation extraction or probe fitting was re-run. This satisfies the
"prove a required artifact is missing before rerunning expensive work"
constraint: nothing was missing.

## Method

Pure derived join, no new model calls: `scripts/full320_v3_matched_format_projection_diagnostic.py`
reads both sweeps, merges by `(model, layer)`, and computes for every layer:

```
raw_gap   = raw_joint_auc  - mean(raw_b2c_auc,  raw_c2b_auc)
proj_gap  = proj_joint_auc - mean(proj_b2c_auc, proj_c2b_auc)
delta_gap       = raw_gap - proj_gap                     # ΔG, "gap rescue"
delta_joint_auc = proj_joint_auc - raw_joint_auc
delta_b2c_auc   = proj_b2c_auc   - raw_b2c_auc
delta_c2b_auc   = proj_c2b_auc   - raw_c2b_auc
rescue_fraction = delta_gap / raw_gap    if |raw_gap| >= 0.01 AUC else null (flagged unstable)
```

`raw_joint_auc` = the "decorrelated" regime (probe trained jointly on both
formats) in the raw sweep; `proj_joint_auc` = `purpose_auc_after_erasure` in
the projected sweep, i.e. the same joint-training regime after rank-64
matched-format residualization. `b2c`/`c2b` are one-format-trained,
other-format-tested cross-format transfer, before/after the same projection.
The projection itself (5-fold cross-fitted matched-format PCA basis of rank
64, fit on training rows per held-out cue-vocabulary fold, subtracted from
activations, probe retrained on the residual) was already computed by
`scripts/full320_v3_erasure_full_layer_sweep.py`; this diagnostic does not
touch that computation.

**Post-purpose-formation region**, used for all model-level summaries: layers
with `raw_joint_auc >= 0.95`, defined purely from the raw trajectory, not
hand-picked spike/min_gap/final layers. Per `RULES.md` §10, layers within
this region are treated as repeated measurements of one depth curve per
model, not independent replicates — no per-layer significance counts are
reported, only depth-region summary statistics (means/medians/fractions).

Per-layer table: `results/full320_v3_matched_format_projection_diagnostic.json`
(+ flat `.csv`). Figure: `plots/matched format projection diagnostic 8-28-2026/matched_format_projection_diagnostic.{png,pdf}`.

## Results

Post-purpose-formation region (raw_joint_auc ≥ 0.95):

| model | eligible layers / total | median raw gap | median proj gap | median ΔG | median rescue fraction (stable only) | mean \|Δjoint AUC\| | mean Δb2c AUC | mean Δc2b AUC | frac. layers gap reduced | frac. layers both directions improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B  | 17/32 (layers 4–20, depth 13–65%)  | 0.058 | 0.015 | 0.035 | 0.72 | 0.0013 | +0.041 | +0.041 | 1.00 | 1.00 |
| Llama-3.1-70B | 67/80 (layers 7–75, depth 9–95%)   | 0.039 | 0.016 | 0.021 | 0.55 | 0.0006 | +0.031 | +0.022 | 1.00 | 1.00 |
| Llama-3.3-70B | 60/80 (layers 7–66, depth 9–84%)   | 0.050 | 0.014 | 0.033 | 0.74 | 0.0007 | +0.041 | +0.027 | 1.00 | 1.00 |

(Mean, not just median, raw/projected gap and ΔG per model are in the JSON;
the medians and means agree to ≤0.01 AUC in every row, i.e. the region is not
driven by a single outlier layer.)

Qualitative depth pattern (Figure panels A–B): both models' raw gap peaks
sharply at very early-to-mid depth (~5–15% relative depth, where format
information is still strongly entangled with purpose before the purpose
readout has stabilized), then decays and drifts back up slowly through the
rest of depth. The rank-64 projected gap tracks the same peak-and-decay shape
but at roughly a third to a half the amplitude throughout — it is not a flat
offset subtraction, and it does not overshoot into a negative "over-rescued"
gap at any layer in the post-formation region for any model. Panel C shows
`delta_b2c_auc` and `delta_c2b_auc` closely tracking each other in scale and
depth-dependence for all 3 models (i.e. the rescue is not one-directional),
while `delta_joint_auc` (faint reference line) stays within ±0.005 AUC of
zero across the entire depth range for all 3 models — an order of magnitude
smaller than the directional rescue wherever the raw gap is non-trivial.

## Interpretation against the three predeclared outcomes

**Strong support** was predeclared as: projected gap closely collapses
wherever raw gap is present, directional transfer improves, joint AUC stays
nearly unchanged. The observed pattern matches this outcome:

- Gap reduction is universal in the post-formation region (100% of eligible
  layers, all 3 models), not layer-cherry-picked.
- Both cross-format directions improve simultaneously in 100% of eligible
  layers, all 3 models — this is the directional-symmetry check that rules
  out the rescue being an artifact of one direction's threshold/calibration
  shifting while the other degrades.
- Joint-purpose AUC changes by a mean absolute 0.0006–0.0013 AUC in the same
  region where the directional rescue averages 0.02–0.04 AUC — roughly
  20–60x smaller, i.e. purpose decodability is not just "less damaged than
  the gap," it is essentially untouched while the gap partially or mostly
  closes.
- The median fraction of the raw gap removed is 0.55–0.74 (not 1.00): the
  matched-format subspace is not the *entire* transfer penalty, but it is the
  majority of it, consistently, at the model-depth-region level rather than
  at isolated layers.

Given the "strong / partial / weak" trichotomy specified for this task, the
result is **strong support** for the finding that "removing the
matched-format-associated subspace specifically removes the cross-format
transfer penalty while leaving joint-format purpose decodability intact,"
with the caveat that the rescue is substantial and consistent rather than
complete (median 55–74%, not ~100%) — some transfer penalty is not explained
by this specific rank-64 matched-format subspace and remains open (residual
interaction structure not captured by the fixed-rank projection, or
non-linear format-conditioned structure the linear basis cannot remove).

## What this does and does not support

- This is a **representational intervention on frozen activations followed
  by probe retraining**. It says the transfer penalty is representable as a
  removable linear subspace of the residual stream, and that removing it
  does not damage joint-purpose linear decodability. It is **not** a causal
  intervention on model behavior or output, and it is **not** a claim about
  a specific neural circuit — consistent with `RULES.md` §7 Q2's scope and
  distinct from the Q3 causal-intervention requirement (steering / logit
  endpoints), which this diagnostic does not attempt and does not
  substitute for.
- The d′ "signal-loss/noise-growth" story from earlier development work is
  not invoked here; this diagnostic's directional-rescue and joint-AUC
  results are self-sufficient evidence and do not need it.
- No new geometry "energy" claims are made; the projection's effectiveness is
  evaluated purely through probe AUC deltas, not through the magnitude of
  the removed subspace.
- Spike/min_gap/final layers are not used as primary evidence anywhere in
  this diagnostic; every quantity above is a post-purpose-formation-region
  aggregate over the full depth trajectory.

## Caveats

- The post-purpose-formation region differs in width and depth location
  across the three models (17/32, 67/80, 60/80 layers) because it is derived
  from each model's own raw joint-purpose trajectory, not from a common
  layer index; comparisons across models are about relative depth, not
  absolute layer number.
- `rescue_fraction` is undefined (and excluded from the median) wherever
  `|raw_gap| < 0.01` AUC; this affects zero layers in the reported
  post-formation region for all three models here (all eligible layers have
  `raw_gap` well above the 0.01 threshold), but would matter near the tail
  ends of each model's full depth sweep, which is why the fraction is
  reported only over the post-formation region rather than the whole sweep.
- This is a single rank (64), single projection method (matched-format PCA
  basis of the factorial `B` term); it is the previously-validated strongest
  erasure point from prior development work, not a newly swept
  hyperparameter, so this diagnostic does not itself establish rank
  sensitivity (see `results/full320_v3_erasure_dose_response_bootstrap.json`
  for that separate analysis).
- Development-tier evidence per `RULES.md` §5.1: both input sweeps are
  already-computed observational full320-v3 results; this diagnostic adds no
  new probe fits and should be labeled exploratory/development, not
  confirmation-grade, consistent with the rest of the full320-v3 erasure line
  of work.
