# Emergency rank-64 C_perp_B specificity run — live status

**Question (single):** after removing the rank-64 B-derived subspace `Q_B`, does
removing the residual interaction subspace `Q_Cperp` reduce the cross-format
purpose-transfer gap *more than* an equal-rank random `Q_B`-orthogonal subspace?

**Decision quantity:** `specific_C = (I_-B - I_-[B,Cperp]) - (I_-B - I_-[B,Rperp])`,
paired 95% payload-block bootstrap CI, integrated over full normalized depth
`q = l/(L-1)`.

Tier: development / exploratory (frozen-activation representational intervention
plus probe retraining). Not behavioral steering, not a confirmation-grade run.

## Scope (deliberately reduced)

| Kept | Dropped |
|---|---|
| rank 64 only | probe retraining at ranks 1/4/16 |
| `raw`, `minus_b`, `minus_b_cperp`, `minus_b_rperp` | `minus_c`, `minus_bc_joint`, `minus_feval`, `minus_fdeploy`, `random_rankmatched` |
| all layers, all 3 probe regimes | — |
| C-capture at r=1,4,16,64 as prefixes of one rank-64 `Q_B` (no extra fits) | — |

Probe contract unchanged: `StandardScaler` + `LogisticRegression(C=0.1, max_iter=3000)`,
leave-one-purpose-family-out, three regimes (joint, benchmark→casual, casual→benchmark).
Cost per layer drops from 495 probe fits to 60.

## Boxes

All three boxes: 32 vCPU (cgroup), Ubuntu 20.04, Python 3.8.10, numpy 1.24.4,
scikit-learn 1.3.2, joblib 1.4.2, huggingface_hub 0.24.7. RAM limits are 240 GB on
the two 70B boxes and 120 GB on the 8B box, so workers are sized 30 / 30 / 16
(measured worker RSS 2.5-2.9 GB at d=8192).

| Box / job | Model | State at 2026-08-29 status refresh | Result |
|---|---|---|---|
| `u2h6griexkcqb6` | llama31_8b | complete, 32/32 layers, 1000 bootstrap reps | Final 8B result below |
| 70B emergency boxes | llama31_70b, llama33_70b | no active `emergency_cperp` jobs visible in current `squeue` | No 70B C-perp result retrieved |

The active cluster jobs currently visible are regular `tae_ixpatch` real/matched
interchange jobs, not emergency C-perp jobs. Do not describe the 70B C-perp
boxes as currently running without a fresh queue check.

8B timings: shard expansion 35 s; one layer alone 305 s; median 199 s under 16-way
parallelism. 70B first layer 1058 s under 30-way parallelism, so ~2.7 waves for 80
layers. Model-to-box assignment is scientifically arbitrary; the smallest bank sits
on the smallest box.

## Result: llama31_8b (complete, 32/32 layers, 1000 bootstrap reps)

| quantity | value | 95% CI |
|---|---:|---|
| `I_raw` | 0.07932 | — |
| `I_-B` | 0.03297 | — |
| `I_-[B,Cperp]` | 0.03758 | — |
| `I_-[B,Rperp]` | 0.03306 | — |
| `extra_C` | -0.00461 | [-0.00547, -0.00361] |
| `extra_random` | -0.00009 | [-0.00017, -0.00001] |
| **`specific_C`** | **-0.00452** | **[-0.00538, -0.00354]** |

`specific_C` is negative and excludes zero: after removing `Q_B`, removing the
residual interaction subspace does **not** improve cross-format purpose transfer;
it degrades it, and by more than the rank-matched random control. Removing `Q_B`
itself removes 58.4% of the raw integrated gap (directional: b2c +0.0466, c2b
+0.0481). Held-out C capture by prefixes of the one rank-64 `Q_B`: r=1 0.020 (81x
isotropic chance), r=4 0.064 (66x), r=16 0.137 (35x), r=64 0.271 (17x).

### Energy diagnostic (the "does it just remove more variance" control)

Pooled over 32 layers x 5 folds:

| subspace | energy removed | scope |
|---|---:|---|
| `Q_B` | 59.34% of raw | all rows |
| `Q_C|B` | 14.91% of post-B residual | all rows |
| `Q_R⊥B` | 1.57% of post-B residual | all rows |
| `Q_C|B` / `Q_R⊥B` | **9.48x** (held-out 9.29x) | per-layer median 10.6x, range 6.3-35.6x |

Both controls are rank 64. So `Q_C|B` strips roughly ten times more residual
activation variance than the rank-matched random direction set and still fails to
close the transfer gap. The energy asymmetry biases *toward* detecting a C effect,
which makes the negative stronger rather than weaker. A residual-PCA control
matched on removed energy is the natural post-hoc follow-up and needs only one
extra frozen-activation condition, no rerun of anything here.

### Gates

| gate | value |
|---|---|
| factorial identity | 1.49e-8 (float32 rounding budget 3.03e-5) |
| `Q_Bᵀ Q_Cperp` | 4.40e-14 |
| `Q_Bᵀ Q_Rperp` | 1.42e-14 |
| held-family violations | 0 |
| Gram basis vs `orth_basis` | passed (backend: gram) |
| legacy `layer_sweep` reproduction | **not exact**: raw 1.12e-3, `minus_b` 7.65e-4, Pearson r 0.999994-0.999999 |

The legacy deviation is a cross-pipeline comparability limit against older
reference JSONs, recorded in `reference_reproduction` with per-layer error curves
and `strict_pass: false`. It does not affect `specific_C`, which contrasts four
conditions that share one probe code path, folds, and items.

### Zero-compute layerwise diagnostics (8B)

`scripts/emergency_cperp_layerwise_diagnostics.py`, output
`results/emergency/emergency_cperp_layerwise_llama31_8b.json`. No probe fitting;
re-scores the stored OOF archive only.

1. **Layerwise trajectory.** `specific_C` per layer is negative in 31 of 32
   layers (single positive layer +0.0050), peak magnitude -0.0175 at q=0.13,
   early-half mean -0.0036, late-half mean -0.0055. The depth integral is not
   hiding a band where removing `Q_C|B` helps.
2. **C capture inside `Q_B` vs the minus-B rescue.** Integrated capture 0.271 vs
   integrated rescue 0.0464. Raw Pearson -0.524, Spearman -0.407; capture peaks
   mid-network (0.152 / 0.327 / 0.286 at first / middle / last layer) while
   rescue rises monotonically (0.0116 / 0.0370 / 0.0503), so the raw sign is a
   depth artifact. Partial correlation given a quadratic in q is +0.195 with
   permutation p = 0.279 (20,000 perms): no reliable association. Layers are not
   independent, so this is descriptive.

## Follow-up: within-`Q_B` C-alignment test (rank 16)

The residual-C story is falsified, so the successor question is narrower: inside
the rank-64 `Q_B`, are the directions most aligned with C the ones carrying the
portability rescue?

    M       = C_train @ Q_B
    M       = U S Vᵀ
    Q_B_rot = Q_B @ V          columns ordered by C alignment

    topC16    = Q_B_rot[:, :16]
    bottomC16 = Q_B_rot[:, -16:]
    randomB16 = Q_B @ R        seeded orthonormal 16-D rotation inside span(Q_B)

All three are rank 16 and live inside the same parent `Q_B`, so the contrast
isolates C-alignment rather than membership in `Q_B`. Only these three conditions
are fitted (45 probe fits per layer); `raw` is reused from the emergency archive
and no earlier condition is recomputed. Reported: `rescue_c = I_raw - I_c`, plus
paired `C_enrichment = rescue_topC16 - rescue_bottomC16` and
`C_vs_random_inside_B = rescue_topC16 - rescue_randomB16`. Both contrasts are
baseline-free algebraically (the `raw` term cancels), which is asserted in the
tests.

Free diagnostics per fold/layer: held-out C capture by each block, activation
energy removed by each block, containment `max|Q_sub - Q_B Q_Bᵀ Q_sub|`,
orthonormality `max|Q_subᵀ Q_sub - I|`, and the C-alignment singular spectrum.
Held family is excluded from `Q_B`, from the rotation fit, and from every probe.

Scripts: `full320_v3_qb_rotation_sweep.py`, `full320_v3_qb_rotation_bootstrap.py`,
`qb_rotation_pod_run.sh`. Coverage: `tests/test_full320_v3_qb_rotation.py`
(10 passed) — rotation orthonormality and identical projector to `Q_B`, descending
alignment order, rank matching and containment of all three blocks, seeded
controls, variance ordering, and paired-bootstrap identities.

### Result: llama31_8b (complete, 32/32 layers, 1000 bootstrap reps)

| quantity | value | 95% CI |
|---|---:|---:|
| `I_raw` | 0.07932 | — |
| `I_topC16` | 0.06368 | — |
| `I_bottomC16` | 0.06930 | — |
| `I_randomB16` | 0.06764 | — |
| `rescue_topC16` | 0.01565 | [0.01467, 0.01664] |
| `rescue_bottomC16` | 0.01002 | [0.00932, 0.01063] |
| `rescue_randomB16` | 0.01168 | [0.01101, 0.01239] |
| **`C_enrichment`** | **0.00562** | **[0.00470, 0.00662]** |
| **`C_vs_random_inside_B`** | **0.00397** | **[0.00316, 0.00479]** |

Both contrasts are positive and exclude zero under the paired payload-block
bootstrap. This supports a development-only internal result that C-aligned
directions inside `Q_B` carry more of the transfer rescue than the least
C-aligned or seeded random rank-16 directions. It is not behavioral steering,
model-output causality, or a confirmation-grade claim.

The completed artifact was assembled from 28 surviving checkpoints plus the
four recomputed missing layers (10–13) against the frozen v3 8B activation bank.
Global checks pass: maximum containment error `1.09e-14`, maximum orthonormality
error `3.19e-14`, and zero held-family basis violations.

### Continuation state

The 8B C-perp sweep and within-`Q_B` rotation follow-up are complete and their
results are secured. The 70B emergency C-perp and rotation outputs remain
unretrieved; do not describe them as running or complete without a fresh queue
check.

## Data path

Pinned freeze `format-EA/full320-v3` rev `d6c3be7be83c469d9a6a2aa4e797f3a7b538efde`,
downloaded from Hugging Face onto each box's **local container disk** and
sha256-verified against `provenance/full320-v3/hf_snapshot.json` before compute:

- `activations/<model>/activations_all_layers.npz`
- `rendered_items_v3_full320_8-17-2026.jsonl` (sha `5528819d…`)
- `results/layer_sweep_full320-v3_<model>.json` (raw AUC reference)

`results/full320_v3_erasure_full_layer_sweep.json` (rank-64 B reference) ships from
the repo. No activation tensor is read over network storage: the compressed bank is
expanded once into per-layer float32 shards on local disk and each layer worker
memory-maps its own shard.

## Speed changes and their gates

1. **Condition/rank pruning** — 8.25x fewer probe fits per layer.
2. **Gram-matrix basis** (`gram_orth_basis`): for `V` of shape `(m, d)` with `m << d`,
   eigendecompose `V Vᵀ` (`m×m`) instead of a full `m×d` SVD, then
   `Q = Vᵀ U / s`. Same uncentered convention and truncation rule as `orth_basis`.
   Gated per run on a real calibration layer: rank equality, min principal cosine
   `> 1-1e-8`, projector agreement `< 1e-8`, transformed-activation agreement `< 1e-8`.
   Falls back to the exact `orth_basis` path automatically if the gate fails.
3. **Layer-level parallelism** with `OMP_NUM_THREADS=1` to avoid BLAS oversubscription.
4. **One projection against `Q_B` reused** by all three removal conditions.

## Sanity gates (run must fail closed)

| # | Gate | Where |
|---|---|---|
| 1 | `F_eval = B + C`, `F_deploy = B - C` | `max_factorial_identity_error` |
| 2 | held family never enters basis or probe training | `held_family_basis_violations == 0` |
| 3 | `Q_Bᵀ Q_Cperp ≈ 0` | `max_qb_t_qcperp` |
| 4 | `Q_Bᵀ Q_Rperp ≈ 0` | `max_qb_t_rperp` |
| 5 | raw AUC reproduces canonical layer sweep | `reference_reproduction.max_raw_auc_abs_error` |
| 6 | rank-64 `minus_b` reproduces canonical B projection | `reference_reproduction.max_rank64_minus_b_auc_abs_error` |
| 7 | Gram basis reproduces `orth_basis` before use | `basis_backend_check.passed` |

Gates 1–4 tolerance `1e-6`; 5–6 tolerance `1e-6` against checked-in references;
7 tolerance `1e-8`.

Local unit coverage: `tests/test_full320_v3_emergency_cperp.py` (6 passed) covers
the Gram/SVD subspace equivalence, rank truncation, capture-prefix nesting,
per-layer leakage and regime masks, and the paired `specific_C` identity.

## Artifacts

Per model, pushed independently as each box finishes:

- `results/emergency_cperp_scores_<model>.npz` (OOF scores, frozen 5-D layout)
- `results/emergency_cperp_scores_<model>.sanity.json` (all gates + timing + run manifest)
- `results/emergency_cperp_benchmark_<model>.json` (one-layer timing)
- `results/emergency_cperp_summary_<model>.json` (bootstrap, >=1000 reps)

- `results/emergency/qb_rotation_scores_llama31_8b.npz` (32-layer OOF scores)
- `results/emergency/qb_rotation_scores_llama31_8b.sanity.json` (rotation gates,
  provenance, and per-layer diagnostics)
- `results/emergency/qb_rotation_summary_llama31_8b.json` (paired payload-block
  bootstrap, 1000 reps)
- `results/emergency/qb_rotation_timing_llama31_8b.json` (layer timings)

Activation banks and shard caches stay on the boxes.
