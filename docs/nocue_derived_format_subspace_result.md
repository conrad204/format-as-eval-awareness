# No-cue-derived format-associated subspace vs. the cross-format purpose-transfer gap

Development evidence, not confirmatory. Llama-3.1-8B-Instruct, frozen full320-v3 activations
(`v3-activations/activations/llama31_8b/activations_all_layers.npz`), all 32 layers, no new forward passes.

## Question

The paper's format-subspace result builds the rank-r basis from the **stated-purpose** matched format
contrast, i.e. from rows that explicitly announce evaluation or deployment. Does the result survive when
the subspace is estimated **entirely from no-cue rows**, where no purpose statement is present?

## Construction

For each held purpose-vocabulary fold and layer l, using only the 3,840 no-cue rows (`render_slot ==
"no_cue"`, `label is None`, `intended_purpose == "none"`):

1. For each payload block g, average the ~6 no-cue renderings per format and form
   `D_g,l = mean(h_no-cue,benchmark,g,l) - mean(h_no-cue,casual,g,l)`.
   Within-payload centroids are used because no-cue rows are not 1:1 paired across formats
   (2,080 benchmark vs 1,760 casual rows); this is the exact matched equivalent available.
2. Stack one `D` row per **stated training (payload_block_id, purpose_pair_id) group** — the same row
   semantics as the existing `factorial_vectors` B matrix — and take an **uncentered SVD**
   (`orth_basis`, the repo's own implementation) to get `Q_nocue,r`.
3. Project stated-purpose activations, `h' = (I - QQ^T) h`, then retrain the **unchanged** probes
   (`StandardScaler` + `LogisticRegression(C=0.1, max_iter=3000, random_state=0)`).

Endpoint, unchanged from the repo: `G_l = AUC_joint,l - 0.5*(AUC_B->C,l + AUC_C->B,l)`, integrated over
normalized depth `q = l/(n_layers-1)` by the trapezoidal rule.

Sanity check: the raw (`r = 0`) per-layer gap curve used here reproduces the pinned
`results/layer_sweep_full320-v3_llama31_8b.json` gap **exactly** (max absolute difference 0.0 across all
32 layers).

## 1-3. Integrated gap, raw vs no-cue-projected

1,000 payload-block bootstrap replicates; one block resample per replicate reused across every layer,
regime, and rank, so ratios are paired (matching `full_layer_gap_erasure_bootstrap.py`). 320 blocks,
7,680 stated rows.

| Quantity | Value | 95% CI |
|---|---:|---|
| Raw integrated gap `G(0)` | **0.07934** | 0.07682, 0.08162 |
| No-cue-projected integrated gap, r=64 | **0.05569** | 0.05368, 0.05758 |
| **Fraction of integrated gap removed, r=64** | **29.80%** | 28.60%, 31.22% |

Rank sweep, with the stated-purpose-derived basis `Q_B` computed on the identical block resamples:

| r | No-cue `G(r)` | No-cue fraction removed (95% CI) | Stated `Q_B` fraction removed (95% CI) | No-cue / stated | Paired difference (95% CI) |
|---:|---:|---|---|---:|---|
| 1 | 0.07996 | **-0.78%** (-1.79%, +0.28%) | 12.78% (11.93%, 13.67%) | -0.06 | -14.4% (-14.56%, -12.54%) |
| 4 | 0.07125 | **10.20%** (9.24%, 11.23%) | 24.61% (23.51%, 25.71%) | 0.41 | -14.4% (-15.35%, -13.40%) |
| 16 | 0.06259 | **21.11%** (20.01%, 22.29%) | 44.81% (43.53%, 46.12%) | 0.47 | -23.7% (-24.97%, -22.36%) |
| 64 | 0.05569 | **29.80%** (28.60%, 31.22%) | 58.52% (57.13%, 59.86%) | 0.51 | -28.7% (-30.25%, -27.03%) |

Two facts, both robust to the block bootstrap: the no-cue-derived subspace removes a **substantial and
clearly nonzero** share of the integrated gap at r >= 4, and it removes **roughly half** of what the
stated-purpose-derived basis removes at every rank >= 4. At r = 1 the no-cue effect is null
(CI spans zero); the single leading no-cue direction does not reproduce the rank-1 `Q_B` effect.

## 4. Per-layer raw vs projected gap (r = 64)

The gap is reduced at **32 / 32 layers**. Full table in
`results/nocue_format_subspace_llama31_8b_r64.csv`; abridged:

| Layer | Depth | Raw `G_l` | Projected `G_l` | Reduction |
|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.0466 | 0.0413 | 0.0052 |
| 3 | 0.10 | 0.2076 | 0.1757 | 0.0320 |
| 5 | 0.16 | 0.0817 | 0.0409 | 0.0407 |
| 8 | 0.26 | 0.0430 | 0.0207 | 0.0223 |
| 11 | 0.35 | 0.0662 | 0.0292 | 0.0370 |
| 16 | 0.52 | 0.0581 | 0.0337 | 0.0243 |
| 21 | 0.68 | 0.0739 | 0.0565 | 0.0174 |
| 26 | 0.84 | 0.0752 | 0.0616 | 0.0136 |
| 30 | 0.97 | 0.0952 | 0.0517 | 0.0434 |
| 31 | 1.00 | 0.0945 | 0.0547 | 0.0399 |

Restricted to the 17 layers where purpose is formed (raw joint AUC >= 0.95, the existing diagnostic's
gate), mean raw gap 0.0600 -> mean projected gap 0.0362.

## 5. Change in joint-format purpose AUC

Integrated change in joint AUC, r=64: **+0.00143** (95% CI +0.00100, +0.00186). Per-layer changes are
within +/-0.003 at every layer with formed purpose. The projection does not damage the joint-format
purpose probe; the gap reduction is not bought by degrading the joint numerator.

## 6. Change in transfer AUCs

Integrated changes, r=64: **B->C +0.01786** (+0.01634, +0.01935), **C->B +0.03230** (+0.03026, +0.03434).
Both cross-format directions improve, and the larger repair is on C->B, the weaker raw direction
(e.g. layer 30: C->B 0.7843 -> 0.8699). Both improve at r=16 and r=64; at r=1 both change by <0.001 with
the wrong sign, consistent with the null at that rank.

## 7. Format AUC after projection

Integrated format AUC after the no-cue projection is **0.9904** (95% CI 0.9885, 0.9920); per-layer range
0.9637–1.0000. Format remains almost perfectly decodable after removing a rank-64 no-cue-derived format
subspace. Whatever this projection does, it is **not** erasure of format information — a caveat that
applies equally to the existing stated-purpose-derived projection.

## Q_B vs Q_nocue geometry (rank 64, per layer, mean over folds)

`results/nocue_subspace_geometry_llama31_8b_r64.json`.

| Diagnostic | Mean over layers | Min | Max |
|---|---:|---:|---:|
| Largest principal cosine | 0.9958 | 0.9939 | 0.9998 |
| Median principal cosine | 0.7686 | 0.7338 | 0.8883 |
| Smallest principal cosine | 0.0134 | 0.0075 | 0.0243 |
| Trace overlap `||Q_B^T Q_nocue||_F^2 / r` | 0.5329 | 0.5005 | 0.6438 |
| Principal cosines > 0.9 (count of 64) | 17.3 | 14 | 31 |
| Held-out stated centered variance in `Q_B` | 0.7560 | 0.6077 | 0.9097 |
| Held-out stated centered variance in `Q_nocue` | 0.6851 | 0.5754 | 0.8232 |

Mean median principal angle ~39.7 degrees. Interpretation, deliberately weak: the two bases share a
substantial but clearly **partial** subspace — a handful of near-identical leading directions
(~17 of 64 cosines above 0.9), a mid-spectrum at ~40 degrees, and a tail that is close to orthogonal.
Trace overlap ~0.53 tracks the ~0.5 effect ratio, which is suggestive but not established as causal.
This is **not** geometric identity, and the shared component is not demonstrated to be "pure format".

## Leakage audit

- No-cue rows are never scored and never enter probe training: 0 non-stated rows in the scored set;
  the label set of the no-cue source rows is `{None}`.
- The basis is rebuilt inside every fold from that fold's training groups only
  (`tests/test_nocue_format_subspace.py` pins the group-key filtering and the centroid contrast).
- **Stated honestly:** the repo's folds hold out purpose **vocabulary families**, not payload blocks. All
  320 payload blocks appear in every fold's training groups, so `Q_nocue` is nearly fold-invariant
  (pairwise trace overlap between fold bases 0.909–0.952 at layer 16), and the held-out fold's *payloads*
  are seen by the basis. This is the same fold semantics the existing `Q_B` pipeline uses, so the
  comparison is apples-to-apples, but it is not a payload-held-out design. Purpose-cue leakage — the
  concern that motivated this analysis — is eliminated by construction, since no-cue rows contain no
  evaluation/deployment statement at all.

## Scope not covered

The two 70B checkpoints were **not** run. Their frozen tensors are ~30 GB uncompressed
(11,520 x 80 x 8,192 fp32) against 15.6 GB of host RAM on this workstation, and the 8B run took
~70 minutes per rank; this is outside the task's "if cheap" allowance and would need the cluster. The
mandated minimum (rank 64, all Llama-3.1-8B layers) plus ranks 1/4/16 are complete.

## Claim boundaries

- Not a test of spontaneous evaluation awareness.
- No-cue rows have no evaluation/deployment labels; nothing here infers a purpose representation from them.
- The narrow question answered: a format-associated subspace identified **without any purpose cue**
  explains a substantial minority (~30% at r=64) of the stated-purpose probe's cross-format transfer
  failure, about half of what the stated-purpose-derived subspace explains.
- Frozen-activation projection plus probe retraining is **not** a behavioral causal intervention.
- Construction is an **uncentered SVD**, matching the repo; it is not centered PCA, so "PCA erasure" is
  the wrong term for it.
- The subspace is a **no-cue-derived format-associated subspace**, not "pure format" — format AUC stays
  at ~0.99 after projection.

## Verdict

This **materially strengthens, with an explicit quantitative haircut**, the paper's central
format-subspace claim.

Strengthens: the qualitative mechanism does not depend on the purpose cue. A subspace estimated with zero
purpose-statement information reduces the cross-format transfer gap at every layer, by 29.8%
[28.6%, 31.2%] of the integrated gap at rank 64, while leaving joint-format purpose AUC intact
(+0.0014) and improving both transfer directions. The most deflationary reading of the original result —
that the basis worked only because it was built from cue-bearing rows and therefore carried purpose
information — is ruled out.

The haircut: it does **not** reproduce the full effect. The stated-purpose-derived basis removes 58.5%,
so the cue-free basis recovers about half (paired difference -28.7% [-30.3%, -27.0%], firmly
excluding equality), and at rank 1 it does nothing. So the effective subspace in the original result is
not identified purely by cue-free format variation: roughly half of that effect lives in directions that
require the stated-purpose contrast to find, consistent with the partial geometric overlap
(trace overlap ~0.53). Any claim that the format subspace is *fully* recoverable without purpose cues
would be unsupported by these numbers.

## Artifacts

| Path | Contents |
|---|---|
| `scripts/full320_v3_nocue_format_subspace.py` | Cross-fitted no-cue basis, projection, probe retraining, bootstrap |
| `scripts/full320_v3_nocue_vs_stated_subspace_compare.py` | Paired `Q_B` vs `Q_nocue` fractions on shared block resamples |
| `scripts/full320_v3_nocue_subspace_geometry.py` | Principal cosines/angles, held-out variance captured |
| `scripts/full320_v3_nocue_subspace_figure.py` | Two-panel figure |
| `tests/test_nocue_format_subspace.py` | Contrast + fold-filtering invariants |
| `results/nocue_format_subspace_llama31_8b_r{1,4,16,64}.{json,csv,md,png}` | Per-rank summaries, per-layer tables |
| `results/nocue_format_subspace_scores_llama31_8b_r{1,4,16,64}.npz` | Per-item OOF scores, geometry, input hashes |
| `results/nocue_vs_stated_subspace_compare_llama31_8b_r{1,4,16,64}.json` | Paired comparisons |
| `results/nocue_subspace_geometry_llama31_8b_r64.json` | Per-layer geometry |
| `results/nocue_format_subspace_figure_llama31_8b.png` | Figure |
