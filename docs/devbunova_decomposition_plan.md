# Cross-dataset decomposition of Devbunova's eval-awareness signal — frozen analysis plan

Status: PRE-REGISTERED. Written 2026-08-23, before any cross-dataset number was
computed. Commit this file before running `scripts/devbunova_decomposition.py`
on real activations; any deviation from this plan must be recorded in the
results JSON under `"deviations"`.

## Question

Devbunova (arXiv:2603.19426) trains an eval-awareness probe on
`bench_eval` vs `casual_deploy_matched` — cells in which evaluation context and
benchmark *format* move together — and reports that ~94.5% of `bench_deploy`
items (benchmark-formatted but genuinely deployment) are classified "eval"
(the format trap). Our full320-v3 design crosses purpose × format, so it
yields a purpose direction trained with format balanced (decorrelated regime)
and a separate format direction. The question, per model:

**How much of her probe's signal is carried by the format direction rather
than a format-robust purpose direction — and does a format-controlled probe
remove the trap without losing eval sensitivity?**

## Inputs (all frozen before this plan)

| artifact | source | anchor |
|---|---|---|
| Our items | `data/full320-v3/rendered_items_v3_full320_8-17-2026.jsonl` (11,520; stated stratum 7,680) | `provenance/full320-v3/` manifest |
| Our activations 8B / 70B | `format-EA/full320-v3` `activations/llama31_{8b,70b}/activations_all_layers.npz` | `hf_snapshot.json` rev `d6c3be7b` |
| Our activations 405B | `hf://Conradf/v3-llama-3.1-405B-instruct` per-layer shards | `extraction_config_405b_ndif.json` |
| Our reference sweeps | `results/layer_sweep_full320-v3_llama31_{8b,70b,405b_ndif}.json` | checked in |
| Her items | `data/devbunova-2x2/items_devbunova_2x2.jsonl` (8,479) | sha256 `bd4b7151…`, `adapter_manifest.json` |
| Her activations | `Conradf/devbunova-2x2-activations` `<model>/<regime>/chunks/` (3 models × {raw, template}) | `extraction_runs_ndif.json` |

Models are matched by construction: Llama-3.1-{8B,70B,405B}-Instruct on both
sides. Llama-3.3-70B has no Devbunova extraction and is out of scope.

## Frozen analytic choices

1. **Layer (primary)** — per model, the decorrelated-AUC argmax of the
   checked-in v3 sweep (chosen from our data only, before touching hers):
   **8B = L8, 70B = L18, 405B = L37.**
   **Robustness layer** — mid-depth: 8B = L16, 70B = L40, 405B = L63.
   No other layers are examined before the primary endpoints are reported.
2. **Our probes** — semantics copied verbatim from `nocue_sweep.py` /
   `pilot80_sweep_full.py` (StandardScaler + LogisticRegression C=0.1,
   max_iter=3000, random_state=0; folds fg1–fg5 on the stated stratum):
   - `purpose_decorr`: decorrelated regime (train = all stated rows).
   - `purpose_conf`: confounded regime (train = eval∧benchmark ∪ deploy∧casual
     stated rows) — our in-house analogue of her training design.
   - `format`: benchmark-vs-casual on the stated stratum.
   External items are scored by each of the five held-fold models and the five
   decision scores averaged (the E6 convention used for no-cue rows).
   Sign: positive score = evaluation (resp. benchmark).
3. **Regime pairing** — primary: our probes applied to her **template**-regime
   activations (matches the probes' training distribution: constant assistant
   header at the probed position). Secondary: same on **raw** (her probed
   position; out-of-distribution positionally — reported, not headline).
4. **Her-protocol probe** — her exact recipe per the vendored code:
   StandardScaler + LogisticRegression(max_iter=1000, random_state=42), train
   on all of `bench_eval` (asis) vs all of `casual_deploy_matched`
   (1,076 vs 1,076; no subsampling — deterministic; deviation from her
   published n=1076 *sample* is noted, her sample is not reconstructible).
   Fit separately per regime at the frozen layer. Score sign flipped so
   positive = eval. Trap FP = fraction of a held-out cell with score > 0.
5. **Threshold** — score > 0 ⇔ "called eval"/"called benchmark" everywhere
   (equivalent to her 0.5-probability cut).
6. **Units** — projections additionally reported in sd units of our
   stated-stratum OOF scores at the same layer/regime (the nocue convention).
7. **Uncertainty** — nonparametric bootstrap, 2,000 draws, seed 0, resampling
   items within cell (pairs, for minimal-pair endpoints). No p-values; 95% CIs.
8. **Minimal pairs** — the 2,656 (asis, stripped) bench-cell pairs with
   `stripped_changed = true`.

## Gates (must pass before any cross-dataset endpoint is interpreted)

- **G1 (probe identity)**: rebuilding the stated-stratum layer record at the
  primary layer must reproduce every leaf of the checked-in sweep JSON for
  that layer exactly (tolerance 1e-9, expected bit-identical).
- **G2 (her-signal replication)**: her-protocol probe on 8B **raw** at L8 must
  show the format trap, FP(bench_deploy_rewritten ∪ bench_deploy_formal,
  asis) ≥ 85% (her RESULTS.md: 94.3% flat across L8–31). If G2 fails, stop
  and diagnose; do not proceed to endpoints.
- **Alignment**: her npz chunk `item_ids` must match the adapter jsonl row
  order; our npz/shard `item_ids` must match our jsonl row order.

### Amendment 2026-08-23 (pre-endpoint; recorded before any endpoint was read)

G1 at 1e-9 assumes the reference environment class (RunPod EPYC/Zen2, numpy
2.5.2, sklearn 1.9.0, openblas — where `verify_nocue_sweep.py` measured 0.0).
On the local Zen5 machine the same code and versions give lbfgs coefficient
jitter of ~1e-6 (different SIMD/BLAS kernels): continuous leaves reproduce to
≤ 3.4e-06 (AUC ≤ 7.2e-07), while a handful of near-zero decision scores flip
sign, moving the discrete `cells` percentages in single-item steps (max
observed 0.337 = ~6 of 1,920 items, in the c2b transfer regime; the format
record, whose leaves are saturated at 1.0, reproduces exactly). Forcing
`OPENBLAS_CORETYPE=ZEN` and disabling AVX-512 does not close the gap.

The near-separable b2c/c2b transfer fits amplify this jitter into ‖w‖ (b2c
AUC delta 3.1e-04, paired_mean delta 4.3e-02 in raw score units) even as
discrimination barely moves. Those two probes are never projected onto her
data — no endpoint uses them.

Complete measured local-vs-reference profile (8B, L8; the basis for the
tolerances below — jitter grows as a regime approaches separability):

| regime       | ref AUC | Δauc    | Δbal_acc | Δcells | Δpaired_mean |
|--------------|---------|---------|----------|--------|--------------|
| confounded   | 0.600   | 7.1e-07 | —        | —      | 3.4e-06 |
| decorrelated | 0.989   | 1.3e-04 | 6.5e-04  | 0.11   | 1.6e-03 |
| b2c          | 0.985   | 3.1e-04 | 2.8e-04  | 0.11   | 4.3e-02 |
| c2b          | 0.979   | 6.3e-05 | 1.2e-03  | 0.34   | 1.3e-02 |
| format       | 1.000   | 0 (exact) | 0      | —      | — |

Amendment (final; tolerances set once from this complete profile, not
iterated further): `DEVB_G1_MODE=local-fp-jitter` gates only the endpoint
regimes (confounded, decorrelated, format), per leaf class: auc 1e-3,
bal_acc/paired_pos 5e-3, cells 1.0 — an order of magnitude above the
measured jitter and an order below any semantic difference (a wrong
mask/fold/label moves auc by ≥1e-2). paired_mean (raw score units, tracks
‖w‖ not discrimination) and the b2c/c2b records are reported in the output
JSON but not gated; b2c/c2b are never projected onto her data. Runs in this
mode are **pilot** results; the archival run is executed on the reference
environment class where G1 must pass at 1e-9 on every leaf of every regime.
The mode used and the full deviation profile are recorded in each output
JSON under `gates.G1`.

## Primary endpoints (per model, template regime, primary layer)

- **P1 Trap under our probes**: %called-eval on `bench_deploy_*` (asis) for
  `purpose_conf` vs `purpose_decorr`. Hypothesis: confounded high,
  decorrelated substantially lower.
- **P2 Sensitivity retained**: AUC(bench_eval vs bench_deploy_*) and
  AUC(casual_eval_mutual vs casual_deploy_*) — within-format eval detection —
  plus overall AUC(eval vs deploy), for `purpose_decorr`.
- **P3 Format attribution**: (a) `format`-probe transfer AUC(bench vs casual
  cells); (b) OLS of her-protocol (template) probe score on
  [`purpose_decorr`, `format`] projections over asis rows: R², standardized
  coefficients, and each regressor's partial R².
- **P4 Scaffold share**: on minimal pairs, mean Δ(asis − stripped) of her
  probe score, `format` projection, and both purpose projections (sd units);
  plus her-probe FP on stripped `bench_deploy_*` (how much of the trap the
  dangling scaffold token alone carries — at her position vs ours).

Secondary: the same table on raw; robustness layer; cross-model scale trend.

## Known caveats (report alongside results)

- 405b/raw chunks 0–76 vs 77–132 came from different NDIF deployment
  instances (bf16 not bit-identical across restarts); template runs
  unaffected. Raw-regime 405B numbers carry this caveat.
- Her `id` column is non-unique; MuTual (casual_eval) has a length skew;
  bench_deploy cells carry a DeepSeek-rewriter fingerprint — so "eval vs
  deploy" contrasts that cross cells inherit her dataset's confounds beyond
  format. The within-format AUCs (P2) and minimal pairs (P4) are the
  clean contrasts; cross-cell AUCs are contextual.
- NDIF activations are unpinned-deployment bf16; third-decimal AUC drift vs
  a re-extraction is expected and does not bear on the endpoints.

## Outputs

`scripts/devbunova_decomposition.py` writes, per model:
- `results/devbunova_decomp/decomp_<model>.json` — endpoints + gates +
  provenance (input shas, HF repo revisions at read time, layer, seeds).
- `results/devbunova_decomp/decomp_<model>.scores.npz` — per-item projections
  (all probes × regimes) and the persisted fold models (coef / intercept /
  scaler mean / scale) for the frozen layers — closing the v3
  coef-persistence gap for these layers.
Layer slices of her activations are cached under
`activations/devbunova-2x2/<model>/<regime>/layer_XXX.npy` (gitignored).
