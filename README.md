# Format-as-eval-awareness: project context

This README is a handoff document, not a marketing summary. It records the scientific scope, the evidence boundaries, the artifacts that are usable, and the work that is still running or unfinished.

## Running now / not finalized yet

Status checked on 2026-08-29. Cluster state is live and can change; a `RUNNING` job is not evidence until its output, manifest, counts, and analysis have completed.

| Job | State | Current status | What it means |
|---|---|---|---|
| `54204383` | RUNNING | Llama-3.1-70B matched-control interchange patch; around layer 34/80, about 73k of 339,840 records in the latest log | Incomplete; no final 70B result or bootstrap |
| `54204384` | RUNNING | Llama-3.3-70B matched-control interchange patch; around layer 34/80, about 72k of 339,840 records in the latest log | Incomplete; no final 70B result or bootstrap |
| `54208103` | RUNNING | Llama-3.1-70B real-swap interchange run; startup/sanity check passed and layer 0 completed | Incomplete; cannot compare against a complete matched run yet |
| `54208125` | PENDING | Llama-3.3-70B real-swap interchange run | Waiting because of `QOSMaxGRESPerUser` |
| `54198919` | FAILED | 8B factorial B/C residual sweep finished layer computation but failed in reference validation because a 32-layer result was compared with a 3-value reference | No residual result exists |
| `54206731` | CANCELLED | Retry of the 8B factorial residual sweep | No result; the retry was canceled before execution |

The incomplete local files below are checkpoints or partial transfers, not final evidence:

```text
results/tae_ixpatch_matched_llama31_70b_partial.jsonl
results/tae_ixpatch_matched_llama33_70b_partial.jsonl
results/tae_ixpatch_full_llama31_8b.job54161731.partial_snapshot.jsonl
results/tae_ixpatch_full_llama31_8b.job54161731.partial_stratified.json
results/tae_ixpatch_full_llama31_8b.job54161731.partial_stratified_v2.json
```

The streaming/checkpoint repair for the factorial residual script is committed in `0926871`. The reference-indexing fix is in `6fe3442`. Neither creates a result by itself.

## What is ready now

The strongest completed new causal artifact is the Llama-3.1-8B full-layer interchange-patching analysis:

```text
results/tae_ixpatch_full_llama31_8b.jsonl
results/tae_ixpatch_matched_llama31_8b.jsonl
results/tae_ixpatch_full_llama31_8b_bootstrap.json
results/tae_ixpatch_full_llama31_8b.manifest.json
results/tae_ixpatch_matched_llama31_8b.manifest.json
docs/interchange_patching_llama31_8b_results.md
plots/jpp/interchange_patch_llama31_8b/
```

It is complete and internally consistent:

- Llama-3.1-8B-Instruct, all 32 layers, rank 16;
- 2,124 rows per layer-condition cell;
- 280 complete payload blocks;
- `real_swap`, `random_control`, and `matched_control` conditions;
- 135,936 real-swap/random-control records and 67,968 matched-control records;
- 2,000 shared payload-block bootstrap replicates;
- activation-bank and item-bank identity recorded in both manifests;
- real hooked forward passes on the model, not frozen-activation projection followed by probe retraining.

The canonical bootstrap is `results/tae_ixpatch_full_llama31_8b_bootstrap.json`. The untracked `results/tae_ixpatch_allcontrols_llama31_8b_bootstrap.json` is a redundant copy and is not the canonical path.

### 8B interchange result

The matched-control joint worst-layer 95% intervals are:

```text
format IIA gap, real_swap - matched_control:  [+0.041, +0.087]
purpose gap, matched_control - real_swap:     [-1.174, -0.825]
```

Safe interpretation:

> In a full-layer real-forward Llama-3.1-8B interchange test, the rank-16 format subspace moves the format readout more effectively than a per-row norm-matched random perturbation. Purpose-readout disruption is depth-dependent and does not support a uniform purpose-protection claim.

This is **development internal causal mediation**. It is not validated behavioral steering, evaluation-awareness confirmation, a universal causal format mechanism, or a multi-model causal replication.

The original random control was rank-matched but not magnitude-matched. The completed matched control addresses that objection. The matched-control uncertainty uses one shared payload-block resample per bootstrap replicate, reused across layers and conditions; layers are a continuous depth trajectory, not independent replicated experiments.

## Scientific question and intended claim

The project studies whether an apparent evaluation-purpose signal in language-model residual streams is actually purpose information, surface format, or a mixture of both.

The target workshop claim is deliberately narrower than “the model has evaluation awareness”:

> Evaluation-context representations contain a format-shared purpose component and a format-conditioned interaction component. Naive probes can collapse onto format; decorrelated training recovers purpose; imperfect cross-format transfer should be explained by the geometry and causal role of the shared and interaction components.

The work is for the TAE (Trust-AI-Eval), NeurIPS 2026 workshop sprint. The project does not attempt to resurrect the earlier failed confirmatory `jProbe` claim. The governing scientific rules are in `RULES.md`.

Use these terms precisely:

- **purpose decodability**: a probe separates stated `evaluation` from stated `deployment` response purpose;
- **format decodability**: a probe separates `benchmark` from `casual` surface format;
- **evaluation awareness**: a stronger behavioral interpretation that is not established merely by an activation probe;
- **internal causal mediation**: an intervention changes a model-internal readout or a mechanistically bounded representation;
- **behavioral steering**: an intervention changes a validated behavioral endpoint with adequate baseline accuracy/calibration and controls.

Current results support the first two terms and some bounded internal causal statements. They do not support a confirmation-grade behavioral-awareness claim.

## Frozen 2 x 2 object

Every canonical payload is rendered in all four cells of a purpose-by-format factorial design:

```text
                 benchmark       casual
 evaluation       h_EB            h_EC
 deployment       h_DB            h_DC
```

With effect coding `p in {+1,-1}` for purpose and `f in {+1,-1}` for format:

```text
h_(p,f) = mu + p*A + f*B + p*f*C + epsilon
```

The terms are:

- `A`: format-shared purpose main effect;
- `B`: format main effect / offset;
- `C`: purpose-by-format interaction.

For matched purpose contrasts:

```text
delta_B = h_EB - h_DB = 2(A + C)
delta_C = h_EC - h_DC = 2(A - C)
A = (delta_B + delta_C) / 4
C = (delta_B - delta_C) / 4
B = ((h_EB + h_DB) - (h_EC + h_DC)) / 4
```

This decomposition is fixed. It must not be redefined after inspecting outcomes.

## Canonical dataset and provenance

The primary freeze is `format-EA/full320-v3`:

```text
HF revision: d6c3be7be83c469d9a6a2aa4e797f3a7b538efde
items SHA-256: 5528819d74e696d1a595062047a829836dfde1d2a45287b06a5ac1ba5d1b6c67
items: 11,520
payload blocks: 320
```

Local canonical paths, when the private HF freeze is materialized:

```text
data/full320-v3/rendered_items_v3_full320_8-17-2026.jsonl
provenance/full320-v3/
```

The 320 payload blocks cover 16 topics, with 10 Grok-4.3 and 10 Gemini-3.1-Pro generated blocks per topic. The rendered bank has 13 fixed templates: seven benchmark/quiz-like and six casual. The stated-purpose stratum contains 7,680 rows. The remaining 3,840 no-cue rows have no explicit purpose statement and are a separate secondary analysis.

The full dataset is matched by construction: the same payload content appears in all four factorial cells. Complete payload blocks and purpose families, not independent rendered rows, are the units for bootstrap and generalization.

Git tracks manifests and revision pins; large datasets and activation archives are private or gitignored. Do not pull a dataset without its pinned revision. The provenance boundary is:

```text
data/<freeze>/             gitignored materialized data
provenance/<freeze>/        tracked revision, render, extraction, and hash anchors
```

The submission inventory is maintained in:

```text
artifacts/submission_sprint_manifest.json
artifacts/submission_figure_manifest.json
```

Both currently declare `development_only` status.

## Completed observational results

Primary activation extraction uses the last prompt token, residual stream output before the final model norm, every transformer block, BF16 inference, and float32 saved activations. Probes are deterministic sklearn logistic regressions with leave-one-purpose-family-out folds.

Pinned models and layer counts:

```text
meta-llama/Llama-3.1-8B-Instruct   32 layers
meta-llama/Llama-3.1-70B-Instruct  80 layers
meta-llama/Llama-3.3-70B-Instruct  80 layers
```

At each model's best decorrelated layer, the checked-in stated-purpose sweep reports approximately:

| model | confounded AUC | decorrelated AUC | benchmark-to-casual AUC | casual-to-benchmark AUC | format AUC |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.600 | 0.989 | 0.951 | 0.940 | 1.000 |
| Llama-3.1-70B | 0.682 | 0.998 | 0.985 | 0.985 | 1.000 |
| Llama-3.3-70B | 0.667 | 0.996 | 0.985 | 0.979 | 1.000 |

The result is observational:

- the natural diagonal training regime is a format trap;
- the decorrelated purpose probe is calibrated across the four cells;
- probes trained entirely in one format transfer to the other at high AUC;
- format is available from the embedding stream, while purpose emerges in mid-depth layers;
- the transfer gap is smaller in the 70B models than in the 8B model.

The full summary is `v3_results_summary.md`. The main reproducible result JSONs are under `results/layer_sweep_full320-v3_*.json` and `results/surface_probes_v3_full320.json`.

### Surface controls

Raw-text controls show that stated purpose is not recoverable from word or character n-grams under the template-held-out controls, while format and template identity remain deliberately recoverable. This blocks a simple bag-of-words explanation for the stated-purpose activation result; it does not establish a universal absence of surface structure.

### Equal-N control

The unequal training-row count in the original decorrelated versus one-format comparison is a known confound. The completed 8B layer-12 equal-N mixed-format control uses four recorded seeds, 320 payload-block bootstrap resamples, and 1,000 bootstrap replicates per seed:

```text
benchmark-only AUC: 0.9588; balanced accuracy: 0.8949
casual-only AUC:    0.9538; balanced accuracy: 0.8775
equal-N mixed AUC:  0.9797; balanced accuracy: 0.9241
```

Artifact:

```text
artifacts/full320_equal_n_llama31_8b_l12.json
```

This is a development observational control for 8B only. An equal-N control for all three models remains missing.

## Completed geometry and erasure analyses

### Full-layer geometry

`results/full320_v3_geometry_full_layer_sweep.json` and the associated figure cover all layers of all three pinned models, ranks 1, 4, 16, and 64, with 10,000 conditional-null draws for ranks 1, 4, and 16. The analysis compares the matched-format subspace against conditional random subspaces drawn from the same top-64 space.

The rank-16 trajectory shows unusually low purpose-direction energy in the matched-format subspace across essentially the full depth profile relative to the conditional isotropic null. The corrected 3-depth by 3-model significance inventory survives both Holm and Bonferroni correction. The full-layer figure is a descriptive continuity trajectory, not 32 or 80 independent significance tests.

The geometry figure and caption are:

```text
plots/jpp/full_layer_geometry_trajectory/
```

Do not call this an all-model A/B/C geometry-and-transfer-correlation result. That required analysis is still missing.

### Format erasure / dose response

The full-layer erasure and dose-response analyses are complete for the three pinned checkpoints. Rank-64 matched-format erasure reduces the integrated cross-format transfer gap by approximately:

```text
Llama-3.1-8B:  58.5%
Llama-3.1-70B: 62.8%
Llama-3.3-70B: 65.3%
```

Relevant artifacts:

```text
results/full320_v3_erasure_full_layer_sweep.json
results/full320_v3_erasure_dose_response_bootstrap.json
results/full320_v3_format_erasure_retrain.json
results/full320_v3_projected_basis_conditional_null.json
results/full320_v3_conditional_null_multiplicity.json
plots/jpp/full_layer_gap_erasure/
```

These are development geometry/erasure results. They do not prove a universal causal format-erasure mechanism.

### No-cue sweeps

No-cue sweeps now exist for the three pinned models and for the unpinned NDIF 405B extension:

```text
results/nocue_sweep_full320-v3_llama31_8b.json
results/nocue_sweep_full320-v3_llama31_70b.json
results/nocue_sweep_full320-v3_llama33_70b.json
results/nocue_sweep_full320-v3_llama31_405b_ndif.json
```

They are secondary development diagnostics. They are not confirmation evidence and must not be silently pooled with the stated-purpose primary sweep.

## Causal/intervention evidence and failures

### Completed 8B real-forward interchange

The completed 8B test swaps the rank-16 matched-format subspace between matched benchmark/casual counterparts during real forward passes. It compares:

- `real_swap`: the structured matched-counterpart swap;
- `random_control`: a rank-matched random orthogonal direction;
- `matched_control`: a random additive perturbation matched to each row's real-swap delta norm.

The purpose-readout effect is depth-dependent. The matched control confirms that the late-layer format-readout effect is direction-specific rather than explained only by perturbation magnitude. The purpose-protection hypothesis does not hold uniformly across depth.

Use `docs/interchange_patching_llama31_8b_results.md` for the complete interpretation and caveats.

### Frozen pilot causal run: negative

The separate pilot80 factorial causal run was pre-registered and analyzed once. Its final decision is:

```text
decision: negative
A_shared: false
B_format: false
C_interaction: true
dose_response: true
endpoint_valid: false
factorial_cube_agreement: true
geometry_to_causality: true
hook_and_reproduction: true
```

The high-interaction selected layers showed clean C effects, but the frozen conjunction also required low-interaction control layers. A and B failed there, and one 8B readout marginally missed the endpoint AUC validity bar. The negative result must remain visible; the informal high-layer peek is not a substitute for the frozen multi-layer gate.

Source: `docs/sealed_prediction_20260823.md` and the corresponding `runs/tae_2026/` manifests.

### Steering diagnostics

The independent no-cue steering study refit the probes in raw residual space, verified probe parity against the live model, and compared matched-norm interventions with random directions. It found:

- format steering causally moves the 70B verbalized logit readout strongly and the 405B readout suggestively;
- format steering changes 8B output formatting above a tight random band, although the 8B verbalized readout is null;
- the decorrelated purpose direction is null-to-weak on these readouts;
- larger-model style changes at the required high dose are dominated by generic perturbation effects.

This supports a bounded claim that models use some format direction on these internal/logit/style diagnostics. It does not validate a behavioral evaluation-awareness endpoint. Source: `docs/steering_results.md`, `docs/steering_prereg.md`, and `results/steering/`.

## External and scale extensions

### 405B NDIF extension

The repository contains a full320 Llama-3.1-405B development extension using NDIF per-layer activation shards:

```text
results/layer_sweep_full320-v3_llama31_405b_ndif.json
results/nocue_sweep_full320-v3_llama31_405b_ndif.json
provenance/full320-v3/extraction_config_405b_ndif.json
```

The NDIF model revision is unpinned. Some raw chunks came from different deployment instances. Treat all 405B results as development-only scale-extension evidence, not a pinned model replication or confirmation-grade result.

### External Devbunova comparison

The external `external/eval-awareness-2x2` archive and `results/devbunova_decomp/decomp_405b.json` support a cross-dataset decomposition of the format trap. The local checkout contains partial activation chunks and is not a matched source-corpus freeze. Treat this as independent external validation/context, not as an additional canonical replication.

The frozen analysis plan and caveats are in `docs/devbunova_decomposition_plan.md`.

### J-lens diagnostics

The J-lens corpus, fits, and geometry/assay artifacts are development mechanistic diagnostics. They are not confirmation evidence and must not be used to promote an unvalidated purpose direction into a behavioral claim. See `runs/tae_2026/jlens_v3/`, `artifacts/jlens*`, and the relevant development log entries.

## What is still missing

The submission inventory intentionally records these gaps:

1. Equal-N mixed-format controls for all three pinned models; only the 8B layer-12 control is complete.
2. A complete all-model A/B/C vector-geometry analysis with transfer-correlation tests.
3. A behaviorally validated causal endpoint with adequate baseline accuracy and calibration.
4. An accepted confirmation bank and sealed confirmation-grade activation extraction.
5. Complete 70B real/matched interchange outputs, shared-block bootstrap, and interpretation.
6. A completed factorial B/C residual decomposition that survives its provenance and reference checks.

Do not turn any missing item into a claim by substituting a nearby diagnostic.

## Claim restrictions

These restrictions apply to every result, caption, README update, and paper draft:

- Say **stated response purpose decodability**, not “evaluation awareness,” unless explicitly discussing the stronger unsupported interpretation.
- Keep AUC and balanced accuracy separate.
- Do not interpret unequal-N cross-format gaps mechanistically.
- Do not treat layers as independent model replicates.
- Bootstrap complete payload blocks or purpose families, preserving the four-cell structure.
- Do not call observational geometry causal.
- Do not call pilot endpoints confirmation endpoints.
- Do not claim validated behavioral steering from internal/logit/style readouts.
- Do not claim a universal format-erasure mechanism.
- Do not claim pinned-checkpoint replication from the unpinned NDIF 405B data.
- Do not hide negative, failed, canceled, or incomplete interventions.
- Do not change layers, seeds, normalization, splits, endpoint wording, or success criteria after seeing outcomes without recording a new exploratory protocol.

## Repository layout

```text
src/eval_format_mvp/       generation, validation, rendering, extraction, CLI
scripts/                   sweeps, geometry, erasure, causal, steering, plots
catalogs/                  format/frame/topic design catalogs
configs/                   frozen and development run configurations
schemas/                   generation, validation, and render schemas
prompts/                   generation and validation prompt templates
tests/                    offline regression and frozen-chain tests
data/                     gitignored materialized HF freezes
provenance/               tracked revision, render, extraction, and hash anchors
results/                   checked-in analysis outputs and development artifacts
plots/                    generated figures
artifacts/                submission/evidence inventories and compact manifests
docs/                      plans, logs, result writeups, and interpretation boundaries
runs/tae_2026/             run-level evidence directories and READMEs
```

The package is `eval-format-mvp`. It requires Python 3.11–3.13 and is configured in `pyproject.toml`.

## Reproducibility and data access

The private HF datasets require access to the `format-EA` organization and a token. `data/` is gitignored. Pull a pinned freeze and verify it against the tracked manifest:

```bash
pip install -e ".[analysis,dev]"
hf auth login

hf download format-EA/full320-v3 --repo-type dataset \
  --revision "$(jq -r .revision provenance/full320-v3/hf_snapshot.json)" \
  --local-dir data/full320-v3

eval-format-mvp verify-manifest-v3 \
  --manifest provenance/full320-v3/render_manifest_v3_full320.json
```

For the older v1 freeze, use the pinned revision in `provenance/v1-8-14-2026/hf_snapshot.json` and the v1 manifest command. Never download an unpinned revision when reproducing evidence.

The canonical v3 generation-to-render chain is self-contained in this repository. New generation requires provider credentials (`OPENROUTER_API_KEY` or Vertex/gcloud); rendering and validation do not.

## Offline checks and plotting

The repository's tests are designed to run without network access or a GPU. On Windows, use a writable project-local pytest temporary directory if the default system temporary directory denies access:

```bash
python -m pytest --basetemp=.pytest-tmp-final
```

The plots-only pipeline for the older checked-in layer-sweep figures is:

```bash
pip install -e ".[analysis]"
python scripts/make_plots.py
python scripts/gap_plots.py
python scripts/make_plot_06a.py
```

The causal and geometry figures have their own `generate.py` files under `plots/jpp/`; each figure directory contains its caption and source. Do not regenerate a figure and silently overwrite a pinned artifact without updating its manifest and hash.

## Operational rules

Cluster work is restricted to the approved `jppatton/langea` subtrees. Inspect `squeue`, `sacct`, logs, and output manifests before changing or stopping jobs. Store large model/cache data under the approved scratch or bighome paths, not in arbitrary home directories. Do not commit credentials, SSH keys, tokens, or private environment files.

The parent agent owns scientific integration. Development, exploratory, and confirmation evidence stay explicitly separated. The paper must be written around the strongest defensible result, including negative and null results, rather than around a predetermined positive claim.

## Key reading order

1. This README for current state and evidence boundaries.
2. `RULES.md` for non-negotiable scientific and operational rules.
3. `v3_results_summary.md` for the primary stated-purpose observational result.
4. `docs/interchange_patching_llama31_8b_results.md` for the completed 8B causal interchange result.
5. `docs/sealed_prediction_20260823.md` for the negative frozen pilot causal result.
6. `docs/steering_results.md` for internal/logit/style steering diagnostics.
7. `docs/development_log.md` and `docs/v2_running_log.md` for chronology and failed/rejected paths.
8. `artifacts/submission_sprint_manifest.json` and `artifacts/submission_figure_manifest.json` for the current inventory and missing-evidence list.
