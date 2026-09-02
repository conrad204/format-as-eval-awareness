# Full320 orthogonalized-C Llama-3.1-70B model extension v1

Prospective new-checkpoint extension after the known positive 8B/Llama-3.3-70B transfer. Final decision: **positive model extension**.

The frozen Llama-3.1-70B high-minus-low effect was **0.54455** (95% payload-block bootstrap CI **[0.53739, 0.55177]**); all 80 blocks were positive, the half-dose/full-dose ratio was 0.5005, and the effect exceeded the frozen absolute-random threshold 0.02759.

| Step | Status | Purpose |
|---|---|---|
| [01_preregistration](01_preregistration/) | `frozen` | Prospectively declared the new Llama-3.1-70B checkpoint extension after disclosing the known positive two-model result. |
| [02_extract_pilot80](02_extract_pilot80/) | `completed` | Deterministic all-layer pilot80 extraction: 1,760 x 80 x 8,192 float32 stored activations. |
| [03_extract_full320](03_extract_full320/) | `completed` | Deterministic all-layer full320 extraction: 2,880 x 80 x 8,192 float32 stored activations. |
| [04_fit_pilot_plan](04_fit_pilot_plan/) | `completed_before_transfer_output` | Fit the unchanged 25-fold pilot80 A/B/C directions, residual scales, random controls, and target readouts for the frozen checkpoint. |
| [05_orthogonalize_C](05_orthogonalize_C/) | `completed_before_transfer_output` | Residualized every frozen C direction against span(A,B); maximum post-transform absolute cosine was 2.98e-9. |
| [06_bind_transfer_plan](06_bind_transfer_plan/) | `completed_before_transfer_output` | Bound copied pilot directions/readouts to all 1,920 stated full320 test rows without refitting. |
| [07_outcome_blind_shakedown](07_outcome_blind_shakedown/) | `passed` | Eight-row, 16-comparison source/target reproduction check passed with maximum absolute difference 0.0; no scientific outcome was computed. |
| [08_complete_causal_run](08_complete_causal_run/) | `completed` | Complete 264,960-record intervention run; every record was written in one invocation and the immutable gate remained intact. |
| [09_frozen_analysis](09_frozen_analysis/) | `positive_model_extension` | One-shot frozen analysis: high-minus-low mean 0.54455; 95% payload bootstrap CI [0.53739, 0.55177]; all 80 blocks positive. |
| [10_local_archive](10_local_archive/) | `verified` | Fifty non-bytecode files / 6,973,721,395 bytes matched independent remote byte sizes and SHA-256 hashes. |

## Start here

- [Scoped conclusion](../../../artifacts/exploratory/full320_orthc_llama31_70b_model_extension_positive_20260827/conclusion.json)
- [Frozen results](../../../artifacts/exploratory/full320_orthc_llama31_70b_model_extension_positive_20260827/extension_results.json)
- [Runner manifest](../../../artifacts/exploratory/full320_orthc_llama31_70b_model_extension_positive_20260827/runner_manifest.json)
- [Local archive manifest](../../../artifacts/exploratory/full320_orthc_llama31_70b_model_extension_positive_20260827/local_archive_manifest.json)
- [Prospective preregistration](../../../configs/full320_orthc_llama31_70b_model_extension_prereg_v1.json)

## Claim boundary

This supports prospective generalization of the narrow fixed internal intervention/readout response to one additional frozen checkpoint. It is not independent confirmation, evaluation awareness, behavioral steering, a scaling law, or a universal Llama mechanism.
