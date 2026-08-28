# Full320 orthogonalized-C transfer v1

Prospective payload-disjoint causal-transfer study. Final decision: **positive replication**, narrowly scoped to the fixed internal intervention/readout contract.

| Step | Status | Purpose |
|---|---|---|
| [01_bank_selection](01_bank_selection/) | `completed` | Prospectively selected 80 full320 payload blocks with zero payload-block overlap with pilot80. |
| [02_extract_8b](02_extract_8b/) | `completed` | Deterministic all-layer Llama-3.1-8B activation extraction for 2,880 transfer rows. |
| [03_extract_70b](03_extract_70b/) | `completed` | Deterministic all-layer Llama-3.3-70B activation extraction for 2,880 transfer rows. |
| [04_plan_8b](04_plan_8b/) | `completed` | Bound the pilot80-trained orthogonalized-C directions, random controls, scales, and readouts to the full320 test rows for 8B. |
| [05_plan_70b](05_plan_70b/) | `completed` | Bound the pilot80-trained orthogonalized-C directions, random controls, scales, and readouts to the full320 test rows for 70B. |
| [06_shakedown_8b](06_shakedown_8b/) | `passed` | Outcome-blind source/target reproduction check: maximum absolute difference 0.0. |
| [07_shakedown_70b](07_shakedown_70b/) | `passed` | Outcome-blind source/target reproduction check: maximum absolute difference 0.0. |
| [08_causal_8b](08_causal_8b/) | `completed` | Complete 264,960-record intervention run; raw SHA-256 aaffc86f… |
| [09_causal_70b](09_causal_70b/) | `completed` | Complete 264,960-record intervention run; raw SHA-256 990b5127… |
| [10_frozen_analysis](10_frozen_analysis/) | `positive_replication` | One-shot frozen analysis: high-minus-low effects 0.7485 (8B) and 0.6817 (70B), all 80 blocks positive in each model. |
| [11_local_archive](11_local_archive/) | `verified` | All 49 run, source-snapshot, and Slurm-log files copied locally and matched byte size plus SHA-256. |

## Start here

- [Final conclusion](../../../artifacts/exploratory/full320_orthc_transfer_positive_20260826/conclusion.json)
- [Frozen results](../../../artifacts/exploratory/full320_orthc_transfer_positive_20260826/transfer_results.json)
- [Evidence manifest](../../../artifacts/exploratory/full320_orthc_transfer_positive_20260826/evidence_manifest.json)
- [Local archive manifest](../../../artifacts/exploratory/full320_orthc_transfer_positive_20260826/local_archive_manifest.json)

## Later prospective checkpoint extension

The separately frozen [Llama-3.1-70B model extension](../full320_orthc_llama31_70b_model_extension_v1/) also passed. It strengthens checkpoint robustness of this narrow response but is not independent source-bank confirmation and does not alter this study's two-model decision.
