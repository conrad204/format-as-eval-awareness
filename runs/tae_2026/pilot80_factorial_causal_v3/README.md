# Pilot80 factorial causal mediation v3

Determinism-patched rerun of the frozen factorial causal protocol. Final decision: **negative**, with a narrow positive high-layer C internal-mediation response.

| Step | Status | Purpose |
|---|---|---|
| [01_determinism_shakedown](01_determinism_shakedown/) | `passed` | Independent repeated extraction was bit-identical for both frozen model paths; maximum absolute difference 0.0. |
| [02_extract_8b](02_extract_8b/) | `completed` | Determinism-patched batch-one all-layer re-extraction for the 8B pilot80 causal bank. |
| [03_extract_70b](03_extract_70b/) | `completed` | Determinism-patched batch-one all-layer re-extraction for the Llama-3.3-70B pilot80 causal bank. |
| [04_plan_8b](04_plan_8b/) | `completed` | Immutable cross-fitted A/B/C intervention plan for 8B. |
| [05_plan_70b](05_plan_70b/) | `completed` | Immutable cross-fitted A/B/C intervention plan for Llama-3.3-70B. |
| [06_causal_8b](06_causal_8b/) | `completed` | Complete 163,200-record causal execution with the frozen reproduction gate intact. |
| [07_causal_70b](07_causal_70b/) | `completed` | Complete 163,200-record causal execution with the frozen reproduction gate intact. |
| [08_frozen_analysis](08_frozen_analysis/) | `negative` | Frozen one-shot decision was negative: C passed at the high layer, but A, B, and full factorial criteria did not jointly pass. |

## Start here

- [Final conclusion](../../../artifacts/exploratory/pilot80_v3_causal_conclusion_20260823/conclusion.json)
- [Evidence manifest](../../../artifacts/exploratory/pilot80_v3_causal_conclusion_20260823/evidence_manifest.json)
- [Development log](../../../docs/development_log.md)
