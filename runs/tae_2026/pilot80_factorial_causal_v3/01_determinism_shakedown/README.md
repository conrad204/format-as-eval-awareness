# 01_determinism_shakedown

- **Status:** `passed`
- **Study:** `pilot80_factorial_causal_v3`

## What happened

Independent repeated extraction was bit-identical for both frozen model paths; maximum absolute difference 0.0.

## Evidence

[`artifacts/exploratory/determinism_shakedown_20260823/run_manifest.json`](../../../../artifacts/exploratory/determinism_shakedown_20260823/run_manifest.json)

## Boundary

This subrun inherits the frozen scope of the v3 determinism-patched protocol. The final negative factorial decision and narrow positive high-layer C response must be reported together; this is not evidence of evaluation awareness.
