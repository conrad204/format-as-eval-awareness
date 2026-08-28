# determinism_shakedown_20260823

- **Status:** `shakedown_passed`
- **Record type:** completed run
- **Canonical source:** [`artifacts/exploratory/determinism_shakedown_20260823`](../../../../artifacts/exploratory/determinism_shakedown_20260823/)

## What happened

Both models pass with enormous margin (bit-identical run_a/run_b activation archives; observed max_abs is exactly 0.0, versus the 0.25 and 0.15625 originally observed). This isolates the source_activation_reproduction gate failure to non-deterministic GPU kernel execution (unpinned TF32/cuBLAS redu…

## Claim boundaries

- This registry entry does not expand the claim scope in the source manifest or conclusion.

## Evidence

- [`70b_shakedown_report.json`](../../../../artifacts/exploratory/determinism_shakedown_20260823/70b_shakedown_report.json)
- [`8b_shakedown_report.json`](../../../../artifacts/exploratory/determinism_shakedown_20260823/8b_shakedown_report.json)
- [`run_manifest.json`](../../../../artifacts/exploratory/determinism_shakedown_20260823/run_manifest.json)
