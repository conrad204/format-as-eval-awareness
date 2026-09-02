# pilot80_factorial_causal_v2_invalid_20260823

- **Status:** `invalid_incomplete_baseline_reproduction_gate_failure`
- **Record type:** invalid run
- **Canonical source:** [`artifacts/exploratory/pilot80_factorial_causal_v2_invalid_20260823`](../../../../artifacts/exploratory/pilot80_factorial_causal_v2_invalid_20260823/)

## What happened

This v2 attempt was `TAE_TECHNICALLY_BLOCKED`: both models failed the unchanged source-activation reproduction gate before a complete causal dataset existed. Its partial raw files remain excluded from inference. A separately preregistered determinism-patched v3 run later passed the same gate and produced the standing complete decision; see [`pilot80_v3_causal_conclusion_20260823`](../pilot80_v3_causal_conclusion_20260823/).

## Claim boundaries

- Never analyze the excluded partial causal records or relax the failed gate.
- Do not report this superseded engineering failure as the current final causal decision; the complete v3 result is negative under the frozen factorial criteria.

## Evidence

- [`70b_job_53274676.err`](../../../../artifacts/exploratory/pilot80_factorial_causal_v2_invalid_20260823/70b_job_53274676.err)
- [`70b_partial_causal_raw.jsonl`](../../../../artifacts/exploratory/pilot80_factorial_causal_v2_invalid_20260823/70b_partial_causal_raw.jsonl)
- [`8b_job_53274675.err`](../../../../artifacts/exploratory/pilot80_factorial_causal_v2_invalid_20260823/8b_job_53274675.err)
- [`8b_partial_causal_raw.jsonl`](../../../../artifacts/exploratory/pilot80_factorial_causal_v2_invalid_20260823/8b_partial_causal_raw.jsonl)
- [`run_manifest.json`](../../../../artifacts/exploratory/pilot80_factorial_causal_v2_invalid_20260823/run_manifest.json)
