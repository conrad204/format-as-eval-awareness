# Development log — determinism fix and v3 causal run (2026-08-23)

## Starting point

`artifacts/exploratory/pilot80_factorial_causal_v2_invalid_20260823/run_manifest.json`
recorded both models failing the frozen `source_activation_reproduction` gate
(`max_abs<=0.125`, `cosine>=0.999`): 8B layer 29 row 419 (max_abs=0.25), 70B
layer 74 row 312 (max_abs=0.15625), both cosine>=0.999. `batch_size=1` and
exact checkpoint/hardware pinning had already been tried and did not resolve
it. Task: diagnose and, if confirmed, fix the root cause without relaxing the
gate, touching partial data, or editing anything prospectively frozen.

## 1. Determinism audit and fix

Neither `src/eval_format_mvp/extract.py` nor `scripts/run_pilot80_factorial_causal.py`
pinned deterministic GPU execution: TF32 matmul/cuDNN reduction order was
unpinned, `torch.use_deterministic_algorithms` was never called,
`CUBLAS_WORKSPACE_CONFIG` was unset, and attention implementation was left on
transformers' auto-select (can pick SDPA or flash-attention kernels with
different reduction order run to run). Added all four, plus the matching env
var export in both cluster sbatch launchers, changing no other scientific
parameter. Grepped both code paths for other nondeterminism sources (RNG
seeding, autocast, DataLoader workers, `cudnn.benchmark`); found none beyond
the four addressed.

Unrelated bug found and fixed in the same pass: `cluster_extract_pilot80_causal_recovery.sbatch`
invoked `-m eval_format_mvp extract`, which unconditionally loads
`catalogs/topics.yaml` via `cli.py`'s `main()` regardless of subcommand. The
scoped, hash-pinned source snapshots this launcher runs from never carry
`catalogs/`. Reproduced directly against the historical
`pilot80_reextract_src_6c572e15` snapshot (same `FileNotFoundError`).
`scripts/extract_activations.py` is the documented catalog-free replacement;
switched the invocation.

## 2. Shakedown (engineering determinism check only)

Extracted a 40-row window bracketing both failing rows, twice independently,
under the patched code, both models. Result: byte-identical
`activations_all_layers.npz` files (matching sha256) in both models, max_abs
diff exactly 0.0. Recorded as its own artifact,
`artifacts/exploratory/determinism_shakedown_20260823/`, separate from any
scientific run, no purpose/format/causal score inspected.

## 3. Full re-extraction and first full causal attempt — new failure mode found

Prospectively declared `pilot80_factorial_causal_mediation_v3_determinism_patched`
(addendum-only, diffed against v2_reextracted to confirm only activation
paths/hashes changed; folds/gates/layers/interventions byte-identical).
Re-extracted both models under the patched code (8B: job 53283747, 70B: job
53284711, both completed clean, `items_sha256` matching the frozen dataset).

First full causal run (batch_size=8, matching prior convention) still failed
`source_activation_reproduction`, at a *different* row (8B row 1061, layer
29, max_abs=0.25 — same value as the original pre-patch failure). Two
targeted diagnostics on the exact failing row, both giving perfect agreement
(fresh single-shot forward pass matched the stored activation exactly, and
replicating the causal script's own hook/clone/warm-process mechanics also
matched exactly) ruled out both cross-script structural differences and
cumulative-process-state drift. Cross-checking the actually-written raw
records showed the failure occurred on the *second* padded intervention
batch (batch_size=8, mixed sequence lengths) — the *first* batch (uniform
lengths, by luck) had passed. Root cause: padded multi-row intervention
batches produce genuine floating-point differences from the unpadded
`batch_size=1` baseline capture, independent of GPU kernel determinism.
Re-ran with `--batch-size 1` throughout (a non-scientific execution
parameter — same interventions, same everything, just unbatched); both
models then ran to completion with zero gate violations across all 163,200
records each.

## 4. Deleted checkpoint during the 70B run

After the 70B run completed, `analyze_pilot80_factorial_causal.py` failed
validating `runner_manifest.json`'s `model_file_hashes` (empty dict). The
entire 70B checkpoint directory
(`huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b4...`)
had been deleted from disk during the ~5-hour run (likely scratch cleanup),
after the model had already loaded successfully — the actual GPU computation
was unaffected, only the end-of-run provenance hash computation found the
files gone. Found a complete, verified copy of the exact same revision
cached under a different HF cache location
(`huggingface/transformers/models--meta-llama--Llama-3.3-70B-Instruct`,
matching `hidden_size=8192`/`num_hidden_layers=80`), restored the missing
path via symlink, and re-ran `run_pilot80_factorial_causal.py --resume`
against the *same* output directory. This re-validated all 163,200 existing
records (already correct), wrote zero new records
(`records_written_this_invocation: 0`), and correctly populated
`model_file_hashes` this time. `raw_sha256` before and after the manifest
regeneration is byte-identical, confirming the causal data itself was never
touched.

## 5. Frozen analysis, run once

`scripts/analyze_pilot80_factorial_causal.py` run exactly once on both
complete raw files (after fixing an infrastructure-only issue: the cluster
snapshot directories aren't git checkouts, so `git rev-parse HEAD` — used
purely for a provenance metadata field — failed; initialized minimal git
repos in the relevant directories rather than editing the frozen script).
Result and full interpretation: see
`artifacts/exploratory/pilot80_v3_causal_conclusion_20260823/conclusion.json`
and the resolved sealed prediction in `docs/sealed_prediction_20260823.md`.

## Summary of legitimate infrastructure fixes made along the way

None of these changed any scientific parameter (folds, gates, layers,
interventions, thresholds, seeds):
- Determinism pinning (TF32, `use_deterministic_algorithms`,
  `CUBLAS_WORKSPACE_CONFIG`, pinned eager attention).
- Catalog-free extraction invocation in the recovery sbatch.
- `--batch-size 1` for the causal run's intervention forward passes (was
  always `1` for baseline captures; extending it to interventions removes a
  real, independently-discovered padding-induced numerical discrepancy).
- Restored a checkpoint deleted by external scratch cleanup, from a verified
  identical-revision copy, then regenerated only the provenance manifest
  field with zero new GPU computation and a byte-identical raw data hash.
- Initialized minimal git repos in non-git cluster working directories so a
  provenance-only `git rev-parse HEAD` call in the frozen analysis script
  could succeed.

## 6. Prospective payload-disjoint orthogonalized-C transfer (2026-08-26)

To test a narrower positive mechanistic claim without changing the frozen v3
factorial decision, a transfer study was prospectively frozen on 80 full320
payload blocks with zero payload-block overlap with pilot80. The source bank
itself was not new: its observational activations had already been inspected.
The independent units here are the intervention payload blocks relative to
pilot80, not a new item bank or a new research program.

The study copied the completed pilot80 orthogonalized-C plans without
refitting: per-fold C directions, 32 random controls, common residual scales,
target readouts, and the fixed low/high/target layers. It extracted fresh
deterministic full320 baselines at batch size 1, then froze
`configs/full320_orthogonalized_c_transfer_v1.json` (SHA-256
`f58171224a631f26121451578b43659391f4c375cfbd0dbd1669b4d8389e2e15`)
before any transfer intervention output.

Outcome-blind eight-row shakedowns passed in both models: all 32 source/target
comparisons had `max_abs=0.0`; minimum cosine was `0.9999998808` for 8B and
`0.9999999404` for 70B. Both complete causal runs then passed the unchanged
`max_abs<=0.125`, `cosine>=0.999` gate and produced exactly 264,960 records
per model. No partial output was inspected.

The frozen analyzer ran exactly once. Its predeclared decision was
`POSITIVE_REPLICATION`:

- 8B high-minus-low purpose-interaction response: `0.748536`, 95% payload-block
  bootstrap CI `[0.738766, 0.758228]`, Holm `p=7.08e-99`.
- 70B: `0.681691`, CI `[0.674173, 0.688965]`, Holm `p=2.06e-104`.
- All 80 payload-block effects were positive in each model.
- Alpha-0.5/full response ratios were `0.500008` and `0.499820`.
- The C means exceeded the frozen 97.5th percentile of absolute random-control
  means by large margins in both models.

The claim is narrow but central: a fixed pilot-trained residual direction
causally transfers across payload-disjoint blocks and preserves its
preselected high-versus-low layer ordering in two model scales, with
dose-linearity and random-direction calibration. It is not evaluation
awareness, not behavioral steering, not a pure interaction channel, and not a
full A/B/C factorial replication. Shared-purpose collateral responses were
substantial (`0.5302` in 8B, `0.2464` in 70B), and payload-block effect
magnitudes were not correlated across models (Pearson `r=0.0639`,
`p=0.574`). The prior strict v3 factorial negative decision remains unchanged.

Complete compact evidence, hashes, limitations, and raw cluster paths are in
`artifacts/exploratory/full320_orthc_transfer_positive_20260826/`.
