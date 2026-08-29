# Matched-format projection reliability check (preliminary, 2026-08-29)

## Status: partial / exploratory — job cancelled before completion

SLURM job `54161245` (`scripts/full320_v3_erasure_full_layer_sweep_reliability.py`
on `uahpc`) was cancelled by request after **Llama-3.1-8B** and
**Llama-3.1-70B** finished all layers but before **Llama-3.3-70B** started.
No final JSON was ever written by the script (it only dumps once, at the
very end, after all 3 models). Everything here was reconstructed from the
job's stderr log (`job_54161245_stderr_partial.log`, included) and merged
against the already-committed raw sweep files. This is exploratory,
2/3-model, log-parsed evidence — not the confirmation-grade artifact.

## Question

Does the rank-64 matched-format projection (already shown to substantially
close the cross-format transfer gap; see
`docs/matched_format_projection_transfer_diagnostic.md`) make the
**both-format purpose probe** more reliable, or just higher-AUC? RULES.md
requires balanced/threshold accuracy be checked separately from AUC, not
treated as interchangeable.

## What's in `reliability_preliminary.json`

Per-layer, for `llama31_8b` (32/32 layers) and `llama31_70b` (80/80 layers):

- `raw_joint_auc` / `raw_joint_bal_acc` — from the already-committed
  `results/layer_sweep_full320-v3_<model>.json` (`decorrelated` regime).
- `proj_joint_auc` / `proj_joint_bal_acc` — parsed from the cancelled job's
  stderr (same rank-64 projection as `results/full320_v3_erasure_full_layer_sweep.json`,
  cross-checked byte-for-byte against that file's `purpose_auc_after_erasure`
  at layer 0 for both models to confirm the log-parse is correct).
- `raw_format_auc` / `proj_format_auc` — same format-sanity-probe comparison
  as the format-persistence addendum in the main diagnostic doc.
- Deltas and the same `raw_joint_auc >= 0.95` post-purpose-formation-region
  filter used throughout the rest of this line of work.

**Not captured** (only exist in the script's final JSON, never written):
b2c/c2b balanced accuracy, `paired_pos`/`paired_mean` minimal-pair
consistency, and per-cell predicted-positive rates for the projected
condition. Recovering these needs a rerun.

## Result

Post-purpose-formation region (raw joint AUC ≥ 0.95):

| model | eligible layers | median raw bal_acc | median proj bal_acc | mean Δ bal_acc | mean \|Δ bal_acc\| | mean \|Δ AUC\| (for comparison) |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B  | 17/32 | 0.9316 | 0.9328 | +0.0006 | 0.0026 | 0.0013 |
| Llama-3.1-70B | 67/80 | 0.9310 | 0.9322 | −0.0008 | 0.0017 | 0.0006 |

Format persistence (all layers, both models — same finding as the main
diagnostic's rank-64 result, confirmed here independently from the
reliability job's own format-sanity probe):

| model | median raw format AUC | median proj format AUC |
|---|---:|---:|
| Llama-3.1-8B  | 1.000 | 0.922 |
| Llama-3.1-70B | 1.000 | 0.954 |

## Interpretation

Balanced accuracy moves by essentially nothing (mean |Δ| 0.0017–0.0026,
same order of magnitude as the AUC change of 0.0006–0.0013) in either
direction across both completed models. This is consistent with, not
contradictory to, the AUC-only finding: the projection is not silently
trading ranking quality for threshold accuracy (or vice versa) — both move
together, negligibly, in the post-purpose-formation region. No evidence of
a reliability gain or loss beyond what the AUC numbers already showed.

This is **2 of 3 models** and **joint-regime bal_acc only** (no b2c/c2b
bal_acc, no paired consistency). Historically this phenomenon has
replicated cleanly across Llama-3.3-70B in every other analysis in this
line of work (see `docs/matched_format_projection_transfer_diagnostic.md`),
so a third-model confirmation is expected but not yet in hand for this
specific reliability question. Do not cite this as confirmation-grade or
as covering all three models.

## Provenance

- Cluster job: `54161245`, `full320_v3_erasure_full_layer_sweep_reliability.py`,
  rank 64, same 5-fold cross-fit / probe as the rest of this line of work.
- Cancelled by explicit user request while `llama33_70b` was loading (no
  layers of that model were scored).
- Raw stderr log preserved verbatim: `job_54161245_stderr_partial.log`.
- Inputs: `results/layer_sweep_full320-v3_llama31_8b.json`,
  `results/layer_sweep_full320-v3_llama31_70b.json` (both already committed,
  untouched by this run).
