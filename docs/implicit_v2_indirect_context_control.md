# Implicit-v2: indirectly worded context control

Revised 2026-09-01. This package is a supporting control for the workshop's
stated-purpose measurement-validity study. The word *implicit* refers only to
indirect wording in the prompt; it does not mean that the relevant status is
absent from the text.

## Design and interpretation

The implicit-v2 scenarios describe evaluative structure in the prompt: blind
comparison, a sealed key, offline rehearsal, and no information leaving the
sandbox. A decoder trained on explicitly stated purpose is applied to these
indirectly worded scenarios and matched live-use scenarios across held-out
scenario families.

This is a generalization-across-wording result. It is not evidence for a
behavioral faculty, spontaneous recognition without textual evidence, or a
status inferred solely from environmental outcomes. The probe is a frozen
readout, not a causal intervention.

## Results

Family-bootstrap results from frozen explicit-purpose probes:

| checkpoint | layer | joint AUC | benchmark to casual | casual to benchmark |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 8 | 0.5980 [0.5269, 0.6697] | 0.5783 [0.5114, 0.6433] | 0.5671 [0.4946, 0.6369] |
| Llama-3.1-70B | 18 | 0.6377 [0.5898, 0.6909] | 0.6317 [0.5776, 0.6867] | 0.6145 [0.5777, 0.6609] |
| Llama-3.3-70B | 18 | 0.6600 [0.6128, 0.7106] | 0.6594 [0.6112, 0.7084] | 0.6264 [0.5863, 0.6678] |

The smaller earlier pilot reached AUC 0.7928 over five fixed scenario pairs
(116/120 matched pairs), so it is retained as development context rather than
as a confirmation result.

## Claim boundary

The defensible claim is that a probe trained on explicitly stated purpose can
separate indirectly worded evaluative scenarios from matched live-use scenarios
at held-family AUC 0.598-0.660 across three checkpoints. This measures wording
generalization for a described situation. It does not establish subjective
awareness, strategic adaptation, behavioral consequences, or causal direction.

## Artifacts and reproduction

- `results/implicit_v2_frozen_r3/` — frozen scenario bank, manifest, and hash.
- `results/implicit_v2_8b_analysis.json` — 8B frozen-probe analysis.
- `results/implicit_v2_llama31_70b_l18_analysis.json` — Llama-3.1-70B analysis.
- `results/implicit_v2_llama33_70b_l18_analysis.json` — Llama-3.3-70B analysis.
- `results/implicit_ea_core_summary.csv` and `results/implicit_ea_core_figure.png` — summary outputs.
- `results/implicit_v2_nocue_control_r{1,4,16,64}.json` — no-cue subspace controls.
- `results/implicit_purpose_fixed_probe/` — earlier fixed-probe pilot package.
- `results/implicit_v2_*_explicit_probe_bank_*.manifest.json` — probe-bank contracts; activation tensors are not duplicated in GitHub.
- `scripts/implicit_v2_build.py`, `scripts/implicit_v2_analyze.py`, and the
  `scripts/implicit_v2_*freeze*`, `scripts/implicit_v2_*nocue*`, and
  `scripts/cluster_*implicit_v2*` runners — generation, freezing, analysis,
  and cluster reproduction code.

The larger implicit-v2 activation files are hosted in the private
[`format-EA/implicit-v2`](https://huggingface.co/datasets/format-EA/implicit-v2)
dataset. Use the recorded manifests and model contracts; do not substitute a
different revision or row order.
