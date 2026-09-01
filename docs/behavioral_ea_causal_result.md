# Causal behavioral evaluation-awareness experiments: corrected record and retractions

Status: corrected 2026-08-31 after an independent adversarial audit (`agent://BrutalEAAuditor`,
transcript `history://BrutalEAAuditor`) and a correct re-analysis of the original data.
This document supersedes the earlier version, which overstated these results.

## What was retracted and why

1. **Retracted:** "Implicit evaluation-like histories causally shift Llama forced-choice
   behavior toward evaluator-favored responses, positive in all three checkpoints."
   The original intervals came from a bootstrap that resampled the 128 crossed
   history x decision cells independently and identically
   (`scripts/analyze_behavioral_ea.py`, `scripts/analyze_behavioral_ea_validation.py`).
   That ignores dependence along both crossed axes and understates generalization
   uncertainty. Correct two-way crossed re-inference is in
   `results/behavioral_ea_crossed_reinference.json`
   (`scripts/recompute_behavioral_ea_crossed_inference.py`, 20,000 replicates, seed 20260905):

   | experiment | checkpoint | coherent log-odds | crossed 95% CI | excludes 0 |
   |---|---|---:|---:|:--:|
   | initial | Llama-3.1-8B | +0.691 | [+0.038, +1.351] | yes |
   | initial | Llama-3.1-70B int8 | +0.357 | [-0.175, +0.940] | no |
   | initial | Llama-3.3-70B int8 | +1.310 | [+0.315, +2.251] | yes |
   | validation | Llama-3.1-8B | +0.449 | [-0.123, +0.994] | no |
   | validation | Llama-3.1-70B int8 | +0.422 | [-0.081, +0.903] | no |
   | validation | Llama-3.3-70B int8 | +0.701 | [-0.610, +1.917] | no |

   On the independent validation bank, **0 of 3** checkpoints have a crossed interval
   excluding zero. The effect does not generalize across history families.

2. **Retracted:** the claim that a focused closed-comparison bank supported the same
   direction. It does not, and the artifact now exists to prove it. Llama-3.1-8B over
   32 domains x 16 decisions gives coherent log-odds **-0.520**, crossed 95% CI
   **[-0.816, -0.221]**, with only **1 of 32** domains positive
   (`results/behavioral_ea_closed_comparison_analysis.json`). The direction reversed.
   The remaining two checkpoints were cancelled once the all-model gate was unreachable,
   so that experiment is reported as a completed negative for 8B and unrun for 70B.

3. **Retracted:** treating the recognition experiment as behavioral evaluation awareness.
   The auditor showed the target is stated in paraphrase: the histories say
   "closed workspace", "concealed criteria", "resulting ordering", "no alternative
   reaches", versus "goes directly", "affects the action", and the answer options
   restate those same facts (`results/behavioral_ea_recognition_frozen/items.jsonl`).
   Withholding nine label words does not make the status latent. That experiment
   measures reading comprehension of an explicitly described situation, and its frozen
   three-model gate failed anyway (`results/behavioral_ea_recognition_analysis.json`,
   `recognition_gate_pass: false`; Llama-3.1-70B crossed CI low 0.703 < 0.750).

## What survives

Only this, and it is not evaluation awareness:

> On these frozen, heavily templated prompts, replacing a closed-comparison-associated
> preceding history with an active-use-associated one changes the next-token A/B log
> odds of author-labelled "evaluator-favored" options. Under correct crossed inference
> the change generalizes across history families in at most 2 of 3 checkpoints in the
> development bank and in 0 of 3 in the independent validation bank.

Additional unresolved threats recorded by the audit: forced two-token A/B softmax is a
response propensity and not a downstream consequence; 70B runs used bitsandbytes int8
while 8B used BF16, so checkpoints are not cleanly comparable; the four experiments were
serial same-day analyses with outcome-informed redesigns; "evaluator-favored" labels were
author-defined rather than externally validated.

## Where the real test lives

The successor experiment, designed against the auditor's requirements, hides the status
instead of describing it and scores an executed environment outcome rather than a token
preference: `results/behavioral_ea_hidden_channel_protocol_DRAFT.json` and
`docs/behavioral_ea_hidden_channel_result.md`.

## Artifacts

- `results/behavioral_ea_crossed_reinference.json` (corrected inference)
- `results/behavioral_ea_closed_comparison_analysis.json` (focused reversal, 8B)
- `results/behavioral_ea_analysis.json`, `results/behavioral_ea_validation_analysis.json`
  (original analyses, superseded inference)
- `results/behavioral_ea_recognition_analysis.json` (failed recognition gate)
- `scripts/recompute_behavioral_ea_crossed_inference.py`
