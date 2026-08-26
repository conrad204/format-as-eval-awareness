# Steering study — results (causal validation of the full320-v3 probes)

Design frozen in `docs/steering_prereg.md`; provenance in
`provenance/full320-v3/steering_run.json`. NDIF, Llama-3.1-{8B,70B,405B}-Instruct,
each model's decorrelated-probe argmax layer (8B L8, 70B L18, 405B L37). All
directions added at one matched norm, Δ_fmt; no-cue items only (nothing about
purpose is stated). Raw data: `Conradf/v3-steering` (HF, tag v3). Per-model
analysis: `results/steering/analysis_*.json`. Plots: `plots/steering 8-26-2026/`.

## The objection this answers

The probes are correlational: a logistic-regression direction that *decodes*
purpose/format need not be one the model *uses*. We add or ablate the probe
directions in the residual stream at the probed layer and measure whether
behavior moves on no-cue items, versus a null band of five random directions
at the same norm.

## Directions and their fidelity

Refit from the published probes at the probed layer and un-standardized into
raw residual space (`scripts/steer_fit_directions.py`). **Probe parity** — the
NDIF deployment's live residual at the probed position, projected on each
direction, vs the checked-in activation projection — is **r = 1.0000 (format),
0.999 (decorrelated purpose), 1.0000 (confounded)** for every model: the model
being steered is byte-for-byte the model that was probed.

Geometry (constant across models): **cos(format, purpose-decorrelated) ≈ −0.01
— orthogonal**; cos(format, purpose-confounded) ≈ 0.87–0.94 — the confounded
purpose direction essentially *is* the format direction.

## Readout 1 — verbalized eval-awareness (logit)

A fixed one-letter "is this an evaluation or a genuine request" prefill (both
letter orders); statistic = signed eval log-odds, effect = condition minus the
same item's unsteered baseline, 320-block cluster bootstrap. "Causal strength"
|z| = |effect| in units of the five random directions' SD, averaged over ±2
dose (sign-agnostic: a used direction may steer either sign); ≥2 clears ~95% of
the random-norm null, ≥1 is suggestive. A direction "moves behavior" if it is
strong (|z|≥2) and sign-flips with dose.

| direction | 8B | 70B | 405B |
|---|---|---|---|
| baseline eval-logodds | −0.33 | +1.95 | +3.38 |
| **format** | 0.3 (null) | **2.2 (strong)** | 1.6 (suggestive) |
| purpose — decorrelated | 0.2 (null) | 0.6 (null) | 1.3 (suggestive) |
| purpose — confounded | 0.3 (null) | **2.0 (strong)** | 1.7 (suggestive) |
| purpose — diff-of-means | 0.4 (null) | 0.6 (null) | **3.1 (strong)** |

Reading (plots 01–03):
- **The format direction is causally used, and it scales.** Null in the 8B,
  strong in the 70B, suggestive in the 405B — the same 8B-null / 70B–405B-active
  pattern the no-cue observational sweep found (+0.6–0.8 sd format pull for
  70B/405B, none for 8B). Dose-monotone with a clean sign flip (plot 01).
- **Sign is model-specific and is *not* the finding.** In the 70B, +format
  *lowers* verbalized eval-awareness (its baseline oddly rates casual items as
  more "eval" than benchmark ones); in the 405B it raises it. Magnitude vs the
  random null is the causal claim, not the sign.
- **The format-orthogonal purpose direction is not causally potent here.**
  Decorrelated purpose is null in the 70B (0.6) and only suggestive in the 405B
  (1.3), while its format-aligned twin (confounded) is strong. The purpose
  signal that moves the verbalized readout is the format-aligned part — the
  probe-confound story, made causal.
- The verbalized readout is intrinsically noisy (a 2-way letter choice is
  perturbation-sensitive; the random band is wide), which is why Readout 2 —
  mechanistically specific to format — matters.

## Readout 2 — answer style (generation)

Greedy no-cue generations under uniform per-condition steering
(`model.edit`, all-position ActAdd; `scripts/ndif_steer_gen.py`), scored for
deterministic style features. Effect vs unsteered baseline, 320-block
bootstrap; random band = the same features under two random directions at the
top dose.

**8B** (base rates: 36 tok, 13% answer-first, 25% markdown): format steering
causally changes *formatting* — **markdown: fmt+8 = −0.072, fmt−8 = +0.050
(sign-flip), against a tight random band of (−0.013, +0.002)** — the model uses
the format direction to make output plainer (benchmark-like) or more formatted
(chat-like). Length and answer-first move too but are matched by random
directions (any large perturbation shortens greedy output), so they are not
format-specific. Coherence holds under format steering (repetition ≈ 0) while
random directions of the same norm degrade it.

**70B / 405B**: at the ±8·Δ_fmt dose needed to move greedy decoding, generic
perturbation dominates for the larger models — markdown and length drop under
*both* +8 and −8 (no sign flip) and track the random-direction span, so the
style readout does not isolate a format-specific effect there. The clean,
direction-specific style result is the 8B's, which is itself notable: **the 8B
uses the format direction to shape output formatting even though its *verbalized*
readout was null** — it acts on format without reporting it. For the larger
models the verbalized logit readout (Readout 1) carries the causal signal;
the style readout corroborates only at the small scale.

## Bottom line

Steering converts **"probes decode format" into "the model uses format"**: the
format direction causally moves both a verbalized eval-awareness readout
(70B strong, 405B suggestive, 8B null — scaling as predicted) and answer style
(formatting, above a tight random null), dose-dependently and above a
matched-norm random baseline, on items where nothing is stated. It does **not**
establish a *separable* causal purpose representation: the format-orthogonal
decorrelated-purpose direction is null-to-weak, while the format-aligned
purpose direction is strong — exactly the confound the probe study flagged,
now demonstrated causally. The 8B is causally inert on these readouts,
matching its flat observational signal.

## Threats and controls

Matched-norm dosing (every direction at Δ_fmt); a five-seed random null band;
sign-flip and dose-monotonicity requirements; a coherence guard (repetition);
probe parity r≈1.0; per-fold direction stability (min cosine 0.98 format,
0.82–0.86 decorrelated purpose). The verbalized question is itself a weak cue,
so only the steered-minus-unsteered delta is interpreted, never absolute rates.
