# Sealed pre-registered prediction — determinism-patched v3 causal run

**Sealed before any inspection of causal readout scores or intervention outcomes.**
8B causal raw data is complete (`pilot80_factorial_causal_8b_v3_detpatch_batch1`,
`records=163200/163200`, gate never violated). 70B is still running. Nobody
on this task has run `scripts/analyze_pilot80_factorial_causal.py` or looked
at any `readout_scores` field from either model's raw file. This document is
committed to git *before* that happens, so its timestamp/commit hash is an
independent, verifiable seal.

## Basis for these predictions

Entirely from offline, activation-geometry-only analyses that never touched
a causal readout score:
1. Isolated single-component interventions (`A_p_alpha1`, `C_p_alpha1`, etc.)
   showed a clean, tight, correctly sign-flipping effect in an informal peek
   at partial 8B data (not used here as evidence, only as prior context).
2. `cos(A, C) ≈ -0.56` at 8B's high-interaction layer (29), `≈ -0.32` at 70B's
   (74) — a real, cross-model-replicated, non-orthogonality between the
   purpose main effect and the interaction direction.
3. A generic-text-fitted Jacobian (zero purpose/format content) predicts the
   real purpose (A) and format (B) readout directions far above a
   random-direction null (z up to 39 at 8B layer 24), and predicts the
   interaction (C) direction consistently the *weakest* of the three at
   every depth where the effect is significant (bootstrap P=1.0 at layers
   15, 20, 24).
4. Raw direction norms: B >> A ≈ C in both models, at both selected layers.

## Predictions

1. **A_shared: PASS** in both models, both layers. High confidence.
   Strongest, most generic, most linearly-explainable signal of the three.
2. **B_format: PASS** in both models, both layers. High confidence. Even
   stronger generic-channel alignment than A.
3. **C_interaction (isolated single-component criterion, the one the frozen
   script actually gates on): PASS**, but with **visibly wider/noisier CIs
   than A or B** in the final bootstrap output. Medium confidence.
4. **factorial_cube_agreement (additive R² > 0.50 for the 8-corner cube):
   FAIL or borderline (close to the 0.50 threshold)**, specifically *because*
   of the A/C non-orthogonality — cube corners inject A, B, and C
   simultaneously, and A's dominant effect will not simply "add" cleanly
   with C's effect if the two are >30% entangled. This is the prediction
   with the most novel content: it says the failure mode (if any) will be
   structural/geometric, not a sign of C being spurious.
5. **dose_response (alpha 0.5 vs 1.0 for C): PASS**, same sign, |half| ≤
   |full|. Lower confidence — closer to a coin flip.
6. **Overall decision: "partial"**, not clean "positive" — driven
   specifically by criterion 4, not by A_shared, B_format, or C_interaction
   failing.
7. **Cross-model agreement will be closer for A_shared and B_format than for
   C_interaction** — following directly from the cross-model geometry check
   showing weaker/less consistent depth-dependence of the A/C entanglement
   in 70B than 8B.

## Falsification

If the actual run shows a clean "positive" (all criteria including
`factorial_cube_agreement` pass), predictions 4 and 6 are wrong and the
A/C entanglement matters less for cube additivity than this reasoning
suggests. If C_interaction itself fails, prediction 3 is wrong and the
isolated-component cleanliness seen in the partial peek did not generalize
to the full, complete-fold, both-model result. Either outcome is reported
here as-is once the real run completes; this file is not edited afterward
except to append a dated "resolution" section with the actual result.

## Resolution — 2026-08-23, both models complete, frozen analysis run once

Actual frozen output (`scripts/analyze_pilot80_factorial_causal.py`, run once,
committed provenance in `pilot80_factorial_causal_v3_analysis/run_manifest.json`
and `causal_results.json`):

```
decision: "negative"
A_shared: false | B_format: false | C_interaction: true | dose_response: true
endpoint_valid: false | factorial_cube_agreement: true | geometry_to_causality: true
hook_and_reproduction: true
```

**Scored against the sealed predictions:**
1. A_shared PASS (high confidence) — **WRONG**. Effect is large and clean at
   both high-interaction layers (8B layer 29 point=1.159, p=0.03; 70B layer 74
   point=1.026, p=0.03), but at the low-interaction control layers it is tiny
   and narrowly misses the random-control p<=0.05 bar (8B layer 5: p=0.091).
   The criterion requires both layers in both models; one narrow miss fails
   the whole conjunction.
2. B_format PASS (high confidence) — **WRONG**, same mechanism: strong at both
   high-interaction layers (8B layer 29 p=0.03; 70B layer 74 p=0.03), but at
   70B's low-interaction layer the point estimate is the wrong sign
   (-0.00055) and at 8B's it has p=0.212.
3. C_interaction PASS (medium confidence) — **CORRECT**. This criterion only
   requires the high-interaction layer, which is exactly where both models
   showed a clean, tight, correctly-signed effect (8B: 0.469, p=0.03; 70B:
   0.783, p=0.03).
4. factorial_cube_agreement FAIL/borderline — **WRONG**, already caught and
   corrected via an informal pre-check before this run (additive R^2 =
   0.9986/0.9976, essentially perfect). The A/C representational overlap does
   not translate into non-additive causal effects; additivity of injected
   perturbations is governed by local linearity of the network's response,
   not by orthogonality of the injected directions. Documented misstep,
   left uncorrected in the predictions section above per the append-only
   policy.
5. dose_response PASS — **CORRECT**.
6. Overall decision "partial" — **WRONG, and wrong in the pessimistic
   direction**: actual is "negative", not even "partial", because "partial"
   itself requires A_shared to be true (which it is not) and additionally
   requires endpoint_valid (which is also false here, driven by one specific
   readout marginally missing its AUC>=0.80 bar in 8B).
7. Cross-model agreement closer for A/B than C — not clearly resolved either
   way; the specific low-interaction-layer failure differs in mechanism
   between models (8B: p-value miss; 70B: wrong sign for B), so this
   prediction is not cleanly confirmed or falsified.

**Net assessment of the prediction exercise itself:** the two predictions I
held with the *highest* confidence (A_shared, B_format) were the two that
turned out wrong, and for a specific, identifiable reason — informal
per-row descriptive statistics at the high-interaction layer looked
enormous and clean, but the frozen criteria also gate on the low-interaction
control layer, which is exactly where main effects are expected to be weak
by construction (that layer was selected for a *low* interaction-to-main-
effect ratio). The informal peeks earlier in this session never checked
that control-layer requirement, so they systematically overstated confidence
in A_shared/B_format. This is a concrete, reportable lesson: single-point,
naive-statistic peeks at a high-signal layer are not a substitute for the
full multi-layer conjunctive gate, even when the peeked numbers are
themselves accurate.
