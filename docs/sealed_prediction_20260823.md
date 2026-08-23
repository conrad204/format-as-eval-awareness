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
