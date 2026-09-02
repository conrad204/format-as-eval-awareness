# No-cue-derived format-associated subspace result

Model: `llama31_8b`; rank 1; 320 complete payload blocks; 1000 paired block-bootstrap replicates.

- Raw integrated gap: **0.0793** (95% CI 0.0768, 0.0816).
- Projected integrated gap: **0.0800** (95% CI 0.0774, 0.0824).
- Fraction removed: **-0.8%** (95% CI -1.8%, 0.3%).
- Integrated change in joint / B->C / C->B AUC: **+0.0001 / -0.0006 / -0.0005**.
- Integrated projected format AUC: **0.9934**.

By the prespecified, conservative rule encoded in this script, this result **weakens** the current format-subspace claim. This classification is about whether a format-associated subspace estimated without purpose cues explains explicitly stated-purpose probe transfer failure; it is not evidence of spontaneous evaluation awareness.

Construction and limits: no-cue rows have no evaluation/deployment labels. The basis is an uncentered SVD of payload-matched benchmark-minus-casual no-cue centroid contrasts, cross-fitted by the existing held-purpose-family folds and projected only before exact probe retraining. Frozen-activation projection is not a behavioral causal intervention. The no-cue-derived subspace is not claimed to be “pure format,” and overlap with the stated-purpose-derived basis is not geometric identity.
