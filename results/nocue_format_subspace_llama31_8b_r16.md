# No-cue-derived format-associated subspace result

Model: `llama31_8b`; rank 16; 320 complete payload blocks; 1000 paired block-bootstrap replicates.

- Raw integrated gap: **0.0793** (95% CI 0.0768, 0.0816).
- Projected integrated gap: **0.0626** (95% CI 0.0606, 0.0644).
- Fraction removed: **21.1%** (95% CI 20.0%, 22.3%).
- Integrated change in joint / B->C / C->B AUC: **+0.0007 / +0.0101 / +0.0249**.
- Integrated projected format AUC: **0.9913**.

By the prespecified, conservative rule encoded in this script, this result **strengthens** the current format-subspace claim. This classification is about whether a format-associated subspace estimated without purpose cues explains explicitly stated-purpose probe transfer failure; it is not evidence of spontaneous evaluation awareness.

Construction and limits: no-cue rows have no evaluation/deployment labels. The basis is an uncentered SVD of payload-matched benchmark-minus-casual no-cue centroid contrasts, cross-fitted by the existing held-purpose-family folds and projected only before exact probe retraining. Frozen-activation projection is not a behavioral causal intervention. The no-cue-derived subspace is not claimed to be “pure format,” and overlap with the stated-purpose-derived basis is not geometric identity.
