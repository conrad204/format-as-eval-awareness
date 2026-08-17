# Consolidation PR summary

This text is the intended PR narrative for merging the fork work back into `conrad204/format-as-eval-awareness`.

## Purpose

Consolidate all work developed in `catears124/format-as-eval-awareness` back into the canonical repository so active development can continue in one place.

At preparation time, the fork was 16 commits ahead and 0 behind upstream base `12462c0f62526c7c121b77147b8b127ab0622f0c`.

## Included work

- frozen pilot-selected layer / revision provenance
- private Hugging Face pilot activation import provenance
- exact 8B pilot endpoint reproduction and selected-layer 70B geometry
- pilot claim-status / surface-form audits
- confirmation-bank generation, validation, assembly, lexical/subword/structural gates, sealing, and immutable one-shot analysis wrappers
- Slurm scripts for generation, validation, sealing, extraction, and analysis
- rejected development-bank artifacts and leakage diagnostics, explicitly preserved as development-only evidence
- conservative forbidden development lexicon
- official Anthropic J-lens pin / provenance
- sealed-confirmation tests
- full consolidation audit document at `docs/fork_consolidation_2026-08-17.md`

## Scientific status

This merge does not promote exploratory pilot evidence to confirmation evidence.

Current status:

- pilot representational results are strong but exploratory;
- rejected candidate banks remain permanently development-only;
- no fresh confirmation bank has been sealed yet;
- no confirmation-grade activation extraction or one-shot confirmatory result exists yet;
- no confirmation-grade causal / mechanistic result exists yet.

The confirmation protocol was hardened prospectively before any confirmation outcomes to include stronger character/subword controls, structural variation, multiple format templates, cue-position balancing, immutable manifests, and fail-closed one-shot execution.

## Reproduced development results

Frozen selected layers:

- Llama-3.1-8B -> L12
- Llama-3.1-70B -> L31
- Llama-3.3-70B -> L17

Selected-layer stated-purpose AUCs recorded in the fork:

- 8B L12: ~0.99175
- 3.1-70B L31: 0.998155
- 3.3-70B L17: 0.995404

Strict double-holdout / transfer values remain exploratory; see committed result JSONs and `docs/fork_consolidation_2026-08-17.md` for exact provenance and interpretation caveats.

## Pilot interpretation corrections

- word-level held-fold TF-IDF ~0.50 is not sufficient to claim all surface shortcuts are eliminated;
- character/subword controls reveal residual scaffold/surface structure;
- old E6 should not be interpreted as a universal benchmark -> evaluation direction;
- pooled second-person AUC partly reflects calibration offsets, so paired / geometric analyses are preferred for representation-alignment claims.

## Rejected confirmation candidates

Rejected / spent candidate banks are included only to preserve provenance and failure analysis.

In particular, the development candidate with rendered-bank SHA:

`ecdc9f715f86bab0a58ad4e472e1bfe43731ca2b355bbed2fc9babc4f01da096`

failed character held-fold gates badly and was never used for Llama confirmation extraction.

## Security

No secrets are intentionally included. Prior exposed provider credentials were treated as compromised and not reused or committed. Local environment files are ignored.

## After merge

Treat `conrad204/format-as-eval-awareness` as the single canonical repository. The `catears124` repo can remain as historical provenance / backup, but active development should continue in the parent repository to avoid future divergence.

## Post-audit development additions

- The selected-layer 8B reproduction was corrected to use `activations_all_layers.npz`; the legacy `activations.npz` archive has materially different L12 values and is not used for evidence.
- Development-only cross-fitted format-erasure diagnostics now cover all three models. The 70B result JSONs retain their pre-correction synthetic-check field; the corrected harness passes independently (`raw purpose AUC=1.0`, removed-format AUC=0.5283).
- An independent FineWeb J-lens corpus is frozen with dataset revision, row IDs, selection seed, hashes, and a pinned official Jacobian-lens fit driver. Fitting is not yet complete.
- Confirmation generation job 52049785 failed before candidate generation on offline cache resolution. Infrastructure-only retry job 52276496 uses unchanged scientific parameters and remains pending GPU resources; no confirmation bank or confirmation activations exist.
- This PR therefore reports development evidence and infrastructure state only. It makes no confirmation-grade or causal claim.
- The completed 3.1-70B all-layer development sweep has 80 layers and decorrelated argmax L31 AUC 0.998160; this remains pilot/development evidence.
