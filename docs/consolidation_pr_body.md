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
- An independent FineWeb J-lens corpus is frozen with dataset revision, row IDs, selection seed, hashes, and the pinned official Jacobian-lens implementation.
- The real 8B two-prompt J-lens smoke completed on H100 job `52308432`: finite nonzero FP32 `[4096,4096]` L12 Jacobian, exact reload, and held-out `apply` all passed.
- The independent 100-prompt development lens completed successfully as job `52310538` (`n_done=100`). Its offline pilot assay over 1,760 pilot rows completed as job `52350692`: L12 purpose-projection AUCs are stated `0.505`, indirect `0.490`, and second-person `0.487`; no-cue format projection is `0.768` at L12, `0.860` at L15, and `0.890` at L24. These are development-only J-space diagnostics, not confirmation or awareness evidence. Full-fit job `52347764` remains running.
- Fresh confirmation generation job `52315277` produced 72/80 payloads and 50/50 frame candidates; semantic validation passed all generated candidates, but the bank is not accepted because payload batch 7 returned the wrong count.
- Confirmation retry `52348268` loaded the exact Stockmark snapshot but returned the wrong candidate count with no accepted candidates. Larger pre-outcome retry `52351168` uses the same prompt/seed/temperature with `max_tokens=16384`; downstream combined-bank, semantic-validation, and lexical-gate jobs are dependency-linked. No confirmation activations or outcomes exist.
- This PR therefore reports development evidence and infrastructure state only. It makes no confirmation-grade or causal claim.
- Repository regression suite passes with the writable project-local temporary directory: 68 tests passed.
- Corrected v2 format-erasure artifacts now exist for all three pilot models, with the same passing synthetic control. Matched-mean purpose / stated→indirect / fresh-format AUCs are 8B `0.9923 / 0.8498 / 0.5000`, 3.1-70B `0.9991 / 0.9243 / 0.5000`, and 3.3-70B `0.9950 / 0.8098 / 0.5000`; all remain development diagnostics.
