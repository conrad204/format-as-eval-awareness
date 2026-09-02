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
- A deterministic 25-block/5-topic, 550-row offline stress assay (`52360914`) reproduced the pattern: L12 stated `0.5153`, indirect `0.4776`, second-person `0.4792`, no-cue format `0.7440`; maxima were stated L20 `0.5594`, second-person L24 `0.5984`, and no-cue format L15 `0.9104`. A fixed 99%-Frobenius-energy J-space decomposition found exact Jacobian rank 4096/4096 (untruncated orthogonal complement zero); truncated J/orthogonal purpose norm fractions moved from L12 `0.621/0.784` to L24 `0.917/0.398`, with corresponding held-fold AUCs L12 `0.805/0.916` and L24 `0.860/0.640`. These are development-only diagnostics.
- The expanded 8-family/16-topic/640-row no-cue battery (`52372658`) shows the format effect is not restricted to MCQ versus short-answer: aggregate benchmark-like minus casual J-space score means are L4 `-1.052`, L9 `-0.551`, L12 `+0.064`, L15 `+0.708`, L20 `+0.544`, and L24 `+1.066`, with substantial family heterogeneity.
- Development causal run `52372860` (20 no-cue rows; L4/L12/L15/L24; equal residual norm) found weaker low-gain margin shifts than full/top-J at L24 and poor first-order prediction at early layers. Gain-matched rerun `52373961` required much larger low-gain deltas and became strongly nonlinear. These are mechanism diagnostics, not causal confirmation.
- Full six-layer geometry controls completed locally with 99%-energy ranks L4/L9/L12/L15/L20/L24 `830/1225/1777/2787/3497/3614`, structured random-subspace controls, bottom singular-subspace controls, and Jacobian gain metrics. Mean 99%-energy top-subspace purpose AUCs were `0.657/0.897/0.805/0.938/0.862/0.860`; bottom-subspace AUCs were `0.751/0.926/0.935/0.943/0.902/0.897` (artifact SHA256 `f2e898d8b6b298b83e38e1aa12718151b8d07d51a496b75efab0d86321e178a7`). The exact Jacobians are full rank, so untruncated orthogonal-complement claims remain invalid.
- Fresh confirmation generation job `52315277` produced 72/80 payloads and 50/50 frame candidates; semantic validation passed all generated candidates, but the bank is not accepted because payload batch 7 returned the wrong count.
- Confirmation retry `52348268` loaded the exact Stockmark snapshot but returned the wrong candidate count with no accepted candidates. Larger pre-outcome retry `52351168` uses the same prompt/seed/temperature with `max_tokens=16384`; downstream combined-bank, semantic-validation, and lexical-gate jobs are dependency-linked. No confirmation activations or outcomes exist.
- This PR therefore reports development evidence and infrastructure state only. It makes no confirmation-grade or causal claim.
- Repository regression suite passes with the writable project-local temporary directory: 68 tests passed.
- Corrected v2 format-erasure artifacts now exist for all three pilot models, with the same passing synthetic control. Matched-mean purpose / stated→indirect / fresh-format AUCs are 8B `0.9923 / 0.8498 / 0.5000`, 3.1-70B `0.9991 / 0.9243 / 0.5000`, and 3.3-70B `0.9950 / 0.8098 / 0.5000`; all remain development diagnostics.
