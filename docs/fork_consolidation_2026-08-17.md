# Fork consolidation — 2026-08-17

This document records the work developed in `catears124/format-as-eval-awareness` before consolidating back into the canonical repository `conrad204/format-as-eval-awareness`.

The goal is to preserve scientific provenance and make clear which results are exploratory development evidence, which protocol elements are prospectively frozen, which candidate banks were rejected, and which confirmation / mechanistic runs remain pending.

## Repository consolidation

- Parent / canonical repository: `conrad204/format-as-eval-awareness`
- Working fork: `catears124/format-as-eval-awareness`
- Parent head before consolidation: `12462c0f62526c7c121b77147b8b127ab0622f0c`
- Fork was synchronized with that upstream head before this PR, preserving fork-side development commits.
- Temporary one-shot upstream-sync workflow was removed after synchronization.
- Going forward, development should happen in the canonical `conrad204` repository rather than maintaining two divergent repos.

## Pilot activation archive

Private Hugging Face dataset used for spent-pilot development analysis:

- Dataset: `format-EA/format_EA_activations_pilot_run`
- Dataset commit: `dff7791d39b2851f53fb517c816732979d2f6cc8`
- Three models:
  - Llama-3.1-8B-Instruct
  - Llama-3.1-70B-Instruct
  - Llama-3.3-70B-Instruct
- 1,760 rendered items per model
- Stored all-layer residual-stream activations
- Semantics: pre-final-norm residual stream, last prompt token after chat template
- Inference: BF16; stored arrays: float32
- Metadata includes rendered prompt text, labels, vocabulary folds, purpose metadata, item IDs, payload/topic IDs, and hashes.

Import provenance is recorded in `provenance/pilot_activation_dataset_import.json`.

## Frozen development-selected layers

The pilot-selected layers were frozen before confirmation extraction:

- Llama-3.1-8B-Instruct -> layer 12
- Llama-3.1-70B-Instruct -> layer 31
- Llama-3.3-70B-Instruct -> layer 17

Selection rule: argmax decorrelated / held-vocabulary stated-purpose AUC on the spent pilot.

These are development-selected layers for future confirmation; they must not be changed after seeing confirmation outcomes.

## Reproduced pilot results

The fork independently reproduced / recorded selected-layer development results from the uploaded activations.

### Llama-3.1-8B, layer 12

- held-vocabulary stated-purpose AUC: ~0.99175
- historical word n-gram baseline: 0.500
- indirect-purpose transfer E4: ~0.866
- second-person transfer E5: ~0.876
- strict vocabulary x payload double holdout E8: ~0.948

Exact reproduction output:

- `results/pilot80_analysis_8b_reproduced_local_selected.json`
- `results/pilot80_layer_sweep_8b_reproduced.json`

### Llama-3.1-70B, layer 31

- held stated-purpose AUC: `0.9981553819444444`
- balanced accuracy: `0.978125`
- `||A|| = 0.721882543396417`
- `||B|| = 4.391039012059936`
- `||C|| = 0.46139412693308207`
- interaction ratio `||C||/||A|| = 0.6391540163338049`
- cosine of benchmark/casual purpose displacements: `0.5277560894019556`
- cosine(A,C): `-0.11584206702327964`
- principal cosine of purpose displacement subspaces: `0.7820142503173273`
- purpose direction projection on B: `0.0009702918611354051`

Artifact:

- `results/pilot80_geometry_31_70b_selected.json`

### Llama-3.3-70B, layer 17

- held stated-purpose AUC: `0.9954036458333333`
- balanced accuracy: `0.9614583333333333`
- `||A|| = 0.38729768277667703`
- `||B|| = 1.8415631620803812`
- `||C|| = 0.1732164738436077`
- interaction ratio `||C||/||A|| = 0.4472437650588462`
- cosine of benchmark/casual purpose displacements: `0.7429994490756138`
- cosine(A,C): `-0.4624840310998772`
- principal cosine of purpose displacement subspaces: `0.9232066321715168`
- purpose direction projection on B: `-0.0010740772901529452`

Artifact:

- `results/pilot80_geometry_33_70b_selected.json`

These are exploratory / development results, not confirmation evidence and not causal evidence.

## Pilot interpretation corrections

The fork work intentionally tightened several claims from the original pilot.

### Surface-form controls

The word-level held-fold TF-IDF baseline at ~0.50 does not prove the absence of all lexical / surface shortcuts.

A character n-gram audit found materially stronger surface decodability on the pilot. Therefore the permanent interpretation is narrower:

- word-identity shortcuts are substantially controlled by vocabulary-disjoint folds;
- the pilot still contains scaffold / character / subword regularities;
- neural AUC remains much higher than these shallow controls, but the pilot should not be described as surface-shortcut-free.

Relevant artifacts include:

- `results/pilot_surface_gates_v2.json`
- `results/pilot_claim_status.json`

### Format effect

The old no-cue E6 effect should not be interpreted as a universal `benchmark format -> evaluation` direction.

Across models, the sign is not stable. The defensible development conclusion is that no stable cross-model format direction was established by the original single benchmark/casual template pair.

### Second-person transfer

Pooled second-person AUC can be depressed by score calibration offsets between loci / sub-strata. Paired ordering, per-locus calibration, and direction / subspace alignment are cleaner tests of representational transfer.

## Confirmation protocol hardening

The fork added substantial confirmation infrastructure so the future confirmation run can be sealed and one-shot rather than an informal rerun of the pilot.

Added / hardened components include:

- `scripts/build_confirmation_bank.py`
- `scripts/generate_confirmation_candidates.py`
- `scripts/validate_confirmation_candidates.py`
- `scripts/assemble_confirmation_bank.py`
- `scripts/build_forbidden_lexicon.py`
- `scripts/lexical_gates.py`
- `scripts/lexical_gates_v2.py`
- `scripts/subword_gates.py`
- `scripts/structural_variation_gate.py`
- `scripts/surface_balance.py`
- `scripts/red_team_lexical_audit.py`
- `scripts/seal_confirmation_bank.py`
- `scripts/prepare_confirmatory_manifest.py`
- `scripts/extract_confirmatory.py`
- `scripts/run_sealed_confirmation.py`
- `scripts/report_confirmation_status.py`
- `scripts/confirmation_analysis.canonical.py`

Cluster / Slurm wrappers were also added for candidate generation, validation, assembly, sealing, activation extraction, and analysis.

## Sealed confirmation wrapper

`run_sealed_confirmation.py` was added to enforce exact confirmatory inputs and prevent accidental repeated analyses.

The wrapper is designed to fail closed on mismatches such as:

- wrong bank hash
- wrong analysis hash
- wrong model ID
- wrong model revision
- wrong frozen layer
- row-count / item-ID mismatch
- existing output
- already-used confirmation sentinel

Tests:

- `tests/test_run_sealed_confirmation.py`
- latest recorded run: 6 passed

Canonical confirmatory analysis SHA-256 recorded in the protocol:

`561560a697bb466e710d969c2bbae11ac050dfe92bcb90ee29b58090a73dd804`

## Confirmation design amendments made before confirmation outcomes

No confirmation activation extraction or confirmatory analysis had occurred when these changes were made.

The protocol was hardened prospectively to require stronger controls, including:

- character n-gram gates
- tokenizer / subword gates
- cross-fold asymmetric cue ownership checks
- stronger structural variation
- cue-position variation
- multiple benchmark/casual formatting variants rather than a single template pair
- final lexical red-team audit before activation extraction
- more explicit immutable bank / manifest hashing

The freeze record is in `docs/analysis_plan_v2.freeze.json`.

Future exploratory development results should not be used as a general lab notebook in the freeze file unless they alter a prospectively frozen test.

## Rejected development confirmation-bank attempts

Two candidate-generation / diagnostic directories are intentionally preserved as development artifacts. They must never be promoted to confirmation data.

### Rejected lexical-leakage candidate

Artifact hash recorded for the rejected rendered bank:

`ecdc9f715f86bab0a58ad4e472e1bfe43731ca2b355bbed2fc9babc4f01da096`

The candidate passed several word-level controls but failed character held-fold gates badly:

- char 3-5gram mean AUC: ~0.88585
- char 3-5gram max fold AUC: ~0.99805
- char 3-6gram mean AUC: ~0.87815
- char 3-6gram max fold AUC: ~0.99620

This bank was rejected before Llama activation extraction.

Preserved under:

- `artifacts/development_rejected_lexical_leakage/`

Diagnostics include:

- `development_char_features.csv`
- `development_char_leakage_report.json`
- `development_char_leakage_report.md`
- lexical / subword gate outputs

### Balanced development candidate

A later structurally balanced candidate and its gate outputs are preserved under:

- `artifacts/development_balanced_v2_candidate/`

These remain development-only artifacts and are not sealed confirmation data.

## Forbidden development lexicon

A conservative development forbidden-lexicon artifact was constructed so confirmation generation can avoid reusing purpose-asymmetric development lexical material:

- `artifacts/dev_forbidden_lexicon.json`

## Confirmation candidate generation / validation plan

The fork prepared a fresh confirmation-generation path using different model lineages from the development data.

Current planned candidate generator:

- Stockmark-2-100B-Instruct
- exact generation revision recorded in scripts / protocol: `98e959a472d0fcaf0c8b6576af19fe3cba7ca4ab`

Current planned independent semantic validator:

- Swallow-70B
- exact revision recorded in the confirmation pipeline / provenance

Target scheduling developed in the fork:

- 80 independent payload blocks
- 16 topics x 5 blocks
- balanced purpose / format / cue strata

At the time of consolidation, the generator job remained pending for GPU resources, so no fresh confirmation bank had been sealed and no confirmation-grade result existed.

## Confirmation scientific status at consolidation

No confirmation-grade activation extraction or one-shot confirmation result has been produced yet.

Therefore current scientific claims must remain:

- pilot representational evidence is strong but exploratory;
- no confirmation-grade positive result yet;
- no confirmation-grade causal / mechanistic result yet.

## J-lens preparation

The official Anthropic Jacobian-lens implementation was pinned for future mechanistic work:

- upstream: `anthropics/jacobian-lens`
- pinned commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e`

Fork provenance:

- `artifacts/jlens_provenance.json`

The intended lens must be fit on an independent generic pretraining-like corpus, not on pilot or confirmation prompts.

The pilot activations alone are not a substitute for fitting the Jacobian lens.

## Planned next development analyses from spent pilot activations

The fork work identified the following high-priority development analyses, none of which should contaminate the sealed confirmation bank:

1. Purpose decoding after cross-fitted removal of format nuisance subspaces.
2. Reverse control: format decoding after removal of purpose subspaces.
3. Stated-purpose -> indirect-purpose transfer before and after format removal.
4. Stated / indirect / second-person direction and low-rank subspace alignment.
5. Cross-locus generalization (noun-only / verb-only / both).
6. Shared cross-format purpose subspace.
7. Consensus low-rank purpose subspace / effective rank.
8. Strict vocabulary x topic/payload double-holdout geometry.
9. More robust uncertainty analyses with topic / frame resampling and probe refitting.
10. Re-analysis of the no-cue format trajectory across depth with paired estimands.
11. Cross-model geometry comparisons between Llama-3.1-8B, Llama-3.1-70B, and Llama-3.3-70B.

These are development analyses only and should be logged separately from prospective confirmation freeze amendments.

## Causal-development direction

The old single-vector causal result from the earlier project was not robust to structural replication. The next causal experiment should therefore not simply reuse that old single-layer interaction vector.

The current prospective direction is to select a causal object on spent development data using held-out stated -> indirect generalization and format-erasure criteria, then test a multi-layer intervention on a separate causal evaluation bank.

Candidate objects include:

- rank-1 consensus purpose direction
- format-residualized rank-1 direction
- rank-4 / rank-8 consensus purpose subspaces
- shared cross-format purpose subspace
- later J-space-projected purpose component

No causal outcome should be used to select the object.

## Security / credential note

A prior broad local search surfaced provider credentials to an agent. Those credentials were treated as compromised and were not reused or committed.

The fork added local environment files to ignore rules. No secrets should be included in this PR.

## Consolidation decision

After this PR is merged, `conrad204/format-as-eval-awareness` should be treated as the single canonical repository.

The `catears124` fork can remain as historical provenance / backup, but active development should continue from the parent repository to avoid future divergence.
