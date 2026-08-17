# Development rejected bank: character leakage diagnosis

Status: `DEVELOPMENT_REJECTED_LEXICAL_LEAKAGE`. This artifact is development-only. It must never be transformed into the confirmation bank or passed to Llama extraction.

## Input and provenance

- Rendered bank SHA256: `ecdc9f715f86bab0a58ad4e472e1bfe43731ca2b355bbed2fc9babc4f01da096`
- Rows: 1,760; payload blocks: 80; stated rows: 960
- Builder: `scripts/build_confirmation_bank.py`, SHA256 `dd007600edf5a24b4ad0f821fb1bf77bf1813cdaee3c537efa522fa361c0b815`
- Builder Git HEAD: `98afd2597daeb0754b8e59b2622f3177ce7156a6`
- Generator: procedural reconstruction, no model/provider, deterministic payload indices 0..79; no random seed
- Diagnostic script: `scripts/diagnose_char_leakage.py`, SHA256 `68e1d1b50575f03a50232804875690c3e87db8db5dd75d3396b91334f0357368`
- Diagnostic feature CSV: `development_char_features.csv`; SHA256 `428bdb25affe0377af2ac23f900f11b437fe40a55191fd7200803fdb7b9ae080`

## Gate result

Predeclared thresholds were mean held-fold AUC <= 0.60 and maximum fold AUC <= 0.70.

| Feature | Mean AUC | Max fold AUC | Status |
|---|---:|---:|---|
| word unigram | 0.5000 | 0.5000 | pass |
| word 1-2gram | 0.5000 | 0.5000 | pass |
| word 1-3gram | 0.5000 | 0.5000 | pass |
| char 3-5gram | 0.6402 | 0.9998 | **fail** |
| char 3-6gram | 0.6261 | 1.0000 | **fail** |
| length/punctuation | 0.5000 | 0.5000 | pass |
| first half | 0.5004 | 0.5018 | pass |
| second half | 0.5000 | 0.5000 | pass |

The original lexical report is preserved at `lexical_gate.json`.

## Root cause

The leakage is in the purpose cue span, not payload identity or row geometry.

Held-fold character AUC, mean/max across folds:

| Scope | Char 3-5 | Char 3-6 |
|---|---:|---:|
| full text | 0.8859 / 0.9980 | 0.8781 / 0.9962 |
| cue sentence only | 0.8667 / 1.0000 | 0.8667 / 1.0000 |
| cue span only | 0.8667 / 1.0000 | 0.8667 / 1.0000 |
| scaffold with cue removed | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| first sentence | 0.8667 / 1.0000 | 0.8667 / 1.0000 |
| second sentence | 0.5000 / 0.5000 | 0.5000 / 0.5000 |

Within the stated stratum:

| Stated locus | Char 3-5 mean/max | Char 3-6 mean/max |
|---|---:|---:|
| noun-only | 0.8109 / 1.0000 | 0.8150 / 1.0000 |
| verb-only | 0.9412 / 1.0000 | 0.8695 / 1.0000 |
| both | 1.0000 / 1.0000 | 1.0000 / 1.0000 |

The dominant mechanism is **morphological/subword generalization across cue words**. The old stem-level rule prevented exact stem reuse but did not prevent reusable character fragments or tokenizer-like pieces. Examples in the top coefficient audit include:

- evaluation-side `-ify` fragments transferring from `certify` to held-out `verify`;
- deployment-side `-ate` fragments transferring from `operate` to held-out `update`;
- `cod`/`ode` transferring through `coder`;
- `ser`/`serve` transferring through `server`, `runner`, and `serve`.

The scaffold-only control is exactly chance. Length/punctuation and half-text controls are also chance. This rules out payload identity, row count, simple length, and the dead context field as the primary leakage mechanism.

A secondary implementation issue was found: the frozen design's lexical control operates on cue sentences, while `scripts/lexical_gates.py` originally fit on full rendered text. The new diagnostic reports both scopes. The cue-only result still fails catastrophically, so correcting the scope does not rescue this bank; it makes the actual cue failure clearer.

## Additional rejection defects

These are independent reasons the candidate was unusable even apart from lexical leakage:

1. Every difference payload block has a duplicate correct option: 16/80 blocks and 176/1,760 rows.
2. `render_bank` constructs a context-containing `body` but never uses it, so payload context is absent from rendered prompts.
3. Fold assignment uses `rank = index % 5`; the payload ID is ignored despite the intended hash-stable assignment rule.
4. The former cross-fold validator checked raw declared cue stems, not normalized stems, actual text differences, character 3-6 ownership, or tokenizer pieces.
5. Semantic validation and rejection logs are empty; no provider-backed semantic validator ran.

## Required construction change

The confirmation generator must use purpose-symmetric vocabulary and syntax. Purpose must be expressed by relational/temporal role, not by role-noun morphology. The next development protocol must:

- construct paired sentences from a shared lexical inventory;
- balance morphology, affixes, clause order, voice, tense, punctuation, and sentence count across labels;
- compute cross-fold ownership for normalized stems, character 3-6grams, and target-tokenizer pieces;
- permit common features only when their eval/deployment counts are symmetric;
- run cue-only and full-text gates separately;
- reject the whole candidate if any construction gate exceeds the frozen thresholds.

No activation extraction was performed on this bank.
