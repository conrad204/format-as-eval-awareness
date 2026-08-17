# v2 running log

Reverse-chronological log of the v2 redesign implementation. Since the
2026-08-17 relocation (entry below) this repo is fully self-contained: all
pipeline code, catalogs, schemas, tests, payload inputs, and frozen artifacts
live here. `~/Projects/personal/probes` is retired as a stale copy.

## 2026-08-17 — Full self-containment: pipeline code moved into this repo

Everything the chain depends on was moved from `~/Projects/personal/probes`
into this repo (user directive: the repo must run standalone on any machine).
Copied byte-identically (every file `cmp`-verified against its source):
`src/eval_format_mvp/` (16 modules new here; `extract.py`/`io.py`/`__init__.py`
were already byte-identical in both repos), `schemas/` (7), `configs/` (8),
`prompts/` (2), `tests/` (8), the payload inputs
`data/payloads.grok43-medium_gpt54.{full80,full.audited,full}.jsonl` +
`data/example_payloads.jsonl`, and `provenance/render_manifest_v2_pilot80.json`.
Smoke files, notebooks, run outputs, and the Colab one-offs were deliberately
left behind (bloat). `pyproject.toml` gained `jsonschema`, a `dev` extra
(pytest), the `eval-format-mvp` console script, and pytest config.

**Path policy change (portability):** the five render manifests' `path` fields
were rewritten from absolute (`/home/cfeng/...`) to repo-relative — content
`sha256` fields untouched, same policy as the catalog canonicalization.
Verification therefore runs from the repo root (elsewhere it fails loudly with
"missing file", never silently). Two test constants were repointed to
content-verified identical files: `test_housekeeping.FROZEN_ITEMS` →
`data/rendered_items_v1_8-14-2026.jsonl` (sha `ebac61db…` = probes'
`outputs/rendered_items.jsonl`) and `test_formats_v3.PILOT80_FROZEN` from a
`Path.home()` absolute path to the same file ROOT-relative. Historical run
records (`provenance/generation_runs/`, `refill_report`, pilot80 analysis
JSONs, extraction configs) keep their original recorded paths — they are
records of where things were at run time, not live references.

**Verification (all from this repo, post-move):**

- Full test suite: 62 passed (matches pre-move).
- All five manifests verify: v1, v2-pilot80, v3-dev, v3-full-a80, v3-full-b80
  — every input/output sha256 recomputed and matched.
- Determinism re-renders, byte-compared against the frozen artifacts:
  v3-full-a80, v3-full-b80, v3-dev, v2-pilot80 all **byte-identical**;
  full160 = byte-concat of the a80+b80 re-renders. The v1 render is *not*
  byte-reproducible by current code (the housekeeping bundle added cue-span
  fields and `stable_hash_v1` frame assignment after the v1 freeze — predates
  the move; probes' own code produces the byte-identical modern output, and
  the frozen v1 file is anchored by its verified manifest sha instead).
- Surface battery re-runs on full160 and v3-dev: output JSON identical to the
  checked-in `results/surface_probes_v3_*.json`.
- Plot scripts (`make_plots`, `gap_plots`, `make_plot_06a`, `pilot_plots_v2`)
  all run clean.

The probes repo was left untouched (its `catalogs/*.yaml` symlinks into this
repo still work), but it is no longer needed by anything here and should be
treated as read-only history.

## 2026-08-17 — Option C: full-crossing schedule + 160-block generation launched

User decision: scale the dev set to 160 blocks × full template crossing
(5,760 items ≈ 3.3× pilot80 extraction). Two parts:

**Full-crossing schedule** (`render-v3 --schedule full`): every rendered
template on every block (11 + alternating anchor = 12 templates × 3 renders
= 36 items/block), making template contrasts within-block paired — the
§8.3 offset forest's paired estimand needs exactly this. New scheme id
`crossed_formats_v3_full` recorded per row (schema enum extended), so the
scheme-locked validator knows which plan reconstructs which rows; the 3+3
rotation lane stays available unchanged under `crossed_formats_v3`. Fold
cycling, contract appearance-rotation, answer-key hashing, and all
F4/aggregate checks carry over; validator reads the schedule off the rows.
Test added (`test_full_crossing_schedule`); suite green.

**Generation run — two aborted starts, both instructive, then
`full160_20260817b` (running):**

1. *Aborted: bare `--per-topic 10` regenerates audit-removed ids.* The
   full80 file's per-topic sample numbers are non-contiguous (audit removed
   7 slots — pb_t001_01/02, pb_t002_02, pb_t004_02, pb_t018_03,
   pb_t032_02/03 — and the 2026-08-16 refill filled them at fresh ids), so
   the raw grid quietly re-targets retired ids. Three such payloads were
   accepted before the run stopped; all three were stripped and the output
   file reset to the frozen 80. **Correct path: `refill-requests
   --per-topic 10 --reserve <pre-audit file>` → `generate --requests`** —
   80 requests at genuinely fresh ids, zero reserved collisions.
2. *Aborted: Gemini 3.1 Pro truncation at `max_output_tokens: 4096`.* The
   error record shows `finish_reason: length` with 3,928 of 4,096 tokens
   consumed by *reasoning*, leaving 154 for JSON. Profile bumped to 8192
   (the model_profile schema max; ~2x observed need). Truncation-on-first-
   Gemini-call is fatal-with-resume by design; two other Gemini payloads
   had already validated fine, confirming the Vertex transport end to end.

3. *Run b stopped at 55/80 on a second Gemini truncation, this time at
   8,192 tokens (7,864 reasoning).* The reasoning-token tail is fat enough
   that no fixed budget fully avoids fatal stops, so the policy changed:
   **generator truncations now consume a generation attempt and retry**
   (`generator_truncation_retry_policy: new_attempt_within_budget_v1`,
   status `generator_truncation_rejection`, excluded from `_feedback` so
   truncation noise never reaches the next prompt; regression test added).
   `model_profile.schema.json` max_output_tokens cap raised 8192 → 32768
   (it predates reasoning models); Gemini profile set to 24,576. Note:
   the new config key means pre-change incomplete run dirs can't resume —
   none existed.

Run `full160_20260817c` (running): the remaining 25 of 80 targeted
requests (55 accepted in b skip via the output file), hash-split Gemini
3.1 Pro (Vertex, gcloud) / Grok 4.3 (OpenRouter), GPT-5.4 validating,
near-dup gate against the full 160 pool (working cross-generator: run b's
first Grok candidate was rejected as a near-dup of an accepted payload and
retried successfully; b accepted 31 Grok / 24 Gemini).
Est. cost ~$2–4 incl. Gemini reasoning tokens. Note: plot-pipeline `.env`
line 2 (`HF_TOKEN =`) has a space before `=` and is not shell-sourceable —
the OpenRouter key is extracted by grep.

**Render decision for 160 blocks (pre-declared):** render the frozen
full80 file and the new-80 file as **two separate `render-v3 --schedule
full` calls** (own manifests), concatenated only for extraction.
Rationale: `assign_block_folds` sorts blocks within a topic, so a combined
160-payload render would silently reassign the *original* 80 blocks' fold
triples away from their pilot80 assignments; per-file rendering keeps the
old blocks' folds identical to pilot80 (comparability) and the BIBD
guarantee holds per file (5 blocks/topic ≥ 3). Item ids are globally
unique across the two files.

**Run c COMPLETE (2026-08-17T10:02Z): 80/80 accepted, zero exhausted, zero
truncations at the 24,576 budget.** Final new-80 split 43 Grok / 37 Gemini;
31 attempt records in c (retries incl. near-dups). Output
`data/payloads.grok43gemini31_gpt54.full160.jsonl` (160 blocks, sha
8d4bbf41…); new-80 subset split out as
`data/payloads.grok43gemini31_gpt54.new80.jsonl` for the two-file render.

**Renders + battery (same day):** `render-v3 --schedule full` on the frozen
full80 (`outputs/rendered_items_v3_full_a80.jsonl`, 2,880 items, sha
554571a2…) and on new80 (`outputs/rendered_items_v3_full_b80.jsonl`, 2,880
items, sha e772009d…), each validator-passed with exactly 240 items per
rendered template per file (within-block pairing achieved); concatenated
as `outputs/rendered_items_v3_full160.jsonl` (5,760). Surface battery on
the concatenation: all four purpose gates 0.500 exactly; template-ID 0.884
(13-way, chance 0.077), stratum 0.786, contract 0.984 — decodable as
expected, reported not gated.

**Artifact relocation (user directive, 2026-08-17): everything lives in
plot-pipeline, never in the probes repo.** All full160 + v3-dev artifacts
moved here and verified post-move (b80 revalidated at its new location;
a80/b80 sha256 match their manifests; full160 = byte-concat of a80+b80):

- `data/payloads.grok43gemini31_gpt54.full160_8-17-2026.jsonl` (160 blocks)
  and `…new80_8-17-2026.jsonl` (the 80 new)
- `data/rendered_items_v3_full_a80_8-17-2026.jsonl`,
  `…_full_b80_8-17-2026.jsonl`, `…_full160_8-17-2026.jsonl` (5,760, the
  extraction input), plus the superseded `…_v3_dev_8-17-2026.jsonl`
- `provenance/render_manifest_v3_full_{a80,b80}.json`,
  `provenance/refill_{requests,report}_full160_8-17-2026.*`,
  `provenance/generation_runs/full160_20260817{,b,c}/` (run configs,
  attempts, errors, run_report)
- `results/surface_probes_v3_full160.json` (and `…_v3_dev.json`)

**Catalogs relocated too (same directive):** the five design catalogs
(`topics.yaml`, `purpose_frames.yaml`, `purpose_frames_v2.yaml`,
`formats.yaml`, `formats_v2.yaml`) now live canonically in
`plot-pipeline/catalogs/`, with symlinks left at
`probes/catalogs/*.yaml` so all pipeline code, tests (62 green through the
symlinks), and stored-path hash verification keep working unchanged.
Manifest `path` fields in `render_manifest_v3_full_{a80,b80}.json` (and
probes' `render_manifest_v2_pilot80.json`) were updated to the canonical
plot-pipeline locations — content sha256s untouched — and
`verify-manifest-v3`/`-v2` pass on all three post-update. The frozen v1/v2
payload inputs (`payloads.grok43-medium_gpt54.*`) stay in the probes repo
as upstream inputs; their content is contained in the full160 file here.

Remaining before GPU: freeze the analysis addendum (leave-one-template-out
endpoints, layer pins, the a80/b80 lineage covariate — new-80 payloads are
Grok+Gemini vs. original all-Grok, so generator lineage should enter the
analysis as a block-level covariate), then extraction (~5,760 prompts × 3
Llama models) + `annotate-tokens` per model.

## 2026-08-17 — Multi-generator generation (Gemini + Grok, GPT-5.4 validating)

`generate_validated_payloads` now accepts several generator profiles
(`--generator-profile` repeatable); each payload slot is assigned one
generator by `stable_hash_by_payload_block_v1` — a hash of the
`payload_block_id` alone, so assignment is insertion-stable (adding/removing
other requests never reassigns a slot, the item-9 discipline) and
resume-stable (retries and rejection feedback for a slot always go back to
the same generator). Every candidate flows through the unchanged single
GPT-5.4 validator, the shared near-dup gate (so cross-generator paraphrases
are caught — the reason this is one run, not two merged runs), and the same
attempt/budget machinery.

Transport: the previously dormant `vertex.structured_caller_for` is now the
default caller resolution — each profile gets the transport matching its
`provider` (OpenRouter key vs. Vertex gcloud auth), so
`gemini31_pro_generator.vertex.yaml` works out of the box; an explicit
`caller` still overrides everything (test injection unchanged).

Robustness decisions: generators must have distinct profile_ids AND distinct
model families (accidental same-lineage dual runs are refused);
per-generator independence from the validator asserted as before;
single-generator runs emit the byte-identical legacy `run_config.json` shape
so existing run directories still resume under the strict-equality check —
only multi-generator runs carry the new `generators` +
`generator_assignment_policy` keys. Attempts and accepted payloads record
`generator_profile` per row; the run report adds `accepted_by_generator`.

Verified: 4 new tests (hash-split provenance end-to-end with a Grok+Gemini
pair, per-block assignment stability, duplicate/same-family rejection,
legacy config-shape preservation); full suite 60 green. README example
updated. Vertex model pinning caveat stands: Gemini 3.1 Pro has no dated
slug on Vertex (`@default`, public preview) — record the run date in any
freeze note that includes Gemini-generated payloads.

## 2026-08-17 — Format catalog v2 + validator implementation

Implements `format_catalog_v2_plan.md` and
`format_catalog_validator_v2_plan.md` end to end. All work uncommitted in the
parent repo working tree alongside the housekeeping bundle. Full test suite
(56 tests incl. 15 new) green; full80 dry-run rendered, validated, hashed,
and surface-gated. No GPU spend.

### Feature: catalog (`catalogs/formats_v2.yaml`)

13 new templates (11 rendered + B7/D6 reserved) + both v1 anchors verbatim
(`legacy: true`, loader cross-checks them byte-for-byte against
`formats.yaml`), contract bank C0–C4, per-template dimension metadata,
in-file F1 exception allowlist, F2 generic-line exemptions, manual banned
stems. Decisions:

- **F1-driven rewordings** (each verified against the frames catalog's
  declared cue stems): C2 "with"→"using"; C3 fully reworded ("Give the final
  response by itself at the end." — original collided with `put`/`on`/`line`);
  C4 "with one field"→"that has one field"; B3 "the following … question"→
  "… question below". Plan doc amended to match.
- **Tier-2 exceptions actually used:** `answer` (scope `*`; MMLU/GSM8K/
  Inspect terminal-field realism + C4 field name), `following` and `with`
  (scoped to B5 only — the deliberate MMLU-preamble fingerprint template).
- D6 authored to dodge `handle`/`request`/`listed` (all fold-owned stems) —
  the lint, not intuition, decided the wording.

### Feature: F1/F2 lints (`src/eval_format_mvp/formats_v2.py`)

Two-tier F1 exactly per validator plan §3.2: hard ban on cue-stem ∪ manual
collisions, exceptions honored only after an R3-style class-symmetry check in
every non-owning fold; legacy anchors demoted to warnings (they do violate:
anchor `with`/`out` reported). F2 = normalized distinctive-line uniqueness
(hard between non-legacy templates) + advisory whole-wrapper TF-IDF cosine
(reuses `dedup.py`; empty wrappers like D1 handled). Lint result on the
shipped catalog: passed, 118 banned stems, 7 exception uses, 4 warnings (2
legacy-F1, 2 expected B2↔v1-anchor line shares — B2 is the declared
successor).

### Feature: schema v3 (`schemas/rendered_item_v3.schema.json`)

New closed schema (`schema_version: 3`, scheme const `crossed_formats_v3`):
template_id / contract_id / template_mode / dimension fields / legacy flag /
nullable cue-span fields all required; `additionalProperties: false` kept.
Tokenizer-derived fields deliberately absent (sidecar, see below). Multi-turn
excluded — own future schema per validator plan §11.

### Feature: renderer + validator (`src/eval_format_mvp/render_v3.py`)

New scheme `crossed_formats_v3`; `render_v2.py` untouched byte-for-byte
(scheme dispatch by module, §8), and the frozen pilot80 jsonl still validates
under the v2 path — regression-tested. Key decisions:

- **Fold BIBD reused unchanged** (`assign_block_folds`): every block keeps
  its pilot80 fold triple; 7 slots cycle the 3 block folds (rotation offset
  hashed per block), guaranteeing topic×fold coverage. v3 uses stated
  third-person pairs only (plan §6); indirect/second-person stay in the v2
  lane.
- **Template rotation:** per topic, consecutive triples over the 6 benchmark
  / 5 deployment rendered templates, anchor alternating — topic-local,
  insertion-stable, same discipline as the BIBD.
- **Contract schedule rotates on per-(topic, template) appearance count**,
  not raw block index — first version used block index; the 3-topic test
  subset exposed uneven coverage (templates appear on a *subset* of blocks,
  so index-rotation isn't a clean cycle). C2 compatibility falls out of
  label metadata alone (no special cases): B4/D3/D5 excluded automatically,
  D4's numbered labels carry C2 into the deployment stratum (§7.3
  resolution).
- **Answer keys (item 17b) via pure per-(block, template) hash.** Traced and
  rejected rank-rotation: per-(topic, template) groups hold only 2–3 blocks,
  which would starve key "D" almost entirely. Hash is uniform in
  expectation, topic-local, uncorrelated with the schedule; validator
  reports the realized distribution and hard-fails only on an absent label
  at ≥16 arrangements. Full80 realized spread e.g. B1 {A36,B24,C27,D27}.
  Anchors keep the v1 `_options` shuffle byte-for-byte (pilot80 bridge).
- **D5 built via `json.dumps`** (the plan's template string is lint/doc
  only) — `str.format` would crash on the literal braces and naive
  interpolation would produce invalid JSON. C0 omits `response_instruction`.
- **Item ids carry template + contract**
  (`item::<block>::<pair|none>::<ev|dp|nc>::<template_id>::<contract>`),
  fixing the v2 scheme's would-be collisions between same-stratum templates.
- **Whitespace rule:** collapse 3+ newlines → 2, strip — so C0's empty slot
  leaves no blank-line fingerprint.
- **Validator derives everything from `build_dataset_plan_v3`** (the same
  function the renderer uses — single source of truth), then requires
  byte-identical reconstruction, then runs F4 fidelity per row (wrapper-line
  residues present, option-style line counts ≥4 — `≥`, not `==`, since
  bodies may contain lookalike lines —, terminal fields, D5 JSON
  round-trip, key-points-at-correct-answer for letter styles) and aggregate
  design checks (compatible contract-cell coverage, C2-in-deployment,
  reserved-never-rendered, topic×fold coverage).

### Feature: F3 surface battery (`src/eval_format_mvp/surface_v3.py`)

Cue-excised (by recorded span, not first-line heuristics), word 1-2-gram +
char_wb 2-5-gram TF-IDF, same probe family as R4 and the sweep control.
Hard gates: purpose under held-template and held-template×held-fold splits,
band [0.35, 0.65]. Report-only: template-ID (held-block), stratum
(held-template), contract (held-block). Full80 results: **purpose gates all
exactly 0.500** — after excision the eval/deploy twins are byte-identical,
so ties everywhere; like the v1 word control this is structural blindness by
design, documented in the module docstring (its teeth are for future lanes
— multi-turn banks, noise — where wrapper text could co-vary with purpose).
Template-ID 0.736 (11-way, chance 0.091), contract 0.931, stratum 0.557 —
formats internally separable, as Sclar et al. predict.

### Feature: token sidecar (`src/eval_format_mvp/annotate_tokens.py`)

Separate `annotate-tokens` pass (lazy transformers import) mirroring
`extract.py`'s prompt construction exactly; records prompt length plus
**both** the content-terminal and prompt-terminal token ids per item — their
divergence is precisely the plan-§7.4 probe-position question, now
measurable instead of assumed. Render/validate stay tokenizer-free
(validator plan §10). Not yet run (needs the HF-gated tokenizer; queue with
the extraction batch).

### Feature: CLI (`cli.py`)

New commands: `lint-formats-v2`, `render-v3`, `validate-rendered-v3`,
`verify-manifest-v3`, `surface-probes-v3`, `annotate-tokens`. v1/v2
commands untouched.

### Feature: tests (`tests/test_formats_v3.py`, 15 tests)

Seeded mutations per validator plan §12.4: injected banned stem → F1 fails;
exception scope leak (`following` outside B5) → fails; duplicated
distinctive line → F2 fails; tampered row → reconstruction fails; forged
option_style → F4 fails. Plus: block composition (21 items, 9+9 stratum
rows + anchor), determinism, contract compatibility invariants, D5
round-trip, anchor-bridging to v1 `_options`, key-label coverage on full80,
excision-twin identity, battery gates, and the pilot80 frozen-chain
regression under the untouched v2 validator.

### Dry-run artifacts (development, not frozen)

`outputs/rendered_items_v3_dev.jsonl` (1,680 items = 80 blocks × 21, sha
856f87cd…), `outputs/render_manifest_v3_dev.json` (hashes verified,
includes both lints + validation report), `outputs/surface_probes_v3_dev.json`.
B4 modes 42/42; per-template items 114–144 (hash-offset spread over the
80-block schedule; deployment exactly 144 each).

### Deviations from the plan docs (all argued, none silent)

1. Contract rotation by appearance count (validator plan said hash+index;
   the plan's own "equally often" intent is served better).
2. Answer-key hash instead of exact rotation (17b "uniform" read as
   uniform-in-expectation; exact balance is impossible under topic-local
   insertion-stability — reasoning in the log entry above).
3. Aggregate contract-coverage check activates at ≥3× compatible-set size
   appearances (not a fixed block count) so smokes/tests don't false-fail.
4. F4 option-line counts use ≥4 rather than ==4 (payload bodies may
   legitimately contain e.g. leading-dash lines).

### Next steps (per plan §9)

1. Freeze the analysis addendum (leave-one-template-out endpoints, §8 hooks,
   layer pins from pilot80) — then promote the dev render to a frozen
   artifact with its manifest.
2. Extraction batch (3 Llama models) + `annotate-tokens` per model
   tokenizer.
3. §5b multi-turn lane: own schema/renderer/validator, prior-exchange bank
   through F1/F2 — deliberately not started (validator plan §11).
