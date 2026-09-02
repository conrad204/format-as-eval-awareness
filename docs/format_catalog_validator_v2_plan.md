# Format Catalog v2 — Validator Implementation Plan

Status: DRAFT plan, 2026-08-17. Companion to `format_catalog_v2_plan.md`,
expanding its §7 ("Pipeline changes required") into a concrete validation
design. Findings below were traced against the parent repo
(`~/Projects/personal/probes`) code as it stands today: `render_v2.py`,
`frames_v2.py`, `validate.py`, `dedup.py`, `schemas/rendered_item_v2.schema.json`,
and the analysis-side controls in `plot-pipeline/scripts/sweep_full.py`.
Line references are to those files at the commit current on this date.

## 1. Why (one paragraph)

The existing validation stack was built for exactly one design — 22 items per
block, two literal format strings, binary `bm`/`cs` item ids, one option
style — and it enforces that design with hardcoded constants, closed schema
enums, and a byte-equality reconstruction check with no version dispatch.
Introducing the v2 format catalog without a validator plan therefore fails in
three tiers: **mechanical breakage** (schema and structural checks reject the
new rows on sight), **spec bugs in the format plan itself** (F1 as written is
unimplementable against our own frames catalog; F3 as written can never pass),
and **one outright crash** (the D5 JSON template breaks `str.format`). Each is
enumerated below with a fix; §12 gives the order of operations.

## 2. Findings summary

| # | Finding | Tier | Where |
|---|---------|------|-------|
| F-1 | F1 banned-stem lint collides with 6 fold-owned stems used by the planned wrappers/contracts | spec bug | §3 |
| F-2 | F3 functional probe reads ~1.0 by construction (cue vocabulary shared across templates) | spec bug | §4 |
| F-3 | D5 JSON template crashes `str.format`; naive interpolation yields invalid JSON | crash | §5.1 |
| F-4 | `item_id` encodes format as binary `bm`/`cs` → duplicate ids across same-stratum templates | crash | §5.2 |
| F-5 | `answer_key` and `_options` are stratum-keyed, wrong for D3/D4/D5 and B4-GSM8K | wrong output | §5.3 |
| F-6 | Schema v2 closed enums + `additionalProperties: false` reject every new field | mechanical | §6 |
| F-7 | Structural validator hardcodes `ITEMS_PER_BLOCK=22`, 2 no-cue items, per-pair 2×2 cells | mechanical | §7 |
| F-8 | Deterministic reconstruction has no scheme dispatch → renderer edits orphan frozen pilot80 | frozen-chain risk | §8 |
| F-9 | §4 schedule rule ("letter contracts in deployment wrappers") vs. dimension table (0 letter-labeled deployment templates) is unvalidatable as written | spec ambiguity | §7.3 |
| F-10 | F2 near-dup: whole-wrapper TF-IDF flags B1-vs-B5 (intentional) and zero-vectors on D1 (empty wrapper) | wrong granularity | §9 |
| F-11 | `terminal_token_id` / `prompt_token_length` inject a tokenizer into the pure render/validate loop | determinism risk | §10 |
| F-12 | Multi-turn lane breaks `text: string` schema, substring cue spans, and first-line heuristics | mechanical | §11 |

## 3. F1 purpose-symmetry lint: two-tier redesign (fixes F-1)

### 3.1 The verified collisions

Running the planned wrapper/contract vocabulary through the actual
`tokenize`/`stem` machinery (`frames_v2.py:64-83`) against
`catalogs/purpose_frames_v2.yaml` produces these collisions. They exist
because the lint deliberately keeps prepositions out of `STOPWORDS`
(`frames_v2.py:42-44`) — when asymmetric they carry purpose signal, so folds
own stems like `on`, `with`, `put`:

| banned stem | owned by | declared in | collides with |
|---|---|---|---|
| `on` | fg1 | vp01/02/05/06/10 ("act **on** this response") | C3 "…**on** its own last line" |
| `with` | fg5 | vp47 ("A grocery trip **with** this reply") | C2, B5 preamble, v1 benchmark anchor |
| `put` | fg5 | vp42/46 ("**put** … into practice") | C3 "**Put** the final response…" |
| `lin` | fg4 | vp32/34 ("The support **line**…") | C3 "…last **line**" |
| `follow` | fg2 | vp12/15 ("will **follow** the final answer") | B3 / B5 "the **following**…" |
| `answ` | fg2 | vp17 ("**answer** sheet for tomorrow's exam") | B1/B5 `Answer:`, B3, C4, v1 casual anchor |

C3 violates three folds at once. Both `legacy: true` anchors violate F1
today. A naive stem-ban either rejects most of the planned catalog or gets
quietly weakened during implementation — the worst outcome, because then F1
certifies nothing.

### 3.2 The redesign

F1 becomes two tiers, both machine-checked:

- **Tier 1 (hard fail):** no purpose-connoting *content* stem may appear in
  any wrapper, contract, or prior-exchange string. Banned list = frame cue
  stems that are open-class content words (review/test/grade/exam/customer/
  user/check/…) ∪ the manual list. Any hit fails the lint, no exceptions.
- **Tier 2 (allowlisted exception):** a fold-owned stem may be granted an
  exception **only if** the lint verifies, with the existing R3 counting code
  (`frames_v2.py:214-234`), that the stem already appears class-symmetrically
  in the frames scaffolds of **every** fold. (`answ` genuinely qualifies:
  "the final answer" saturates both classes across pairs.) Exceptions live in
  a dedicated block **inside `catalogs/formats_v2.yaml`** — one entry per
  token with a one-line justification — so they are sha256-hashed into the
  render manifest and cannot drift silently. An exception whose symmetry
  precondition stops holding fails the lint at that point, not at review time.

### 3.3 Rewording where nothing is lost

Cheaper than exceptions wherever realism is not load-bearing:

- **C3** → e.g. `"Give the final response by itself at the end."` (drops
  `put`, `on`, `line` in one stroke; verify `end` against the catalog before
  freezing).
- **B3** instruction line: drop "following" — e.g.
  `"Answer the multiple choice question below."` (still needs the `answ`
  Tier-2 exception; `below` must be checked).
- `Answer:` / `A:` terminal fields and the B5 preamble stay as-is under
  Tier 2 — they are the MMLU/GSM8K signatures whose realism is the point of
  those templates.

### 3.4 Longer term

vp17's real cue is the bigram "answer sheet," not the unigram "answer"; the
unigram-level declaration is the defect. Fix the declaration granularity in
the confirmation-bank folds (new vocabulary, non-Claude lineage, per the
frozen plan) — do **not** edit the frozen v2 frames catalog, which would break
pilot80 comparability.

## 4. F3 functional probe: excise cue spans, split by expectation (fixes F-2)

As specified ("TF-IDF probe predicting purpose from **full renders** under a
held-out-template split must be ≈ chance") the check can never pass: every
render contains the `{purpose}` cue sentence, cue vocabulary is shared across
all templates, so holding out a *template* holds out no cue vocabulary — the
probe reads the cue and scores ~1.0 by design. The frames analogue R4
(`frames_v2.py:263`) only works because its split holds out vocabulary folds.

Redesign:

1. **Excise cue text by recorded span, not position.** Cue spans are already
   logged per render (`_cue_annotations`, `render_v2.py:169`). The probe input
   is the render with the cue span removed. Retire the
   `text_without_first_line` heuristic (`sweep_full.py`) — it also breaks
   outright for the §5b multi-turn lane, where the cue sits in the final turn.
2. **Word- and char-level TF-IDF** on the excised text (the rev-2 lesson:
   word-level alone missed the 0.72 char-level channel).
3. **Per-probe expectation table**, pre-declared:

   | probe (on excised text) | split | expectation | role |
   |---|---|---|---|
   | purpose | held-template | ≈ chance (0.35–0.65) | **hard gate** |
   | purpose | held-template × held-fold | ≈ chance | strictest variant, hard gate |
   | template-ID | held-block | ≈ 1.0 | expected; report only |
   | stratum | held-template | decodable | expected; report only |
   | contract-ID | held-block | decodable | expected; report only |

4. The same excision mechanism feeds the analysis-side n-gram control in the
   sweep, so dev lint and permanent control stay one probe family (the R4
   pattern: same vectorizer, `LogisticRegression C=1.0, random_state=0`).

## 5. Renderer changes (fixes F-3, F-4, F-5)

1. **D5 must not go through `str.format`.** Templates render via
   `str(template).format(...)` (`render_v2.py:144`); D5's literal JSON braces
   make `.format()` parse `{"input":` as a field name and raise. Even escaped,
   interpolating a body containing newlines/quotes into a JSON string literal
   yields invalid JSON. Build D5 as `json.dumps` of a dict (purpose line
   prepended outside the object); F4 asserts `json.loads` round-trips on every
   D5 render and that the parsed object's fields equal the payload fields.
2. **`item_id` gains `template_id` and `contract_id` segments.** The current
   scheme (`render_v2.py:147-151`) encodes format only as `bm`/`cs`; two
   renders of one block/pair/purpose under two same-stratum templates collide
   and the validator's own duplicate check (`render_v2.py:234`) kills the
   render. New shape:
   `item::<block>::<pair|none>::<ev|dp|nc>::<template_id>::<contract_id>`.
3. **`answer_key` keys off template metadata, not stratum.** Replace
   `answer_key if format_name == "benchmark" else None` (`render_v2.py:165`)
   with logic driven by the template's declared `option_style`: letter styles
   → letter key; numbered → number key; bullets/JSON-array/absent → `null`
   plus `correct_answer` text (unchanged). B4's GSM8K mode is option-free and
   gets `null` despite being benchmark-stratum.
4. **Option-style renderers** (dotted, paren, bullets, numbered, `choice:`,
   JSON array, absent) as pure functions selected by catalog metadata; F4
   asserts declared style ↔ rendered text per render (§7.2).
5. **Item 17b answer-key shuffle is hash-deterministic**, keyed like
   `assign_block_folds` (`render_v2.py:51`) on
   `(payload_block_id, template_id)` — never RNG — or reconstruction equality
   (§8) fails on every row.
6. **Whitespace normalization rule, declared once:** after slot substitution,
   collapse runs of 3+ newlines to 2 and strip outer whitespace. Otherwise the
   blank line left where C0's empty `{contract}` sat is itself a contract
   fingerprint.

## 6. Schema v3 (fixes F-6)

`rendered_item_v2.schema.json` has `additionalProperties: false` and closed
enums for `render_slot`, `cue_style`, `format` — every new field is rejected
on sight. Author `rendered_item_v3.schema.json` (`schema_version: 3`), keeping
`additionalProperties: false` (it is doing its job):

- New required: `template_id`, `contract_id` (`null` for `legacy: true`
  anchors with embedded contracts), `stratum` alias note (the binary `format`
  field stays, per plan §2.1), and the declared dimensions
  (`heading_style`, `option_style`, `serialization`, `register`,
  `terminal_field`) copied from the catalog per render.
- `cue_char_span` (present-or-null) becomes required — §4 depends on it.
- Tokenizer-derived fields (`prompt_token_length`, `terminal_token_id`) are
  **not** in this schema; they live in the annotation sidecar (§10).
- Multi-turn rows do **not** reuse this schema (§11).

## 7. Structural validator: derive invariants, don't hardcode them (fixes F-7, F-9)

### 7.1 The single-source-of-truth rule

`validate_rendered_v2` asserts `ITEMS_PER_BLOCK == 22` (`render_v2.py:44,243`),
exactly 2 no-cue items spanning `{benchmark, casual}` (`:254`), and complete
2×2 cells per pair (`:264`) — all false under the new per-block design (3+3
stratum draw, one no-cue per (block, template), ≈21 items). Do not replace
these constants with new constants. The validator calls the same plan function
the renderer uses (`build_block_plan_v3`) and compares the observed block
composition against the plan's output. One function defines the design; the
validator can then never drift from the renderer.

### 7.2 New invariants (data-driven, read from the catalog)

- Per block: exactly 3 benchmark + 3 deployment new templates + 1 anchor
  (alternating), template rotation topic-local, hash-rotated,
  insertion-stable (same discipline as `assign_block_folds`).
- Per (block, template): {eval, deploy} renders of one stated pair + one
  no-cue render.
- **Template × contract compatibility:** C2 only where the template's
  declared `option_style` produces labels; B4-GSM8K and D1/D2 only with
  letter-free contracts. The compatibility matrix is catalog data, not code.
- **Contract-schedule balance:** each compatible (template, contract) cell
  appears equally often across blocks, hash-rotated.
- **Answer-key uniformity** (item 17b): keys ≈ uniform over A–D within each
  letter-labeled template.
- **`reserved: true` templates (B7/D6) never appear in rendered rows**; their
  catalog entries must exist and be hashed into the manifest.
- **`legacy: true` anchors excluded** from contract-balance and factorial
  counts; included in per-block count.
- **F4 metadata fidelity:** re-derive spans (§7.3 of the format plan) and
  assert declared `heading_style` / `option_style` / `terminal_field` /
  `serialization` match the rendered text; for D5, JSON round-trip (§5.1).

### 7.3 Resolve the C2 contradiction before coding it

Plan §4 requires letter contracts in deployment-shaped wrappers, but the
dimension table gives deployment zero letter-labeled templates — only D4's
numbered labels can satisfy C2. Pick one, pre-declare it, and encode it in the
compatibility matrix: either (a) C2's wording is "option label" and D4's
numbers qualify (recommended — no new template needed), or (b) the §4
schedule rule is relaxed to "C2 appears in ≥1 deployment template." An
unvalidatable sentence in the plan becomes a silent judgment call in code.

## 8. Reconstruction versioning: scheme dispatch (fixes F-8)

`validate_rendered_v2` re-renders with the *current* code and requires dict
equality (`render_v2.py:269-282`) under a single hardcoded `SCHEME_V2`.
Editing the renderer in place therefore orphans the frozen pilot80 dataset —
exactly the frozen-chain guarantee this project depends on. `validate.py:54-65`
already solved this for v1 with scheme dispatch; render_v2 dropped the
pattern. Restore it:

- A renderer registry keyed on `frame_assignment_scheme`:
  `crossed_bibd_v2` → the current `render_payload_v2` code path, preserved
  byte-for-byte; the new catalog renders under a new id (e.g.
  `crossed_formats_v3`).
- Acceptance test: after all v3 changes land, `validate-rendered-v2` still
  passes on the frozen pilot80 jsonl, unmodified.
- Manifest inputs gain `formats_v2.yaml` (and the contract bank, if a separate
  file) so `verify_render_manifest_v2` (`render_v2.py:373`) hashes them.

## 9. F2 near-dup: line-level, not whole-wrapper (fixes F-10)

`dedup.py`'s TF-IDF cosine was built for payload text. On wrapper text it
fails twice: D1's wrapper is the empty string (zero vector — also audit
`cosine()` for divide-by-zero), and B1 vs B5 differ by one preamble line *by
design*, so whole-wrapper cosine flags an intentional pair. Redefine F2 at
line level:

- Normalize each wrapper line (lowercase, collapse whitespace, drop slot
  markers). A **distinctive line** is one occurring in exactly one template.
- Hard fail: two templates sharing a non-generic line. Generic terminal
  fields (`Answer:`, `A:`) and empty lines are exempt via a small explicit
  list in the catalog.
- Keep the TF-IDF cosine as an **advisory warning** only, with empty wrappers
  skipped.

## 10. Tokenizer metadata: separate annotation pass (fixes F-11)

`terminal_token_id` and `prompt_token_length` (format plan §7.2) would inject
a transformers dependency and cross-machine variance into a currently pure,
dependency-free deterministic loop — and reconstruction equality would
inherit both. Instead:

- Render and structural validation stay tokenizer-free.
- A separate `annotate-tokens` pass writes a sidecar jsonl keyed by `item_id`
  with its own manifest: pinned tokenizer id + revision + sha256 of the
  tokenizer files, input jsonl hash, output hash.
- The §7.4 verification task (which token position `extract.py` actually
  probes after `apply_chat_template`) lives in this pass's report — settled
  empirically, recorded in the extraction config, as the plan requires.

## 11. Multi-turn lane: isolated, never shoehorned (fixes F-12)

Message arrays break the `text: string` schema, substring-based cue spans,
first-line heuristics, and single-message reconstruction simultaneously. The
format plan already mandates "never pooled silently" for analysis; extend
that to validation: own schema (`rendered_item_multiturn_v1.schema.json`,
messages array with `message_role` + per-message cue spans), own renderer,
own validator and manifest, own scheme id. The prior-exchange bank (≥8
exchanges) runs through the same F1 tiers (§3.2) and F2 line check (§9) as
wrappers.

## 12. Sequencing and acceptance criteria

1. **Amend `format_catalog_v2_plan.md` first** — it currently commits §7 to
   the broken F1/F3 specs and the C2 contradiction: adopt §3 (two-tier F1),
   §4 (excision F3), §7.3 (C2 resolution), and reword C3/B3 per §3.3.
2. Schema v3 + scheme-dispatch registry (§6, §8). Gate: frozen pilot80 still
   validates unchanged.
3. Renderer changes (§5) + catalog authoring with dimension metadata,
   exceptions block, and source-commit pins. Gate: D5 JSON round-trips; no
   `item_id` collisions on a full dry-run render.
4. Lints F1/F2/F4 (§3, §9, §7.2). Gate: full catalog passes with every Tier-2
   exception justified in-file; a seeded mutation test (inject a banned stem,
   a duplicated line, a wrong option style) fails each lint.
5. Structural validator rework (§7). Gate: dry-run render of the full
   schedule passes; each invariant has a seeded-violation test.
6. Render for real; run the §4 probe battery **before any GPU spend** (format
   plan §9.2). Gate: purpose-on-excised-text ≈ chance at word and char level.
7. Token annotation pass (§10), then freeze the analysis addendum and extract.
8. Multi-turn lane (§11) only after primary results, per format plan §9.5.
