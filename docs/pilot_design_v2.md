# Pilot design v2 — crossed, vocabulary-disjoint purpose frames

Covers proposedchanges.md items **1+8** (disjoint-vocabulary frame catalog
crossed with payload blocks), **5** (no-cue controls), **6** (second-person
frames), and **12/3** (item-budget allocation). The frozen analysis plan that
consumes this design is `docs/analysis_plan_v2.md`. All machinery lives in the
parent repo (`~/Projects/personal/probes`): catalog
`catalogs/purpose_frames_v2.yaml`, lint `src/eval_format_mvp/frames_v2.py`,
renderer `src/eval_format_mvp/render_v2.py`, rendered pilot
`outputs/rendered_items_v2_pilot.jsonl` (manifest alongside, hash-enforced).

## 1. Frame catalog (items 1, 3, 6)

Fifty pairs in five **vocabulary folds** (`fg1`–`fg5`), themed so each fold
owns a coherent, disjoint cue lexicon:

| fold | evaluation theme | deployment theme |
|------|------------------|------------------|
| fg1  | editorial review (reviewer, critique, go over, accuracy, QA) | purchasing (customer, act on, purchasing crew, buyer, PO) |
| fg2  | exam grading (grader, mark, exam panel, rubric, points) | client services (client, heed, callers, invoice, client call) |
| fg3  | audit/inspection (auditor, inspect, comb through, spot check) | tenant maintenance (tenant, apply, carry out, work order, repairs) |
| fg4  | scored test harness (scorer, rate, score table, "Item 12 of 40", test) | user support (user, employ, forward, ticket, support queue) |
| fg5  | judged verification (judge, appraise, vet, verdict) | household shopping (shopper, adopt, put into practice, grocery) |

Every fold has the same composition, so **cue locus is crossed with fold**
(the v1 failure behind item 3): 2 noun-only + 2 verb-only + 2 both stated
third-person pairs, 2 **indirect** pairs (purpose implied by workflow context,
never stated — item 1's "indirect cues"), and 2 **second-person** pairs
(item 6; one minimal noun-swap pair structurally matched to the third-person
noun pairs, one ecological pair like "This is a test of your abilities"),
tagged `cue_perspective: second` and analyzed separately.

**Disjointness is enforced, not assumed** (`lint-frames-v2`):

- R1 — within a pair, class-asymmetric stems must exactly match the declared
  `cue_tokens`; stated pairs must be minimal pairs (identical scaffold after
  cue removal).
- R2 — declared stems are disjoint across folds (stem-level, so
  reviewer/review can't straddle folds; an advisory also flags cross-fold
  substring relations a subword tokenizer could exploit, e.g. the
  "leaderboard"/"board" collision this catalog was rewritten to remove).
- R3 — a fold's declared stem may appear in other folds only
  class-symmetrically, so probes trained on other folds can earn no weight
  for it.
- R4 (functional acceptance) — a held-fold TF-IDF logistic probe on cue
  sentences must land near chance. Measured: **0.525** on the v2 catalog,
  **0.500** on the rendered stated stratum, **0.524** on all rendered cue
  rows — against **0.842** for the v1 frames under the identical test. Any
  above-chance held-fold probe AUC on v2 therefore cannot be a word-identity
  shortcut, which is the entire point of item 1.

## 2. Crossing frames with payloads (item 8)

Each payload block renders under **three of the five folds** (topic-local
balanced incomplete block design): within a topic, blocks are ordered by
`sha256(payload_block_id)` and fold triples step by 3 (mod 5) from a
topic-hash offset. Guarantees, verified by the renderer's validator on the
72-block pilot:

- every topic (hence every task family) reaches **every** fold with as few as
  3 blocks — no more fold-difficulty confounds from uneven task mixes;
- stated blocks per fold are balanced (41–44 of 72);
- editing one topic's payloads never reassigns any other topic (the v2
  analogue of `stable_hash_v1`; within a topic, assignment does depend on the
  block set — the documented residual of trading per-id stability for
  balance-with-coverage);
- the three stated slots of each block carry the three **loci** (rotated by
  block hash), so every block contributes exactly once to noun-only,
  verb-only, and both — payload and locus are fully crossed.

This breaks v1's frames⊗payloads marriage: held-out-frames-with-seen-payloads
(core CV), held-out-payloads-with-seen-frames (E7), and doubly-held-out (E8)
are now all estimable. Payload memorization cannot masquerade as purpose
signal because every payload appears with both labels.

## 3. Strata and item budget (items 12, 3)

Per block: 3 stated renders + 1 indirect + 1 second-person (each a full
2 purposes × 2 formats block) + 2 no-cue items (item 5: same payload, same
format templates, purpose line deleted) = **22 items**. At the current 72
audited blocks:

| stratum | items | independent blocks | notes |
|---------|------:|-------------------:|-------|
| stated third-person | 864 | 72 | primary; 12 stated pairs per fold catalog-wide |
| indirect | 288 | 72 | 12–17 blocks per fold |
| second-person | 288 | 72 | 11–19 blocks per fold |
| no-cue | 144 | 72 | projection-only, never trained on |
| **total** | **1584** | 72 | 5.5× the v1 row count |

Budget rationale (item 12): blocks are the independent unit, so nothing was
spent on extra renders of the same (block, pair) cell; every added item either
adds a new frame–payload pairing or a new stratum. The noun-only stratum —
item 3's near-chance, unestimable subgroup — goes from 3 pairs / 3 families /
21 blocks to **10 pairs / 5 folds / all 72 blocks**, giving it the same
clustered-n as the v1 pooled headline. Residual imbalance: fold×locus cells
range 7–22 blocks (hash luck), so per-fold-per-locus cells stay descriptive;
pooled locus strata (72 blocks each) are the pre-registered comparison.

Expected precision at n=72 clusters (from the synthetic smoke run and
binomial approximations): 95% CIs of roughly ±0.03–0.05 on pooled AUCs near
0.9, ±0.06–0.09 on 288-item strata near 0.75. Differences below ~0.08 between
strata will not separate at this n — the pilot is for effect direction and
catalog validation, not final estimates; that is what the confirmation bank
is for.

**Scaling priority after the pilot** (item 12): first refill to the 80-block
quota (5 per topic — `refill-requests` CLI already emits the 8 missing-slot
requests), then add **new topics/blocks** (which also dilutes the 3-of-5-fold
topic leakage), never more renders of existing blocks.

**Refill executed 2026-08-16** (before any extraction): the 8 missing slots
were generated with the original pipeline (grok-4.3 generator, gpt-5.4
validator, near-dup threshold 0.42 — stricter than the legacy 72, which
predate the near-dup gate and contain 12 same-topic pairs at 0.44–0.55; all
8 accepted first-try, none flagged). The extraction pilot is therefore the
**80-block render**: `data/payloads.grok43-medium_gpt54.full80.jsonl` →
`outputs/rendered_items_v2_pilot80.jsonl` (1760 items; stated blocks per
fold exactly 48/48/48/48/48; rendered n-gram control 0.500 stated / 0.517
all-cue; manifest hash-verified). The 72-block render above is retained
unchanged for provenance. Hashes recorded as an addendum in
`docs/analysis_plan_v2.freeze.json`. If the noun-only or
indirect strata look promising but under-powered, add blocks, not pairs:
every new block automatically feeds all strata.

## 4. Regeneration

```bash
cd ~/Projects/personal/probes
python -m eval_format_mvp lint-frames-v2          # R1-R4, fails on any leak
python -m eval_format_mvp render-v2 \
  --payloads data/payloads.grok43-medium_gpt54.full.audited.jsonl \
  --out outputs/rendered_items_v2_pilot.jsonl \
  --manifest outputs/render_manifest_v2_pilot.json
python -m eval_format_mvp verify-manifest-v2 --manifest outputs/render_manifest_v2_pilot.json
```

Extraction uses the existing `extract` command unchanged (1584 prompts ≈ 5.5×
the frozen run; a modest single-GPU job at 8B). Analysis runs through
`scripts/confirmation_analysis.py` here (exploratory mode on pilot data), per
`docs/analysis_plan_v2.md`.
