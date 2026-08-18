# Pilot80 Analysis Notes — 2026-08-16 (rev 2)

Notes on the v2 pilot run (80 payload blocks, 1,760 items, three Llama models
extracted and analyzed). Endpoint definitions first, then the assessment of the
results in `results/v2/pilot80_analysis_*.json`.

**Rev 2 (same day):** incorporates an external AI review whose checkable claims
were independently re-verified against this repo (see "Review verification"
at the end). Main changes: the endpoints are exploratory here, not
preregistered; the lexical-control claim is narrowed (character n-grams reach
~0.72); the four causes are downgraded to hypotheses; CI conditionality is
stated; the E6 estimand is flagged as nonstandard; the paired 480/480 result is
elevated to the headline.

## Headline (narrowed)

On this synthetic, highly templated development set, linear probes reliably
decode explicitly stated evaluation-vs-deployment purpose across held-out
declared cue vocabularies and held-out topic groups. **All three models rank
the evaluation twin above the deployment twin for 480/480 stated minimal pairs
at the selected layers** (verified by recomputation for 8B; reviewer-verified
for all three) — same payload, same wrapper, one sentence changed, scored by a
probe that never saw the pair's vocabulary fold.

What this does **not yet establish:** a unique abstract "purpose direction,"
latent evaluation awareness, or generalization to genuinely novel language.
Word-identity shortcuts are ruled out; broader surface-form regularities are
not (see below).

## What the E's are

"E" stands for **endpoint** — the analysis outcomes defined in the frozen
analysis plan (`docs/analysis_plan_v2.md`, endpoints table), computed by
`scripts/confirmation_analysis.py`. Important status note: on the pilot these
are **confirmation-plan endpoints applied exploratorily** — the plan itself
designates all pilot analyses exploratory and reserves confirmatory status for
the future sealed bank. The plan, code, and pilot results first appear in the
same commit, so the repo does not independently establish prospective timing;
"preregistered" applies (prospectively) only to the eventual confirmation run.
E1 is primary (the pass/fail gate for confirmation); the rest are secondary and
reported with CIs regardless of outcome.

| Endpoint | Definition | Notes |
|---|---|---|
| **E1** | Stated-purpose decoding: held-vocabulary-fold AUC on stated third-person items, reported next to the cue-sentence n-gram control | **Primary.** Gate: probe AUC − n-gram AUC ≥ 0.05 with 95% CI excluding 0 (proposedchanges items 1–2) |
| **E2** | E1 split by cue locus: noun-only / verb-only / both | The stratum analysis that was impossible in v1, where noun-only rode on 3 sentences (item 3) |
| **E3** | Cross-format transfer: train benchmark → test casual (b2c) and reverse (c2b), within stated | |
| **E4** | Indirect cues: stated-trained probes scored on items where purpose is only implied by context; plus indirect-only CV | CI lower bound > 0.5 upgrades the estimand from "stated" to "implied" purpose (item 4) |
| **E5** | Second-person cues ("you are being evaluated") under third-person-trained probes; plus second-only CV | Transfer vs own-CV gap is the "same representation?" readout (item 6) — but see the calibration caveat below |
| **E6** | No-cue format shift: purpose-neutral items projected on the purpose direction; standardized mean difference (benchmark − casual) and AUC | Positive = benchmark reads as more evaluative (item 5). Sign-flipped across models; estimand nonstandard — see below |
| **E7** | Payload holdout (topic groups), frames seen | The split v1's confounded design could never make (item 8) |
| **E8** | Doubly held out: new vocabulary fold AND new topic group | Strictest generalization number; *approximate* (not exact) analogue of what v1's 0.935 measured |

Also reported: per-fold stated AUCs with block-clustered CIs (item 12).

## What the numbers say

| Endpoint (held-fold unless noted) | v1 (decorr., 8B / 70B) | 8B L12 | 3.1-70B L31 | 3.3-70B L17 |
|---|---|---|---|---|
| E1 stated AUC | 0.935 / 0.974 | **0.992** | **0.998** | **0.995** |
| word n-gram control (cue) | 0.842 | 0.500 | 0.500 | 0.500 |
| char n-gram control (cue, 2–5, best config) | — | ~0.72 (text-level, model-independent) | | |
| paired eval > deploy (stated, selected layer) | — | 480/480 | 480/480 | 480/480 |
| E8 doubly held out (fold × topic) | — | 0.948 | 0.983 | 0.986 |
| E3 transfer b2c / c2b | 0.845 / 0.741 · 0.858 / 0.920 | 0.782 / 0.787 | 0.870 / 0.954 | 0.955 / 0.818 |
| E4 indirect-cue transfer | — | 0.866 | 0.899 | 0.769 |
| E6 no-cue format shift (nonstandard SD units) | — | **−1.52** | **−1.32** | **+0.21** |

Two corrections to the framing "more accurate across all models": the
improvement is concentrated in *held-fold stated decoding* (E1); cross-format
transfer (E3) did **not** uniformly improve, and the new hard strata sit well
below 0.99.

## What the lexical control does and does not rule out

*Technical:* the reported n-gram control is word-level TF-IDF on the cue
sentence. It lands at exactly 0.500 by construction: held-fold cue words are
out of vocabulary, so out-of-fold test vectors are all-zero and the scores are
constant. That means **word-identity shortcuts are controlled — near-disabled
by design — but the control cannot see other surface regularities.** A
character-level TF-IDF probe (2–5-grams, best of several configs) on the same
cue sentences and the same held-fold split reaches **~0.72 AUC** (reproduced;
config-sensitive: 0.44–0.72 across reasonable settings). Full-prompt word
TF-IDF is 0.503. Structurally this is unsurprising: the 30 stated pairs are
**six scaffold templates crossed with five vocabularies** — e.g. every fold's
noun-only frame is literally "A {noun} will read this reply." with only the
noun swapped — so a held-out fold shares its entire sentence scaffold with
training, and sub-word regularities in the swapped stems partially generalize.
The neural probe at 0.992–0.998 still clearly outperforms 0.72, so E1 remains
meaningful, but the correct statement is: **simple word-identity shortcuts no
longer explain the result** — not "keyword cheating is impossible" and not
"the probe found exactly the semantic direction."
*Plain:* we proved the probe isn't matching whole words. We have not proved it
isn't exploiting subtler spelling-level patterns — a letter-pattern detector
gets partway there (0.72), because our "different vocabularies" live inside
identical sentence frames. The real probe is far better than that detector,
so something beyond spelling is being read — but calling it "the concept" is
more than the data shows. Action: report the character-level control alongside
the word-level one, and make the confirmation frames vary sentence structure,
not just the swapped words.

## Why the signal got stronger — four hypotheses (not identified causes)

These are ranked by plausibility; too many things changed simultaneously
(training size, frame construction, vocabulary rules, payloads, selected
layers) for a causal decomposition without a factorial ablation. The earlier
"+0.044 from the split, +0.013 remaining" arithmetic overstated what E8 can
support — E8 holds out whole topic groups, while v1's folds coupled vocabulary
families to individual payload blocks but generally retained related topics,
so they are not exactly apples-to-apples.

**H1. The split got easier in a legitimate but important way.**
*Technical:* v1's folds held out frames and payload blocks simultaneously (the
item-8 confound), so its 0.935 measured generalization to new wordings *and*
new questions at once. v2's E1 holds out only vocabulary — every test payload
was seen in training under other frames, by design of the crossed BIBD. The
nearest (imperfect) comparator is E8: 0.948 vs v1's 0.935 for 8B, suggesting
much of the E1 jump reflects the easier split rather than a better-read
representation.
*Plain:* the old test was two exams at once (new wording + new questions); the
new headline number is just the new-wording exam. On the closest thing to the
combined exam, v2 is only modestly better.

**H2. More and more-diverse training data pushed the probe toward more
abstract features.**
*Technical:* each training set went from ~8 cue pairs sharing one lexicon to
24 pairs across four mutually disjoint declared vocabularies, with 3.3× the
stated items. Under L2 regularization (C=0.1), whole-word features no longer
generalize across training folds, so weight concentrates on features shared
across vocabularies — semantic ones, but potentially also scaffold- and
sub-word-level ones (see the char-ngram result).
*Plain:* the new data punishes whole-word cheating, so the probe had to move to
features that work across wordings. Those are probably more conceptual — but
the letter-pattern detector shows "works across our wordings" is a weaker bar
than "understands the concept."

**H3. Cleaner minimal pairs.**
*Technical:* the v2 lint rules force eval/deploy twins to share identical
scaffolds except the declared stems, so class means differ only along cue
content — less nuisance variance than v1's looser paraphrase pairs. (The same
property is what makes the scaffold-repetition caveat possible.)
*Plain:* the sentence pairs are surgical twins — good for isolating the cue,
but it also means the dataset is six sentence skeletons in five costumes.

**H4. Layer selection optimism.**
*Technical:* L12/L31/L17 were argmax-chosen on this same pilot, so E1 carries
selection bias — handled correctly by the plan (pin via addendum, then confirm
on fresh data). v1 also argmax'd, so this doesn't explain the *difference*,
but it inflates the pilot's absolute numbers slightly.
*Plain:* we picked the best-scoring layer after seeing the scores; the
confirmation run exists to remove that flattery.

## How much is noise?

*Technical:* the reported CIs are **conditional**: the cluster bootstrap
resamples payload blocks but does not refit probes, so it measures test-block
variability given the fitted probes, the five fixed vocabulary folds, the 30
repeated frame pairs, one frame author, and the selected layer. For E7/E8,
blocks are nested in 16 topics but blocks (not topics) are resampled;
reviewer's topic-cluster recomputation widened the 8B E8 CI from ~0.931–0.965
to ~0.924–0.970 (not overturning, but wider than reported), and a frame-pair
bootstrap left E1 high (~0.980–0.999 for 8B). Training-set and frame-author
variability remain unmeasured. Within those conditions the headline is not
noise: E1's clustered CI is ±0.005 at 80 blocks, per-fold spread is
0.980–0.9997, three independently extracted models agree, and the paired
result is 480/480. Where models disagree beyond their CIs, believe the
disagreement: E4 spans 0.769–0.899 and E6 flips sign entirely.
*Plain:* the error bars answer "would this replicate with new questions?" —
and yes, solidly. They cannot answer "would this replicate with new sentence
styles, a different frame author, or a re-tuned pipeline?", because those were
sampled exactly once. The secondary numbers where the three models contradict
each other should be treated as unsettled regardless of their error bars.

## E6: no stable cross-model format direction

*Technical:* the sign chain is internally consistent (label 1 = evaluation,
positive scores = evaluation side, E6 = benchmark − casual; no downstream sign
error found on review). But the magnitude's units are nonstandard: the
denominator is the SD of *all* no-cue scores pooled across both format groups
(and held fixed during bootstrap), which mixes between-group variance into the
scale — conventional pooled-within-group d for 8B is ≈ −2.33 and paired d_z ≈
−2.66, vs the reported −1.52. Either rename the reported quantity or move to a
paired estimand. Substantively, two models project benchmark-formatted neutral
prompts strongly toward *deployment* and the third shows ~nothing; with n=1
template pair per format class, the defensible conclusion is simply: **no
stable cross-model format direction was found.** The single-template confound
(response-contract line, headings, etc.) makes any stronger reading —
including "benchmark reads as deployment" — unsupported.
*Plain:* the code isn't lying about the direction, but the effect size is in
home-made units, and the three models don't tell one story. With only one
benchmark template and one casual template, we can't distinguish "format
effect" from "this particular template's quirks." Verdict: unresolved, needs
multiple templates.

## E5: transfer gap partly reflects calibration, not representation

*Technical:* second-person pooled transfer AUCs (0.876 / 0.938 / 0.891) sit
*below* both of their own by-locus AUCs (8B: noun 0.939, context 0.966 —
visible in the endpoint JSON), a Simpson-style signature of score offsets
between sub-strata; paired ordering is meanwhile ~100%. So the transfer vs
own-CV gap mixes representation alignment with per-frame calibration offsets.
Direction cosines or cross-calibrated paired effects would be the cleaner
"same representation?" readout.
*Plain:* the probe puts second-person items on the right side of each other
almost perfectly, but different sentence styles land at different baseline
heights, which drags down the pooled score. The current gap number therefore
understates how well the direction transfers — measure the angle between
directions instead.

## Plot guide: what each figure in this folder shows

Two scripts write into this folder, and their numbering overlaps (a known fix
item). **Set A** = `scripts/pilot_plots_v2.py`, five endpoint-centric figures
built from the stated-only layer sweeps (`pilot80_layer_sweep_*.json`) and the
endpoint reports. **Set B** = `scripts/pilot80_plots.py`, eleven figures: v1
counterparts 01–07/06a built from the full-regime sweeps
(`pilot80_sweep_full_*.json`, all computed on held **vocabulary** folds within
the stated stratum), plus new endpoint figures 08–10. Set B deliberately keeps
the v1 colors (8B blue, 3.3-70B orange; new 3.1-70B violet) for side-by-side
comparability with the frozen v1 plots; Set A colors 3.1-70B green — one of
the inconsistencies queued for fixing.

### Set A (`pilot_plots_v2.py`)

- **01_stated_signal_by_depth** — *Technical:* the E1 quantity (stated
  third-person held-vocabulary-fold AUC) at every layer, one curve per model;
  dots mark each model's argmax layer (12/31/17), which is what the plan pins
  for confirmation. This is the layer-selection instrument. *Plain:* how well
  the "purpose" reading works at each depth of the network, with the best
  depth per model dotted — the depth we commit to using from now on.
- **02_endpoint_forest** — *Technical:* point + 95% block-cluster CI for E1,
  its n-gram control, E3 both directions, E7, and E8, three models per row.
  Caveat: the title "Every generalization split clears chance" is contradicted
  by its own n-gram row at exactly 0.500 (a control, not a generalization
  split — but the title doesn't say so); retitle pending. *Plain:* one picture
  of all the main pass/fail numbers with their error bars — every real test
  sits far above coin-flip, while the word-counting baseline sits exactly at
  coin-flip.
- **03_strata** — *Technical:* forest of the harder strata: E2 by cue locus
  (noun/verb/both), E4 indirect (stated-trained transfer vs indirect-only own
  CV), E5 second-person (transfer vs own CV). Transfer rows score cue styles
  the probe never trained on. *Plain:* how the probe does on the harder cases —
  cues living in a single word, cues only implied by context, and cues
  addressed to the model itself — including whether a probe trained on plain
  statements can read them anyway.
- **04_nocue_format_shift** — *Technical:* E6 forest — the benchmark−casual
  shift of no-cue items along each model's purpose direction, in the
  (nonstandard, pooled-SD) units discussed above. Same content as Set B's 09.
  *Plain:* when a prompt says nothing about its purpose, does merely looking
  like a quiz move the model's internal reading? The three models disagree,
  which is the finding.
- **05_per_fold** — *Technical:* E1 broken out per held vocabulary fold
  (fg1–fg5, labeled with their lexicons), with clustered CIs — the item-12
  robustness view. *Plain:* the headline number re-computed five times, each
  time hiding a different vocabulary from training — showing no single
  wording family carries the result.

### Set B (`pilot80_plots.py`) — v1-counterpart figures

- **01_purpose_signal_by_depth** — *Technical:* per-layer purpose AUC under
  two training regimes: decorrelated (solid; purpose balanced against format,
  tested on the held vocabulary fold) and confounded (dashed; trained only on
  the eval-benchmark + deploy-casual cells, where purpose and format move
  together). Peak layers annotated. Counterpart of v1 plot 01, now with the
  stronger vocabulary-disjoint holdout. *Plain:* solid = what the model
  actually knows about purpose; dashed = what a sloppy probe learns when
  "test-looking" and "evaluation" always co-occur in training.
- **02_confound_signature** — *Technical:* at each model's peak layer, the
  percent of prompts in each of the four purpose×format cells that the probe
  calls "evaluation," for confounded vs decorrelated training. The confounded
  probe's calls track the format axis; the decorrelated probe's track the
  purpose axis. *Plain:* the smoking-gun bar chart — the naive probe says
  "evaluation" whenever the prompt looks like a quiz regardless of what the
  prompt says its purpose is; the properly trained probe follows the actual
  purpose sentence.
- **03_transfer_by_depth** — *Technical:* E3 by depth — probes trained on one
  format only, tested on the other (b2c left, c2b right), still on held
  vocabulary folds. *Plain:* does the purpose reading learned from quiz-shaped
  prompts still work on chat-shaped prompts, and vice versa, at every depth.
- **04_ranking_vs_threshold** — *Technical:* at the peak layer, AUC (solid)
  vs balanced accuracy at the fixed threshold 0 (pale) for decorrelated, b2c,
  c2b. AUC is threshold-free; balanced accuracy is not, so the gap between the
  bars isolates calibration shift under format change. *Plain:* after a format
  switch the probe still ranks evaluation above deployment nearly as well, but
  its yes/no cutoff drifts — scores shift wholesale without scrambling their
  order. (Same phenomenon as the E5 calibration caveat above.)
- **05_paired_consistency_by_depth** — *Technical:* per layer, the fraction
  of the 480 stated minimal pairs (same block, same pair, same format) where
  the eval twin outscores the deploy twin, under held-fold probes; annotation
  marks the best layer's 480/480. *Plain:* the headline paired result as a
  curve — change one sentence, hold everything else fixed, and count how often
  the score moves the right way.
- **06_signal_split_by_depth** — *Technical:* per model, decorrelated AUC
  (solid) vs the mean of the two E3 transfer AUCs (dashed), with the
  difference shaded. Caveat (review-confirmed): the shading implies an
  additive decomposition of "signal," but AUCs from different fitted probes
  don't decompose — read the shaded band as "cross-format AUC drop," nothing
  more; reframe pending. *Plain:* solid = purpose signal when training saw
  both formats; dashed = what survives training on only one; the gap is how
  much the reading depends on familiar formatting — but it's a comparison of
  two separate probes, not a pie chart of one probe's signal.
- **06a_signal_split_explainer** — *Technical:* single-panel annotated
  version of 06 for 3.3-70B only (matching the v1 explainer's model), with
  arrows marking the widest gap and the near-zero-gap layer. Same
  decomposition caveat as 06. *Plain:* a slowed-down, one-model walkthrough of
  figure 06 for readers seeing the solid/dashed comparison for the first time.
- **07_format_bound_gap** — *Technical:* the 06 difference curve plotted
  directly (decorrelated − mean transfer AUC) by depth, all models on one
  axis; zero line = fully format-invariant reading. Same caveat: it's an AUC
  difference between differently trained probes. *Plain:* one line per model
  tracking how format-dependent the purpose reading is at each depth — near
  zero means a format-blind reading at that depth.

### Set B — new v2-only figures

- **08_generalization_ladder** — *Technical:* bar chart with cluster-CIs at
  each model's pinned layer for: n-gram control, E1, E7, E8, E4 transfer, E5
  transfer — the holdouts ordered roughly by strictness. Caveat: the caption
  says "pre-registered endpoints"; per the timing note above these are
  confirmation-plan endpoints applied exploratorily — caption fix pending.
  *Plain:* the whole generalization story on one ladder — from "held-out
  wordings" through "held-out wordings and topics at once" to "cue styles never
  trained on" — with the word-counting baseline at coin-flip for scale.
- **09_nocue_format_shift** — *Technical:* E6 forest with directional axis
  annotations ("reads as deployment" / "reads as evaluation"); duplicate
  content of Set A's 04 (one of the two should go). Same nonstandard-units
  caveat. *Plain:* identical story to Set A figure 04: formatting alone moves
  neutral prompts, but not in a consistent direction across models.
- **10_cue_locus** — *Technical:* E2 bars with cluster-CIs — stated held-fold
  AUC when the cue differs only in the noun, only in the verb, or both, 80
  blocks per stratum (v1's noun-only rode on 3 pairs in one family and was
  unestimable). *Plain:* how small the cue can get before the probe loses it —
  even a single swapped noun ("reviewer"→"customer") is enough, though it's
  the weakest of the three.

## Presentation/plot fixes (all verified in code)

- `pilot_plots_v2.py` figure 02 title "Every generalization split clears
  chance" is contradicted by its own n-gram row at exactly 0.500 — retitle.
- `pilot80_plots.py` figure 08's caption says "Pre-registered endpoints" —
  same overstatement as the notes' original preregistration language; reword
  to exploratory application of the confirmation-plan endpoints.
- `pilot80_plots.py` figure 06 shades (decorrelated − mean cross-format) AUC as
  a "format-bound" *component*; AUCs from different fitted probes are not
  additive signal quantities — rename to "cross-format AUC drop," drop the
  decomposition framing.
- Llama-3.1-70B is green (`#1baf7a`) in `pilot_plots_v2.py` and purple
  (`#8a5cd6`) in `pilot80_plots.py` — unify.
- The two figure sets share numbers (two "01", two "02"…) in one folder —
  renumber or split folders.
- The v1 column of the results table above now includes both models' E3 values
  (8B 0.845/0.741; 3.3-70B 0.858/0.920 — the earlier revision omitted the
  latter under an "8B / 70B" header).

## Recommended order before confirmation (revised)

1. **Rewrite pilot claims as exploratory** (this revision) and narrow the
   headline to cross-lexicon decodability of explicit cue semantics; elevate
   the 480/480 paired result.
2. **Strengthen surface-form controls:** add character n-gram (and other
   surface) controls to the permanently reported set alongside the word-level
   one; for the confirmation frames, vary sentence *structure* across folds,
   not just the swapped stems, so the scaffold-repetition loophole closes.
3. **Fix uncertainty reporting:** supplement block-only CIs with topic- and
   frame-aware resampling, and probe refitting inside the bootstrap where
   feasible.
4. **Correct E6** to a paired, conventionally scaled estimand; report the
   pilot E6 as "no stable cross-model format direction."
5. **Item 10 (format variants) before sealing anything** — E6 and E3 both
   hang on one template pair; add cue-position variation (item 7) in the same
   render.
6. **Close items 13/14/17b** (audit trail, deterministic solvers, answer-key
   balance) — the confirmation bank needs fresh payloads.
7. **Amend the freeze plan / pin layers (12/31/17) via addendum, commit the
   full pilot provenance** (parent-repo render, refill logs, catalogs were
   uncommitted as of 2026-08-16), then **build the sealed confirmation bank
   from a different model lineage** and spend the one-shot `--confirmatory`
   run.
8. **Then cross-family (Qwen3) and natural-data (Section C, items 19–22)
   validation** — after the construct/statistical fixes above, since a
   transferred direction inherits whatever ambiguity it still carries.

## 405B and Qwen3

**405B FP8 — skip for now.**
*Technical:* E1 is at ceiling (0.992–0.998), so a 405B point is uninformative
there; the strata where scale could matter (E4, E8) show no clean monotone
trend between 8B and the two 70Bs, so one more point won't establish one. FP8
also perturbs activations, so any 405B-vs-70B difference would conflate scale
with quantization — a clean comparison wants bf16 (~810GB, multi-node).
*Plain:* the test is too easy for the models we already have — a bigger model
can't score higher than ~perfect. Only revisit if the paper needs a scaling
story, after confirmation, as labeled exploration.

**Qwen3 — yes, but after the fixes above.**
*Technical:* the strongest remaining external-validity threats are now (a)
surface-form/scaffold shortcuts and (b) "Llama-family quirk." Qwen3 addresses
(b); it cannot address (a), so run it after the control/frame fixes or the
transfer result inherits the ambiguity. Small checkpoints (e.g. 1.7B/8B/32B)
run through the existing `extract.py` on modest hardware. Gotcha: Qwen3's chat
template inserts `<think>` blocks by default — render with
`enable_thinking=False` (or use the 2507 Instruct variants) and smoke-test
that the last-prompt-token position means the same thing. Keep them
exploratory (or add via plan addendum with their own pilot-argmax layers) so
the Llama confirmation stays clean.
*Plain:* a different model family answers "is this a Llama habit?" — the
question reviewers will ask. But run it after tightening the controls, so
we're transferring a direction worth transferring.

## Review verification (rev 2 appendix)

External review claims checked against this repo on 2026-08-16:

- **Reproduced:** char TF-IDF 2–5-gram cue control ≈ 0.72 (reviewer: ~0.71;
  config-sensitive, 0.44–0.72); word-level cue control exactly 0.500;
  full-prompt word control 0.503; 8B 480/480 paired ordering at layer 12
  (refit from activations); v1 3.3-70B E3 = 0.858/0.920.
- **Verified by inspection:** scaffold reuse across folds in
  `catalogs/purpose_frames_v2.yaml` (6 stated scaffolds × 5 vocabularies);
  E6 denominator = pooled SD over both groups, fixed during bootstrap
  (`confirmation_analysis.py`); bootstrap resamples blocks without refitting;
  plan labels pilot exploratory while plan+results share one commit; both
  plot-code issues; color mismatch; duplicate figure numbering; E5 pooled <
  both by-locus AUCs in the 8B endpoint JSON.
- **Taken on reviewer's authority (plausible, not reproduced):** 70B paired
  480/480 counts; topic-cluster and frame-pair bootstrap CI widths; pooled
  Cohen's d ≈ −2.33 and d_z ≈ −2.66 for 8B E6; E5 paired ordering
  100%/100%/95%.
