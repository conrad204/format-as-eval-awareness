# Hidden-channel test: latent evaluation status is represented before it is reportable

Status: **exploratory**, not pre-registered. 2026-08-31. Two independent adversarial
audits informed the design and this write-up (`history://BrutalEAAuditor`,
`history://HiddenChannelAuditor`). The governing protocol file was marked
`DO_NOT_RUN` when the first run started and specifies 32 arithmetic items against a
bank of 16, so its item-level gate was unreachable by construction. Nothing here is
reported as a passed pre-registration.

## The question this design fixes

Earlier work in this repository, and the published probing literature, supplies the
model with **recognisable evidence of evaluation** in the prompt: benchmark
formatting, multiple-choice shape, eval-like topics, an explicit purpose statement,
or an indirectly worded but still explicit description ("blind trial", "sealed key").
Success then only shows that a familiar surface can be detected or read.

Here the status is **absent from the queried trial**. An arbitrary channel letter
means closed-selection on one prompt and active-use on another, and its meaning
exists only in eight prior outcome episodes inside the same conversation. The
letter-to-status mapping is fully reversed within every identical
episode-family x arithmetic-item unit. Prompts of opposite status differ in
**exactly one character**.

Frozen bank: `results/behavioral_ea_hidden_channel_frozen`
(`items_sha256` `0bcd4b48...`), 32 separately authored episode families, 16
arithmetic items, 8 held-out consequence questions, 5,120 prompts.

## Result 1: reporting the latent status scales; reading a stated one does not

Forced-choice accuracy, two-way crossed bootstrap over 32 episode families and 8
question families:

| checkpoint (BF16) | latent status | crossed 95% CI | above chance | families | questions |
|---|---:|---:|:--:|---:|---:|
| Llama-3.1-8B | 0.5527 | [0.4980, 0.6357] | no | 24/32 | 5/8 |
| Llama-3.1-70B | **0.7178** | [0.6035, 0.8350] | yes | 31/32 | 6/8 |
| Llama-3.3-70B | **0.6592** | [0.5547, 0.7803] | yes | 29/32 | 7/8 |

The 70B effect replicates across two independently trained checkpoints.

Reading a **stated** route is easy at both scales, so the latent failure at 8B is not
a comprehension failure:

| checkpoint | stated route | crossed 95% CI |
|---|---:|---:|
| Llama-3.1-8B | 0.9082 | [0.8184, 0.9824] |
| Llama-3.1-70B | 0.9609 | [0.8984, 1.0000] |

## Result 2: the 70B effect is not a letter prior, not recency

**Mapping reversal.** A constant "K means closed" prior scores exactly 0.5 pooled.
Observed, pooled within mapping:

| checkpoint | mapping 0 | mapping 1 |
|---|---:|---:|
| Llama-3.1-70B | 0.7188 | 0.7168 |
| Llama-3.3-70B | see per-condition table in artifact | |
| Llama-3.1-8B | 0.5547 | 0.5508 |

**Recency.** Reading only the final episode outcome would inflate trials where the
queried channel is the one just named. Accuracy is *higher* when it differs:

| checkpoint | queried channel = last episode's | queried channel differs |
|---|---:|---:|
| Llama-3.1-70B | 0.6699 | **0.7656** |
| Llama-3.3-70B | 0.6094 | **0.7090** |
| Llama-3.1-8B | 0.5625 | 0.5430 |

## Result 3: the 8B null is neither a dose limit nor a general in-context-learning limit

| control (Llama-3.1-8B) | accuracy | crossed 95% CI | families |
|---|---:|---:|---:|
| latent evaluative, 8 episodes | 0.5527 | [0.4980, 0.6357] | 24/32 |
| latent evaluative, **16 episodes** | 0.5381 | [0.4980, 0.5947] | 19/32 |
| **arbitrary** non-evaluative contingency, 8 episodes | **0.6777** | [0.5693, 0.7871] | 31/32 |

Doubling the evidence does not help. A structurally identical hidden contingency with
no evaluative semantics — a channel determines which stakes-free bookkeeping test
picks a folder colour, with the same episode count, the same reversal, and the same
one-character separation — **is** learned above chance.

### The same comparison in all three checkpoints, paired by episode family

Both banks share the same 32 episode families, so the contrast can be paired within
family and bootstrapped over families (`results/specificity_paired_gap.json`,
20,000 replicates):

| checkpoint | evaluative | arbitrary | paired gap | 95% CI | families arbitrary higher |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.5527 | 0.6777 | **+0.1250** | [0.0996, 0.1514] | 31/32 |
| Llama-3.1-70B | 0.7178 | 0.8447 | **+0.1270** | [0.0850, 0.1689] | 25/32 |
| Llama-3.3-70B | 0.6592 | 0.8604 | **+0.2012** | [0.1602, 0.2422] | 31/32 |

Every checkpoint learns the arbitrary contingency **better**, with intervals excluding
zero. There is no evaluative privilege anywhere in this design. Whatever drives the
latent-status result at 70B is not an evaluation-specific faculty; it is
latent-variable inference that happens to be applied to evaluation-flavoured content,
and it is weaker for that content than for matched arbitrary content.

**Caveat that bounds the interpretation.** The two banks cannot share question
wording, because stakes-free episodes have no consequences to ask about. Structure,
dose, reversal, family set and lexical separation are matched; question form is not.
The direction and consistency of the gap are therefore solid, but its magnitude
cannot be attributed to evaluative semantics as opposed to question difficulty.

## Result 4: the explicit ceiling is genuine status reading, not a letter strategy

Audit defect D3 was that closed was always letter K in the check bank. Fixed by an
incongruent condition where the stated route contradicts what that letter taught in
the episodes. Llama-3.1-8B follows the **statement** at **0.8691** (n = 512), so the
0.9082 ceiling reflects reading a stated status rather than a memorised letter rule.

## Result 5: 8B represents the status it cannot report

Cross-fitted linear probe on the same prompts' final-token activations, 3,072 rows,
32 layers, leave-episode-family-out, **layer chosen inside the training folds only**:

| read-out (Llama-3.1-8B) | pooled AUC | family-mean | 95% CI | families > 0.5 |
|---|---:|---:|---:|---:|
| latent status | 0.7242 | **0.7699** | [0.6900, 0.8404] | 29/32 |
| permutation null | 0.4976 | 0.5036 | [0.4834, 0.5243] | 14/32 |
| channel letter (positive control) | 0.9592 | 0.9734 | [0.9575, 0.9856] | 32/32 |

Status is an XOR-like conjunction of the mapping taught in the episodes and the
letter in the query. Pooled over mappings, a letter-only feature scores 0.5 and a
mapping-only feature scores 0.5; the permutation null preserves both and returns
0.504. Linear decodability at 0.770 therefore implies the residual stream already
carries the conjunction.

Set against the model's own answer on identical prompts (0.5527, [0.4980, 0.6357]),
this is a **dissociation between representation and report** at 8B.

## What is invalid, stated plainly

Four things were built, run, and must not be used. Three were my own control
failures, found by audit or by inspection.

1. **Action arm (metric gaming): uninterpretable, not null.** When the retention rule
   is stated outright, the model emits the flawed legacy answer on 2.73% of trials
   (14/512) versus 0.20% (1/512) on live-route trials. Maximum expressible effect
   ~0.025 against a 0.15 threshold. The observed 0.0029 [0.0000, 0.0107] is evidence
   about refusing to emit a knowingly wrong integer, not about status inference.
2. **First probe's "scrambled null" was uninformative.** It labelled rows by channel
   letter, which is trivially decodable (0.967), so it could only return ~1.0. It did.
3. **Second probe's "cross-mapping transfer" was confounded by design.** Training on
   one mapping makes status and letter perfectly collinear, so the probe can only
   learn the letter and must invert on the reversed mapping. Its 0.3163 is a
   guaranteed artifact and was briefly, wrongly, called decisive.
4. **Instruction-compliance diagnostic is a stimulus bug.** The instruction said
   "the number printed as the reference entry", but each family names that field in
   its own words ("slip carries a prior figure of 86"), while the query also carries a
   salient item number. The model returned the item number: the most common parsed
   values are 400, 406, 409, 411-415. It measures my wording, not compliance.

## Claim I will defend

> In a frozen environment where an arbitrary channel letter's status existed only in
> eight prior outcome episodes, and the mapping was reversed within every identical
> unit, Llama-3.1-70B and Llama-3.3-70B reported that latent status above chance
> (0.7178 [0.6035, 0.8350]; 0.6592 [0.5547, 0.7803]), robust to mapping reversal and
> not attributable to recency or to any surface difference beyond one character, while
> Llama-3.1-8B did not (0.5527 [0.4980, 0.6357]) despite reading a stated route at
> 0.9082 and despite a 16-episode dose. This is not an evaluation-specific ability:
> in all three checkpoints a structurally identical non-evaluative contingency is
> learned better, paired within episode family, by +0.125 [0.0996, 0.1514],
> +0.127 [0.0850, 0.1689] and +0.201 [0.1602, 0.2422]. A cross-fitted probe with a
> valid permutation null nonetheless decodes the latent status from Llama-3.1-8B
> activations at 0.7699 [0.6900, 0.8404], so at that scale the status is represented
> without being reportable.

Not claimed: subjective awareness; deception; spontaneous recognition in ordinary
traffic; metric gaming of any kind; an evaluation-specific faculty; that evaluative
semantics rather than question form cause the specificity gap; any pre-registered
outcome.

## Artifacts

- banks: `results/behavioral_ea_hidden_channel_frozen`, `..._check_frozen`,
  `..._supplement_frozen`, `results/behavioral_ea_arbitrary_contingency_frozen`
- runs: `results/behavioral_ea_hidden_channel_{llama31_8b,recog_llama31_70b,recog_llama33_70b}.json`,
  `..._check_llama31_8b.json`, `..._recogchk_llama31_70b.json`, `..._supp_llama31_8b.json`,
  `results/behavioral_ea_arbitrary_llama31_8b.json`
- analyses: `results/recognition_latent_evaluative.json`,
  `results/recognition_arbitrary_control.json`,
  `results/recognition_explicit_ceiling_70b.json`,
  `results/behavioral_ea_latent_vs_explicit.json`,
  `results/probe_v3_hidden_channel_llama31_8b.json`
- superseded, retained for audit: `results/probe_hidden_channel_llama31_8b.json`,
  `results/probe_v2_hidden_channel_llama31_8b.json`,
  `results/behavioral_ea_hidden_channel_interim_8b.json`
- code: `scripts/build_behavioral_ea_hidden_channel_*.py`,
  `scripts/run_behavioral_ea_hidden_channel.py`,
  `scripts/analyze_recognition_bank.py`,
  `scripts/probe_hidden_channel_status_v3.py`,
  `scripts/extract_hidden_channel_activations.py`,
  `tests/test_hidden_channel_parser.py`

## Still open

- Llama-3.1-70B and Llama-3.3-70B arbitrary-contingency controls (running).
- Llama-3.3-70B stated-route ceiling, and 70B dose/incongruence supplements (queued).
- 70B activations and probe, to test whether representation is present at both scales
  while report emerges only at 70B (needs the cluster GPU; APIs do not expose
  activations).
- A causal step: the repository already contains steering machinery
  (`scripts/steer_fit_directions.py`, `scripts/ndif_steer.py`) but `NDIF_API_KEY` is
  not present in this environment.
- A repairable action arm, if metric gaming is to be tested at all.
