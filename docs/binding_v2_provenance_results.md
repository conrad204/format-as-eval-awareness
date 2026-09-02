# binding_v2 and provenance_v1/v2/v3: results

Two experiments, both frozen before activations existed, both run on
`Llama-3.1-8B`, `Llama-3.1-70B`, `Llama-3.3-70B`.

- **binding_v2** hardens `binding_v1` on the two weaknesses its own writeup did
  not close: format was a single template pair, and the label was solvable by
  token coreference.
- **provenance_v1/v2** replace the assigned purpose variable with data origin,
  which we do not control.

All probes are the repository's existing `LogisticRegression(C=0.1)` +
`StandardScaler` convention. Inference units are the relational family (8),
the scaffold template (11), or the underlying text pair (1,000).

---

## 1. binding_v2: the semantics-ablated twin

`binding_v1` decoded a "bound response purpose" at 0.966 AUC. I flagged that its
label was computable by checking whether the nonce in the final sentence matches
the nonce in the sentence carrying discard semantics — pure coreference, no
representation of consequence required.

binding_v2 adds a **neutral twin**: identical structure, identical bag-of-words
invariance, identical difficulty of the matching computation, but the two roles
differ by an arbitrary non-consequential tag (`square` vs `round` filing)
instead of discarded vs persistent effects. If decoding were coreference, both
arms decode equally. Audit confirms both arms sit at exactly 0.500 for every
surface baseline (n-gram, length multiset, payload identity, arm, hop, template).

Family-held-out probes, joint-format, family-mean AUC:

| model | layer | consequence h0 | neutral twin h0 | verdict |
|---|---|---|---|---|
| Llama-3.1-8B | 14 | **0.7896** CI [0.678, 0.903], 8/8 families, p=0.0039 | 0.5419 CI [0.506, 0.584], 5/8, p=0.36 | separated |
| Llama-3.1-70B | 18 | **0.6977** CI [0.604, 0.795], 7/8, p=0.0352 | 0.5062 CI [0.489, 0.526], 4/8, p=0.64 | separated |
| Llama-3.3-70B | 18 | **0.7371** CI [0.631, 0.833], 7/8, p=0.0352 | 0.5072 CI [0.489, 0.524], 5/8, p=0.36 | separated |

**The ablation control passes.** The neutral twin is at chance in all three
checkpoints while the consequence arm decodes well above it. The decoded variable
is not reducible to token identity matching: stripping consequence semantics
while preserving the matching computation destroys decodability.

This is the strongest available defense of the `binding_v1` construct, and it was
obtained by trying to break it, not to confirm it.

### Indirection depth degrades sharply

Same probes, deeper reference chains (h1 = via a bound attribute, h2 = via an
attribute-bound team):

| split | 8B | 3.1-70B | 3.3-70B |
|---|---|---|---|
| within consequence h0 | 0.7896 | 0.6977 | 0.7371 |
| within consequence h1 | 0.6903 | 0.5535 | 0.5654 |
| within consequence h2 | 0.5937 | 0.5152 | 0.5153 |
| train h0 → test h1 | 0.6360 (p=0.0039) | 0.5202 (p=0.0039) | 0.5306 (p=0.0352) |
| train h0 → test h2 | 0.5592 (p=0.0039) | 0.5037 (p=0.14) | 0.5073 (p=0.14) |
| neutral h0 → h1 | 0.5057 (p=0.36) | 0.5047 (p=0.14) | 0.5025 (p=0.64) |

Cross-hop transfer is statistically above chance in the consequence arm and
absent in the neutral arm, but the effect is small: 0.64 at best and 0.50–0.53 in
both 70B checkpoints. **This does not support an abstract bound variable that
survives indirection.** It supports a mostly surface-anchored representation with
a weak abstract component in the 8B model. That is a negative result for the
"abstract use-role variable" framing I sketched earlier.

### The frozen stated-purpose probe does not transfer

Applying the persisted explicit-purpose probes with no retraining:

| model | consequence h0 | neutral h0 |
|---|---|---|
| Llama-3.1-8B | 0.5118 CI [0.499, 0.526] | 0.4891 |
| Llama-3.1-70B | 0.5094 CI [0.498, 0.521] | 0.4986 |
| Llama-3.3-70B | 0.5046 CI [0.494, 0.518] | 0.4988 |

Chance. The stated-purpose direction and the relational-binding variable are
effectively unrelated, replicating the near-orthogonality already in
`docs/binding_v1_results.md` and confirming it is not an artifact of one template.

---

## 2. Format subspace with a held-out template

`binding_v1`'s format subspace removed 75–85% of the transfer gap, but "format"
there was exactly one template pair (`bench_mmlu_bare_v2` / `deploy_bare_v2`), so
the basis could have captured one template's idiosyncratic direction.

binding_v2 assigns 6 benchmark × 5 casual templates over 30 balanced
combinations, and the test holds out a **whole template**: every complete
factorial block using it is removed from both the rank-64 uncentered-SVD basis
and from probe training.

| model | layer | raw gap | projected gap | fraction removed | templates improved | format AUC raw → projected |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 14 | 0.0299 | 0.0058 | **80.6%** | 10/11 | 0.960 → 0.606 |
| Llama-3.1-8B | 31 | 0.0227 | 0.0069 | **69.7%** | 10/11 | 0.942 → 0.506 |
| Llama-3.1-70B | 18 | 0.0175 | 0.0021 | **88.0%** | 11/11 | 0.998 → 0.298 |
| Llama-3.3-70B | 18 | 0.0192 | 0.0033 | **82.7%** | 10/11 | 1.000 → 0.448 |

Joint-format AUC costs 0.006–0.017 while cross-format AUC *rises* 0.007–0.009.

**This materially strengthens the paper's format-subspace claim.** The effect
size (70–88%) matches the original 75–85% while surviving a generalization test
the original could not run: the basis transfers to surface realizations it never
saw. The single-template-pair objection is answered.

Caveat, stated plainly: the absolute gaps are small (0.017–0.030) because the
binding label decodes weakly at the item-pooled level, so a large percentage is a
large share of a small quantity.

---

## 3. provenance: replacing the assigned label with data origin

Every purpose result in this project, binding_v2 included, decodes a variable an
experimenter assigned. provenance tests a variable we do not control: did the
text actually come from an evaluation corpus, or from a real deployment log?

Two banks were built. **v1 is a pilot and should not be cited as a deployment
test**: I built it from `no_robots` and OASST, then audited the sources and found
both are *human-authored data created for instruction tuning*, not organic
traffic. v2 replaces the negative class with **WildChat-1M**, real ChatGPT
interaction logs (ODC-By, revision `7d6490e4…`).

Design: 500 eval-corpus texts (MMLU + ARC-Challenge) and 500 WildChat first user
turns, nearest-neighbor length-matched (mean difference **0.01 characters**, max
sorted-pair difference **1 character**), each rendered once under a benchmark
scaffold and once under a casual scaffold, so provenance ⟂ format by
construction. Frozen stated-purpose probes are applied with **no retraining**.

### The confound is large and is reported, not hidden

Surface-text baselines on v2 provenance:

| baseline | AUC |
|---|---|
| word 1–2 gram, grouped 5-fold | **0.9615** |
| char 3–5 gram, grouped 5-fold | **0.9657** |
| word 1–2 gram, eval-source swap | 0.9375 |
| length only | 0.4751 |

Real eval items and real user turns are lexically almost separable. Length is
not the confound; topic and wording are.

### Result

Frozen stated-purpose probe, provenance AUC, joint regime, all layers:

| model | max AUC (layer) | median over layers | min |
|---|---|---|---|
| Llama-3.1-8B | 0.7277 (L23, q=0.74) | 0.5391 | 0.2431 |
| Llama-3.1-70B | 0.7431 (L23, q=0.29) | 0.6441 | 0.3913 |
| Llama-3.3-70B | 0.7546 (L15, q=0.19) | 0.6152 | 0.3851 |

Maxima are selected over 32/80 layers and are therefore optimistic; medians are
the honest summary. Worst eval-source × deployment-source pair stays above 0.70
at the best layer (0.7216 / 0.7031 / 0.7266).

**Reading this correctly:** a direction fit purely on synthetic stated cues does
carry above-chance information about real corpus provenance, but at **0.73–0.75
it is far below the 0.96 bag-of-words ceiling**, and several depths invert below
0.5 (0.243 minimum in the 8B). This is *not* evidence of a privileged internal
"am I being evaluated" representation. It is consistent with the direction partly
tracking content that also distinguishes the corpora. I am not claiming
spontaneous evaluation awareness, and this bank cannot establish it.

### The measurement-validity signal is the real finding

Because format is crossed with provenance, the scaffold effect and the true
provenance effect are measurable on the same score scale:

| model | benchmark-format AUC | casual-format AUC | mean gap | layers where \|format effect\| > \|provenance effect\| |
|---|---|---|---|---|
| Llama-3.1-8B | 0.766 | 0.688 | +0.049 | 13/32 |
| Llama-3.1-70B | 0.860 | 0.737 | +0.171 | 48/80 |
| Llama-3.3-70B | 0.861 | 0.752 | +0.179 | 48/80 |

The provenance × format score interaction has a bootstrap CI excluding zero at
22/32 (8B), 71/80, and 71/80 layers, block-resampled over complete text pairs
with one shared block sample across layers and regimes.

**At roughly half of all depths in both 70B checkpoints, merely reformatting a
prompt moves the eval-ness score as much as or more than genuinely changing
whether the text came from a benchmark or a real user.** That is a
measurement-validity result about the probe readout on real text, and it is the
part of this run that most supports the paper's thesis that format is entangled
with purpose decoding.

### Cross-fitted no-cue format subspace on real text

Rank-64 `Q_nocue` bases, estimated per original purpose-family fold from full320
no-cue matched-format contrasts only (no provenance row enters the basis), then
projected out of the frozen probe weights:

| model | layer | raw AUC | projected AUC | Δ AUC (CI) | probe weight fraction in Q | external variance captured |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 14 | 0.4683 | 0.4321 | −0.036 [−0.042, −0.031] | 0.0042 | 0.226 |
| Llama-3.1-8B | 31 | 0.6617 | 0.6948 | +0.033 [+0.029, +0.038] | 0.0022 | 0.161 |
| Llama-3.1-70B | 18 | 0.3913 | 0.3590 | −0.032 [−0.040, −0.025] | 0.0099 | 0.261 |
| Llama-3.3-70B | 18 | 0.4835 | 0.4483 | −0.035 [−0.042, −0.028] | 0.0091 | 0.233 |

The no-cue format subspace holds 16–26% of external activation variance but only
**0.2–1.0% of the frozen purpose probe's weight norm**, and removing it changes
provenance AUC by at most 0.036. This independently reproduces, on out-of-domain
real text, the near-orthogonality found by the earlier no-cue analysis
(trace overlap 0.53, cosine ≈ 0.02): the format-associated subspace and the
stated-purpose direction are largely distinct objects.

---

## 3b. provenance_v3: two independent deployment-log corpora

`provenance_v2`'s stated weakness was a single deployment-log source. With LMSYS
access granted, `provenance_v3` adds a second, independent log corpus, making
leave-one-log-corpus-out a real test rather than a relabeling:

| class | sources |
|---|---|
| evaluation corpus | MMLU (250) + ARC-Challenge (250) |
| deployment log | WildChat-1M (250) + LMSYS-Chat-1M (250) |

Length matching is exact: eval-vs-deployment mean difference **0.000**
characters, max sorted-pair difference **0.0**, and the two log corpora differ by
1.31 characters.

**The two log corpora are deliberately not pooled.** WildChat is general free
ChatGPT use; LMSYS is Chatbot Arena, where users deliberately compare and probe
models — arguably a mild evaluation context in its own right. The audit shows
they are nearly as separable from each other (**0.9607**) as evaluation is from
deployment (**0.9586**), so treating "deployment" as one population would have
been wrong.

Audit ceilings: word 1–2 gram 0.9586, char 3–5 gram 0.9652, source-pair holdout
0.9223, **leave-one-log-corpus-out 0.9424**, length only 0.4558.

### 2 x 2 source robustness

Frozen probe, no retraining, best layer per model, all four eval-source ×
log-corpus pairs:

| model | max AUC (layer) | median | worst source pair | all four pairs |
|---|---|---|---|---|
| Llama-3.1-8B | 0.7017 (L27) | 0.5342 | 0.6745 | 0.674–0.725 |
| Llama-3.1-70B | 0.7371 (L24) | 0.5899 | 0.6916 | 0.692–0.786 |
| Llama-3.3-70B | 0.7257 (L31) | 0.5587 | 0.6615 | 0.662–0.786 |

The provenance signal survives swapping **both** the evaluation source and the
deployment-log corpus. This is the robustness upgrade v3 was built to deliver,
and it holds. Magnitudes match v2 (0.73–0.75) and remain far below the 0.96
surface ceiling, so the interpretation limits in §3 are unchanged.

### Arena traffic vs general use, inside the deployment class

Both log corpora carry label 0, which allows a prediction the label itself cannot
make: a direction tracking something genuinely eval-like should rank arena
traffic above general use. AUC > 0.5 means arena scored as more evaluation-like:

| model | mean over layers | median | range | layers CI > 0.5 | layers CI < 0.5 |
|---|---|---|---|---|---|
| Llama-3.1-8B | 0.5126 | 0.5281 | 0.428–0.603 | 4/32 | 5/32 |
| Llama-3.1-70B | **0.5498** | 0.5520 | 0.470–0.587 | **13/21** | **0/21** |
| Llama-3.3-70B | **0.5447** | 0.5425 | 0.479–0.603 | **7/21** | **0/21** |

In both 70B checkpoints the effect is small but strictly one-directional: never
significantly reversed at any depth. The 8B is inconsistent (4 layers up, 5
down), i.e. null.

**How much this is worth:** at AUC ≈ 0.55 it is a weak effect, and arena and
general traffic also differ in topic and register, so content alone could produce
it. It is *consistent with* the direction tracking something about
model-testing context beyond the corpus label, and it is the only result here
that points at all toward evaluation-like sensitivity without a stated cue. It
is not sufficient to claim evaluation awareness, and I do not claim it.

### No-cue projection replicates

| model | layer | raw AUC | projected | Δ AUC | probe weight in Q | external variance in Q |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 14 | 0.5089 | 0.4747 | −0.034 | 0.0042 | 0.234 |
| Llama-3.1-8B | 31 | 0.6415 | 0.6707 | +0.029 | 0.0022 | 0.165 |
| Llama-3.1-70B | 18 | 0.3996 | 0.3650 | −0.035 | 0.0099 | 0.261 |
| Llama-3.3-70B | 18 | 0.4418 | 0.4152 | −0.027 | 0.0091 | 0.238 |

Same picture as v2 on an independent bank: the no-cue format subspace holds
16–26% of external variance but 0.2–1.0% of the probe's weight norm.

---


## 4. Effect on the paper's central claim

| claim | effect | basis |
|---|---|---|
| format subspace explains cross-format purpose-transfer failure | **strengthened** | 70–88% gap removal now survives whole-template holdout across 11 templates and 3 checkpoints |
| `binding_v1`'s bound-purpose variable is not lexical coreference | **strengthened** | semantics-ablated twin at chance (0.51–0.54) while consequence arm decodes 0.70–0.79 |
| an abstract bound variable survives indirection | **weakened** | cross-hop transfer 0.50–0.64; near-collapse in both 70B checkpoints |
| stated-purpose direction generalizes to other purpose-like variables | **unchanged / negative** | frozen probe at chance on binding_v2 (0.505–0.512) |
| formatting contaminates eval-ness readout on real text | **newly supported** | format effect ≥ provenance effect at 48/80 layers in both 70B models |
| provenance result is robust across independent corpora | **newly supported** | all four eval-source × log-corpus pairs 0.66–0.79 in every checkpoint |
| the model spontaneously represents being evaluated | **not established; one weak positive** | provenance AUC 0.73–0.75 vs 0.96 surface ceiling; arena-over-general AUC ≈ 0.55, one-directional in both 70B models, null in 8B |

## 5. Claim boundaries

- Corpus provenance is not the model's own judgment about being evaluated.
- Frozen-activation probe scoring is not behavior and not a causal intervention.
- `provenance_v3` adds a second log corpus (LMSYS), so cross-log-corpus
  robustness is now tested and holds; both log corpora are nonetheless public
  research releases, not private production traffic.
- The arena-over-general effect (AUC ≈ 0.55) is small and confounded with topic
  and register. It is suggestive, not decisive, and needs a content-matched
  design to become evidence.
- Provenance and lexical content are inseparable in natural corpora; the audit
  ceiling (0.96) bounds every provenance interpretation here.
- The projections use uncentered SVD, not centered PCA, and the subspace is
  called format-associated, never "pure format".
- `provenance_v1` is a human-authored-instruction pilot, retained for
  transparency; v2 has genuine deployment-log provenance and v3 adds a second
  independent log corpus.

## 6. Artifacts

Banks (frozen, hash-gated, audited pre-activation):
`results/binding_v2_frozen/`, `results/provenance_v1_frozen/`,
`results/provenance_v2_frozen/` (+ `SOURCE_LICENSES.txt`),
`results/provenance_v3_frozen/`.

Results: `results/binding_v2_{model}_analysis.json`,
`results/binding_v2_{model}_template_subspace.json`,
`results/provenance_v{1,2,3}_{model}_frozen_probe.json`,
`results/provenance_v{2,3}_{model}_qnocue_r64.json`,
`results/provenance_v3_{model}_log_corpus_contrast.json`,
`results/binding_v2_provenance_{binding,subspace,provenance}.csv`,
`results/provenance_v3_{provenance,log_corpus_contrast}.csv`,
`results/binding_v2_provenance_figure.png`,
`results/provenance_v3_figure.png`.

Code: `scripts/binding_v2_{build,audit,analyze,format_subspace,provenance_report}.py`,
`scripts/provenance_v1_{build,audit,score}.py`,
`scripts/provenance_v2_{build,audit,format_projection}.py`,
`scripts/provenance_v3_{build,audit,log_corpus_contrast,report}.py`, plus the
`cluster_*binding_v2*`, `cluster_*provenance_v*` SLURM launchers.

A `write_text` newline bug was found and fixed during this run: on Windows it
translated `\n` to `\r\n`, so a frozen bank's on-disk bytes did not match its own
recorded `items_sha256` and the extraction integrity gate correctly refused the
job. All banks now write bytes and verify.
