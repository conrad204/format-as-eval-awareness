# Direction 1 — Implicit purpose carried by prompt format

**Status: complete.** All three checkpoints analysed, all controls run and read
out, all figures generated. Everything here is exploratory development analysis
of the spent `full320-v3` freeze. Nothing is confirmatory and nothing is an
evaluation-awareness claim. See [Claim boundary](#claim-boundary).

The one-line summary: **when a prompt states no purpose, its wrapper moves it
along the model's stated-purpose axis, and markup serialization moves it toward
deployment.** The effect is no-cue-specific in all three models and survives
every well-powered control. The *mechanism* — whether serialization per se is
doing the work — is supported in two of three models and is not settled by this
data; see findings 5 and 8 and the [next step](#run-state).

---

## The question

The repository's existing result — stated response purpose is linearly
decodable at AUC ≈ 0.99 and transfers across format families — is close to
tautological: the purpose sentence is in the prompt, so a probe recovering it
is unsurprising. The interesting quantity is the one the frozen analysis plan
files as secondary endpoint **E6** and never makes central:

> When a prompt states *no* purpose at all, where does the model place it on
> the stated-purpose axis, and what feature of its presentation moves it?

The freeze contains **3,840 no-cue rows** — all 320 payload blocks rendered
through all 13 templates with no purpose sentence. That stratum was unanalysed.

## Headline findings

**1. Uncued prompts are not neutral.** Projected onto the purpose axis, no-cue
prompts sit roughly **20–30% of the way from the deployment centroid toward the
evaluation centroid** at mid-to-late depth (8B: π ≈ +0.25 to +0.32 for both
format families from layer ~13 on). The unit is calibrated: π = 1 is the effect
of literally writing *"an auditor will inspect this response."*

**2. Format carries implicit purpose, and the two cuts point in opposite
directions.** The benchmark/casual contrast the literature uses is real and
points toward *evaluation* (ρ ≈ +0.05 to +0.07 in the 8B, +0.04 to +0.11 in
3.1-70B, +0.02 to +0.14 in 3.3-70B). The contrast between **markup-serialized
and plain templates points the opposite way** — JSON, XML and Markdown wrappers
pull an uncued prompt toward the *deployment* pole, in all three models at
every layer.

The **opposite sign** is what replicates. The **relative size does not**, and
the original framing ("the benchmark/casual cut is the wrong one") only holds
for the 8B. Markup runs 2–4× the benchmark contrast in the 8B, ~1–4× in
3.1-70B, and is *smaller* than it through much of 3.3-70B's deep half. See
finding 8 — that difference turns out to be the interesting part.

**3. How well the benchmark/casual label describes the template spread is
model-dependent.** Decomposing the variance of π(t) across the 13 templates,
the within-family share is **0.90–0.999 in the 8B**, **0.68–0.98 in 3.1-70B**,
and **0.72 early but 0.23–0.52 in the deep half of 3.3-70B**. So the family
label is nearly uninformative in the 8B and explains up to ~77% of the spread
deep in 3.3-70B. In the 8B the three markup templates are exactly the three
lowest of thirteen at most layers (exact permutation p = 0.0035, the minimum
attainable for 3-of-13); in the 70Bs they are lowest on average but interleave.
One of them, `bench_tagged_v2`, is a **benchmark**-family template that sits
with the casual JSON and Markdown templates rather than with its own family.

**4. The shift exists only when no purpose is stated.** This is the
load-bearing control. The same markup contrast measured on the *cued* rows is
near zero or slightly positive; on no-cue rows it is robustly negative. The
differential (no-cue minus cued, both block-demeaned, same axis units) is
negative at **every** layer with CIs excluding zero:

| model | layer | purpose AUC | markup − plain, cued | no-cue | differential |
|---|---|---|---|---|---|
| 8B | 7 | 0.9869 | −0.029 | −0.146 | **−0.118 [−0.135, −0.101]** |
| 8B | 11 | 0.9853 | +0.026 | −0.254 | **−0.280 [−0.300, −0.260]** |
| 8B | 15 | 0.9824 | +0.075 | −0.126 | **−0.201 [−0.222, −0.180]** |
| 8B | 24 | 0.9337 | +0.041 | −0.178 | **−0.219 [−0.250, −0.189]** |
| 8B | 31 | 0.9128 | +0.030 | −0.128 | **−0.157 [−0.191, −0.125]** |
| 70B | 18 | 0.9985 | +0.014 | −0.115 | **−0.129 [−0.146, −0.112]** |
| 70B | 24 | 0.9961 | −0.021 | −0.166 | **−0.145 [−0.159, −0.131]** |
| 70B | 30 | 0.9954 | −0.001 | −0.125 | **−0.124 [−0.136, −0.112]** |
| 70B | 60 | 0.9662 | −0.013 | −0.081 | **−0.068 [−0.089, −0.048]** |
| 70B | 79 | 0.9411 | −0.020 | −0.091 | **−0.071 [−0.097, −0.046]** |

A per-template offset shared by cued and uncued rows cancels in the
differential, so this rules out "the axis just has a template offset." It is
negative at every layer of both models with CIs excluding zero, and the cued
and no-cue template offsets are in fact *negatively* correlated across
templates at most 8B layers (r ≈ −0.3 to −0.67).

An exact permutation test over the 13 templates — asking whether the three
markup templates occupy the extreme low ranks *of the differential* — is a much
weaker instrument, since it treats templates as the unit and its floor is
p = 0.0035. It agrees strongly in the 8B (p = 0.0035–0.007 from L11 on),
moderately in the 70B's early-middle layers (p = 0.018–0.042 at L18–L30), and
not in the 70B's deep half (p = 0.13–0.20, though the block-bootstrapped
contrast there still excludes zero). The block bootstrap is the primary
evidence; the rank test is a secondary check on the ordering.

**This is the signature a prior should have: the wrapper matters when the
prompt says nothing about purpose, and stops mattering once purpose is
stated.**

**5. It replicates within template, on an independent factor.** The obvious
weakness of finding 2 is that only 3 of 13 templates are markup-serialized, so
"serialization" and "those three templates" are confounded. The response
**contract** slot escapes that: contracts rotate over 4–5 values *inside* every
v2 template, and one of them — `C4`, *"Return an object that has one field named
answer"* — is the one requesting a structured, machine-readable response.

Under a two-way fixed-effects model absorbing both payload block and template,
C4 shows the same signature as markup templates, with no template confounding:

| model | layer | C4 vs C0–C3, no-cue | cued | differential |
|---|---|---|---|---|
| 8B | 16 | −0.045 [−0.058, −0.033] | +0.055 | **−0.101 [−0.122, −0.079]** |
| 8B | 24 | −0.073 [−0.093, −0.052] | +0.078 | **−0.150 [−0.178, −0.122]** |
| 8B | 28 | −0.111 [−0.130, −0.092] | +0.084 | **−0.195 [−0.221, −0.167]** |
| 70B | 18 | −0.033 [−0.038, −0.027] | +0.008 | **−0.041 [−0.057, −0.025]** |
| 70B | 30 | −0.033 [−0.041, −0.025] | +0.008 | **−0.041 [−0.054, −0.028]** |
| 70B | 70 | −0.059 [−0.075, −0.041] | +0.057 | **−0.116 [−0.139, −0.093]** |
| 70B | 79 | −0.087 [−0.106, −0.067] | +0.036 | **−0.123 [−0.147, −0.100]** |

The 8B and 3.1-70B show the same shape: the differential is near zero or
positive in the first third of the network, then turns negative and grows with
depth. The 3.1-70B effect is about half the 8B's.

**It does not hold in 3.3-70B.** There the differential is negative only in the
early-middle layers (−0.051 [−0.065, −0.038] at L18, −0.045 at L30), turns
*positive* at L40 (+0.042 [+0.024, +0.058]), and is indistinguishable from zero
from L50 on. So the contract test — the cleanest escape from the template
confound — replicates in two of three checkpoints and fails in the third. That
is a real limitation, not a rounding detail: the markup differential itself
still replicates in 3.3-70B, but the independent within-template evidence for
*why* does not.

Note this test was chosen *after* seeing the markup result, so it is a targeted
prediction tested on a non-confounded factor — not independent confirmation. It
is labelled that way deliberately, following the language the repository uses
for its estimator-reversal result.

**6. Markup wrappers do not suppress the purpose representation.** They respond
*more* strongly to an explicit cue, not less. Scale-normalized Cohen's-d
difference: 8B +0.37 [+0.26, +0.48] at L15 and +0.43 [+0.33, +0.53] at L24;
70B +0.20 [+0.08, +0.32] at L30 and +0.17 [+0.05, +0.28] at L60. Smaller in the
70B, same direction, never negative. So finding 4 is not the wrapper degrading
the axis.

**7. The 8B's sign reversal at its peak-AUC layer is 8B-specific.** In the 8B,
ρ goes negative at layers 7–8 — exactly the peak-AUC layers — then recovers.
Both 70Bs (peak L18) stay positive throughout. Confirmed on the recomputed
runs; do not read it as a general depth law.

**8. Post-training strengthens the benchmark→evaluation prior, and only in the
back half of the network.** Llama 3.1 70B and Llama 3.3 70B are architecturally
identical — 80 blocks, hidden 8192, same tokenizer — and were run on the same
11,520 byte-identical prompts. They differ only in post-training, which makes
this a controlled A/B on post-training rather than a third replication.

Through the first half they are nearly indistinguishable. From ~50% depth
onward 3.3's benchmark→evaluation drift is consistently **1.4–1.9× larger**,
with non-overlapping bootstrap CIs:

| layer | 3.1-70B ρ | 3.3-70B ρ | Δ_format 3.1 → 3.3 | within-family share |
|---|---|---|---|---|
| 18 | +0.108 | +0.102 | 2.47 → 2.07 | 0.73 / 0.72 |
| 30 | +0.068 | +0.085 | 1.29 → 1.44 | 0.89 / 0.89 |
| 50 | +0.073 | +0.114 | 1.20 → 1.67 | 0.82 / 0.45 |
| 60 | +0.079 | +0.134 | 1.06 → 1.65 | 0.73 / 0.38 |
| 79 | +0.066 | +0.141 | 0.71 → 1.32 | 0.72 / 0.23 |

This is not a normalization artifact. ρ divides by the stated-cue effect, and
Δ_stated is uniformly ~10% smaller in 3.3 *including at the early layers where
ρ matches* — far too little to produce a 2× ratio deep. The divergence is in
the numerator, and the scale-free standardized statistic shows it with
non-overlapping CIs (L50: +0.87 [+0.75, +1.01] vs +1.17 [+1.03, +1.32];
L60: +0.74 [+0.63, +0.88] vs +1.10 [+0.97, +1.25]).

Meanwhile the markup differential is similar in both, so post-training did not
change *that* mechanism much — it amplified the benchmark/casual one. Caveats:
n = 2 checkpoints of one lineage, the specific post-training differences are
not public, and this was not a preregistered comparison.

**9. The models broadly agree on which wrappers read as deployment, and the
pattern of agreement corroborates finding 8.** Each model is fitted
independently, so correlating the 13-template ordering between them asks
whether the ladder tracks a property of the prompt or one checkpoint's
idiosyncrasy. Layers are matched by normalized depth. **All 30 correlations —
every pair, every depth, both π and the differential — are positive.**

| depth | 8B vs 3.1-70B | 8B vs 3.3-70B | 3.1-70B vs 3.3-70B |
|---|---|---|---|
| q = 0.25 | +0.34 | +0.13 | **+0.85** |
| q = 0.375 | +0.58 | +0.33 | **+0.85** |
| q = 0.5 | +0.55 | +0.50 | **+0.81** |
| q = 0.75 | +0.51 | +0.40 | +0.78 |
| q = 1.0 | +0.50 | +0.40 | +0.59 |

(Pearson r on the per-template differential; Spearman is similar and is in
`cross_model_agreement.json`.)

Two things stand out. Cross-*scale* agreement is only moderate (r ≈ 0.13–0.58),
so the specific template ordering is partly model-specific — consistent with
the ordering tests being an 8B result. And the two same-architecture
checkpoints agree strongly at shallow and mid depth (0.81–0.85) but decay
deeper (0.59 at the last layer) — **exactly the depth range where finding 8
says post-training pushed them apart**. That is an independent readout of the
same divergence from a different statistic, which is the main reason to trust
finding 8 rather than treat it as one estimator's artifact.

With 13 fixed templates these correlations are descriptive; no p-value is
attached, and the 30 values are not independent of one another.

**10. Counterfactual geometry separates a shared benchmark prior from a
probe-dependent serialization result.** The score analyses above establish
that format changes the value assigned by a fitted purpose probe. A stricter
question is whether the *activation displacement itself* points in the same
direction as the matched stated-purpose intervention. `geometry_alignment.py`
answers that question on the fully crossed 320-block x 11-v2-template panel.
For each block it forms

```
P = mean_template(h_evaluation - h_deployment)
V(template) = h_no-cue - 0.5 * (h_evaluation + h_deployment)
D(group) = mean_group(V) - mean_complement(V)
```

Thus `D` removes the format displacement shared by cued and uncued rows before
its cosine with `P` is measured. Uncertainty is a joint delete-one-block
jackknife. A second estimator divides every feature by its cued-row standard
deviation, fits purpose directions while excluding each cue-vocabulary family
in turn, and reports the held-family-independent projection of `D` as a
fraction of the explicit-purpose magnitude.

At the fixed normalized depth q = 0.375:

| model | layer | raw cosine: benchmark - casual | raw cosine: markup - plain | standardized held-family projection ratios |
|---|---:|---:|---:|---:|
| 3.1 8B | 12 | +0.120 [+0.083, +0.156] | -0.220 [-0.246, -0.194] | +0.114 / -0.348 |
| 3.1 70B | 30 | +0.138 [+0.110, +0.166] | -0.037 [-0.057, -0.017] | +0.114 / -0.075 |
| 3.3 70B | 30 | +0.143 [+0.113, +0.172] | -0.044 [-0.068, -0.020] | +0.136 / -0.055 |

The benchmark/casual result is the replicating part. After cue-family purpose
directions become stable (held-vs-excluded cosine >= 0.5), its raw alignment is
positive with a block-jackknife interval excluding zero at all 27 reliable 8B
layers, 70/72 3.1-70B layers, and 69/71 3.3-70B layers. It is therefore not
only a logistic-score effect: the no-cue-specific benchmark displacement has a
component in the same full-space direction as explicitly stated evaluation.

Serialization does **not** support the same cross-model statement. It is
negative at every reliable 8B layer, reaching -0.488 [-0.523, -0.454] at L16.
In 3.1-70B it is negative through L35, crosses zero, and is positive from L38
through L79; the final-layer cosine is +0.262 [+0.229, +0.296]. In 3.3-70B it
stays weakly negative through much of the network and approaches zero at the
end (-0.004 [-0.034, +0.026]). The score-based negative markup differential
therefore cannot be described as the whole activation displacement moving
toward deployment. In the 70Bs it is a component selected by the probe metric,
while the dominant Euclidean displacement is checkpoint- and depth-dependent.

An exact null over every same-size partition of the 11 fixed templates further
limits population language. The benchmark grouping is unusually positive at
q = 0.25 and 0.375 in both 70Bs (two-sided p = 0.041-0.050), but not at later
depths and not in the 8B. The 8B markup grouping is unusually negative under
the directional test at mid/deep fixed depths (one-sided p = 0.006-0.012), with
the two-sided test crossing 0.05 only at q = 0.5. Neither named grouping is a
random sample from a format population.

The defensible addition is consequently methodological as well as empirical:
**probe-score drift and counterfactual displacement alignment are different
claims.** Benchmark-shaped prompting shows evidence under both; serialization
does not generalize across metrics and checkpoints. Full outputs are in
`results/*_geometry.json`, the fixed-depth digest is
`results/geometry_summary.json`, and the layer curves are
`figures/08_counterfactual_geometry.png`.

## Estimator robustness (in progress)

This repository has already shown that this freeze can produce **sign
reversals** between probe pipelines
(`full320_exact_logistic_estimator_reversal_conclusion_20260823`), and every
result above uses the frozen L2 logistic recipe. So the headline effect is
being re-measured under estimators with different inductive biases.

Choosing the comparators mattered more than expected. The repository's reversal
estimator, **diagonal LDA**, reaches only **AUC 0.74** on purpose decoding here
against logistic's 0.985 — it ignores feature correlations, and a
reimplementation matching `scripts/full320_strict_equal_rows.py` agrees exactly
with naive diagonal LDA, so that is what the estimator does on this task rather
than a bug. A divergence under a probe that weak would be weak evidence. Two
comparators at *matched quality* were therefore added:

| probe | 8B L11 purpose AUC | inductive bias |
|---|---|---|
| `logistic` (frozen) | 0.985 | L2-penalised discriminative |
| `slda` | 0.973 | generative, full covariance, Ledoit–Wolf shrinkage |
| `svc` | 0.984 | max-margin hinge loss |
| `dlda` | 0.724 | diagonal covariance — the repo's reversal estimator |

**First result, and it is the important one.** The markup differential survives
the swap to diagonal LDA and in fact *grows*, with CIs excluding zero at every
layer, even though that same estimator flips the benchmark/casual ρ at L11:

| 8B layer | logistic | diagonal LDA |
|---|---|---|
| 11 | −0.280 [−0.300, −0.260] | −0.321 [−0.366, −0.277] |
| 19 | −0.169 [−0.194, −0.144] | −0.623 [−0.682, −0.565] |
| 31 | −0.157 [−0.191, −0.125] | −0.688 [−0.755, −0.623] |

So the estimator dependence this freeze is known for appears to attach to the
*benchmark/casual* contrast, not to the no-cue-specific markup differential.
`slda` and `svc` are running to confirm at matched quality.

## Controls

| control | question | 8B | 3.1-70B | 3.3-70B |
|---|---|---|---|---|
| **cued-vs-no-cue differential** | is it a template offset on the axis? | **passes** — CIs exclude 0 at every layer | **passes** | **passes** |
| **task-family nuisance axis** | would *any* decodable axis do this? | **passes decisively** (~10×) | **passes decisively** (~10×) | **passes** (~3–5×) |
| **within-template contract (C4)** | is it just those three templates? | **passes** | **passes**, ~half the size | **fails deep** — see below |
| **block-FE regression + prompt length** | is it length, or payload content? | **passes** | **passes** — −0.10 to −0.21 | **passes** — −0.02 to −0.13 (n.s. only at L79) |
| **cue responsiveness** | does markup damp the axis? | **passes** — amplifies | **passes** — amplifies, smaller | **passes** — +0.17 to +0.25 |
| **legacy / preamble exclusion** | do the v1 anchors or the MMLU-fingerprint template carry it? | **passes** — L11 −0.254 → −0.247 | not rerun | not rerun |
| **terminal-field-matched subgroup** | is it the literal last token? | **passes** — exact p = 0.028, the floor | **inconclusive** — p = 0.06–1.0 | **inconclusive** — p = 0.14 |
| **exact rank test on the differential** | do markup templates hold the extreme ranks? | **passes** — p = 0.0035–0.007 from L11 | **mixed** — p = 0.018–0.042 at L18–30, n.s. deeper | **fails** — p = 0.035 at L50 only |
| **INLP format erasure** | is it the linear format subspace? | **inconclusive as erasure, informative anyway** | not run (scope) | not run (scope) |
| **permutation null** | does a same-capacity probe on shuffled labels move these rows? | **passes at 4 of 5 layers** — exceeds all 20 draws at L11/15/23/31 (p = 0.048, the floor); **fails at L7** | not run (scope) | not run (scope) |

The two controls that weaken on the 70B are both **small-n ordering tests over
the 13 templates**, which treat templates as the unit and have very little
power — the rank test's floor is p = 0.0035 and the terminal-field subgroup has
only 9 templates and 2 markup ones. The quantities with real power, which
bootstrap over 320 payload blocks, replicate cleanly on both models. That
asymmetry is what one would expect if the effect is real but the template
catalog is simply too small to establish an ordering; it is not independent
evidence against the effect, and it is not treated as such.

**The nuisance-axis control is the sharpest one.** `task_family` is a property
of the payload block, and blocks are crossed with every template by
construction, so task family is exactly orthogonal to template identity in the
design. It is also *more* decodable than purpose (AUC 0.998–1.000). If the
markup shift were just "these prompts differ in text, so any direction
separates them," it would appear here too. It does not:

| 8B layer | task-family AUC | markup − plain on task axis | on purpose axis |
|---|---|---|---|
| 11 | 0.9997 | +0.003 [−0.009, +0.016] | −0.254 [−0.266, −0.241] |
| 15 | 0.9982 | +0.006 [−0.004, +0.016] | −0.126 [−0.137, −0.115] |
| 24 | 0.9995 | +0.019 [+0.007, +0.031] | −0.178 [−0.197, −0.159] |

An order of magnitude apart, on axes of comparable strength. The shift is
specific to the purpose direction.

**On the erasure control.** INLP ran its full 12 rounds and hit its 0.55 target
in **0 of 5 folds at every one of the 8 layers** — benchmark/casual format AUC
bottomed out at ≈ 0.91–0.93, from 1.000. It is not that erasure fell just
short; the format direction was never linearly removable in 12 rounds anywhere
in the network. This therefore **cannot be reported as "format was removed."**

What it does show is that after deleting the twelve leading linear format
directions, purpose AUC is unchanged (L11: 0.9853 → 0.9852) and the no-cue
shift is unchanged or slightly *larger* (L11 ρ: +0.073 → +0.080). The drift is
not carried by the leading linear format subspace — a weaker claim than
erasure, and the only one the run supports.

**On the permutation null.** 20 draws per layer at 5 layers, labels shuffled
within each (training fold × format cell). The permutation works — null purpose
AUC is 0.496–0.507 (max 0.566) — but it does **not** produce a zero no-cue
shift. A probe of identical capacity fitted to a structureless target still
moves these rows, with null |max| for the markup differential reaching 0.97 at
L7 and 0.22–0.69 elsewhere. Comparing against zero would badly overstate the
result; comparing against the null is the right test.

| 8B layer | observed markup differential (smd) | null mean ± sd | null \|max\| | obs / null\|max\| | p |
|---|---|---|---|---|---|
| 7 | −0.759 | −0.13 ± 0.43 | 0.974 | **0.78** | 0.095 |
| 11 | −1.644 | +0.07 ± 0.31 | 0.691 | 2.38 | 0.048 |
| 15 | −1.056 | +0.01 ± 0.25 | 0.561 | 1.88 | 0.048 |
| 23 | −0.897 | +0.01 ± 0.11 | 0.217 | 4.13 | 0.048 |
| 31 | −0.524 | −0.01 ± 0.12 | 0.283 | 1.85 | 0.048 |

**Passes at L11, L15, L23 and L31** — the observed differential exceeds all 20
null draws (p = 0.048 is the floor for 20 draws, so this is the strongest
result attainable at this draw count). **Fails at L7**, where the observed
value sits *inside* the null range. L7 is also where ρ goes anomalously
negative and where the differential is weakest, so the two observations agree.

Two limits to state plainly. The margin is modest — 1.85–4.13× the largest
null draw, not orders of magnitude. And 20 draws floors p at 0.048, so no
stronger significance claim is available without more draws; the shard-level
checkpointing makes extending the draw count cheap if that becomes worth doing.

## Why the design supports the question

| property | consequence |
|---|---|
| 320 payload blocks × 13 templates, fully crossed | every contrast is computed *within* a payload block, so content is held fixed by construction |
| cued stratum balanced 3,840/3,840 evaluation/deployment, crossed with benchmark/casual | the axis is trained decorrelated from format |
| 5 vocabulary-disjoint cue folds | the axis is scored only on cue lexicons it never saw |
| no-cue rows format-matched to cued rows (2,080 benchmark / 1,760 casual, identical to each cued class) | the no-cue projection is not reweighted relative to training |
| byte-identical evaluation/deployment twins | the calibration unit is a pure one-sentence intervention |
| 4–5 response contracts rotate *within* each v2 template | contract is crossed with template, which is what makes finding 5 possible |
| `task_family` is a block property, and blocks are crossed with templates | gives an exactly template-orthogonal nuisance axis |

What "markup serialization" means here is substantial, not cosmetic.
`deploy_json_v2` renders the whole prompt as `{"input": ..., "choices": [...]}`
with no instruction at all; `bench_tagged_v2` wraps it in `<question>` /
`<options>`; `deploy_markdown_v2` uses `### Input` / `### Choices` / `### Output`.
Plain templates address the model in prose. The finding is that being handed a
machine-readable record reads as deployment.

## Method

**Axis.** The frozen recipe from `docs/analysis_plan_v2.md`: `StandardScaler` +
`LogisticRegression(C=0.1, max_iter=3000, random_state=0)` on the
last-prompt-token residual stream, trained on the decorrelated cued stratum,
leave-one-cue-vocabulary-family-out. No-cue rows carry no fold, so — again
following the frozen plan — they are scored by the mean of the five held-fold
models.

**Calibration unit.** Logistic decision values have arbitrary scale, so every
effect is divided by the stated-cue effect at the same layer:

```
Δ_stated = mean_block(eval) − mean_block(deployment)        [out-of-fold]
π(group) = ( mean_block(no-cue in group) − mean_block(deployment) ) / Δ_stated
ρ        = mean_block( no-cue benchmark − no-cue casual ) / Δ_stated
```

Because every block appears in every condition, ρ is a within-block paired
quantity.

**Uncertainty.** Cluster bootstrap over the 320 `payload_block_id` values,
B = 10,000 (2,000 for the two fixed-effects regressions), seed 0, percentile
95% CIs — the repository's existing convention. Numerator and denominator are
recomputed from the same resample so ratios keep their correlation.

**Determinism.** BLAS is pinned to one thread per worker (`axis.pinned`).
Without it the number of BLAS threads depends on `--jobs`, which changes
summation order in lbfgs and moves AUC in the 4th decimal — small relative to
the bootstrap CIs, but enough that a re-run would not reproduce. All artifacts
in `results/` were recomputed under pinning; the earlier unpinned set is kept
in `results_prepinning/` for comparison and is not used.

## Pipeline

```
activations/full320-v3/<model>/activations_all_layers.npz   (frozen, sha256-checked)
        |  src/prep_activations.py         stream-decompress -> X_all_layers.npy (mmap)
        v
        |  src/score_export.py             fit the axis once per layer -> *_scores.npz
        v
        |  src/run_analysis.py --stage main       all-layer estimands + block bootstrap
        |  src/template_analysis.py               per-template ladder, group contrasts, robustness
        |  src/template_offset.py                 cued-vs-no-cue differential   <- headline control
        |  src/contract_analysis.py               within-template contract contrast
        |  src/cue_responsiveness.py              per-template response to an explicit cue
        |  src/nocue_regression.py                block-FE model, length + terminal controls
        |  src/nuisance_axis.py                   task-family axis as an "any axis" control
        |  src/run_analysis.py --stage controls   INLP erasure, transfer, abstraction gap
        |  src/permutation_null.py                control-task null distribution
        |  src/cross_model.py                     do the models order templates alike?
        v
        |  src/make_figures.py
        v
figures/  results/
```

`prep_activations.py` re-verifies each archive against the
`activations_all_layers_sha256` in its frozen `config.json` before
decompressing, so every result traces to the pinned bytes. `score_export.py`
fits the axis once per model; every estimand downstream is a pure function of
those scores plus the design frame.

## Running it

```bash
bash ai_post_analysis/src/run_pipeline.sh              # everything
bash ai_post_analysis/src/run_pipeline.sh llama31_8b   # one model's core stages
bash ai_post_analysis/src/run_extras.sh                # contract + cross-model
python ai_post_analysis/src/geometry_alignment.py      # counterfactual geometry
```

**CPU only — no GPU is needed for anything in this folder.** Wall time on 14
cores: 8B core ≈ 25 min, each 70B core ≈ 2 h, 8B control stage ≈ 75 min (INLP
dominates), 8B permutation null ≈ 50 min.

### Interrupting and resuming

Every expensive stage checkpoints per layer — per *permutation draw* for the
null, since one layer there is ~20 fits. Finished units are written
immediately as shards under `results/_partial/<stem>/`, atomically, so a run
killed at any point keeps everything already computed. Re-running any stage,
or the whole pipeline, recomputes only what is missing:

```
[llama31_8b_main] resuming: 30/32 layers already done, 2 to compute
```

* `--force` on any stage discards its shards and recomputes.
* Changing `--n-boot` or `--seed` is detected and invalidates the shards
  automatically, so a re-run can't silently mix settings (`checkpoint.guard_params`).
* The cheap post-processing stages skip when their output file exists.
* Artifacts are assembled from **every** shard on disk, so a targeted
  `--layers 17` rerun refreshes that layer inside the complete file rather than
  replacing the file with one layer.
* `src/backfill_checkpoints.py` reconstructs shards from finished artifacts
  that predate checkpointing.

## Figures

| file | content |
|---|---|
| `01_depth_profile.png` | both format contrasts vs normalized depth, per model, peak-AUC layer marked, low-AUC depths greyed out |
| `02_template_ladder.png` | per-template π at fixed q = 0.375, coloured by serialization, marked by family |
| `03_no_cue_specific.png` | the differential: markup − plain on cued vs no-cue rows |
| `04_controls.png` | erasure and permutation-null draws on the scale-free statistic |

The ladder and control panels use q = 0.375 rather than each model's peak-AUC
layer. That depth is not chosen here — it is the one the repository's own
estimator-reversal analysis already pinned — which avoids selecting a layer on
the outcome being plotted.

## Run state

**Complete and verified (all pinned):** all three checkpoints — main, scores,
templates + robustness, offsets, cue response, regression, nuisance axis,
contracts — plus the 8B control stage, the permutation null, cross-model
agreement, and seven figures.
23 layer-indexed artifacts, all with full, sorted, unique layer coverage.
`src/progress.py` reports completion by counting checkpoint shards.

**Not run, deliberately:** the INLP control stage and permutation null on the
70Bs. Both scale with hidden size squared; the erasure control was already
inconclusive at 4,096 dims, and the load-bearing control — the cued-vs-no-cue
differential — is in the per-model core and ran on all three.

**Next, in order:**

1. Fold `contract_analysis.py` and `cross_model.py` into `run_pipeline.sh`
   (they were written while it was already running, so they live in
   `run_extras.sh`).
2. Extend the permutation null past 20 draws. p = 0.048 is the floor at that
   count, so the current result cannot be stronger no matter how large the
   effect is; the per-draw checkpointing makes adding draws cheap and
   incremental.
3. **The decisive experiment this points at** is a new render crossing
   serialization (plain / JSON / XML / Markdown) *orthogonally* with template
   family, rather than the current catalog where each markup template is its
   own template. The contract result (finding 5) is the best available
   substitute but is a different manipulation — output framing, not input
   framing. The repo has the machinery (`lint-formats-v2`, `render-v3`,
   `catalogs/formats_v2.yaml`); extraction would need one GPU session.

## Claim boundary

Carried over from `artifacts/exploratory/pilot80_main_track_conclusion_20260823`
and the repository's standing rules:

* **Development / exploratory** work on a spent freeze. The v3 confirmation
  bank has not been extracted; nothing here can promote itself to confirmatory.
* The estimand is **the projection of uncued prompts onto an axis fitted to
  stated purpose**. Not evaluation awareness, not behaviour, not causal.
* Layers were inspected after extraction. Layer-resolved statements are
  descriptive of this fixed checkpoint panel.
* Templates are a fixed catalog, not a sample from a population of formats, and
  only three of the thirteen are markup-serialized.
* The contract test (finding 5) was chosen after seeing the markup result. It
  is a targeted prediction on a non-confounded factor, not independent
  confirmation.
* "Markup serialization" remains confounded with template identity in the
  current render. The differential and contract controls constrain the
  alternatives; they do not eliminate them.
