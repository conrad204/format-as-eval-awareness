# binding_v1: response purpose as a factorial relational binding

Status: development evidence. Frozen dataset, pre-registered analysis structure,
3/3 checkpoints. Inference unit is the relational-template **family** (n = 8),
never the individual prompt.

## What the experiment does

The earlier controlled study **states** the response purpose. `implicit-v2`
replaced the statement with an implied context, but its two classes still
contrasted broad semantic clusters (offline / historical / simulated versus
live / current / operational). A reviewer could therefore argue that the
learned "purpose" direction merely overlaps a global
`offline/comparison <-> live/action` semantic axis.

`binding_v1` removes that explanation **by construction**. Every prompt
contains BOTH use-role descriptions:

- a measurement role whose consequences are discarded, and
- an operational role whose consequences persist.

Two arbitrary nonce routes are named. The design crosses

- `m` = which route holds which role,
- `s` = which route the current response belongs to,
- `f` = benchmark versus casual response format,

and the label is the interaction

```
p = m * s
```

so the measurement role is bound to the current response in exactly half the
cells. Consequently evaluation-like words, deployment-like words, offline/live
semantics, either route name, the role assignment alone, and the selected route
alone are **all** uninformative. Only the relation predicts the label.

Frozen bank: `results/binding_v1_frozen/`
(`items.jsonl` SHA-256 `bdfd3d4360d1704511f7c3451432c5127b1dbecad75e73654ab4d301f1018efd`),
8 families x 64 payloads x 2 mappings x 2 routes x 2 formats = **4096 items**,
frozen and committed **before** any activation existed.

### Dataset audit (`results/binding_v1_frozen/audit_report.json`)

Exact factorial balance holds: every mapping x selection x format cell occurs
exactly once per payload, all 512 blocks have 8 items, and the label equals the
deterministic interaction for all 4096 items. Under
leave-one-entire-family-out CV, **every** leakage baseline is exactly
**0.500**: word 1-2gram TF-IDF, character 3-5gram TF-IDF, setup-text-only
1-3gram, nonce-route identity, mapping alone, selection alone, format alone,
and payload identity. Role-description strings and nonce route tokens are
purpose-balanced by construction. Name-mention counts are also balanced: every
cell contains one route named twice and one named once, so no count cue exists.

## Results

Extraction reuses the paper's pipeline unchanged (same chat template, same
last-prompt-token position, same block-output convention, same pinned
checkpoints). Llama-3.1-8B: all 32 layers. Both 70B models: pre-specified
validation layer 18, inherited from the old explicit study.

### A. Frozen explicit purpose probe -> bound purpose: fails at the pre-specified layer, but transfers in the deeper half at 70B

The persisted explicit-purpose probes (joint, benchmark-trained,
casual-trained; refit only from the original explicit rows, never from
binding_v1) were first evaluated at the **pre-specified** validation layer 18
inherited from the explicit study. There the result is flatly negative:

| model | layer | family-mean AUC | family bootstrap CI | families > 0.5 | sign test |
|---|---|---|---|---|---|
| Llama-3.1-8B | 18 | 0.527 | [0.504, 0.562] | 6/8 | 0.145 |
| Llama-3.1-70B | 18 (pre-specified) | 0.506 | [0.496, 0.516] | 7/8 | 0.035 |
| Llama-3.3-70B | 18 (pre-specified) | 0.506 | [0.496, 0.517] | 4/8 | 0.637 |

Full-depth extraction of both 70B models changes this picture. Layer 18 is only
22.8% of an 80-layer network, and the effect lives in the back half:

| model | relative depth | family-mean AUC | family bootstrap CI | families > 0.5 |
|---|---|---|---|---|
| Llama-3.1-70B | 50% (L40) | 0.582 | [0.529, 0.634] | 7/8 |
| Llama-3.1-70B | 70% (L56) | 0.569 | [0.517, 0.620] | 7/8 |
| Llama-3.1-70B | 100% (L79) | 0.587 | [0.541, 0.632] | 7/8 |
| Llama-3.3-70B | 50% (L40) | 0.587 | [0.526, 0.659] | 7/8 |
| Llama-3.3-70B | 60% (L48) | 0.602 | [0.539, 0.672] | 7/8 |
| Llama-3.3-70B | 90% (L72) | 0.581 | [0.523, 0.649] | 7/8 |

In both 70B checkpoints a broad contiguous band from roughly 45% depth to the
final layer shows family-mean AUC around 0.57-0.62 with family bootstrap CIs
excluding 0.5 and 7/8 families above chance. The best single layers are 0.601
[0.552, 0.647] (Llama-3.1-70B L78) and 0.624 [0.554, 0.703] (Llama-3.3-70B L44),
both with 7/8 families and exact family sign-test p = 0.035.

**Two honesty caveats.** First, only layer 18 was pre-registered; the deeper
band was examined after seeing results, so the "best layer" CI and p-value are
optimistically biased by selection on the same data. The defensible claim is the
broad band, not any single layer. Second, Llama-3.1-8B stays at chance at
**every** one of its 32 layers (maximum family-mean AUC 0.527), so this is a
70B-only effect, not a general one.

The paired factorial interaction statistic
`I = (s(+,+) - s(+,-) - s(-,+) + s(-,-)) / 4` follows the same depth pattern.

For contrast, the same frozen probes on `implicit-v2` reach family-mean AUC
0.60-0.72 at shallow depth in the 8B alone. Since `binding_v1` balances away the
global offline/live semantic axis, the reading is that much of the earlier
shallow fixed-probe transfer rode on that broad axis, while a weaker but real
bound-purpose component survives in the deeper half of the 70B models.

### B. Bound purpose is decodable from held-out relational families

New probes trained on binding_v1 with leave-one-entire-family-out CV (identical
recipe: train-only `StandardScaler`, `LogisticRegression(C=0.1)`, seed 0)
recover the interaction well above chance.

| model | layer | joint AUC | bench->casual | casual->bench | transfer gap | families gap > 0 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 14 (45%) | 0.802 | 0.688 | 0.656 | +0.130 | 8/8 |
| Llama-3.1-8B | 31 (100%) | 0.703 | 0.621 | 0.592 | +0.096 | 8/8 |
| Llama-3.1-70B | 18 (23%) | 0.748 | 0.585 | 0.561 | +0.175 | 8/8 |
| Llama-3.1-70B | 30 (38%) | 0.939 | - | - | +0.089 | 8/8 |
| Llama-3.1-70B | 79 (100%) | 0.879 | - | - | +0.106 | 7/8 |
| Llama-3.3-70B | 18 (23%) | 0.793 | 0.621 | 0.584 | +0.191 | 8/8 |
| Llama-3.3-70B | 38 (48%) | 0.966 | - | - | +0.084 | 8/8 |
| Llama-3.3-70B | 79 (100%) | 0.916 | - | - | +0.162 | 8/8 |

At full depth the 70B models decode the bound purpose far better than layer 18
suggested: peaks of **0.939** (Llama-3.1-70B, L30) and **0.966**
(Llama-3.3-70B, L38). Decoding is near chance through roughly the first fifth of
each network and rises sharply to a mid-depth peak. **The original
format-conditioned transfer phenomenon reappears everywhere**: the cross-format
mean AUC stays well below the joint-format AUC, and the gap is positive in 8/8
families at essentially every layer past the onset in all three checkpoints.

### C. Factorial geometry

Matched contrasts over complete factorial cells give the mapping main effect
`M`, selected-route main effect `S`, format main effect `F`, the bound-purpose
interaction `P = mean over cells of (m*s) h`, and `C = m*s*f`.

| layer | \|P\| | \|M\| | \|S\| | \|F\| | cos(P, explicit probe) | cos(P_bench, P_casual) | mean cos(P) across families |
|---|---|---|---|---|---|---|---|
| 8 | 0.002 | 0.00 | 0.00 | 0.70 | +0.022 | +0.61 | -0.01 |
| 13 | 0.041 | 0.00 | 0.00 | 1.08 | +0.019 | +0.92 | +0.27 |
| 18 | 0.120 | 0.01 | 0.01 | 1.54 | +0.022 | +0.91 | +0.29 |
| 31 | 0.705 | 0.07 | 0.08 | 6.79 | +0.011 | +0.91 | +0.31 |

Three points. First, `|P|` grows steadily with depth while the additive route
and selection main effects stay near zero, which is the signature expected if
the network encodes the binding rather than its parts. Second, `P` is stable
across formats (cosine about 0.91 from layer 12 on) and only partially shared
across families (about +0.3), matching the family-level variability in B.
Third, `cos(P, frozen explicit probe)` is about +0.02 at every depth: the bound
purpose direction is nearly orthogonal to the old explicit-purpose direction,
which independently explains why A fails. Format dominates the representation
in norm at every depth.

Design orthogonality does not imply geometric orthogonality, and cosine alone
does not establish semantic identity.

### D. A format-associated subspace explains most of the cross-format gap

A format basis built only from matched benchmark-minus-casual block contrasts of
the **training** families, cross-fitted leave-one-family-out and projected out
with `I - QQ^T` before refitting the binding probes:

Llama-3.1-8B, all-layer sweep:

| layer | raw gap | rank 16 gap | rank 64 gap | rank-64 delta gap (family CI) | families reduced | rank-64 joint AUC |
|---|---|---|---|---|---|---|
| 12 | +0.114 | +0.090 | +0.017 | -0.096 [-0.138, -0.051] | 8/8 | 0.746 (raw 0.751) |
| 16 | +0.125 | +0.088 | +0.027 | -0.098 [-0.141, -0.057] | 8/8 | 0.772 (raw 0.776) |
| 20 | +0.128 | +0.089 | +0.033 | -0.096 [-0.138, -0.054] | 8/8 | 0.749 (raw 0.753) |
| 24 | +0.115 | +0.094 | +0.022 | -0.093 [-0.141, -0.045] | 8/8 | 0.715 (raw 0.724) |
| 31 | +0.096 | +0.069 | +0.024 | -0.072 [-0.109, -0.036] | 8/8 | 0.693 (raw 0.703) |

Both 70B checkpoints at the pre-specified validation layer 18:

| model | raw gap | rank 16 gap | rank 64 gap | rank-64 delta gap (family CI) | families reduced | rank-64 joint AUC |
|---|---|---|---|---|---|---|
| Llama-3.1-70B | +0.175 | +0.109 | +0.042 | -0.134 [-0.185, -0.082] | 8/8 | 0.756 (raw 0.748) |
| Llama-3.3-70B | +0.190 | +0.132 | +0.043 | -0.147 [-0.195, -0.096] | 8/8 | 0.803 (raw 0.793) |

Removing a rank-64 format-associated subspace removes roughly 75-85% of the
cross-format transfer gap in 8/8 families in **3/3 checkpoints**, and the effect
is monotone in rank (rank 1 does almost nothing). Joint decoding of the bound
purpose is preserved throughout, and in both 70B models it improves slightly
(0.748 to 0.756 and 0.793 to 0.803), so the projection is not simply destroying
signal. Response-format variation therefore contributes substantially to
cross-format failure even when purpose is identified only as a relational
interaction.

## Plain answers

- **Did the old explicit-purpose probe generalize to the relationally bound
  purpose variable?** Partly, and only at scale and depth. At the pre-specified
  layer 18 and at every 8B layer it is at chance. In the deeper half of both 70B
  models it reaches about 0.57-0.62 family-mean AUC with CIs excluding 0.5 and
  7/8 families above chance, though that band was identified post hoc.
- **Was the effect consistent across template families?** For the frozen probe
  in the deep 70B band, 7/8 families. For binding-trained probes and for the
  transfer gap, 8/8 families.
- **Did it replicate across models?** The deep-band fixed-probe transfer
  replicates in 2/2 of the 70B models but is absent in the 8B. The positive
  binding-decodability and transfer-gap results replicate in 3/3.
- **Did benchmark/casual format still affect transfer?** Yes, strongly and in
  8/8 families in every model.
- **Did the format-associated subspace explain any of it?** Yes: roughly 75-85%
  of the gap at rank 64, in 8/8 families in 3/3 checkpoints, with joint decoding
  preserved (and slightly improved in both 70B models).

## What this establishes

1. A bound response-purpose variable, identified only by the interaction of
   role assignment with the selected response route, is linearly decodable from
   frozen activations and generalizes across entirely held-out relational
   framings in 3/3 checkpoints.
2. That decodability cannot be attributed to the presence of evaluation-like
   versus deployment-like semantic content, to either route name, to the role
   assignment alone, or to the selected route alone, because all of those are
   balanced by construction and every text and metadata baseline sits exactly at
   chance under family-held-out CV.
3. The format-conditioned cross-format transfer deficit reproduces in this
   relational regime, and a cross-fitted format-associated subspace accounts for
   most of it.
4. At 70B scale, a direction learned purely from **explicitly stated** purpose
   carries measurable information about the **relationally bound** purpose in
   the deeper half of the network (family-mean AUC about 0.57-0.62, 7/8
   families, CIs excluding chance), without any retraining. The alignment is
   weak, not an identity: `cos(P, explicit probe)` stays near +0.03-0.04, and
   the effect is absent in the 8B and at the pre-specified layer 18. So the two
   constructs partially overlap deep in large models rather than being the same
   direction.

## What this does NOT establish

- It is not proof of evaluation awareness, and not evidence that a model
  spontaneously knows it is being evaluated. `p` is an experimental variable,
  not an internal "awareness" quantity.
- It does not show that a single explicit-purpose direction transfers to
  inferred use roles. Here it did not.
- The bound variable may reflect a broader representation of consequence,
  counterfactuality, or operational role. That is not a defect if those
  properties are part of what defines the assigned use role, but no unique
  semantic essence of "purpose" is identified.
- Probing a frozen representation is not a behavioral endpoint and not a causal
  intervention. Per `RULES.md` this is internal representational evidence, not
  behavioral steering.
- Family-level inference rests on 8 independent relational families; the 4096
  prompts are matched replications, not 4096 independent contexts.

## Artifacts

- Frozen bank and audit: `results/binding_v1_frozen/`
- Analyses: `results/binding_v1_llama31_8b_analysis.json`,
  `results/binding_v1_llama31_70b_analysis.json`,
  `results/binding_v1_llama33_70b_analysis.json`
- Format subspace: `results/binding_v1_llama31_8b_format_subspace.json`,
  `results/binding_v1_llama31_70b_format_subspace.json`,
  `results/binding_v1_llama33_70b_format_subspace.json`
- Table: `results/binding_v1_primary_results.csv`
- Figure: `plots/jpp/binding_v1/binding_v1.png`
- Scripts: `scripts/binding_v1_build.py`, `scripts/binding_v1_audit.py`,
  `scripts/binding_v1_analyze.py`, `scripts/binding_v1_format_subspace.py`,
  `scripts/binding_v1_report.py`
- Launchers: `scripts/cluster_extract_binding_v1_8b_all_layers.sbatch`,
  `scripts/cluster_extract_binding_v1_70b_l18.sbatch`,
  `scripts/cluster_analyze_binding_v1.sbatch`,
  `scripts/cluster_binding_v1_format_subspace.sbatch`
