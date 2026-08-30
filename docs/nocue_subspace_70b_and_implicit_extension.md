# No-cue format subspace: 70B extension and implicit-v2 application

Companion to `docs/nocue_derived_format_subspace_result.md`, which established the
Llama-3.1-8B result and explicitly left two things uncovered: the two 70B
checkpoints (ruled out there on workstation RAM) and the application to
`implicit-v2`. Both were run on the cluster with the same scripts and are
reported here. Development evidence, not confirmatory.

## 1. The cue-free result replicates at 70B scale

`scripts/full320_v3_nocue_format_subspace.py`, unchanged, all 80 layers per
model, 1,000 payload-block bootstrap replicates. Endpoint is the same integrated
cross-format transfer gap `G` over normalized depth.

| model | raw `G` | r=1 | r=4 | r=16 | r=64 fraction removed (95% CI) |
|---|---:|---:|---:|---:|---|
| Llama-3.1-8B (prior work) | 0.07934 | -0.8% | 10.2% | 21.1% | **29.80%** [28.60, 31.22] |
| Llama-3.1-70B | 0.05495 | -13.2% | -5.3% | 10.8% | **25.82%** [24.15, 27.62] |
| Llama-3.3-70B | 0.06308 | -5.8% | +1.6% | 17.8% | **32.73%** [31.47, 34.03] |

Two things replicate in 3/3 checkpoints. A format-associated subspace estimated
with **zero purpose-statement information** removes a substantial, clearly
nonzero share of the integrated gap at rank 64 (26-33%), and the effect is
strongly rank-dependent: rank 1 does nothing or is slightly counterproductive,
and the effect only becomes clear from rank 16. The 70B raw gaps are smaller
than the 8B's, so this is a proportional reduction of a smaller quantity.

The prior document's quantitative haircut therefore stands unchanged and now
applies across scale: the cue-free basis explains a substantial minority of the
transfer failure, not all of it.

## 2. Removing the cue-free format subspace improves implicit cross-format decoding

`scripts/implicit_v2_nocue_subspace_control.py`. `Q_nocue` is built only from
original no-purpose benchmark-minus-casual contrasts and **excludes all 40
payload blocks used by implicit-v2**, so no implicit payload contributes to the
basis. Implicit probes remain leave-one-entire-scenario-family-out. Inference
unit is the scenario family (n = 6).

Llama-3.1-8B, rank 64:

| layer | projected joint AUC | projected cross-format | projected gap | change in gap (family CI) |
|---:|---:|---:|---:|---|
| 8 | 0.968 | 0.899 | +0.070 | **-0.088** [-0.099, -0.076] |
| 16 | 0.944 | 0.743 | +0.201 | **-0.043** [-0.056, -0.031] |
| 24 | 0.938 | 0.721 | +0.217 | **-0.054** [-0.063, -0.046] |
| 31 | 0.985 | 0.842 | +0.143 | **-0.053** [-0.060, -0.046] |

Rank monotonicity holds here too (at layer 8: -0.013 at r=1, -0.054 at r=4,
-0.059 at r=16, -0.088 at r=64). Joint-format decoding of the implicit context
is untouched (0.968 and 0.985 at layers 8 and 31, matching the unprojected
values), so the gap reduction is not bought by degrading the numerator.

At layer 8 the projection removes roughly 56% of the implicit cross-format gap
(+0.158 to +0.070); at mid and late depth it removes 20-27%. So the same
cue-free format-associated directions that reduce the *explicit* transfer gap
also reduce the *implicit* one.

## Claim boundaries

- Not evidence of spontaneous evaluation awareness; `Q_nocue` carries no
  purpose labels by construction.
- Frozen-activation projection plus probe refitting is not a behavioral causal
  intervention.
- Uncentered SVD, matching the repository; not centered PCA.
- The subspace is format-**associated**, not "pure format": as in the 8B
  analysis, format itself remains highly decodable after projection.
- The 8B row in table 1 is quoted from the prior document for comparison and was
  not recomputed here.

## Artifacts

- `results/nocue_format_subspace_llama31_70b_r{1,4,16,64}.json`
- `results/nocue_format_subspace_llama33_70b_r{1,4,16,64}.json`
- `results/implicit_v2_nocue_control_r{1,4,16,64}.json`
- Launchers: `scripts/cluster_run_nocue_format_subspace.sbatch`,
  `scripts/cluster_run_implicit_v2_nocue_control.sbatch`
