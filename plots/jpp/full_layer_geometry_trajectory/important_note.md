# What "rank 16" means in this figure

"Rank 16" is the dimensionality of the candidate **format subspace** `Q_format(r)`
being tested at that point in the analysis, not a layer index or a model
hyperparameter.

`Q_format(r)` is an orthonormal basis for the top-`r` principal components of the
**format contrast** structure (the benchmark-vs-casual difference directions, `B`,
from the 2x2 factorial decomposition -- see `factorial_vectors` in
`scripts/format_erasure_development.py`). Rank 16 means: take the top 16 dimensions
of that format-shared subspace, and ask how much of the purpose-probe direction's
energy,

```
E_purpose→format(16) = ||Q_format(16)^T · w_purpose||^2
```

falls inside that 16-dimensional slice.

## Why rank 16 specifically

This sprint tests a fixed ladder of ranks -- `1, 4, 16, 64` -- spanning:

- rank 1: does even a single format direction absorb any purpose signal?
- rank 64: a generous format subspace, sized to match the empirical top-64-PC
  subspace used elsewhere in the analysis.

Rank 16 is the largest rank in that ladder where the conditional-null control is
still well-defined and the effect is unambiguous:

- rank 1 is underpowered (only 3/9 or 5/9 cells reach nominal significance, and only
  0/9 or 1/9 survive Holm/Bonferroni correction across the pre-registered 27-cell
  family) -- an underpowered-at-tiny-magnitude effect, not a sign reversal.
- ranks 4 and 16 are fully robust: 9/9 cells survive both Holm and Bonferroni
  correction, under both the full-space isotropic null and the high-variance-
  subspace conditional null (`results/full320_v3_conditional_null_multiplicity.json`).
- rank 64 has no conditional-null control at all: a random rank-64 subspace of the
  64-dimensional top-PC space *is* the whole space, so that specific null is
  undefined there.

So rank 16 is the headline rank for this figure because it is the largest,
best-powered rank where the apples-to-apples conditional-null test is both
meaningful and fully corrected-significant.

## What the result means

Even letting the "format subspace" eat up 16 whole dimensions, the purpose
direction's energy inside it stays far below the exact chance level (`16/d`) at
essentially every depth of every checkpoint tested (`llama31_8b`, `llama31_70b`,
`llama33_70b`) -- not just the 3 pre-registered pinned depths. Purpose information
is not hiding inside even a generously-sized format-shared subspace.

## What this does NOT claim

This full-layer trajectory is descriptive, not a new significance test. Adjacent
layers share almost all their variance (residual-stream continuity), so the
per-layer values plotted here are repeated measurements of one continuous depth
trajectory per model, not independent replicates. No per-layer significance count
or multiple-testing correction is computed across layers -- doing so would be
exactly the layers-as-independent-replicates pseudoreplication the project's
operational rules (`RULES.md`) prohibit. The only corrected, confirmatory claim
remains the pre-registered 3-depth, 27-cell family in
`results/full320_v3_conditional_null_multiplicity.json`; this figure exists only to
show that result holding continuously across depth, i.e. that the 3 pinned depths
were not a favorable cherry-pick from an otherwise noisy profile.
