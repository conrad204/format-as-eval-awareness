# Transport intervention branch — FROZEN, closed

Status: **closed, negative result**. No further transforms, tuning, or mechanism
search on this branch past this point.

## Preregistered prediction (verbatim, as given by the user before any result was inspected)

> A. Casual→benchmark: test coordinate rotation. ... if it selectively restores the
> casual→benchmark signal term / AUC, that is strong evidence that coordinate
> misalignment explains the signal-loss failure mode.
>
> B. Benchmark→casual: test covariance mismatch. ... The predicted result is
> selective reduction of the benchmark→casual noise term with relatively little
> change in mean separation.
>
> ... If both selective predictions hold across the three models, that becomes the
> deepest explanatory result in the paper. If not, the current exact decomposition
> remains the defensible endpoint.

Formalized as, per-cell:

```
interaction_procrustes = ΔAUC_procrustes(c2b) − ΔAUC_procrustes(b2c)   [predicted > 0]
interaction_coral      = ΔAUC_coral(b2c)      − ΔAUC_coral(c2b)        [predicted > 0]
```

Practical rescue threshold, predeclared: **+0.02 AUC**.

## Two real bugs found and fixed during development (documented, not hidden)

1. **Procrustes O(d³) full-rank SVD on a rank-deficient matrix.** Initial
   implementation formed the full d×d matrix `M = Xsc.T @ Xtc` and took its full
   SVD — 385s/fit at d=4096, infeasible at d=8192 (n_pairs=240 training block
   centroids means the true rank is ≤239, so a full d×d SVD wastes O(d³) computing
   mostly-zero singular values). Replaced with an exactly-orthogonal construction
   restricted to a shared PCA subspace (K=239, fit on pooled no-cue training data),
   verified via norm-preservation to machine precision (max deviation 1.4e-14 on
   synthetic data) and via a synthetic recovery test before trusting real-data
   results. Cost dropped to ~1-20s/fit.

2. **CORAL under-regularization amplifying sampling noise.** Initial
   regularization (eigenvalue floor at 0.1% of max, condition number ≤1000)
   produced 3-5x noise INFLATION on real data instead of the predicted reduction —
   traced to n_rows (~1300-1550) << d (4096-8192), meaning small eigenvalues of the
   sample covariance are mostly sampling noise, and the whitening inverse-sqrt
   amplifies them. Fixed by (a) restricting CORAL to the same shared PCA subspace
   as Procrustes (statistically honest: covariance from ~1500 rows cannot
   determine more than ~1500 dimensions regardless of method) and (b) tightening
   the eigenvalue floor to condition number ≤20. Verified against a realistic
   synthetic low-rank+noise control (recovers target covariance trace correctly)
   before trusting real-data results. Cost dropped from ~180-286s/fit to ~0.2-2.5s/fit.

3. **(Caught before any result was reported.)** A missing
   `model_result["layers"][str(L)] = layer_result` line meant three full pipeline
   runs computed and printed correct results but never persisted them to the JSON
   artifact (`"layers": {}` on disk despite correct console output). Caught by an
   immediate post-write integrity check, fixed, and all three models were rerun
   end-to-end. Reruns reproduced the console-verified numbers exactly (deterministic,
   confirmed bit-for-bit on printed CI bounds), confirming the original bug was a
   save-path omission, not a computation error.

## Result: neither prediction holds, and the pattern is internally inconsistent

Full grid: 3 models × 3 fixed depths (27%/50%/final, not retrospectively chosen) ×
2 interaction tests = 18 cells. 1,000-rep block bootstrap over the 80 held-out
payload blocks per cell (paired: baseline and transform scored on the identical
resampled block set).

| | mean interaction | sign-positive | sig. predicted direction | sig. WRONG direction | clears +0.02 |
|---|---|---|---|---|---|
| Procrustes (predicted c2b > b2c) | +0.0089 | 5/9 | 5/9 | 2/9 | 2/9 |
| CORAL (predicted b2c > c2b) | +0.0024 | 4/9 | 3/9 | 4/9 | 2/9 |

Per-model average interaction (Procrustes): Llama-3.1-8B +0.004, Llama-3.1-70B
**−0.007** (wrong sign), Llama-3.3-70B +0.030. Per-model average (CORAL):
Llama-3.1-8B +0.006, Llama-3.1-70B −0.001, Llama-3.3-70B +0.003. No model-level
sign agreement for Procrustes; all three CORAL model-averages are within noise
of zero.

CORAL in particular shows **more cells significantly contradicting the
prediction (4/9) than supporting it (3/9)** — not merely underpowered null
evidence, but a pattern actively inconsistent with a stable underlying effect.

**Verdict:** both aggregate mean interactions fall far short of the predeclared
+0.02 practical threshold, and the per-cell pattern does not consistently favor
the predicted direction in either hypothesis. Per the preregistration: the simple
transport hypotheses (rotation explains casual→benchmark signal loss; covariance
alignment explains benchmark→casual noise growth) are **not supported**. This is
reported as a negative result, not spun as partial support.

## Defensible endpoint

`format_probe_signal_noise_decomposition.py` (exact d′ signal/noise decomposition,
bootstrapped, real-AUC-validated, replicated 9/9 in sign) remains the paper's
mechanistic-level result. The two failure modes it identifies (signal loss for
casual→benchmark transfer, noise growth for benchmark→casual transfer) are
established and quantified; **what specific representational change produces
that asymmetry remains open** — this transport-intervention branch ruled out two
natural candidate explanations (coordinate rotation, covariance mismatch) without
identifying a replacement. Report accordingly: describe the two failure modes as
established, and state plainly that their underlying cause is not yet identified.

## Frozen artifact hashes (sha256)

See `artifacts/transport_intervention_freeze_hashes.json` for the machine-readable
version. Snapshot:

- `scripts/format_transport_intervention.py`: `aadb643a2e31da164a7c126500be7632cb67a18a8d8a6b7c00a499aaa70698ca`
- `scripts/format_probe_signal_noise_decomposition.py`: `c611191dd0edef01074b6cf08460b8119edea02a73807246b894a14d7816c3dc`
- `artifacts/format_transport_intervention.json`: `e18966387ae0d99921840a101f33601f282a5aea284e312350fb750c5176764c`
- `artifacts/format_probe_signal_noise_decomposition.json`: `e236b39cbcea1aa543543e607751cbe5ff9c27417a223086bc5885874b86ffc5`
- `plots/jpp 8-27-2026/transport_intervention_results/generate.py`: `56dd8887d34f6ee7251639c5aee4f6cc11ad6cc73139e11fac8cb737eafce879`
- `plots/jpp 8-27-2026/transport_intervention_results/transport_intervention_results.pdf`: `300b5a5c65308c8fbdd3e5f4664d78ccae9bdc13dcab6fc73215e2d8eee6e5d0`
- `plots/jpp 8-27-2026/probe_signal_noise_decomposition/generate.py`: `c8daffc215b9a08c7c06b6bf5a6696ca7a3098c7e9d1d36ace54ac257f5243fc`
- `plots/jpp 8-27-2026/probe_signal_noise_decomposition/probe_signal_noise_decomposition.pdf`: `c72e8639e86f6e7e3ebd40f9a37ca7d8fcb962cfcfe02ab221c70c9ae0d2a2c8`

**Branch closed. No further work on transport/rotation/covariance mechanism search.**
