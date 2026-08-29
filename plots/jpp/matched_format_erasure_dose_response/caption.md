**Figure.** Cross-format transfer gap collapses with matched-format-PCA erasure rank
while purpose AUC stays essentially flat, across all three checkpoints and all three
pinned depths.

Per-fold cross-fitted (5 held-out purpose-family folds) matched-format PCA erasure
(orth_basis(B, r), the un-averaged per-group evaluation/deployment x benchmark/casual
contrast basis) at increasing rank r, then retrained the same balanced 2x2 purpose
probe on the erased activations. Purpose AUC moves by at most ~1 point from its raw
(unerased) value at any depth, model, or rank tested, while the transfer gap falls by
18-72% (largest reductions at the early-layer spike, smallest at the final layer of
Llama-3.1-70B). Rank grid differs by model (8B: 1/4/16/64; both 70Bs: 1/16, a thinned
grid used for compute-cost reasons at 8192 hidden dimensions) -- not padded or
interpolated to match.

Caveat, not shown in this figure: this erasure method does NOT drive format itself to
chance-level decodability (format_auc_after_erasure stays well above 0.5 at every
rank tested here) -- see the null-projection and direction-geometry analyses in this
same results/ directory for why the gap still falls substantially despite incomplete
format erasure. This figure documents the raw dose-response only, not the mechanism.

Source: results/full320_v3_format_erasure_retrain.json (schema_version 2). Generated
by generate.py in this directory (reads only the pinned JSON, no activation
extraction or probe refitting in this script).
