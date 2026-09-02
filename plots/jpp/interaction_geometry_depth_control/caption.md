**Figure.** Interaction geometry appears related to transfer failure, but the
relationship is largely explained by depth.

Across layers, stronger format-dependence of the purpose representation initially
appears associated with a larger cross-format transfer gap. However, both quantities
vary systematically with model depth. After removing their depth dependence, the
association largely disappears in two of three models, and a stricter layer-to-layer
change test is non-significant in all three. We therefore treat the raw relationship as
primarily depth-confounded rather than evidence that interaction geometry drives the
transfer gap.

**Panel 1 (raw relationship).** All post-spike layers (relative depth ≥ 0.15) from all
three models, format-dependence of the purpose representation vs. cross-format transfer
gap. Lines are per-model least-squares fits.

**Panel 2 (after removing the effect of depth).** The same points after residualizing
both quantities on relative depth (rank-transform each variable, linearly regress the
ranked x- and y-values on ranked depth, plot the residuals) — the identical procedure
used for the pinned partial-Spearman statistics — then standardized within each model
(each model's residuals z-scored using that model's own mean and standard deviation,
not pooled across models). Standardization is a per-model, per-axis affine rescale; it
does not change point order, relative spacing, or any within-model correlation — it only
puts models with different residual scales on comparable axes. Every point is shown at
its (within-model-standardized) value; none is excluded, clipped, or annotated as
off-scale. The cloud is centered near the origin with no consistent diagonal trend,
though a handful of layers near the depth ≥ 0.15 cutoff and the final layer (mostly
Llama-3.1-70B) show visibly larger residuals than the rest — an expected edge effect of
rank-residualization at the boundary of the ranked-depth axis, not noise or an error
(verified: excluding these boundary-adjacent points strengthens, not weakens,
Llama-3.1-70B's correlation, so they are not driving the significant result — see
below).

**Panel 3 (layer-to-layer changes).** Layer-to-layer first differences of the same two
quantities, in depth order — the stricter test. Changes in format-dependence do not
consistently co-occur with changes in the transfer gap.

Per-model statistics (from the pinned analysis, not re-derived):
- Llama-3.1-8B: depth-adjusted r = 0.15 (p = 0.45); layer-to-layer r = −0.03 (p = 0.90)
- Llama-3.3-70B: depth-adjusted r = 0.06 (p = 0.65); layer-to-layer r = 0.12 (p = 0.33)
- Llama-3.1-70B: depth-adjusted r = 0.46 (p < .001); layer-to-layer r = 0.21 (p = 0.09)

Llama-3.1-70B's depth-adjusted correlation remains significant, but this does not
replicate under the stricter layer-to-layer test and is not seen in the other two
models — reported as a non-replicating, model-specific residual, not a generalizable
mechanism. Ad hoc sensitivity check (not a pinned statistic, computed only for this
caption, not plotted): excluding the highest-leverage boundary layers strengthens, not
weakens, this correlation (r=0.52, p<.0001, n=63), so it is not a boundary-point
artifact; it is a genuine broad-layer residual pattern specific to this model. Panel 3
is what shows this pattern does not survive at the sharper layer-to-layer scale, which
is the basis for not treating it as a replicated mechanism.

Raw correlations (for reference, all post-spike layers): Llama-3.1-8B r = 0.74,
Llama-3.3-70B r = 0.94, Llama-3.1-70B r = 0.85.

Source data: artifacts/format_interaction_geometry_full320.json,
artifacts/format_interaction_geometry_partial_corr.json.
Plotting script: generate.py (same folder); every plotted statistic is asserted against
the pinned artifact to 1e-6 before the figure is drawn.
