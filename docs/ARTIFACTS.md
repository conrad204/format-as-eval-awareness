# Workshop artifact map

This is a reproducibility map, not a manuscript. The canonical full320-v3
item order and model revisions are recorded in `provenance/full320-v3/`.
Large activation tensors are hosted in the private
[`format-EA/full320-v3`](https://huggingface.co/datasets/format-EA/full320-v3)
dataset and are not duplicated in GitHub.

| analysis | source artifact | generating code | caveat |
|---|---|---|---|
| Stated-purpose AUC and balanced accuracy | `v3-activations/results/` | `scripts/format_bound_regrowth_bootstrap.py`, `scripts/full320_plots.py` | held-purpose-family evaluation; layer peaks are descriptive |
| Cross-format transfer and complete layer trajectories | `v3-activations/results/` | `scripts/format_bound_regrowth_bootstrap.py` | transfer is observational; layers are not independent replications |
| Equal-N control | `results/full320_equal_n_8b*`, `artifacts/` | `scripts/full320_equal_n.py`, `scripts/full320_strict_equal_rows.py` | 8B-only development control |
| Matched-format-associated rank ladder | `results/full320_v3_erasure_*`, `artifacts/` | `scripts/full320_v3_format_erasure_retrain.py`, `scripts/full_layer_gap_erasure_bootstrap.py`, `scripts/full320_v3_erasure_extend_ranks.py` | not pure-format or PCA erasure |
| No-cue format-associated controls | `results/*nocue*`, `artifacts/` | `scripts/full320_v3_nocue_format_subspace.py`, `scripts/full320_v3_nocue_subspace_geometry.py`, `scripts/nocue_sweep.py` | purpose statements are excluded when estimating the basis |
| Real-forward interchange | `results/format_transport_intervention*`, `artifacts/format_transport_intervention.json` | `scripts/format_transport_intervention.py` | downstream internal format readout; not behavioral causality |
| Purpose/format geometry and null controls | `results/full320_v3_*geometry*`, `results/*conditional*`, `artifacts/format_*` | `scripts/full320_v3_direction_geometry.py`, `scripts/full320_v3_conditional_hv_null.py`, `scripts/full320_v3_projected_basis_conditional_null.py` | descriptive geometry and multiplicity-controlled nulls |
| Provenance-v3 specificity tournament | `results/ea_provenance_specificity_*.json`, `provenance/` | `scripts/provenance_v3_*`, `scripts/ea_provenance_direction_specificity.py` | negative external-validity evidence |
| Binding-v1 and binding-v2 supporting controls | `results/binding_v1_*`, `results/binding_v2_*`, `artifacts/` | `scripts/binding_v1_*`, `scripts/binding_v2_*` | supporting relational-binding controls |
| Implicit-v2 indirectly worded context control | `results/implicit_v2_frozen_r3/`, `results/implicit_v2_*`, `results/implicit_purpose_fixed_probe/`, `docs/implicit_v2_indirect_context_control.md` | `scripts/implicit_v2_build.py`, `scripts/implicit_v2_analyze.py`, `scripts/implicit_v2_nocue_subspace_control.py`, `scripts/implicit_purpose_fixed_probe.py` | described context generalization; not latent or inferred status |
| Direct Jacobian-lens diagnostic | `artifacts/jlens_100/`, `artifacts/jlens_1000/` | `scripts/jlens_geometry_controls.py`, `scripts/jlens_purpose_decomposition.py` | 8B-only scalar internal readout; not behavior |

## Hash and model contract

The full320 rendered-item hash is recorded in `provenance/full320-v3/`.
Model revisions, activation position, dtype, row order, and exclusions are
recorded in the corresponding manifests. Reproduction must use those records
rather than a newly rendered bank.
