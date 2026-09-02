# Workshop artifact map

All headline values use the frozen full320-v3 item order and the pinned model revisions recorded in `provenance/full320-v3/`. Activation tensors are hosted in [`format-EA/full320-v3`](https://huggingface.co/datasets/format-EA/full320-v3); they are not duplicated in GitHub.

| paper object | source artifact | generating code | uncertainty unit / caveat |
|---|---|---|---|
| Main purpose AUC and balanced-accuracy table | `v3-activations/results/layer_sweep_full320-v3_{llama31_8b,llama31_70b,llama33_70b}.json`; `results/full320_*` summaries | `scripts/format_bound_regrowth_bootstrap.py`, `scripts/ea_core_evidence_report.py` | held-purpose-vocabulary-family evaluation; layer peak is descriptive; rows are not independent across layers |
| Cross-format B→C and C→B transfer | same layer-sweep JSON files | `scripts/format_bound_regrowth_bootstrap.py` | held-purpose-family probe; transfer is observational |
| Complete layer trajectories | `v3-activations/results/layer_sweep_full320-v3_*.json` | `scripts/format_bound_regrowth_bootstrap.py` | all transformer blocks; do not treat layers as independent replications |
| Main depth, calibration, and transfer figures | `paper/figures/main_purpose_by_depth.png`, `main_auc_balanced_accuracy.png`, `main_cross_format_transfer.png` | `paper/scripts/make_publication_figures.py` | generated from pinned layer-sweep result arrays |
| Equal-N control | `results/full320_equal_n_8b*`, `artifacts/*equal*n*` | `scripts/full320_equal_n*.py`, matching test files | 8B-only development control; complete payload blocks and fixed seeds |
| Matched-format-associated rank ladder | `results/full320_v3_erasure*`, `artifacts/*format*subspace*` | `scripts/full320_v3_format_erasure_retrain.py`, `scripts/full_layer_gap_erasure_bootstrap.py` and rank scripts | payload/block bootstrap; call it a matched-format-associated subspace, not pure format or PCA erasure |
| No-cue format-associated control | `results/nocue_*`, `results/*nocue*`, `artifacts/*nocue*` | `scripts/implicit_v2_nocue_subspace_control.py` and full320 no-cue scripts | purpose statements excluded when estimating the basis; explains only a minority of the gap |
| Real-forward interchange | `results/format_transport_intervention*`, `artifacts/format_transport_intervention.json` | `scripts/format_transport_intervention.py` and companion figure scripts | downstream internal format readout only; not behavioral causality |
| Geometry and conditional nulls | `results/format_*geometry*`, `results/*conditional*`, `artifacts/format_*` | corresponding `scripts/format_*geometry*.py`, null/bootstrap scripts | descriptive geometry and multiplicity-controlled nulls; no causal behavioral claim |
| Provenance_v3 specificity tournament | `results/ea_provenance_specificity_{llama31_8b,llama31_70b,llama33_70b}.json` and `provenance/` manifests | `scripts/provenance_v3_*`, `scripts/ea_provenance_direction_specificity.py` | MMLU/ARC/WildChat/LMSYS source audits; negative external-validity evidence, not spontaneous awareness |
| Binding_v2 supporting control | `results/binding_v2_*`, `artifacts/*binding*` | `scripts/binding_v2_*` and matching analysis scripts | semantics-ablated neutral twin and template-held-out format control; appendix evidence |
| Direct Jacobian-lens diagnostic | `artifacts/jlens*`, `results/jlens*` | `scripts/jlens_*`; `paper/scripts/make_publication_figures.py` | 8B-only development diagnostic; scalar internal readout, not behavior |

## Hash and model contract

The full320 rendered-item hash is recorded in `provenance/full320-v3/` and the paper source. Model revisions, activation position, dtype, row order, and exclusions are recorded in the corresponding manifests. Regeneration must use those records rather than a newly rendered bank.
