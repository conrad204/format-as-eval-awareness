# Core paper result: format-robust representational evaluation awareness

Status: local working synthesis, 2026-08-31. Not committed or pushed.

## One-sentence claim

**Llama models form a format-robust internal representation of stated evaluation context, but conventional evaluation-awareness probes can collapse onto benchmark format; a content-matched purpose-by-format factorial design identifies the representation and separates it from the format-associated component that causes probe failure.**

This is a positive claim of **representational evaluation awareness** under the recognition/representation convention used by Needham et al., Nguyen et al., Heidari et al., and Li et al. It is not a claim of spontaneous, strategic, conscious, or behaviorally consequential awareness.

## Why this is novel

Devbunova showed that a conventional benchmark-versus-chat probe can be a format detector and that pooled decorrelated training repairs classification. This repository adds four identification steps that change the scientific conclusion from “format confound” to “format-robust EA representation”:

1. **Exact factorial matching.** Each payload appears under both stated purposes and both formats. Content, topic, and generator provenance are identical across cells rather than matched by proxy.
2. **Held-vocabulary semantic generalization.** Leave-one-purpose-family-out folds require the model representation to transfer across purpose vocabularies. Template-held-out word and character n-gram controls are exactly at AUC 0.500.
3. **Cross-format training, not pooled-format classification.** A purpose probe trained entirely on benchmark-format activations transfers to casual format, and the reverse probe transfers back. The model therefore contains a shared purpose signal rather than two unrelated format-specific lookup rules.
4. **Mechanistic decomposition.** Format is present from layer 0; purpose emerges during transformer computation. A cross-fitted matched-format subspace explains 59–65% of the residual transfer deficit across the three checkpoints while containing unusually little purpose-direction energy. The stricter no-cue-derived rank-64 basis removes 29.8% in the completed 8B analysis, showing a weaker but cue-independent format-associated component. A real-forward matched interchange shows that the matched-format subspace moves the format readout more than a norm-matched random intervention.

The novel object is not “an EA direction.” It is the **factorial geometry of evaluation awareness**: a format-shared semantic purpose component plus a format-conditioned component that determines whether ordinary probes generalize.

## Primary positive evidence

Best decorrelated layer, with layer selected entirely inside the controlled full320 experiment:

| checkpoint | joint purpose AUC | benchmark to casual | casual to benchmark | matched pairs correctly ordered |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 0.9888 | 0.9513 | 0.9401 | 1.000 |
| Llama-3.1-70B | 0.9984 | 0.9846 | 0.9850 | 1.000 |
| Llama-3.3-70B | 0.9959 | 0.9845 | 0.9787 | 1.000 |

The natural-diagonal probe instead classifies the held-out cells by format: deployment rendered as a benchmark is called evaluation 91.5–100% of the time. Thus the same activation bank contains both a misleading format solution and a nearly format-invariant evaluation-context solution; the training design determines which one the probe recovers.

Depth separates the components:

- format AUC is at least 0.99 at layer 0;
- purpose AUC crosses 0.95 only after transformer computation;
- purpose and cross-format transfer peak at approximately 23–26% normalized depth;
- the transfer deficit is smaller in both 70B checkpoints than in 8B.

Rank-64 matched-format subspace removal with probe retraining removes approximately 58.5%, 62.8%, and 65.3% of the integrated cross-format transfer gap in the three checkpoints. This is a frozen-activation representation diagnostic, not behavioral causal evidence.

## New specificity tournament: the external-provenance bridge fails

Protocol was written to `results/ea_specificity_protocol_20260831.json` before computing the control outcomes. The controlled best layers were fixed at L8, L18, and L18. Directions were estimated only from full320 and applied without fitting to provenance_v3. The primary endpoint was the minimum signed AUC across all four evaluation-source by deployment-log-source pairs.

| checkpoint | purpose | purpose minus format projection | stated-row format | no-cue format | purpose null p |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.502 | 0.530 | **0.680** | **0.663** | 0.416 |
| Llama-3.1-70B | 0.233 | 0.253 | **0.679** | **0.668** | 0.802 |
| Llama-3.3-70B | 0.285 | 0.306 | **0.654** | **0.654** | 0.723 |

The gate failed in every checkpoint. At the layers chosen by the controlled EA experiment, real corpus provenance aligns with format directions, not the controlled purpose direction. One hundred payload-block sign-flip directions span very wide provenance AUC ranges, so arbitrary synthetic directions can also align with corpus differences. The previously reported late-layer peak provenance AUCs of 0.70–0.74 are descriptive, provenance-selected peaks and are not selective evidence for the controlled EA representation.

Consequences:

- Do not use provenance_v3 as the positive EA result.
- Do not say the controlled direction zero-shot identifies real evaluation provenance.
- Retain provenance as a negative external-validity analysis showing why corpus-origin classification is not sufficient evidence of EA.
- The positive claim rests on exact factorial identification and cross-format semantic transfer.

## Role of binding_v2

`binding_v2` is a specificity control, not the positive experiment. Its semantics-ablated construction shows that consequence semantics remain decodable while arbitrary relational binding and cross-hop transfer largely collapse. The frozen purpose probe stays near chance on that unrelated variable. This blocks the interpretation that any matched binary role automatically produces the claimed EA geometry.

## Role of causal results

The completed 8B real-forward interchange supports one bounded mechanism: swapping the matched format subspace moves the format readout more than a per-row norm-matched random perturbation. Purpose-readout effects vary by depth and do not establish uniform protection or behavioral EA. The sealed factorial pilot was negative and remains negative. Neither result is required for the primary positive representational claim.

## Recommended title and abstract claim

### Title

**Factorial Probing Reveals Format-Robust Evaluation Awareness in Language Models**

Alternative, more conservative:

**Separating Evaluation Awareness from Benchmark Format in Language-Model Representations**

### Abstract-level result

> We study whether activation probes that distinguish evaluation from deployment recover evaluation context or benchmark format. In a content-matched purpose-by-format factorial design, naturally confounded probes classify by format. Factorially decorrelated probes instead recover a semantic evaluation-context representation with held-family-out AUC 0.989–0.998. Probes trained entirely in one format transfer to the other with AUC 0.940–0.985, establishing a format-shared representation rather than pooled format-specific rules. Format is decodable from the embedding stream, whereas evaluation context emerges at intermediate depth. A cross-fitted matched-format subspace explains 59–65% of the remaining integrated transfer deficit; a stricter no-cue-derived basis removes 29.8% in the completed 8B analysis. These results provide evidence of format-robust representational evaluation awareness and show that probe training geometry determines whether it is recovered or replaced by a format shortcut. External corpus-provenance controls do not validate transfer to spontaneous real-world recognition.

## Claim boundary

Use “evaluation awareness” only with an immediate operational definition:

> We use evaluation awareness in the representational sense established in prior probing work: internally distinguishing stated evaluation from deployment context. We do not infer subjective awareness, spontaneous recognition without contextual evidence, strategic adaptation, or behavioral inconsistency.

The positive paper is therefore not “models spontaneously know they are being tested.” It is:

> **Models semantically represent evaluation context across radically different surface realizations; standard probes often fail to recover that representation because format supplies an easier correlated solution.**

## Local artifacts

- `results/ea_specificity_protocol_20260831.json`
- `scripts/ea_provenance_direction_specificity.py`
- `scripts/cluster_ea_provenance_specificity.sbatch`
- `results/ea_provenance_specificity_{model}.json`
- `scripts/ea_core_evidence_report.py`
- `results/ea_core_evidence_summary.csv`
- `results/ea_core_evidence_figure.png`

No artifact in this analysis was committed or pushed.
