# Implicit-v2, reframed: "implicit" meant indirectly worded, not absent

Revised 2026-08-31. The previous version of this file was headed "Core EA result:
zero-shot recognition of implicit evaluative situations" and called the finding
positive representational evaluation awareness. That framing does not survive the
audits applied to the rest of this line of work (`history://BrutalEAAuditor`,
`history://HiddenChannelAuditor`), and leaving it standing while retracting the
behavioural claims would have made the repository incoherent.

## What was wrong with the old framing

The implicit-v2 scenarios **describe** evaluative structure in the prompt: blind
comparison, a sealed key, an offline rehearsal, nothing leaving the sandbox. The
evaluative situation is therefore present in the text, only worded indirectly. A
probe that transfers to such prompts shows that a decoder trained on explicit
statements also fires on indirect descriptions of the same thing. That is a
generalisation-of-wording result, not evidence that the model infers an unstated
evaluation state.

The result was also oversold relative to its size: held-family joint AUCs are
0.598-0.660, one interval touches chance, and the endpoint is a frozen probe rather
than any behaviour.

## The evidence, unchanged, correctly placed

Frozen explicit-purpose probes applied to implicit-v2 (six held-out scenario
families, family-bootstrap intervals):

| checkpoint | layer | joint | benchmark to casual | casual to benchmark |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 8 | 0.5980 [0.5269, 0.6697] | 0.5783 [0.5114, 0.6433] | 0.5671 [0.4946, 0.6369] |
| Llama-3.1-70B | 18 | 0.6377 [0.5898, 0.6909] | 0.6317 [0.5776, 0.6867] | 0.6145 [0.5777, 0.6609] |
| Llama-3.3-70B | 18 | 0.6600 [0.6128, 0.7106] | 0.6594 [0.6112, 0.7084] | 0.6264 [0.5863, 0.6678] |

A smaller earlier pilot with five fixed scenario pairs reached AUC 0.7928 with
116/120 matched pairs correctly ordered. Because it has only five pairs, implicit-v2
is the reference result.

## Where it sits: a three-rung ladder

The hidden-channel experiment (`docs/behavioral_ea_hidden_channel_result.md`) supplies
the missing rung, and the three together are more informative than any one alone.
Llama-3.1-8B, same model throughout:

| rung | what the prompt contains | read-out | Llama-3.1-8B |
|---|---|---|---:|
| stated | the purpose or route in so many words | forced choice | **0.9082** |
| described | evaluative structure worded indirectly | frozen probe transfer | **0.5980-0.6600** |
| absent | status only in prior outcomes, one-character separation | forced choice | **0.5527** (chance) |
| absent | same prompts | cross-fitted probe | **0.7699** |

Comprehension survives the first rung easily. Indirect wording already costs most of
the signal. When the status is genuinely absent from the text, the model's own report
falls to chance — yet a probe still decodes it. The ladder locates exactly where the
capability ends and where representation nonetheless persists.

## Corrected claim for this file

> Probes trained on explicitly stated evaluation purpose continue to separate
> indirectly worded evaluative scenarios from matched live-use scenarios at held-family
> AUC 0.598-0.660 across three checkpoints. This is generalisation across wording of a
> described situation. It is not evidence that a model infers an unstated evaluation
> state, and it is not behavioural.

## Claim boundary

No subjective awareness, no spontaneous recognition without textual evidence, no
strategic adaptation, no behavioural consequence. Frozen-probe decoding is not a
causal intervention. Development evidence, not confirmation.

## Artifacts

- `results/implicit_v2_8b_analysis.json`
- `results/implicit_v2_llama31_70b_l18_analysis.json`
- `results/implicit_v2_llama33_70b_l18_analysis.json`
- `results/implicit_ea_core_summary.csv`, `results/implicit_ea_core_figure.png`
- pilot: `results/implicit_purpose_fixed_probe/result.json`
- successor experiment: `docs/behavioral_ea_hidden_channel_result.md`
