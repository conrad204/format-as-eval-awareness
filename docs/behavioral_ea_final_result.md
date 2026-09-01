# RETRACTED: "Llama models exhibit behavioral evaluation awareness in implicit closed-comparison settings"

Retracted 2026-08-31 after independent adversarial audit (`history://BrutalEAAuditor`).

The headline was not warranted. Three specific defects, each verifiable in this repository:

1. The recognition task states its own answer in paraphrase, so it measures reading
   comprehension of an explicitly described situation, not recognition of a latent
   evaluation status (`results/behavioral_ea_recognition_frozen/items.jsonl`).
2. Its own frozen three-model gate failed
   (`results/behavioral_ea_recognition_analysis.json`, `recognition_gate_pass: false`).
   The retracted document reported the failure and then asserted the headline anyway.
3. The claimed adaptation support does not survive correct crossed inference, and the
   focused follow-up reversed direction (`results/behavioral_ea_crossed_reinference.json`,
   `results/behavioral_ea_closed_comparison_analysis.json`).

Corrected record: `docs/behavioral_ea_causal_result.md`.
Successor experiment with hidden status and executed outcomes:
`docs/behavioral_ea_hidden_channel_result.md`.
