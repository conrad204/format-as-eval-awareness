# Vendored copy — not our code

Upstream: https://github.com/viliana-dev/eval-awareness-format
Commit:   b6a5000cba8cf99d9a7e313f6371b589d9761b05 ("Fix title, subtitle, workshop name, remove CoQA", 2026-04-01)
Vendored: 2026-08-19 (git history stripped; contents otherwise unmodified)
License:  MIT (see LICENSE)

Code for Devbunova, "Is Evaluation Awareness Just Format Sensitivity?"
(arXiv:2603.19426, ICLR 2026 ICBINB workshop). Companion dataset is vendored
at external/eval-awareness-2x2/ and verified byte-identical to
hf.co/datasets/viliana-dev/eval-awareness-2x2 @ a50e4c983e7e66ebbe8f160e3549cc98e29271d4
(see provenance/devbunova-2x2/hf_snapshot.json).

Why vendored — the two facts our replication depends on:
1. Extraction applies NO chat template: every extraction path tokenizes raw
   text (`tokenizer(batch, padding=True)`, default add_special_tokens=True →
   BOS, right padding, last real token via attention_mask.sum()-1). See
   scripts/evaluate_probe_cross.py:140-178. The probed position is therefore
   the final content token (the ":" of "Answer:"/"Solution:" in benchmark
   cells) — NOT a constant assistant header as in our chat-template pipeline.
2. The confounded-probe protocol behind the paper's 94.5% (RESULTS.md: 94.3%)
   Bench-Deploy false-positive rate: StandardScaler fit on train,
   LogisticRegression(max_iter=1000, random_state=42), train bench_eval(0) vs
   casual_deploy_matched(1), score fraction of bench_deploy predicted 0.
   See scripts/evaluate_probe_cross.py main().

Our bridge into this data: scripts/adapt_devbunova.py.
