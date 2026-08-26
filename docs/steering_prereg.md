# Steering study — pre-registration (written 2026-08-26, before any steered run)

## Question

The v3 probes are correlational. The objection: directions found by logistic
regression need not be *used* by the model. Test: add or ablate the probe
directions in the residual stream at the probed layer, on **no-cue** items
(nothing about purpose is stated), and ask whether behavior moves.

## What is fixed before running

**Models** (NDIF, bf16): Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct,
Llama-3.1-405B-Instruct — the three 3.1-family models of full320-v3.

**Layer**: each model's decorrelated-probe argmax from the checked-in sweeps
(8B L8/32, 70B L18/80, 405B L37/126). One layer per model; no layer search.

**Directions** (`scripts/steer_fit_directions.py`), refit at that layer on the
checked-in activations with the sweep's exact fits (StandardScaler +
LogisticRegression C=0.1, folds fg1–fg5), un-standardized (`w/σ`), unit-
normalized per fold, fold-averaged:

| name | probe | contrast (sign: + = benchmark / evaluation) |
|---|---|---|
| `fmt` | format probe | benchmark vs casual, stated stratum |
| `pur_dec` | decorrelated purpose probe | evaluation vs deployment, formats balanced |
| `pur_conf` | confounded purpose probe | evaluation::benchmark vs deployment::casual |
| `fmt_dm`, `pur_dm` | difference of class means | same contrasts, not a probe |
| `rand0`, `rand1` | seeded isotropic Gaussian | equal-or-larger-norm controls |

Regression check before use: refit out-of-fold scores must correlate r ≥ 0.999
with the checked-in `results/nocue_scores/*.scores.npz` at that layer and match
its AUC within 1e-3 (bit-identity is not attainable across CPUs for the
near-separable purpose probes; the format probe reproduces to ~1e-6).

**Dose unit** Δ: each real direction's own training-contrast gap in projection
(mean of positive class − mean of negative class on the stated stratum, raw
residual units). Doses k ∈ {−2, −1, +1, +2} × Δ. Random directions are dosed at
max(Δ_fmt, Δ_pur_dec, Δ_pur_conf). Ablation = mean-ablation: the component
along v is set to the no-cue mean projection m.
Added after fitting the directions but before any steered run: `pur_dec`'s
own Δ is small in absolute units (0.10 / 0.25 / 0.49 for 8B / 70B / 405B, vs
0.46 / 0.65 / 1.43 for `fmt`), because the un-standardized LR direction
weights low-variance coordinates. So `pur_dec` is also dosed **norm-matched**
to `fmt` (`pur_dec±1n`, `±2n`: α = k·Δ_fmt), answering "at the same
perturbation norm as format steering, does the purpose direction do
anything?" `pur_dm` (difference of means) is included in the logit readout.

**Positions**: "tail" = the probed last prompt token (constant assistant-header
token) and every later position (prefill tokens / generated tokens, via
`tracer.all()`). Secondary "@all" (logit readout, `fmt` only): every non-BOS
prompt position too.

**Items** (seeded, block-stratified, `scripts/steer_make_plan.py`): 4 no-cue
items per payload block for the logit readout (2 benchmark + 2 casual; 1,280),
2 per block for generation (640; 405B may use 1 per block if throughput
requires — decided from the first-chunk timing, before looking at effects).

**Readouts**
1. *Verbalized eval-awareness (logit)*: fixed assistant prefill ending in a
   one-letter verdict (Option A/B = evaluation/genuine, both letter orders);
   statistic = log-odds(eval letter vs genuine letter) at the final position,
   averaged over the two orders.
2. *Answer style (generation, greedy, 128 tokens)*: response length; whether
   the response opens with a bare answer (letter / short answer) vs prose;
   markdown density; meta/evaluation mentions; degenerate-repetition flag.
3. *Sandbagging proxy*: accuracy vs `answer_key` / `correct_answer`.
4. *Probe parity*: the deployment's pre-intervention residual at the probed
   position is saved for baseline rows and compared with the checked-in
   activations (per-item cosine, projection correlation).
Refusal is not measurable on these benign items and is out of scope.

**Statistics**: per condition, mean effect vs the shared baseline (same item,
same prefill), reported as paired differences with block-cluster bootstrap 95%
CIs over the 320 payload blocks (2,000 draws, seed 0 — the `nocue_sweep.py`
convention). Dose–response = the four dose means per direction.

## Predictions (stated in advance)

- P1 `fmt` dose–response on style: +Δ pushes casual items toward benchmark-
  style output (shorter, answer-first), −Δ pushes benchmark items toward prose;
  monotone in dose. Random directions at equal norm: no monotone effect.
- P2 `fmt` and `pur_conf` increase the verbalized-eval log-odds with dose;
  `pur_dec` is the open question — if the decorrelated purpose direction moves
  verbalized eval-awareness on no-cue items, the model uses a purpose
  representation separable from format. No prediction on its magnitude.
- P3 Mean-ablating `fmt` shrinks the benchmark-vs-casual gap in the readouts
  relative to baseline and to random-direction ablation.
- P4 Scale: the no-cue sweep found +0.6–0.8 sd format pull on the purpose
  direction for 70B/405B and none for 8B; steering effects on the purpose
  readout are expected to be weaker in the 8B.
- P5 Accuracy (sandbagging proxy): exploratory, no directional prediction.

## Decision rules

A direction "moves behavior" if its |k|=2 effect has a bootstrap CI excluding
0, the four doses are monotone, and the effect exceeds the largest random-
direction effect at the same norm. Coherence guard: effects are only reported
at doses where the degenerate-repetition rate stays within 5 points of
baseline; if a dose fails the guard it is reported as degraded, not steered.
Negative results are reported as such.
