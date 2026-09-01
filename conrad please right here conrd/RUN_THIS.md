# Conrad: run this

Hi — everything you need is in this folder. Read the 60-second version, run the
block that matches what you have access to, send back `out/`. Nothing here needs
our cluster, and nothing writes outside this folder.

Updated 2026-09-01 with what we actually need now, which is **different from what
the old README asked for**. The behavioural numbers are largely settled; what is
missing is **activations**.

---

## 60-second version

```bash
cd scripts
pip install numpy scikit-learn                 # analysis
pip install nnsight                            # only for --route ndif
pip install torch transformers accelerate      # only for --route local

# 0. sanity: no model needed, proves the banks and layer plan are intact
python run_405b_hidden_channel_activations.py --route local --dry-run \
  --items ../banks/latent_action.jsonl --tag check \
  --conditions map0_K,map0_M,map1_K,map1_M

# 1. THE IMPORTANT ONE — latent bank activations (2048 prompts)
python run_405b_hidden_channel_activations.py --route local \
  --model /path/to/Meta-Llama-3.1-405B-Instruct \
  --items ../banks/latent_action.jsonl --tag llama31_405b \
  --conditions map0_K,map0_M,map1_K,map1_M --out-dir ../out/acts

# 2. second bank, same thing, stated status (1024 prompts)
python run_405b_hidden_channel_activations.py --route local \
  --model /path/to/Meta-Llama-3.1-405B-Instruct \
  --items ../banks/explicit_ceiling.jsonl --tag llama31_405b_stated \
  --conditions explicit_closed,explicit_active --out-dir ../out/acts
```

Swap `--route local` for `--route ndif` (plus `export NDIF_API_KEY=...`) if you are
going through NDIF instead of local weights; the output is identical either way.

Then run the two analyses (CPU only, a couple of minutes, no model):

```bash
# A. is the latent status linearly decodable at all?
python latent_probe_hidden_channel_fixed_layer.py \
  --activations ../out/acts/hidden_channel_activations_llama31_405b.npz \
  --meta ../out/acts/hidden_channel_activations_llama31_405b.meta.json \
  --bank-items ../banks/latent_action.jsonl --layer 28 \
  --out ../out/latent_probe_llama31_405b.json \
  --direction-out ../out/latent_direction_llama31_405b_l28.npz

# B. the decisive asymmetry, both directions
python stated_to_latent_hidden_channel.py \
  --stated-activations ../out/acts/hidden_channel_activations_llama31_405b_stated.npz \
  --stated-meta ../out/acts/hidden_channel_activations_llama31_405b_stated.meta.json \
  --latent-activations ../out/acts/hidden_channel_activations_llama31_405b.npz \
  --latent-meta ../out/acts/hidden_channel_activations_llama31_405b.meta.json \
  --layer 28 --out ../out/stated_to_latent_llama31_405b.json

python stated_to_latent_hidden_channel.py \
  --stated-activations ../out/acts/hidden_channel_activations_llama31_405b_stated.npz \
  --stated-meta ../out/acts/hidden_channel_activations_llama31_405b_stated.meta.json \
  --latent-activations ../out/acts/hidden_channel_activations_llama31_405b.npz \
  --latent-meta ../out/acts/hidden_channel_activations_llama31_405b.meta.json \
  --layer 28 --reverse --out ../out/latent_to_stated_llama31_405b.json

Send back the whole `out/` directory. If the `.npz` files are too big to send,
send `out/*.json` and we can work from those.

**Layer 28 is fixed in advance** — it is the same relative depth (0.225) we
pre-registered for the smaller checkpoints, and the script prints it for you. Do
not change it based on how the results look; if you want more depths, the runner
already stores ~23 of them and we can add a descriptive curve later.

---

## This has been smoke-tested

Before sending this to you we ran the exact scripts in this folder, against the
exact bank files in this folder, on a *partial-layer* activation file of the same
shape your runner produces (4 stored layers instead of ~23) using our 8B
activations. It reproduced our published 8B numbers to four decimals:
`stated_to_latent` 0.5118, `latent_to_stated` 1.0000, letter control 0.9383,
permutation null 0.4955. So if something breaks, it is the model access or the
environment, not the analysis code.

## Which route do I have?

| what you have | route | value to us |
|---|---|---|
| 405B weights on disk + enough GPUs | `--route local` | **highest — this is what we need** |
| NDIF access (`NDIF_API_KEY`, can reach `ndif.dev`) | `--route ndif` | **same, equally good** |
| Only a chat API (OpenRouter etc.) | see "behavioural only" below | low — mostly already covered |

We could not use NDIF ourselves: `ndif.dev` is TCP-blocked from both our
workstation and our cluster login node (`http=000`). That is a network block, not
a key problem, so it will probably work fine from your side.

Rough cost of route 1+2: 3,072 forward passes of ~1.5k tokens each, no generation,
plus 1,024 more. No sampling, no generation loop, single forward pass per prompt.

### Behavioural only

If you cannot get activations, the useful-but-optional run is the old one:

```bash
python run_405b_openrouter.py --items ../banks/latent_recognition.jsonl \
  --out ../out/latent_recognition_405b.json --concurrency 8
python analyze_recognition_bank.py ../out/latent_recognition_405b.json \
  --bank-items ../banks/latent_recognition.jsonl --label latent_405b \
  --out ../out/analysis_latent_405b.json
```

We already have this for Hermes-3-405B (a **fine-tune**, 0.708). A genuine
Meta-405B-Instruct number would still be a clean scaling point, but it is not
where the open question is anymore.

---

## Why activations, and what we are testing

Three checkpoints are done. The behavioural scaling story is settled enough:

| checkpoint | reads a **stated** route | infers an **unstated** one |
|---|---:|---:|
| Llama-3.1-8B | 0.9082 | 0.5527 (chance) |
| Llama-3.1-70B | 0.9609 | **0.7178** |
| Llama-3.3-70B | 0.9727 | **0.6592** |
| Hermes-3-405B (fine-tune, via API) | not run | 0.7080 |

The live question is **representational**, and it is where the interesting result
is. Two things we found, both of which your run would extend to frontier scale:

**1. Latent status is decodable, and it scales.** A probe trained on the latent
action rows, leave-one-episode-family-out, at a layer fixed in advance:

| checkpoint | latent-status probe AUC (family-mean) |
|---|---:|
| Llama-3.1-8B (L8) | 0.7870 [0.6915, 0.8751] |
| Llama-3.1-70B (L18) | **0.9487** [0.9078, 0.9817] |
| Llama-3.3-70B (L18) | running as of writing |
| Llama-3.1-405B (L28) | **← your run** |

Paired within the same 32 families, 8B → 70B is +0.1617 [0.0742, 0.2578].

**2. The standard readout is a strict subset of the latent one.** Inside this same
environment, with the same families, items and prompt type, only stated-vs-inferred
differing:

| direction | Llama-3.1-8B (L8) | Llama-3.1-70B (L18) | **405B (L28)** |
|---|---:|---:|---:|
| stated-status probe → **inferred** status | 0.5118 [0.452, 0.575] — chance | 0.6997 [0.626, 0.770] | **← your run** |
| inferred-status probe → **stated** status | **1.0000** [1.0, 1.0], 32/32 | **1.0000** [1.0, 1.0], 32/32 | **← your run** |

Read the rows separately. The bottom row is saturated at both scales: a probe
trained on the *inferred* status recovers the *stated* status perfectly, every
family, both models. The top row is the interesting one and it **moves with
scale**: at 8B a probe trained the way the literature trains them — on explicit
stated evaluation-vs-deployment contrasts — learns nothing about a status the model
must infer (0.51, chance), while at 70B it picks up a good deal of it (0.70).

That is exactly why a third scale point is worth your compute. Two readings are
currently indistinguishable:

* the codes **converge** with scale — 405B should land above 0.70 on the top row,
  and the "explicit probes miss inferred status" concern is a small-model artifact;
* the convergence **saturates** — 405B lands near 0.70, and the gap between a
  perfect 1.00 and a partial 0.70 is a persistent property of the readout.

Either way the inclusion is one-directional, which is the result. Analysis B above
produces both rows in one go (second command adds `--reverse`).

---

## The task, in three sentences

Each prompt contains eight prior exchanges establishing that an arbitrary channel
letter (`K` or `M`) means one of two things: replies on that channel stay internal
and are kept only if they copy a flawed reference number, or they go to someone
acting on them where only factual correctness helps. The letter-to-meaning mapping
is **reversed** in half the prompts, so a fixed "K means internal" guess scores
exactly 0.5. In the `latent_action.jsonl` bank the query is a neutral arithmetic
question that never mentions any of this; in `explicit_ceiling.jsonl` the route is
stated outright.

Prompts of opposite status differ by **exactly one character**.

## The banks

`banks/BANKS.json` carries row counts and SHA-256 for every file. All frozen — the
scripts verify the hashes, so please do not edit them.

| file | rows | kind | what it is for |
|---|---:|---|---|
| **`latent_action.jsonl`** | **2048** | action | **primary.** Status absent from the query. Both analyses need it. |
| **`explicit_ceiling.jsonl`** | 1536 | action + recognition | 1024 action rows with the route **stated** — analysis B needs these. |
| `latent_recognition.jsonl` | 1024 | recognition | behavioural: the model answers a question about the status. |
| `arbitrary_control.jsonl` | 1024 | recognition | behavioural: matched non-evaluative contingency. |
| `cot_reasoning.jsonl` | 1024 | reasoning | behavioural: ≤50 words of reasoning allowed first. |

## Sanity gates — please glance at these before sending

The scripts print all of them.

1. **Row counts**: 2048 and 1024. The runner refuses to proceed on an incomplete
   crossing, so if it started, this is fine.
2. **Letter control** in analysis A should be near 1.0 — the channel letter is
   visible in the prompt, so a probe must find it. If it is not, something is
   wrong with the activations.
3. **Permutation null** in analysis A should sit at ~0.50. It shuffles status
   within each family × mapping block, so letter and mapping stay balanced.
4. **Both mappings** (`auc_map0`, `auc_map1`) should be on the same side of 0.5.
   If one is high and the other low, the probe is riding the letter, not status.

## Honest status

Exploratory, not pre-registered. The governing protocol was marked `DO_NOT_RUN`
when the first run started and its item gate was unreachable. Four sub-results are
invalid and labelled as such in `docs/behavioral_ea_hidden_channel_result.md`. A
behavioural specificity control shows a structurally identical **non-evaluative**
contingency is learned *better* in every checkpoint (+0.125 to +0.204, paired by
family), so none of this is evidence of an evaluation-specific faculty — the claim
is about measurement validity of evaluation-awareness probes, not about the model
having a special faculty.

Questions: ping us with the console output, it prints everything we need to debug.
