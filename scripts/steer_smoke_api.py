"""NDIF API smoke test for the steering driver (run on the pod, 8B by default).

Checks, in one short session each:
  T1 baseline trace: last-token residual at the probed layer + candidate logits
  T2 per-row additive steering at positions >= start: verifies the layer output
     moved by exactly V (bf16 rounding) and downstream logits changed
  T3 mean-ablation formula: projection after intervention equals target m
  T4 logits_to_keep=1 forward kwarg (memory relief for the 8B deployment)
  T5 generate with tracer.all() and per-row vectors (greedy, 24 tokens), timed
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from steer_common import (MODELS, PREFILLS, candidate_ids, chat_prompt, encode_rows,
                          get_tokenizer, load_items, load_keys, prefill_token_count)

model_key = sys.argv[1] if len(sys.argv) > 1 else "8b"
items_path = sys.argv[2]
spec = MODELS[model_key]
L, d = spec["layer"], spec["hidden"]

ndif_key, _ = load_keys()
from nnsight import CONFIG, LanguageModel

CONFIG.API.APIKEY = ndif_key
tok = get_tokenizer(model_key)
rows = load_items(items_path)
nc = [r for r in rows if r["render_slot"] == "no_cue"]
sub = [nc[0], nc[1], nc[2000], nc[2001]]  # 2 benchmark + 2 casual
texts = [chat_prompt(tok, r["text"]) for r in sub]
enc = encode_rows(tok, texts)
B, T = enc["input_ids"].shape
cand = candidate_ids(tok)
print("candidates", cand, [tok.decode([c]) for c in cand])
for name, (pf, ev, dep) in PREFILLS.items():
    print(name, "prefill tokens:", prefill_token_count(tok, texts[0], pf))

model = LanguageModel(spec["hf_id"])
assert int(model.config.num_hidden_layers) == spec["n_layers"]

rng = np.random.default_rng(0)
Vnp = rng.standard_normal((B, d)).astype(np.float32)
Vnp /= np.linalg.norm(Vnp, axis=1, keepdims=True)
V = torch.tensor(Vnp)
alpha = torch.tensor([0.0, 25.0, 0.0, 25.0])
abl = torch.tensor([0.0, 0.0, 0.0, 0.0])
m = torch.zeros(B)
start = torch.full((B,), T - 1, dtype=torch.long)


def run(label, fn):
    t0 = time.time()
    out = fn()
    print(f"{label}: {time.time() - t0:.1f}s", flush=True)
    return out


def t1():
    with model.session(remote=True):
        res = list().save()
        with model.trace(enc):
            o = model.model.layers[L].output
            h = o[0] if isinstance(o, (tuple, list)) else o
            res.append(h[:, -1, :].detach().cpu())
            res.append(model.output.logits[:, -1, :].detach().cpu()[:, cand].float())
    return res


r1 = run("T1 baseline", t1)
h_pre, lg_pre = r1[0], r1[1]
print("  h_pre", h_pre.shape, h_pre.dtype, "norms", h_pre.float().norm(dim=1).tolist())
print("  cand logits", lg_pre.tolist())


def steer_body(o, V, alpha, abl, m, start):
    h = o[0] if isinstance(o, (tuple, list)) else o
    Tn = h.shape[1]
    Vf = V.to(h.device)
    if Tn > 1:
        mask = (torch.arange(Tn, device=h.device)[None, :] >= start.to(h.device)[:, None]).float()
    else:
        mask = torch.ones(h.shape[0], 1, device=h.device)
    hf = h.float()
    proj = torch.einsum("btd,bd->bt", hf, Vf)
    delta = (alpha.to(h.device)[:, None] - abl.to(h.device)[:, None] * (proj - m.to(h.device)[:, None])) * mask
    h_new = (hf + delta[:, :, None] * Vf[:, None, :]).to(h.dtype)
    return h_new


def t2():
    with model.session(remote=True):
        res = list().save()
        with model.trace(enc):
            o = model.model.layers[L].output
            h_new = steer_body(o, V, alpha, abl, m, start)
            if isinstance(o, (tuple, list)):
                model.model.layers[L].output = (h_new,) + tuple(o[1:])
            else:
                model.model.layers[L].output = h_new
            o2 = model.model.layers[L + 1].input  # what the next layer actually receives
            h2 = o2[0] if isinstance(o2, (tuple, list)) else o2
            res.append(h2[:, -1, :].detach().cpu())
            res.append(model.output.logits[:, -1, :].detach().cpu()[:, cand].float())
    return res


import cloudpickle, steer_smoke_api  # noqa: E402  (ship steer_body's module by value)

cloudpickle.register_pickle_by_value(steer_smoke_api)
r2 = run("T2 steer add", t2)
h_post, lg_post = r2[0], r2[1]
diff = (h_post.float() - h_pre.float())
print("  |h_post-h_pre| per row", diff.norm(dim=1).tolist(), "(expect ~0,25,0,25)")
print("  cos(diff, V) rows 1,3:", [float(torch.nn.functional.cosine_similarity(diff[i], V[i], dim=0)) for i in (1, 3)])
print("  logit change", (lg_post - lg_pre).abs().max(dim=1).values.tolist())


def t3():
    abl3 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    m3 = torch.tensor([5.0, 5.0, 5.0, 5.0])
    with model.session(remote=True):
        res = list().save()
        with model.trace(enc):
            o = model.model.layers[L].output
            h_new = steer_body(o, V, torch.zeros(B), abl3, m3, start)
            if isinstance(o, (tuple, list)):
                model.model.layers[L].output = (h_new,) + tuple(o[1:])
            else:
                model.model.layers[L].output = h_new
            o2 = model.model.layers[L + 1].input
            h2 = o2[0] if isinstance(o2, (tuple, list)) else o2
            res.append(h2[:, -1, :].detach().cpu())
    return res


r3 = run("T3 ablate", t3)
proj_after = torch.einsum("bd,bd->b", r3[0].float(), V)
proj_before = torch.einsum("bd,bd->b", h_pre.float(), V)
print("  proj before", proj_before.tolist(), "after", proj_after.tolist(), "(expect 5.0)")


def t4():
    with model.session(remote=True):
        res = list().save()
        with model.trace(enc, logits_to_keep=1):
            lg = model.output.logits
            res.append(lg.shape)
            res.append(lg[:, -1, :].detach().cpu()[:, cand].float())
    return res


try:
    r4 = run("T4 logits_to_keep", t4)
    print("  logits shape", r4[0], "match T1:", torch.allclose(r4[1], lg_pre, atol=0.05))
except Exception as exc:  # noqa: BLE001
    print("  T4 failed:", type(exc).__name__, str(exc)[:300])


def t5():
    with model.generate(enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id, remote=True) as tracer:
        with tracer.all():
            o = model.model.layers[L].output
            h_new = steer_body(o, V, alpha, abl, m, start)
            if isinstance(o, (tuple, list)):
                model.model.layers[L].output = (h_new,) + tuple(o[1:])
            else:
                model.model.layers[L].output = h_new
        out = model.generator.output.save()
    return out


r5 = run("T5 generate steered x24", t5)
print("  out shape", tuple(r5.shape))
for i in range(B):
    print(f"  row{i} alpha={alpha[i].item()}:", repr(tok.decode(r5[i, T:], skip_special_tokens=False))[:200])


def t6():
    with model.generate(enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id, remote=True) as tracer:
        out = model.generator.output.save()
    return out


r6 = run("T6 generate baseline x24", t6)
for i in range(B):
    print(f"  row{i} base:", repr(tok.decode(r6[i, T:], skip_special_tokens=False))[:200])
print("rows 0,2 (alpha=0) identical to baseline:", bool(torch.equal(r5[0], r6[0])), bool(torch.equal(r5[2], r6[2])))
print("SMOKE DONE")
