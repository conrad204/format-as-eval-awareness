"""Edit-based generation steering driver (full320-v3 no-cue items) → HF.

The logit driver (scripts/ndif_steer.py) steers the residual at the probed
position with a per-row vector inside a session. For *generation* that pattern
fails on NDIF: the intervention only takes effect across decoding steps under
`tracer.all()`, and a session that bundles many `generate()`+`tracer.all()`
blocks silently drops the collected outputs. The reliable mechanism is
`model.edit()` — a persistent edit that applies on every forward — followed by
standalone remote `generate()` calls whose output is bound after the `with`
block and appended in client scope (verified: steers every step, collects in a
loop; a helper *function* around the generate breaks nnsight's source capture,
so the loop body is kept inline).

Because an edit is one global intervention, generation is steered *uniformly
per condition* (all rows get the same direction·dose) and at *all positions*
(ActAdd), unlike the logit readout's per-row tail steering. One condition =
one HF chunk = all gen items for that condition. Doses run a stronger ladder
than the logit readout (greedy tokens are discrete; ±2·Δ_fmt rarely flips
them), with random directions at the top dose as the null band.

  add:     h <- h + alpha·v            (alpha = k·Δ_fmt)
  ablate:  h <- h - (h·v - m)·v        (component along v set to no-cue mean m)

Run from repo root (keys from env / $STEER_ENV / .env):
  python scripts/ndif_steer_gen.py --model 8b --plan work/plan_8b.json
  python scripts/ndif_steer_gen.py --model 8b --plan work/plan_8b.json --limit-conds 1
  python scripts/ndif_steer_gen.py --model 8b --plan work/plan_8b.json --verify-chunk fmt+8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steer_common import (MODELS, chat_prompt, encode_rows, get_tokenizer,  # noqa: E402
                          load_items, load_keys, sha256_file)

HF_REPO = "Conradf/v3-steering"
MAX_RETRIES = 6
BACKOFF_S = 30
EOS_IDS = (128001, 128008, 128009)

# Generation conditions: a per-condition (uniform) dose ladder. k multiples of
# Δ_fmt; randoms only at the top dose (null band there). "_abl" = mean-ablation.
GEN_LADDER = (
    ("base", None, "none", 0.0),
    ("fmt+2", "fmt", "add", 2.0), ("fmt-2", "fmt", "add", -2.0),
    ("fmt+8", "fmt", "add", 8.0), ("fmt-8", "fmt", "add", -8.0),
    ("pur_dec+8", "pur_dec", "add", 8.0), ("pur_dec-8", "pur_dec", "add", -8.0),
    ("pur_conf+8", "pur_conf", "add", 8.0), ("pur_conf-8", "pur_conf", "add", -8.0),
    ("rand0+8", "rand0", "add", 8.0), ("rand1+8", "rand1", "add", 8.0),
    ("fmt_abl", "fmt", "ablate", 0.0), ("pur_dec_abl", "pur_dec", "ablate", 0.0),
)


def chunk_name(prefix, cid):
    safe = cid.replace("+", "p").replace("-", "m")
    return f"{prefix}/chunks/cond_{safe}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--items", default="inputs/rendered_items_v3_full320_8-17-2026.jsonl")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--run-tag", default="v3")
    ap.add_argument("--gen-batch", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--limit-conds", type=int, default=None)
    ap.add_argument("--verify-chunk", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = MODELS[args.model]
    L, d = spec["layer"], spec["hidden"]
    plan = json.loads(Path(args.plan).read_text())
    assert plan["model_key"] == args.model and plan["layer"] == L
    unit = float(plan["dose_unit_value"])
    gen_batch = args.gen_batch or spec["gen_batch"]

    items_path = Path(args.items)
    rows_all = load_items(items_path, expect_sha=plan["items_sha256"])
    by_id = {r["item_id"]: r for r in rows_all}
    gen_items = plan["gen_items"][: args.limit_items] if args.limit_items else plan["gen_items"]
    dnpz_path = Path(plan["directions_npz"])
    assert sha256_file(dnpz_path) == plan["directions_npz_sha256"]
    dnpz = np.load(dnpz_path)
    assert int(dnpz["layer"]) == L
    dirs = {k[3:]: dnpz[k].astype(np.float32) for k in dnpz.files if k.startswith("v::")}
    m_nocue = plan["nocue_mean"]

    prefix = f"{args.run_tag}/{args.model}/gen"
    tok = get_tokenizer(args.model)
    # pre-encode item batches once (shared across conditions)
    texts = [chat_prompt(tok, by_id[i]["text"]) for i in gen_items]
    batches = [list(range(i, min(i + gen_batch, len(gen_items)))) for i in range(0, len(gen_items), gen_batch)]
    max_new = args.max_new_tokens or int(plan["gen"]["max_new_tokens"])
    conds = list(GEN_LADDER)
    if args.limit_conds:
        conds = conds[: args.limit_conds]
    print(f"{spec['hf_id']} L{L} [gen/edit] -> {args.repo}/{prefix}", flush=True)
    print(f"{len(gen_items)} items, {len(batches)} batches of {gen_batch}, {len(conds)} conditions, "
          f"unit Δ_fmt={unit:.3f}", flush=True)

    if args.dry_run:
        print("conditions:", [c[0] for c in conds]); print("DRY RUN OK"); return

    ndif_key, hf_token = load_keys()
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    if args.verify_chunk is not None:
        from huggingface_hub import HfFileSystem

        with HfFileSystem(token=hf_token).open(f"datasets/{args.repo}/{chunk_name(prefix, args.verify_chunk)}", "rb") as f:
            dat = np.load(io.BytesIO(f.read()))
        print({k: (dat[k].shape, str(dat[k].dtype)) for k in dat.files})
        for i in range(0, len(dat["item_ids"]), max(1, len(dat["item_ids"]) // 6)):
            ids = [int(t) for t in dat["gen_ids"][i] if t >= 0 and int(t) not in EOS_IDS]
            print(f"  {dat['item_ids'][i]}: {tok.decode(ids)[:110]!r}")
        print("VERIFY OK"); return

    import torch
    from nnsight import CONFIG, LanguageModel

    CONFIG.API.APIKEY = ndif_key
    model = LanguageModel(spec["hf_id"])
    assert int(model.config.num_hidden_layers) == spec["n_layers"]

    enc_batches = [encode_rows(tok, [texts[j] for j in b]) for b in batches]

    manifest = {"run_tag": args.run_tag, "model_key": args.model, "hf_id": spec["hf_id"], "layer": L, "mode": "gen",
                "method": "model.edit uniform per-condition, all-position ActAdd; standalone remote generate",
                "plan_sha256": sha256_file(Path(args.plan)), "dose_unit_value": unit, "gen_batch": gen_batch,
                "conditions": [c[0] for c in GEN_LADDER], "n_items": len(gen_items), "eos_ids": EOS_IDS,
                "gen": plan["gen"], "nnsight_version": __import__("nnsight").__version__,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(manifest, indent=1).encode()),
                    path_in_repo=f"{prefix}/manifest_{manifest['started_at'].replace(':', '')}.json",
                    repo_id=args.repo, repo_type="dataset")

    done = set(api.list_repo_files(args.repo, repo_type="dataset"))
    for ci, (cid, dname, mode, k) in enumerate(conds):
        if chunk_name(prefix, cid) in done:
            print(f"[{ci+1}/{len(conds)}] {cid}: already done", flush=True); continue
        v = torch.from_numpy(dirs[dname]).float() if dname else torch.zeros(d)
        alpha = float(k) * unit
        m = float(m_nocue.get(dname, 0.0))
        t0 = time.time()
        outs = None
        for attempt in range(MAX_RETRIES):
            try:
                # Build the persistent edit for this condition (inline; nnsight
                # captures the block source, so no helper function here).
                with model.edit() as med:
                    o = model.model.layers[L].output
                    h = o[0] if isinstance(o, (tuple, list)) else o
                    vv = v.to(h.device)
                    if mode == "add":
                        h[:] = h + alpha * vv
                    elif mode == "ablate":
                        proj = torch.einsum("...d,d->...", h.float(), vv)
                        h[:] = (h.float() - (proj - m)[..., None] * vv).to(h.dtype)
                    # mode "none": no edit
                collected = []
                for enc in enc_batches:
                    with med.generate(enc, remote=True, max_new_tokens=max_new,
                                      do_sample=False, pad_token_id=tok.pad_token_id) as tracer:
                        saved = med.generator.output.save()
                    collected.append(saved.detach().cpu())
                outs = collected
                break
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {str(exc)[:300]}"
                print(f"{cid} attempt {attempt+1} failed: {msg}", flush=True)
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(BACKOFF_S * (2 ** attempt))

        gen = np.full((len(gen_items), max_new), -1, np.int32)
        plen = np.zeros(len(gen_items), np.int32)
        for b, enc, out in zip(batches, enc_batches, outs):
            out = out.numpy()
            Tp = int(enc["input_ids"].shape[1])
            assert out.shape[0] == len(b) and Tp <= out.shape[1] <= Tp + max_new
            assert np.array_equal(out[:, :Tp], enc["input_ids"].numpy())
            for j, gi in enumerate(b):
                g = out[j, Tp:]
                cut = len(g)
                for ti, t in enumerate(g):
                    if int(t) in EOS_IDS:
                        cut = ti + 1
                        break
                gen[gi, :cut] = g[:cut]
                plen[gi] = Tp
        buf = io.BytesIO()
        np.savez(buf, gen_ids=gen, prompt_len=plen,
                 item_ids=np.array(gen_items), cid=np.array([cid] * len(gen_items)),
                 alpha=np.float32(alpha), direction=np.str_(dname or ""), mode=np.str_(mode),
                 layer=np.int16(L), model_id=np.str_(spec["hf_id"]))
        buf.seek(0)
        for attempt in range(MAX_RETRIES):
            try:
                api.upload_file(path_or_fileobj=buf, path_in_repo=chunk_name(prefix, cid),
                                repo_id=args.repo, repo_type="dataset")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == MAX_RETRIES - 1:
                    raise
                buf.seek(0); time.sleep(BACKOFF_S * (2 ** attempt))
        j0 = 0
        ids0 = [int(t) for t in gen[j0] if t >= 0 and int(t) not in EOS_IDS]
        print(f"[{ci+1}/{len(conds)}] {cid} done in {time.time()-t0:.0f}s; e.g. {tok.decode(ids0)[:80]!r}", flush=True)
    print(f"STEER PASS DONE: {spec['hf_id']} [gen]")


if __name__ == "__main__":
    main()
