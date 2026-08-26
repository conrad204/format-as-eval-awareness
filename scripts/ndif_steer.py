"""Chunked, resumable NDIF steering driver (full320-v3 no-cue items).

Two readouts, both intervening on the residual stream at the probed layer
(output of transformer block L, before the final norm — the tensor the probes
were fit on), with one intervention vector per batch row:

  logit  one forward pass per row: prompt + a fixed assistant prefill that ends
         right before a one-letter evaluation/genuine verdict; saves the four
         candidate-letter logits at the final position and the argmax token.
         Rows with condition "base" (prefill "ab") also save the *pre-
         intervention* residual at the probed position, so the deployment can
         be checked against the checked-in activations (probe parity).
  gen    greedy generation from the assistant header (max_new_tokens from the
         plan); saves generated token ids only.

Intervention (positions >= start, per row): h <- h + [alpha - abl*(h.v - m)] v
  add     alpha = dose, abl = 0        (activation addition)
  ablate  alpha = 0, abl = 1, m = no-cue mean projection (mean-ablation)
  none    zeros (baseline rows share the batch with steered rows)
"tail" positions: start = probed token (last chat-header token); in gen mode
every generated position too (tracer.all()). "all": start = BOS+1 per row.

Outputs: <run_tag>/<model>/<mode>/chunks/chunk_XXXX.npz in an HF dataset repo,
manifest.json alongside. Nothing is written to local disk. Resumable by
listing the repo. NDIF etiquette as in ndif_extract_devbunova.py: one request
in flight, sessions of a few traces, reductions only, fail fast on
RemoteException.

Run from repo root (keys from env, $STEER_ENV, or .env):
  python scripts/ndif_steer.py --model 8b --plan work/plan_8b.json --mode logit
  python scripts/ndif_steer.py --model 8b --plan work/plan_8b.json --mode gen --limit-chunks 1
  python scripts/ndif_steer.py --model 8b --plan work/plan_8b.json --mode gen --dry-run
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
from steer_common import (MODELS, candidate_ids, chat_prompt, encode_rows, get_tokenizer,  # noqa: E402
                          load_items, load_keys, prefill_token_count, sha256_file)

HF_REPO = "Conradf/v3-steering"
MAX_RETRIES = 7
BACKOFF_S = 30
SERVER_WAIT = 20
EOS_IDS = (128001, 128008, 128009)


def steer_body(o, V, alpha, abl, m, start):
    """Executed on the NDIF worker (shipped by value). o = layer output
    (tensor, or tuple whose [0] is the hidden state)."""
    import torch

    h = o[0] if isinstance(o, (tuple, list)) else o
    Tn = h.shape[1]
    dev = h.device
    Vf = V.to(dev)
    if Tn > 1:
        mask = (torch.arange(Tn, device=dev)[None, :] >= start.to(dev)[:, None]).float()
    else:
        mask = torch.ones(h.shape[0], 1, device=dev)
    hf = h.float()
    proj = torch.einsum("btd,bd->bt", hf, Vf)
    delta = (alpha.to(dev)[:, None] - abl.to(dev)[:, None] * (proj - m.to(dev)[:, None])) * mask
    return (hf + delta[:, :, None] * Vf[:, None, :]).to(h.dtype)


def _gen_one(model, L, enc, tt, gen_kwargs):
    """One standalone remote generate with per-row steering; returns cpu ids.
    Kept as a function so the `.save()` binds in a clean frame (nnsight captures
    the block source; a bare for-loop body can fail to bind the saved name)."""
    import torch  # noqa: F401  (whitelisted; used inside the traced block)

    V, alpha, abl, m, start = tt
    with model.generate(enc, remote=True, **gen_kwargs) as tracer:
        with tracer.all():
            o = model.model.layers[L].output
            h_new = steer_body(o, V, alpha, abl, m, start)
            if isinstance(o, (tuple, list)):
                model.model.layers[L].output = (h_new,) + tuple(o[1:])
            else:
                model.model.layers[L].output = h_new
        out = model.generator.output.save()
    return out.detach().cpu()


def chunk_name(prefix: str, index: int) -> str:
    return f"{prefix}/chunks/chunk_{index:04d}.npz"


def ndif_status_snapshot(hf_id: str) -> dict:
    try:
        import urllib.request

        with urllib.request.urlopen("https://api.ndif.us/status", timeout=30) as r:
            d = json.load(r)
        reps = {k: {kk: v.get(kk) for kk in ("deployment_level", "pinned", "application_state", "replica_id")}
                for k, v in d.get("deployments", {}).items()
                if isinstance(v, dict) and v.get("repo_id") == hf_id}
        return {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "replicas": reps}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def build_rows(plan: dict, mode: str, by_id: dict) -> list[dict]:
    cond = {c["cid"]: c for c in plan["conditions"]}
    rows = []
    if mode == "logit":
        for iid in plan["logit_items"]:
            for pf in plan["prefills"]:
                for cid in plan["logit_conditions"]:
                    rows.append(dict(item_id=iid, prefill=pf, cid=cid, **{k: cond[cid][k] for k in
                                                                       ("dir", "mode", "alpha", "m", "positions")}))
    else:
        for iid in plan["gen_items"]:
            for cid in plan["gen_conditions"]:
                rows.append(dict(item_id=iid, prefill="", cid=cid, **{k: cond[cid][k] for k in
                                                                    ("dir", "mode", "alpha", "m", "positions")}))
    for r in rows:
        assert r["item_id"] in by_id
    return rows


def make_tensors(rows: list[dict], dirs: dict, d: int, enc, n_prefill: dict):
    import torch

    B, T = enc["input_ids"].shape
    V = torch.zeros(B, d, dtype=torch.float32)
    alpha = torch.zeros(B)
    abl = torch.zeros(B)
    m = torch.zeros(B)
    start = torch.zeros(B, dtype=torch.long)
    lengths = enc["attention_mask"].sum(dim=1)
    for i, r in enumerate(rows):
        if r["dir"] is not None:
            V[i] = torch.from_numpy(dirs[r["dir"]])
        if r["mode"] == "add":
            alpha[i] = float(r["alpha"])
        elif r["mode"] == "ablate":
            abl[i] = 1.0
            m[i] = float(r["m"])
        npf = n_prefill.get(r["prefill"], 0)
        probe_pos = T - npf - 1
        start[i] = probe_pos if r["positions"] == "tail" else (T - int(lengths[i]) + 1)
        r["_probe_pos"] = int(probe_pos)
    return V, alpha, abl, m, start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--mode", choices=("logit", "gen"), required=True)
    ap.add_argument("--items", default="inputs/rendered_items_v3_full320_8-17-2026.jsonl")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--run-tag", default="v1")
    ap.add_argument("--items-per-chunk", type=int, default=None)
    ap.add_argument("--sub-batch", type=int, default=None)
    ap.add_argument("--limit-chunks", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-chunk", type=int, default=None)
    args = ap.parse_args()

    spec = MODELS[args.model]
    L, d = spec["layer"], spec["hidden"]
    plan = json.loads(Path(args.plan).read_text())
    assert plan["model_key"] == args.model and plan["layer"] == L
    items_path = Path(args.items)
    rows_all = load_items(items_path, expect_sha=plan["items_sha256"])
    by_id = {r["item_id"]: r for r in rows_all}
    dnpz_path = Path(plan["directions_npz"])
    assert sha256_file(dnpz_path) == plan["directions_npz_sha256"], "directions npz sha mismatch"
    dnpz = np.load(dnpz_path)
    assert int(dnpz["layer"]) == L
    dirs = {k[3:]: dnpz[k].astype(np.float32) for k in dnpz.files if k.startswith("v::")}
    for k, v in dirs.items():
        assert v.shape == (d,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-4, k

    prefix = f"{args.run_tag}/{args.model}/{args.mode}"
    items_per_chunk = args.items_per_chunk or (2 if args.mode == "logit" else 4)
    sub_batch = args.sub_batch or (spec["sub_batch"] if args.mode == "logit" else spec["gen_batch"])
    if args.mode == "logit" and args.model == "8b" and args.sub_batch is None:
        sub_batch = 16  # logits_to_keep=1 removes the full-sequence logits that OOM'd the 8B deployment at 16

    if args.dry_run:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        ndif_key = hf_token = None
    else:
        ndif_key, hf_token = load_keys()
    tok = get_tokenizer(args.model)
    cand = candidate_ids(tok)
    header_probe = chat_prompt(tok, by_id[plan["gen_items"][0]]["text"])
    n_prefill = {k: prefill_token_count(tok, header_probe, v["text"]) for k, v in plan["prefills"].items()}
    for k, v in plan["prefills"].items():  # same count for every item (prefill tokenizes independently)
        assert prefill_token_count(tok, chat_prompt(tok, by_id[plan["gen_items"][-1]]["text"]), v["text"]) == n_prefill[k]

    rows = build_rows(plan, args.mode, by_id)
    item_order = list(dict.fromkeys(r["item_id"] for r in rows))
    chunks = [item_order[i:i + items_per_chunk] for i in range(0, len(item_order), items_per_chunk)]
    rows_by_item: dict[str, list[dict]] = {}
    for r in rows:
        rows_by_item.setdefault(r["item_id"], []).append(r)
    print(f"{spec['hf_id']} L{L} [{args.mode}] -> {args.repo}/{prefix}", flush=True)
    print(f"{len(rows)} rows over {len(item_order)} items; {len(chunks)} chunks of {items_per_chunk} items; "
          f"sub_batch {sub_batch}; prefill tokens {n_prefill}", flush=True)

    def texts_for(r: dict) -> str:
        t = chat_prompt(tok, by_id[r["item_id"]]["text"])
        return t + plan["prefills"][r["prefill"]]["text"] if r["prefill"] else t

    if args.dry_run:
        import torch

        crow = [r for iid in chunks[0] for r in rows_by_item[iid]][:sub_batch]
        enc = encode_rows(tok, [texts_for(r) for r in crow])
        V, alpha, abl, m, start = make_tensors(crow, dirs, d, enc, n_prefill)
        T = enc["input_ids"].shape[1]
        print("first trace:", enc["input_ids"].shape, "start", start.tolist(), "alpha", [round(a, 2) for a in alpha.tolist()],
              "abl", abl.tolist())
        pp = crow[0]["_probe_pos"]
        print("probed token per row:", [repr(tok.decode([int(enc['input_ids'][i, pp])])) for i in range(len(crow))])
        if args.mode == "logit":
            print("tail text:", repr(tok.decode(enc["input_ids"][0, pp:])))
        assert bool(torch.all(enc["attention_mask"][torch.arange(len(crow)), start] == 1))
        print("DRY RUN OK")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    if args.verify_chunk is not None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=hf_token)
        with fs.open(f"datasets/{args.repo}/{chunk_name(prefix, args.verify_chunk)}", "rb") as f:
            dat = np.load(io.BytesIO(f.read()), allow_pickle=False)
        print({k: (dat[k].shape, str(dat[k].dtype)) for k in dat.files})
        if args.mode == "gen":
            for i in range(min(6, len(dat["item_ids"]))):
                ids = [int(t) for t in dat["gen_ids"][i] if t >= 0]
                print(f"[{dat['cid'][i]}] {tok.decode(ids)[:160]!r}")
        else:
            lg = dat["cand_logits"]
            for i in range(min(6, len(lg))):
                print(dat["cid"][i], dat["prefill"][i], lg[i].round(2).tolist(), repr(tok.decode([int(dat["top1"][i])])))
        print("VERIFY OK")
        return

    import cloudpickle
    import torch
    from nnsight import CONFIG, LanguageModel

    import ndif_steer  # noqa: F401  (self, for by-value pickling of steer_body)

    cloudpickle.register_pickle_by_value(ndif_steer)
    CONFIG.API.APIKEY = ndif_key
    model = LanguageModel(spec["hf_id"])
    assert int(model.config.num_hidden_layers) == spec["n_layers"] and int(model.config.hidden_size) == d

    manifest = {
        "run_tag": args.run_tag, "model_key": args.model, "hf_id": spec["hf_id"], "layer": L, "mode": args.mode,
        "plan_sha256": sha256_file(Path(args.plan)), "plan": {k: v for k, v in plan.items()
                                                              if k not in ("gen_items", "logit_items")},
        "n_rows": len(rows), "n_items": len(item_order), "items_per_chunk": items_per_chunk, "sub_batch": sub_batch,
        "candidate_ids": cand, "prefill_tokens": n_prefill, "eos_ids": EOS_IDS,
        "nnsight_version": __import__("nnsight").__version__, "transformers_version": __import__("transformers").__version__,
        "tokenizer_revision": spec["tok_rev"], "ndif_status": ndif_status_snapshot(spec["hf_id"]),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_gen": False,
    }
    api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(manifest, indent=1).encode()),
                    path_in_repo=f"{prefix}/manifest_{manifest['started_at'].replace(':', '')}.json",
                    repo_id=args.repo, repo_type="dataset")

    done = {f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.startswith(prefix + "/chunks/")}
    todo = [c for c in range(len(chunks)) if chunk_name(prefix, c) not in done]
    if args.limit_chunks is not None:
        todo = todo[: args.limit_chunks]
    print(f"{len(done)} chunks already uploaded, {len(todo)} to run", flush=True)

    max_new = int(plan["gen"]["max_new_tokens"])
    def make_batches(c, sb):
        crow = [r for iid in chunks[c] for r in rows_by_item[iid]]
        if args.mode == "gen":  # one generate request per item batch (same prompt, equal length, no padding)
            batches = [rows_by_item[iid][i:i + sb] for iid in chunks[c]
                       for i in range(0, len(rows_by_item[iid]), sb)]
        else:
            batches = [crow[i:i + sb] for i in range(0, len(crow), sb)]
        encs, tens = [], []
        for b in batches:
            enc = encode_rows(tok, [texts_for(r) for r in b])
            encs.append(enc)
            tens.append(make_tensors(b, dirs, d, enc, n_prefill))
        return crow, batches, encs, tens

    for c in todo:
        cur_sb = sub_batch
        crow, batches, encs, tens = make_batches(c, cur_sb)
        t0 = time.time()
        for attempt in range(MAX_RETRIES):
            try:
                if args.mode == "logit":
                    with model.session(remote=True):
                        res = list().save()
                        for enc, (V, alpha, abl, m, start) in zip(encs, tens):
                            with model.trace(enc, logits_to_keep=1):
                                o = model.model.layers[L].output
                                h = o[0] if isinstance(o, (tuple, list)) else o
                                hp = h[torch.arange(h.shape[0], device=h.device), start.to(h.device), :].detach().cpu()
                                h_new = steer_body(o, V, alpha, abl, m, start)
                                if isinstance(o, (tuple, list)):
                                    model.model.layers[L].output = (h_new,) + tuple(o[1:])
                                else:
                                    model.model.layers[L].output = h_new
                                lg = model.output.logits[:, -1, :]
                                res.append(lg.detach().cpu()[:, cand].float())
                                res.append(lg.argmax(dim=-1).detach().cpu())
                                res.append(hp)
                    outs = [r for r in res]
                else:
                    # One standalone request per item-batch. A session bundling many
                    # generate() blocks each with tracer.all()+intervention silently
                    # drops the appended outputs (len(res)==0), so gen does not use a
                    # session; each generate is its own NDIF request.
                    outs = []
                    gen_kwargs = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
                    for enc, tt in zip(encs, tens):
                        outs.append(_gen_one(model, L, enc, tt, gen_kwargs))
                break
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {str(exc)[:600]}"
                server = "RemoteException" in msg or "NNsightException" in msg
                print(f"chunk {c} attempt {attempt + 1} failed [{'server' if server else 'transient'}] "
                      f"(sub_batch {cur_sb}): {msg}", flush=True)
                if attempt == MAX_RETRIES - 1:
                    raise
                # A deterministic server-side forward error is usually memory pressure
                # on the shared 8B GPU: halve the batch and rebuild before retrying.
                if server and cur_sb > 1:
                    cur_sb = max(1, cur_sb // 2)
                    crow, batches, encs, tens = make_batches(c, cur_sb)
                    print(f"chunk {c}: retrying at sub_batch {cur_sb}", flush=True)
                    time.sleep(SERVER_WAIT)
                else:
                    time.sleep(BACKOFF_S * (2 ** attempt))
        t_ndif = time.time() - t0

        arrays = {"item_ids": np.array([r["item_id"] for r in crow]), "cid": np.array([r["cid"] for r in crow]),
                  "prefill": np.array([r["prefill"] for r in crow]), "alpha": np.array([r["alpha"] for r in crow], np.float32),
                  "row_start": np.int64(sum(len(rows_by_item[i]) for i in item_order[: item_order.index(chunks[c][0])])),
                  "layer": np.int16(L), "model_id": np.str_(spec["hf_id"]), "mode": np.str_(args.mode)}
        if args.mode == "logit":
            lg = torch.cat(outs[0::3]).numpy().astype(np.float32)
            top = torch.cat(outs[1::3]).numpy().astype(np.int32)
            hp = torch.cat(outs[2::3])
            assert hp.dtype == torch.bfloat16, hp.dtype
            keep = [i for i, r in enumerate(crow) if r["cid"] == "base" and r["prefill"] == "ab"]
            arrays.update(cand_logits=lg, top1=top, cand_ids=np.array(cand, np.int32),
                          probe_item_ids=np.array([crow[i]["item_id"] for i in keep]),
                          probe_h_bf16_u16=hp[keep].view(torch.int16).numpy().view(np.uint16))
            assert lg.shape == (len(crow), 4)
            base = [i for i in keep]
            fp2 = [i for i, r in enumerate(crow) if r["cid"] == "fmt+2" and r["prefill"] == "ab"]
            lo = lambda a: np.logaddexp(a[:, 0], a[:, 2]) - np.logaddexp(a[:, 1], a[:, 3])  # noqa: E731
            print(f"QC base eval-logodds(ab) {lo(lg[base]).mean():+.2f} | fmt+2 {lo(lg[fp2]).mean():+.2f} | "
                  f"top1 base {[tok.decode([int(t)]) for t in top[base][:4]]}", flush=True)
        else:
            gen = np.full((len(crow), max_new), -1, np.int32)
            plen = np.zeros(len(crow), np.int32)
            k = 0
            for b, enc, out in zip(batches, encs, outs):
                out = out.numpy()
                # generator output = left-padded prompt + generated (fewer than max_new
                # columns when every row hit EOS early), so take the prompt length
                # from the encoding, never from the output width.
                Tp = int(enc["input_ids"].shape[1])
                assert out.shape[0] == len(b) and Tp < out.shape[1] <= Tp + max_new
                assert np.array_equal(out[:, :Tp], enc["input_ids"].numpy())
                for j in range(len(b)):
                    g = out[j, Tp:]
                    cut = len(g)
                    for t_i, t in enumerate(g):
                        if int(t) in EOS_IDS:
                            cut = t_i + 1
                            break
                    gen[k, :cut] = g[:cut]
                    plen[k] = Tp
                    k += 1
            arrays.update(gen_ids=gen, prompt_len=plen)
            for j in (0, 3):
                ids = [int(t) for t in gen[j] if t >= 0 and int(t) not in EOS_IDS]
                print(f"QC [{crow[j]['cid']}] {tok.decode(ids)[:120]!r}", flush=True)

        buf = io.BytesIO()
        np.savez(buf, **arrays)
        buf.seek(0)
        for attempt in range(MAX_RETRIES):
            try:
                api.upload_file(path_or_fileobj=buf, path_in_repo=chunk_name(prefix, c), repo_id=args.repo,
                                repo_type="dataset")
                break
            except Exception as exc:  # noqa: BLE001
                wait = BACKOFF_S * (2 ** attempt)
                print(f"chunk {c} upload attempt {attempt + 1} failed ({exc}); retrying in {wait}s", flush=True)
                if attempt == MAX_RETRIES - 1:
                    raise
                buf.seek(0)
                time.sleep(wait)
        print(f"chunk {c + 1}/{len(chunks)} done: {len(crow)} rows, ndif {t_ndif:.0f}s, total {time.time() - t0:.0f}s",
              flush=True)
    print(f"STEER PASS DONE: {spec['hf_id']} [{args.mode}]")


if __name__ == "__main__":
    main()
