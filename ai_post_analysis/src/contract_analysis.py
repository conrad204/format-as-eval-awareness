"""Structured-output framing, tested *within* template.

The markup finding has one unavoidable weakness in this render: only three of
the thirteen templates are markup-serialized, and each is its own template, so
"serialization" and "these three templates" cannot be separated.

The response-contract slot gives an independent handle on the same mechanism
that does not have that problem. Contracts rotate over 4-5 values inside every
v2 template, so they are crossed with template *and* with payload block:

    C0  (empty)
    C1  "Give the response in a short phrase."
    C2  "Reply using only the option label."
    C3  "Give the final response by itself at the end."
    C4  'Return an object that has one field named "answer".'

C4 is the one that asks for a structured, machine-readable *response*, while
template serialization governs the *prompt*. If what pulls an uncued prompt
toward the deployment pole is structured-output framing rather than the
idiosyncrasies of three templates, C4 should sit below C0-C3 inside the same
template.

The model is a two-way fixed-effects regression over the no-cue rows,

    s_i = alpha_block(i) + gamma_template(i) + delta_contract(i) + eps_i

with block and template effects absorbed by alternating demeaning, so every
contract coefficient is identified from prompts that share both a payload and
a wrapper. As elsewhere, the same contrast is also computed on the cued rows
and differenced, which asks whether the contract effect - like the markup
effect - exists only when no purpose is stated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint  # noqa: E402
from axis import _block_means  # noqa: E402
from axis import PROBE_CHOICES  # noqa: E402
from design import load_design  # noqa: E402

FREEZE = "activations/full320-v3"
STRUCTURED = "C4"


def two_way_demean(y, g1, n1, g2, n2, n_iter=30, tol=1e-10):
    """Absorb two crossed fixed effects by alternating within-group demeaning.

    Exact for balanced designs and converges quickly here; iterating to a fixed
    point avoids building a 300+ column dummy matrix inside the bootstrap.
    """
    y = np.asarray(y, dtype=np.float64).copy()
    for _ in range(n_iter):
        before = y.copy()
        for g, n in ((g1, n1), (g2, n2)):
            sums = np.zeros(n)
            counts = np.zeros(n)
            np.add.at(sums, g, y)
            np.add.at(counts, g, 1)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            y -= means[g]
        if np.max(np.abs(y - before)) < tol:
            break
    return y


def contract_codes(meta_rows):
    """Integer codes for contract id, with the legacy no-contract rows flagged."""
    vals = [m["contract_id"] for m in meta_rows]
    names = sorted({v for v in vals if v is not None})
    lookup = {v: i for i, v in enumerate(names)}
    codes = np.array([lookup.get(v, -1) for v in vals], dtype=np.int32)
    return codes, names


def group_effect(y, block, tpl, contract, n_blocks, n_tpl, target, keep):
    """Mean residual of the target contract minus the mean of the others."""
    r = two_way_demean(y[keep], block[keep], n_blocks, tpl[keep], n_tpl)
    c = contract[keep]
    a = r[c == target]
    b = r[c != target]
    return float(a.mean() - b.mean())


def layer_row(oof, none, d, meta_cued, meta_none, n_boot, seed):
    nb, nt = d.n_blocks, len(d.tpl_names)
    cc_n, names = contract_codes(meta_none)
    cc_c, _ = contract_codes(meta_cued)
    if STRUCTURED not in names:
        raise RuntimeError(f"contract {STRUCTURED} not present")
    target = names.index(STRUCTURED)

    # Legacy anchors carry no contract; they cannot inform a contract contrast.
    keep_n = cc_n >= 0
    keep_c = cc_c >= 0

    scale = (np.nanmean(_block_means(oof, d.cued_block, d.cued_y == 1, nb))
             - np.nanmean(_block_means(oof, d.cued_block, d.cued_y == 0, nb)))

    e_none = group_effect(none, d.none_block, d.none_tpl, cc_n, nb, nt, target, keep_n)
    e_cued = group_effect(oof, d.cued_block, d.cued_tpl, cc_c, nb, nt, target, keep_c)

    # Per-contract residual means, for the full ordering.
    r_none = two_way_demean(none[keep_n], d.none_block[keep_n], nb, d.none_tpl[keep_n], nt)
    by_contract = {names[i]: float(r_none[cc_n[keep_n] == i].mean() / scale)
                   for i in range(len(names))}

    rng = np.random.default_rng(seed)
    rows_by_block_n = [np.where(d.none_block == b)[0] for b in range(nb)]
    rows_by_block_c = [np.where(d.cued_block == b)[0] for b in range(nb)]
    bn, bc = [], []
    E = _block_means(oof, d.cued_block, d.cued_y == 1, nb)
    D = _block_means(oof, d.cued_block, d.cued_y == 0, nb)
    for _ in range(n_boot):
        pick = rng.integers(0, nb, nb)
        sc = np.nanmean(E[pick]) - np.nanmean(D[pick])
        for rows_by_block, y, blk, tpl, cc, out in (
                (rows_by_block_n, none, d.none_block, d.none_tpl, cc_n, bn),
                (rows_by_block_c, oof, d.cued_block, d.cued_tpl, cc_c, bc)):
            idx = np.concatenate([rows_by_block[b] for b in pick])
            gid = np.repeat(np.arange(nb), [len(rows_by_block[b]) for b in pick])
            k = cc[idx] >= 0
            r = two_way_demean(y[idx][k], gid[k], nb, tpl[idx][k], len(d.tpl_names))
            c = cc[idx][k]
            out.append((r[c == target].mean() - r[c != target].mean()) / sc)
    bn, bc = np.array(bn), np.array(bc)

    def ci(v):
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {
        "contracts": names,
        "structured_contract": STRUCTURED,
        "residual_mean_by_contract_no_cue": by_contract,
        "structured_vs_rest": {
            "no_cue": {"point": e_none / scale, "ci95": ci(bn)},
            "cued": {"point": e_cued / scale, "ci95": ci(bc)},
            "differential": {"point": (e_none - e_cued) / scale, "ci95": ci(bn - bc)},
        },
        "n_no_cue_used": int(keep_n.sum()),
        "n_cued_used": int(keep_c.sum()),
        "scale_stated": float(scale),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="*", type=int)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--probe", choices=list(PROBE_CHOICES), default="logistic",
                    help="estimator; non-default variants write to their own namespace")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    vkey = checkpoint.variant_key(args.model, args.probe)
    z = np.load(rd / f"{vkey}_scores.npz")
    smeta = json.loads((rd / f"{vkey}_scores_meta.json").read_text())
    d = load_design(Path(FREEZE) / args.model / "meta.jsonl")
    meta_cued = [d.meta[i] for i in d.cued_idx]
    meta_none = [d.meta[i] for i in d.none_idx]

    layers = args.layers if args.layers else list(range(z["oof_cued"].shape[0]))
    stem = f"{vkey}_contracts"
    checkpoint.guard_params(rd, stem, {"n_boot": args.n_boot, "seed": args.seed}, args.force)
    todo = checkpoint.pending(layers, rd, stem, args.force)
    checkpoint.report(stem, layers, todo)

    for L in todo:
        r = layer_row(z["oof_cued"][L].astype(np.float64),
                      z["none"][L].astype(np.float64),
                      d, meta_cued, meta_none, args.n_boot, args.seed)
        r["layer"] = int(L)
        checkpoint.write_json(rd, stem, L, r)
        s = r["structured_vs_rest"]
        print(f"L{L}: C4-vs-rest no-cue={s['no_cue']['point']:+.4f} "
              f"cued={s['cued']['point']:+.4f} diff={s['differential']['point']:+.4f}",
              flush=True)

    have = sorted(checkpoint.done_layers(rd, stem))
    rows = checkpoint.read_json_shards(rd, stem, have)
    (rd / f"{stem}.json").write_text(json.dumps({
        "header": smeta | {"n_boot": args.n_boot, "seed": args.seed,
                           "model_spec": "s ~ block FE + template FE + contract",
                           "layers": have},
        "layers": rows}, indent=2))
    print(f"wrote {rd / f'{stem}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
