"""Item-level model of where an uncued prompt lands on the purpose axis.

Reads the exported per-item scores and fits, over the 3,840 no-cue rows,

    s_i = alpha_block(i) + b_markup * markup_i + b_bench * benchmark_i
          + g * z(prompt_tokens_i) + eps_i

Payload-block fixed effects are absorbed by within-block demeaning, so every
coefficient is identified from contrasts between renderings of *the same*
semantic payload. Coefficients are rescaled into stated-cue units by the same
block-level eval-minus-deployment difference used elsewhere, and uncertainty is
a cluster bootstrap over payload blocks.

Two confounds motivate the extra terms. Prompt length varies by template, and
the readout position is the last prompt token, so the literal terminal token
differs by template; `terminal_field` is therefore also reported as a
restricted subgroup where markup and plain templates share the same ending.
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
from stats_utils import rank_exact_p  # noqa: E402

FREEZE = "activations/full320-v3"


def demean_by_group(M, groups, n_groups):
    """Subtract each group's mean from every column (fixed-effect absorption)."""
    M = np.asarray(M, dtype=np.float64)
    single = M.ndim == 1
    if single:
        M = M[:, None]
    sums = np.zeros((n_groups, M.shape[1]))
    counts = np.zeros(n_groups)
    np.add.at(sums, groups, M)
    np.add.at(counts, groups, 1)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    out = M - means[groups]
    return out[:, 0] if single else out


def fit_within(y, D, groups, n_groups):
    yd = demean_by_group(y, groups, n_groups)
    Dd = demean_by_group(D, groups, n_groups)
    beta, *_ = np.linalg.lstsq(Dd, yd, rcond=None)
    return beta


def analyse_layer(oof, none, d, tok_none, n_boot, seed):
    feats = d.tpl_feature_table
    markup_t = np.array([feats[t]["serialization"] != "plain" for t in d.tpl_names])
    bench_t = d.tpl_format.astype(bool)

    markup = markup_t[d.none_tpl].astype(float)
    bench = bench_t[d.none_tpl].astype(float)
    zlen = (tok_none - tok_none.mean()) / tok_none.std()

    nb = d.n_blocks
    designs = {
        "markup_only": np.column_stack([markup]),
        "markup_plus_family": np.column_stack([markup, bench]),
        "markup_family_length": np.column_stack([markup, bench, zlen]),
    }
    names = {"markup_only": ["markup"],
             "markup_plus_family": ["markup", "benchmark_family"],
             "markup_family_length": ["markup", "benchmark_family", "z_prompt_tokens"]}

    E = _block_means(oof, d.cued_block, d.cued_y == 1, nb)
    D_ = _block_means(oof, d.cued_block, d.cued_y == 0, nb)
    scale = np.nanmean(E) - np.nanmean(D_)

    point = {}
    for key, X in designs.items():
        b = fit_within(none, X, d.none_block, nb)
        point[key] = {n: float(v / scale) for n, v in zip(names[key], b)}

    # cluster bootstrap over blocks
    rng = np.random.default_rng(seed)
    boot = {k: {n: [] for n in names[k]} for k in designs}
    block_rows = [np.where(d.none_block == b)[0] for b in range(nb)]
    for _ in range(n_boot):
        pick = rng.integers(0, nb, nb)
        rows = np.concatenate([block_rows[b] for b in pick])
        gid = np.repeat(np.arange(nb), [len(block_rows[b]) for b in pick])
        sc = np.nanmean(E[pick]) - np.nanmean(D_[pick])
        for key, X in designs.items():
            b = fit_within(none[rows], X[rows], gid, nb)
            for n, v in zip(names[key], b):
                boot[key][n].append(v / sc)
    ci = {k: {n: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
              for n, v in d_.items()} for k, d_ in boot.items()}

    # per-serialization breakdown and terminal-field-matched subgroup
    pi = {}
    for i, t in enumerate(d.tpl_names):
        m = d.none_tpl == i
        pi[t] = float((np.nanmean(_block_means(none, d.none_block, m, nb)) - np.nanmean(D_)) / scale)
    by_serial = {}
    for s in sorted({feats[t]["serialization"] for t in d.tpl_names}):
        ts = [t for t in d.tpl_names if feats[t]["serialization"] == s]
        by_serial[s] = {"templates": ts, "mean_pi": float(np.mean([pi[t] for t in ts]))}

    no_term = [t for t in d.tpl_names if feats[t]["terminal_field"] is None]
    term_test = rank_exact_p([pi[t] for t in no_term],
                             [feats[t]["serialization"] != "plain" for t in no_term])

    tok_by_tpl = {t: float(tok_none[d.none_tpl == i].mean()) for i, t in enumerate(d.tpl_names)}
    lengths = np.array([tok_by_tpl[t] for t in d.tpl_names])
    pis = np.array([pi[t] for t in d.tpl_names])
    r = float(np.corrcoef(lengths, pis)[0, 1])

    return {
        "coefficients_stated_units": point,
        "coefficients_ci95": ci,
        "pi_by_template": pi,
        "by_serialization": by_serial,
        "terminal_field_none_subgroup": {
            "templates": no_term, "rank_test": term_test,
        },
        "mean_prompt_tokens_by_template": tok_by_tpl,
        "corr_pi_vs_length_across_templates": r,
        "scale_stated": float(scale),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="*", type=int)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--force", action="store_true",
                    help="discard existing per-layer checkpoints and recompute")
    ap.add_argument("--probe", choices=list(PROBE_CHOICES), default="logistic",
                    help="estimator; non-default variants write to their own namespace")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    vkey = checkpoint.variant_key(args.model, args.probe)
    z = np.load(rd / f"{vkey}_scores.npz")
    smeta = json.loads((rd / f"{vkey}_scores_meta.json").read_text())
    d = load_design(Path(FREEZE) / args.model / "meta.jsonl")
    tok_none = z["token_lengths"][d.none_idx].astype(np.float64)

    layers = args.layers if args.layers else list(range(z["oof_cued"].shape[0]))
    stem = f"{vkey}_regression"
    checkpoint.guard_params(rd, stem,
                            {"n_boot": args.n_boot, "seed": args.seed}, args.force)
    todo = checkpoint.pending(layers, rd, stem, args.force)
    checkpoint.report(stem, layers, todo)

    for L in todo:
        r = analyse_layer(z["oof_cued"][L].astype(np.float64),
                          z["none"][L].astype(np.float64),
                          d, tok_none, args.n_boot, args.seed)
        r["layer"] = int(L)
        checkpoint.write_json(rd, stem, L, r)
        print(f"L{L}: markup={r['coefficients_stated_units']['markup_family_length']['markup']:+.4f}",
              flush=True)

    have = sorted(checkpoint.done_layers(rd, stem))
    rows = checkpoint.read_json_shards(rd, stem, have)
    out = {"header": smeta | {"n_boot": args.n_boot, "seed": args.seed,
                              "model_spec": "s ~ block FE + markup + benchmark_family + z(prompt_tokens)"},
           "layers": rows}
    path = rd / f"{args.model}_regression.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
