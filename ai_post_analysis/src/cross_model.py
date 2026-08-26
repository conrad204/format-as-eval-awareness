"""Do the three models order the templates the same way?

Each model is fitted independently, so agreement between them about *which*
wrappers read as deployment is evidence that the ladder tracks a property of
the prompt rather than one checkpoint's idiosyncrasy.

Models differ in depth (32 vs 80 blocks), so layers are matched by normalized
depth q = layer / (n_layers - 1), taking each model's nearest layer to the
requested q. Two quantities are compared:

  pi(t)            the raw no-cue placement
  differential(t)  the no-cue-specific part, offset_none(t) - offset_cued(t),
                   which is what the headline claim rests on

Correlations over 13 templates are descriptive - the templates are a fixed
catalog, not a sample - so Spearman is reported alongside Pearson and no
p-value is attached to a correlation over so few, non-independent, points.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

Q_POINTS = (0.25, 0.375, 0.5, 0.75, 1.0)


def nearest_layer(rows, n_layers, q):
    target = q * (n_layers - 1)
    return min(rows, key=lambda r: abs(r["layer"] - target))


def load(rd: Path, model: str, kind: str):
    p = rd / f"{model}_{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="ai_post_analysis/results")
    ap.add_argument("--models", nargs="*",
                    default=["llama31_8b", "llama31_70b", "llama33_70b"])
    args = ap.parse_args()
    rd = Path(args.results_dir)

    data = {}
    for m in args.models:
        tpl, off = load(rd, m, "templates"), load(rd, m, "offsets")
        if tpl is None:
            continue
        n = tpl["header"]["n_transformer_layers"]
        data[m] = {"n_layers": n, "templates": tpl,
                   "offsets": {r["layer"]: r for r in off["layers"]} if off else None}
    if len(data) < 2:
        print("need at least two models with results; nothing to compare")
        return 0

    tpl_names = sorted(next(iter(data.values()))["templates"]["layers"][0]["pi_by_template"])
    out = {"header": {"models": list(data), "q_points": list(Q_POINTS),
                      "templates": tpl_names,
                      "note": "13 fixed templates; correlations are descriptive"},
           "comparisons": []}

    for q in Q_POINTS:
        vecs_pi, vecs_diff, layers = {}, {}, {}
        for m, dm in data.items():
            row = nearest_layer(dm["templates"]["layers"], dm["n_layers"], q)
            layers[m] = row["layer"]
            vecs_pi[m] = np.array([row["pi_by_template"][t] for t in tpl_names])
            off = (dm["offsets"] or {}).get(row["layer"])
            if off:
                vecs_diff[m] = np.array([off["offset_none_by_template"][t]
                                         - off["offset_cued_by_template"][t]
                                         for t in tpl_names])
        entry = {"q": q, "layer_by_model": layers, "pairs": []}
        for a, b in combinations(vecs_pi, 2):
            pair = {"models": [a, b],
                    "pi_pearson": float(np.corrcoef(vecs_pi[a], vecs_pi[b])[0, 1]),
                    "pi_spearman": float(spearmanr(vecs_pi[a], vecs_pi[b]).statistic)}
            if a in vecs_diff and b in vecs_diff:
                pair["differential_pearson"] = float(
                    np.corrcoef(vecs_diff[a], vecs_diff[b])[0, 1])
                pair["differential_spearman"] = float(
                    spearmanr(vecs_diff[a], vecs_diff[b]).statistic)
            entry["pairs"].append(pair)
        out["comparisons"].append(entry)

    path = rd / "cross_model_agreement.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    for e in out["comparisons"]:
        for p in e["pairs"]:
            extra = (f" | differential r={p['differential_pearson']:+.3f} "
                     f"rho={p['differential_spearman']:+.3f}"
                     if "differential_pearson" in p else "")
            print(f"q={e['q']:<6} {p['models'][0]:>13} vs {p['models'][1]:<13} "
                  f"pi r={p['pi_pearson']:+.3f} rho={p['pi_spearman']:+.3f}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
