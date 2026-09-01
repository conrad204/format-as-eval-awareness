"""Paired comparison of two recognition banks that share their episode families.

Two banks measured on the same 32 families can be contrasted within family, which
removes family difficulty from the comparison. This is the estimator already used
for the evaluative-versus-arbitrary specificity gap inside
`verify_hidden_channel_package.py`; it lives here so any paired contrast (dose-8
versus dose-16, evaluative versus arbitrary) uses one code path and writes one
artifact shape.

Per family, accuracy is the mean of `target_logit_margin > 0` over that family's
rows in the bank; the statistic is the mean over families of (B - A), with a
family bootstrap for its interval. Rows may be restricted per side with
`--a-prefix` / `--b-prefix` when a run file carries several banks.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_accuracy(rows: list[dict], prefix: str | None) -> dict[str, float]:
    by: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "recognition":
            continue
        if prefix and not r["condition"].startswith(prefix):
            continue
        by[r["episode_family"]].append(float(r["target_logit_margin"] > 0))
    if not by:
        raise RuntimeError(f"no recognition rows for prefix {prefix!r}")
    return {k: float(np.mean(v)) for k, v in by.items()}


def family_value(rows: list[dict], prefix: str | None, minus: str | None) -> dict[str, float]:
    """Per-family accuracy, or a within-family contrast when `minus` is given."""
    high = family_accuracy(rows, prefix)
    if minus is None:
        return high
    low = family_accuracy(rows, minus)
    return {f: high[f] - low[f] for f in sorted(set(high) & set(low))}


def load(paths: list[Path]) -> dict[str, dict]:
    out = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_source"] = str(path)
        data["_sha256"] = sha256(path)
        out[data["model_key"]] = data
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", type=Path, required=True, help="baseline runs")
    ap.add_argument("--b", nargs="+", type=Path, required=True, help="comparison runs")
    ap.add_argument("--a-prefix", default=None, help="condition prefix selecting the baseline bank")
    ap.add_argument("--b-prefix", default=None, help="condition prefix selecting the comparison bank")
    ap.add_argument("--a-minus-prefix", default=None,
                    help="make the baseline side a within-family contrast: a-prefix minus this prefix")
    ap.add_argument("--b-minus-prefix", default=None,
                    help="make the comparison side a within-family contrast: b-prefix minus this prefix")
    ap.add_argument("--a-label", default="a")
    ap.add_argument("--b-label", default="b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    side_a, side_b = load(args.a), load(args.b)
    rng = np.random.default_rng(args.seed)
    models = {}
    for key in sorted(set(side_a) & set(side_b)):
        a = family_value(side_a[key]["rows"], args.a_prefix, args.a_minus_prefix)
        b = family_value(side_b[key]["rows"], args.b_prefix, args.b_minus_prefix)
        families = sorted(set(a) & set(b))
        if not families:
            raise RuntimeError(f"{key}: banks share no episode family")
        diff = np.array([b[f] - a[f] for f in families])
        idx = rng.integers(0, len(families), (args.n_bootstrap, len(families)))
        draws = diff[idx].mean(axis=1)
        lo, hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
        models[key] = {
            "a_source": side_a[key]["_source"], "a_sha256": side_a[key]["_sha256"],
            "b_source": side_b[key]["_source"], "b_sha256": side_b[key]["_sha256"],
            "a_prefix": args.a_prefix, "b_prefix": args.b_prefix,
            "a_minus_prefix": args.a_minus_prefix, "b_minus_prefix": args.b_minus_prefix,
            args.a_label: float(np.mean([a[f] for f in families])),
            args.b_label: float(np.mean([b[f] for f in families])),
            "paired_gap": float(diff.mean()), "paired_gap_ci95": [lo, hi],
            "families_b_higher": int((diff > 0).sum()), "n_families": len(families),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }

    result = {"schema_version": 1, "label": args.label, "n_bootstrap": args.n_bootstrap,
              "seed": args.seed,
              "inference": "paired within shared episode family, family bootstrap",
              "gap_definition": f"mean over families of ({args.b_label} - {args.a_label})",
              "models": models}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, rec in models.items():
        print(f"[{args.label}] {key}: {args.a_label} {rec[args.a_label]:.4f} -> "
              f"{args.b_label} {rec[args.b_label]:.4f} | gap {rec['paired_gap']:+.4f} "
              f"CI {[round(x, 4) for x in rec['paired_gap_ci95']]} | excludes 0 {rec['ci_excludes_zero']}"
              f" | families higher {rec['families_b_higher']}/{rec['n_families']}", flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
