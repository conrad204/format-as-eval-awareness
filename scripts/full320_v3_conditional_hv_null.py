"""High-variance-subspace conditional null for the matched-format geometry result.

No activations, no probes, no npz reads -- reuses only two already-saved scalars
per (model, depth): purpose_energy_in_top64_pcs (P = ||V64 @ w_purpose||^2, from
results/full320_v3_direction_geometry.json) and the observed
E_purpose_in_format_subspace(r) at r in {1,4,16} (from
results/full320_v3_direction_geometry.csv).

Distinguishes:
  (A) purpose specifically avoids FORMAT directions
  (B) purpose simply avoids dominant high-variance directions in general

Method: draw a random rank-r orthonormal subspace R uniformly WITHIN the fixed
empirical top-64 PC subspace (i.e. a uniformly random r-dim subspace of R^64),
and ask how much of the purpose direction's energy a generic (format-unrelated)
high-variance sub-direction would capture by chance.

Rotational-invariance shortcut: for a uniformly random rank-r subspace of R^64,
the captured-energy distribution of any fixed vector c in R^64 depends only on
||c||^2, not on c's specific orientation (the Grassmannian's uniform measure is
rotation-invariant, so we can rotate any c to a canonical axis without changing
the projection-energy distribution). So instead of needing purpose's actual
64-dim coordinate vector (which was not persisted), a stand-in c0 with the same
norm (||c0||^2 = P, the already-saved purpose_energy_in_top64_pcs) is exactly
equivalent in distribution. This is why this control needs no rerun of anything.

Exactly:  ||R^T c||^2 / ||c||^2  ~  Beta(r/2, (64-r)/2)  for any fixed c != 0.
So the conditional null for the *full-space* quantity E_purpose_in_format(r) is
P * Beta(r/2, (64-r)/2) -- reported both as the exact analytic form and via an
explicit 10k-draw Monte Carlo (as requested), which will agree with it up to
sampling noise as a sanity check on the shortcut.

r=64 is excluded: a random rank-64 subspace of R^64 is the whole space (trivial).
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

N_DRAWS = 10_000
RANKS = (1, 4, 16)
HV_DIM = 64  # ambient dim for this conditional null: the top-64 PC subspace itself
SEED = 0


def draw_captured_energy_fraction(c0: np.ndarray, rank: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """||R^T c0||^2 for n_draws uniformly random rank-`rank` orthonormal R in R^{len(c0)}."""
    d = len(c0)
    out = np.empty(n_draws)
    for i in range(n_draws):
        G = rng.standard_normal((d, rank))
        R, _ = np.linalg.qr(G)
        out[i] = np.sum((R.T @ c0) ** 2)
    return out


def main():
    geom = json.load(open("results/full320_v3_direction_geometry.json"))
    csv_rows = list(csv.DictReader(open("results/full320_v3_direction_geometry.csv")))
    obs_by_cell = {(r["model"], r["depth"], int(r["rank"])): float(r["E_purpose_in_format_subspace"])
                   for r in csv_rows}

    rng = np.random.default_rng(SEED)
    results = []
    for model, mdata in geom["by_model"].items():
        for depth, ddata in mdata["by_depth"].items():
            P = ddata["purpose_energy_in_top64_pcs"]  # ||c||^2, already saved -- no rerun needed
            c0 = np.zeros(HV_DIM)
            c0[0] = np.sqrt(P)  # any fixed vector of this norm is distributionally equivalent (see docstring)
            for r in RANKS:
                observed = obs_by_cell[(model, depth, r)]
                draws = draw_captured_energy_fraction(c0, r, N_DRAWS, rng)
                k = int(np.sum(draws <= observed))
                p_empirical = (k + 1) / (N_DRAWS + 1)
                # exact analytic cross-check (should match p_empirical within Monte Carlo noise)
                p_analytic = float(beta.cdf(observed / P, r / 2, (HV_DIM - r) / 2)) if P > 0 else float("nan")
                cond_null_mean = P * r / HV_DIM
                below_conditional_null = observed < np.median(draws)
                results.append({
                    "model": model, "depth": depth, "rank": r,
                    "purpose_energy_in_top64_pcs_P": P,
                    "observed_E_purpose_in_format": observed,
                    "conditional_null_mean": cond_null_mean,
                    "conditional_null_median_empirical": float(np.median(draws)),
                    "conditional_null_std_empirical": float(draws.std()),
                    "n_draws": N_DRAWS,
                    "exceedances_leq_observed": f"{k}/{N_DRAWS}",
                    "p_below_conditional_null_empirical": p_empirical,
                    "p_below_conditional_null_analytic": p_analytic,
                    "below_conditional_null": bool(below_conditional_null),
                })
                print(f"{model:<12} {depth:<8} r={r:<3} P={P:.5f} obs={observed:.5f} "
                      f"cond_null_mean={cond_null_mean:.5f} p_emp={p_empirical:.4f} "
                      f"p_analytic={p_analytic:.4f}")

    n_below = sum(r["below_conditional_null"] for r in results)
    n_sig_below = sum(r["p_below_conditional_null_empirical"] < 0.05 for r in results)
    summary = {
        "n_cells_total": len(results),
        "n_cells_below_conditional_null_median": n_below,
        "n_cells_significantly_below_p0.05": n_sig_below,
    }
    print(f"\n{n_below}/{len(results)} cells below conditional-null median; "
          f"{n_sig_below}/{len(results)} significant at p<0.05")

    out = {
        "schema_version": 1,
        "mode": "development",
        "purpose": "High-variance-subspace conditional null: is purpose energy in the matched-"
                   "format subspace below chance EVEN CONDITIONAL ON already being confined to "
                   "the top-64 (dominant-variance) subspace? Distinguishes format-specific "
                   "avoidance from generic high-variance avoidance. Zero rerun of activations "
                   "or probes -- uses only already-saved purpose_energy_in_top64_pcs and "
                   "E_purpose_in_format_subspace values, via the rotational-invariance shortcut "
                   "documented in this script's module docstring.",
        "compared_to": "results/full320_v3_direction_geometry_significance.json ('full-space "
                        "isotropic null'); this file is the 'high-variance-subspace conditional null'.",
        "n_draws": N_DRAWS, "ranks_tested": list(RANKS),
        "excluded_rank_64_reason": "a random rank-64 subspace of the 64-dim top-PC space is the whole space",
        "cells": results,
        "summary": summary,
    }
    Path("results/full320_v3_conditional_hv_null.json").write_text(json.dumps(out, indent=2))
    print("wrote results/full320_v3_conditional_hv_null.json")


if __name__ == "__main__":
    main()
