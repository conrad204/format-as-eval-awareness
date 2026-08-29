"""Exact analytic significance test for the matched-format-subspace geometry result
(results/full320_v3_direction_geometry.csv). Zero new model/probe compute -- this
is a closed-form post-hoc test applied to already-computed energy values.

For a uniformly random unit vector v in R^d, ||Q^T v||^2 for any fixed rank-r
orthonormal subspace Q follows Beta(r/2, (d-r)/2) exactly (projection of a
uniform point on the (d-1)-sphere onto a fixed r-dim subspace), with mean r/d.
This gives a parameter-free null for "is E_purpose_in_format(r) below chance",
per (model, depth, rank) cell, with no simulation and no free parameters beyond
the already-fixed r and the model's known hidden dimension d.

p = P(X <= observed) under Beta(r/2, (d-r)/2): small p means the purpose
direction's energy in the matched-format subspace is significantly BELOW what
an unrelated random direction would show by pure high-dimensional chance --
directly answering the "near-zero cosines aren't surprising in high-D" concern
with a real test rather than an eyeballed magnitude.
"""
import csv
import json
from pathlib import Path

from scipy.stats import beta

DIMS = {"llama31_8b": 4096, "llama31_70b": 8192, "llama33_70b": 8192}
CSV_PATH = "results/full320_v3_direction_geometry.csv"


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    out_rows = []
    for r in rows:
        model, depth, rank = r["model"], r["depth"], int(r["rank"])
        d = DIMS[model]
        obs = float(r["E_purpose_in_format_subspace"])
        a, b = rank / 2, (d - rank) / 2
        p = float(beta.cdf(obs, a, b))
        out_rows.append({
            "model": model, "depth": depth, "rank": rank, "dim": d,
            "observed_E_purpose_in_format": obs, "chance_null_mean_r_over_d": rank / d,
            "p_below_chance": p, "significant_at_0.05": p < 0.05,
            "significant_at_bonferroni_0.05_over_36": p < 0.05 / len(rows),
        })

    by_rank = {}
    for rank in sorted({row["rank"] for row in out_rows}):
        cells = [row for row in out_rows if row["rank"] == rank]
        by_rank[str(rank)] = {
            "n_cells": len(cells),
            "n_significant_0.05": sum(c["significant_at_0.05"] for c in cells),
            "n_significant_bonferroni": sum(c["significant_at_bonferroni_0.05_over_36"] for c in cells),
            "max_p": max(c["p_below_chance"] for c in cells),
            "worst_cell": max(cells, key=lambda c: c["p_below_chance"]),
        }

    result = {
        "schema_version": 1,
        "mode": "development",
        "purpose": "Exact analytic (Beta distribution) significance test of whether purpose-"
                   "direction energy inside the matched-format PCA subspace is below the "
                   "high-dimensional chance level r/d. Zero new compute -- post-hoc test of "
                   "results/full320_v3_direction_geometry.csv values.",
        "n_multiple_comparisons": len(rows),
        "cells": out_rows,
        "summary_by_rank": by_rank,
    }
    Path("results/full320_v3_direction_geometry_significance.json").write_text(json.dumps(result, indent=2))
    print("wrote results/full320_v3_direction_geometry_significance.json")
    for rank, s in by_rank.items():
        print(f"rank={rank}: {s['n_significant_0.05']}/{s['n_cells']} cells p<0.05, "
              f"{s['n_significant_bonferroni']}/{s['n_cells']} survive Bonferroni, "
              f"worst cell = {s['worst_cell']['model']}/{s['worst_cell']['depth']} "
              f"p={s['worst_cell']['p_below_chance']:.4f}")


if __name__ == "__main__":
    main()
