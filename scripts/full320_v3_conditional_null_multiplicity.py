"""Multiple-testing correction for the 27 high-variance-subspace conditional-null
tests in results/full320_v3_conditional_hv_null.json. Zero new compute -- pure
post-hoc correction of already-computed p-values.
"""
import json
from pathlib import Path

from statsmodels.stats.multitest import multipletests

ALPHA = 0.05


def main():
    d = json.load(open("results/full320_v3_conditional_hv_null.json"))
    cells = d["cells"]
    p_emp = [c["p_below_conditional_null_empirical"] for c in cells]
    p_ana = [c["p_below_conditional_null_analytic"] for c in cells]

    def correct(pvals, label):
        rej_bonf, p_bonf, _, _ = multipletests(pvals, alpha=ALPHA, method="bonferroni")
        rej_holm, p_holm, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
        nominal = sum(p < ALPHA for p in pvals)
        return {
            "p_source": label,
            "n_tests": len(pvals),
            "nominal_p<0.05": nominal,
            "bonferroni_significant": int(sum(rej_bonf)),
            "holm_significant": int(sum(rej_holm)),
        }

    overall_emp = correct(p_emp, "empirical")
    overall_ana = correct(p_ana, "analytic")

    by_rank = {}
    for r in (1, 4, 16):
        idx = [i for i, c in enumerate(cells) if c["rank"] == r]
        sub_emp = [p_emp[i] for i in idx]
        # Correction is applied within the full 27-test family (not re-corrected per rank
        # subset), then we report how many of that rank's cells survive the family-wise
        # correction, so the breakdown is consistent with the overall correction above.
        rej_bonf_full, _, _, _ = multipletests(p_emp, alpha=ALPHA, method="bonferroni")
        rej_holm_full, _, _, _ = multipletests(p_emp, alpha=ALPHA, method="holm")
        by_rank[str(r)] = {
            "n_cells": len(idx),
            "nominal_p<0.05_empirical": sum(p < ALPHA for p in sub_emp),
            "bonferroni_significant_within_family_of_27": int(sum(rej_bonf_full[i] for i in idx)),
            "holm_significant_within_family_of_27": int(sum(rej_holm_full[i] for i in idx)),
        }

    result = {
        "schema_version": 1,
        "mode": "development",
        "purpose": "Holm and Bonferroni correction (alpha=0.05, m=27) of the high-variance-"
                   "subspace conditional-null p-values. No new compute.",
        "alpha": ALPHA,
        "overall_empirical_p": overall_emp,
        "overall_analytic_p": overall_ana,
        "by_rank_within_family_of_27": by_rank,
    }
    Path("results/full320_v3_conditional_null_multiplicity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
