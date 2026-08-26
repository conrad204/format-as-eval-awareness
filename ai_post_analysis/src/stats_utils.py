"""Small statistics helpers shared across the Direction 1 analyses."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def rank_exact_p(vals, group_mask):
    """Exact two-sided permutation test that a group sits at one extreme.

    With 13 templates the full set of group assignments is small enough to
    enumerate, so the small-n ordering claim is tested rather than eyeballed.
    The statistic is the difference in group means; the p-value is the fraction
    of assignments whose statistic is at least as extreme in absolute value.
    """
    vals = np.asarray(vals, dtype=float)
    g = np.asarray(group_mask, dtype=bool)
    k, n = int(g.sum()), len(vals)
    obs = vals[g].mean() - vals[~g].mean()
    cnt = tot = 0
    for c in combinations(range(n), k):
        m = np.zeros(n, dtype=bool)
        m[list(c)] = True
        tot += 1
        if abs(vals[m].mean() - vals[~m].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return {"observed_diff": float(obs), "p_two_sided": cnt / tot,
            "n_permutations": tot, "n_group": k, "n_total": n}
