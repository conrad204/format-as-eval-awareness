"""Round-trip acceptance check for sweep coefficient persistence.

1. Regression: the modified sweep's JSON must equal the checked-in reference.
2. Round-trip: rebuilding OOF decision scores from the persisted coef /
   intercept / scaler arrays must reproduce every regime's AUC at every layer.
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

items, npz_path, sweep_json, coefs_path, reference_json = sys.argv[1:6]

rows = [json.loads(l) for l in open(items) if l.strip()]
a = np.load(npz_path)
A = a["X"]
y = np.array([int(r["label"]) for r in rows])
fam = np.array([r["purpose_family_id"] for r in rows])
fmt = np.array([r["format"] for r in rows])

new = json.load(open(sweep_json))
ref = json.load(open(reference_json))


def assert_superset(a, b, path=""):
    """Every leaf in the reference must appear unchanged in the new output.

    The checked-in reference predates the n-gram-control fields the current
    script adds, so the new JSON may have extra keys but must never disagree.
    """
    if isinstance(b, dict):
        for k in b:
            assert k in a, f"missing {path}/{k}"
            assert_superset(a[k], b[k], f"{path}/{k}")
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_superset(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a} != {b}"


assert_superset(new, ref)
print(f"regression: {sweep_json} matches every value in {reference_json} "
      f"({len(new)} layers)")

C = np.load(coefs_path, allow_pickle=False)
assert [str(x) for x in C["item_ids"]] == [r["item_id"] for r in rows]
folds = [str(f) for f in C["fold_families"]]

worst = 0.0
checked = 0
for record in new:
    L = record["layer"]
    X = A[:, L, :].astype(float)
    for regime in ("confounded", "decorrelated", "b2c", "c2b", "format"):
        yy = (fmt == "benchmark").astype(int) if regime == "format" else y
        te = C[f"mask::{regime}::eval"]
        s = np.full(len(rows), np.nan)
        for f in folds:
            tem = (fam == f) & te
            if not tem.any():
                continue
            tag = f"layer{L:02d}::{regime}::fold_{f}"
            z = (X[tem] - C[f"{tag}::scaler_mean"]) / C[f"{tag}::scaler_scale"]
            s[tem] = z @ C[f"{tag}::coef"] + C[f"{tag}::intercept"]
        m = ~np.isnan(s)
        auc = float(roc_auc_score(yy[m], s[m]))
        stored = record[regime]["auc"]
        worst = max(worst, abs(auc - stored))
        assert abs(auc - stored) < 1e-9, (L, regime, auc, stored)
        checked += 1
print(f"round-trip: {checked} layer x regime AUCs reproduced from persisted "
      f"coefficients (max |delta| = {worst:.2e})")
