"""Regression check for nocue_sweep.py.

The no-cue sweep recomputes the stated-stratum probes with code copied from the
scripts that produced the checked-in v3 layer sweeps. Every stated-stratum leaf
in the reference JSON must therefore reappear unchanged; anything else means the
probes moved and the no-cue projections are not comparable to the published
numbers. Extra keys (the "nocue" block) are expected and ignored.

Usage: verify_nocue_sweep.py new_sweep.json reference_sweep.json [tol]
"""

import json
import math
import sys

new_path, ref_path = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9

new = json.load(open(new_path))
ref = json.load(open(ref_path))
assert len(new) == len(ref), f"layer count {len(new)} != {len(ref)}"

worst = (0.0, "")
n_leaves = 0


def compare(a, b, path):
    global worst, n_leaves
    if isinstance(b, dict):
        for k in b:
            assert k in a, f"missing {path}/{k}"
            compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(b, list):
        assert len(a) == len(b), f"length {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(b, float):
        n_leaves += 1
        if math.isnan(b):
            assert math.isnan(a), f"{path}: expected nan, got {a}"
            return
        d = abs(a - b)
        if d > worst[0]:
            worst = (d, path)
    else:
        n_leaves += 1
        assert a == b, f"{path}: {a} != {b}"


for i, (a, b) in enumerate(zip(new, ref)):
    compare(a, b, f"L{b['layer']}")

status = "PASS" if worst[0] <= tol else "FAIL"
print(f"{status}: {len(ref)} layers, {n_leaves} leaves, "
      f"max abs deviation {worst[0]:.3e} at {worst[1] or '-'} (tol {tol:.0e})")
sys.exit(0 if worst[0] <= tol else 1)
