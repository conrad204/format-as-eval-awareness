#!/usr/bin/env python3
"""Per-layer purpose-score scale for the interchange tradeoff panel.

`purpose_disruption_gap` in the interchange bootstrap is a difference of mean
absolute shifts in the frozen purpose probe's DECISION SCORE. That is a
scale-dependent quantity: the probe score's spread varies by layer and by
checkpoint, so raw values are not comparable across either axis, and a panel
that overlays three models across depth silently mixes units.

This script emits, per (model, layer), the standard deviation of the baseline
purpose score, and the disruption gap expressed in units of that SD:

    gap_sd(layer) = ( mean|dpurpose|_control - mean|dpurpose|_real ) / sd_baseline(layer)

Positive = the real swap perturbs the purpose readout LESS than a
magnitude-matched random perturbation, i.e. purpose is relatively protected.

This is the same score-SD convention already used for the no-cue displacement
metric, so the paper reports one normalization rather than two.

Usage:
  python scripts/interchange_purpose_scale.py llama33_70b \
      results/tae_ixpatch_full_llama33_70b.jsonl \
      results/tae_ixpatch_matched_llama33_70b.jsonl \
      -- results/tae_ixpatch_purposescale_llama33_70b.json
  # add --even-only for an even-layer grid
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    argv = list(sys.argv[1:])
    even_only = "--even-only" in argv
    if even_only:
        argv.remove("--even-only")
    sep = argv.index("--")
    model = argv[0]
    inputs = [Path(p) for p in argv[1:sep]]
    out_path = Path(argv[sep + 1])

    base: dict[int, list[float]] = defaultdict(list)          # baseline purpose scores
    shift: dict[tuple[int, str], list[float]] = defaultdict(list)  # |delta purpose|

    for p in inputs:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if even_only and r["layer"] % 2:
                    continue
                c = r["condition"]
                if c not in ("real_swap", "matched_control"):
                    continue
                L = r["layer"]
                if c == "real_swap":
                    base[L].append(r["purpose_score_baseline"])
                shift[(L, c)].append(abs(r["purpose_score_patched"] - r["purpose_score_baseline"]))

    layers = sorted(L for L in base if (L, "matched_control") in shift)
    by_layer = {}
    for L in layers:
        sd = float(np.std(base[L], ddof=1))
        real = float(np.mean(shift[(L, "real_swap")]))
        ctrl = float(np.mean(shift[(L, "matched_control")]))
        by_layer[str(L)] = {
            "baseline_purpose_score_sd": sd,
            "mean_abs_shift_real": real,
            "mean_abs_shift_control": ctrl,
            "purpose_gap_raw": ctrl - real,
            "purpose_gap_sd_units": (ctrl - real) / sd if sd > 0 else float("nan"),
        }

    out = {
        "schema_version": 1,
        "mode": "development",
        "purpose": "per-layer purpose-probe score scale and SD-normalized disruption gap; "
                   "positive = real swap perturbs purpose LESS than the magnitude-matched control",
        "model": model,
        "even_layer_grid": even_only,
        "n_layers": len(layers),
        "inputs": [str(p) for p in inputs],
        "by_layer": by_layer,
    }
    out_path.write_text(json.dumps(out, indent=2))
    vals = [by_layer[str(L)]["purpose_gap_sd_units"] for L in layers]
    print(f"{model:12s} layers={len(layers):3d}  gap_sd range [{min(vals):+.3f}, {max(vals):+.3f}]  "
          f"sd(baseline) range [{min(by_layer[str(L)]['baseline_purpose_score_sd'] for L in layers):.2f}, "
          f"{max(by_layer[str(L)]['baseline_purpose_score_sd'] for L in layers):.2f}]")


if __name__ == "__main__":
    main()
