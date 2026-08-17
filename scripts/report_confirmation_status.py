"""Report every predeclared endpoint and conservative claim status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ENDPOINTS = (
    "E1_stated_auc", "E1_ngram_auc", "E1_auc_minus_ngram", "E2_stated_locus",
    "E3_transfer_by_format", "E4_indirect_transfer", "E5_second_transfer",
    "E6_nocue_format_shift_sd", "E7_payload_holdout", "E8_double_holdout", "per_fold_stated",
)


def ci_excludes_zero(entry: dict) -> bool:
    ci = entry.get("ci95")
    return isinstance(ci, list) and len(ci) == 2 and (ci[0] > 0 or ci[1] < 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        endpoints = report.get("endpoints", {})
        models[path.stem] = {
            "config": report.get("config", {}),
            "endpoints": {name: endpoints.get(name, {"status": "missing"}) for name in ENDPOINTS},
        }
    claim_status = {}
    for name, report in models.items():
        e = report["endpoints"]
        e1 = e.get("E1_auc_minus_ngram", {})
        e4 = e.get("E4_indirect_transfer", {})
        e5 = e.get("E5_second_transfer", {})
        e8 = e.get("E8_double_holdout", {})
        claim_status[name] = {
            "A_vocabulary_general_stated_purpose": bool(e1.get("point", -1) >= 0.05 and ci_excludes_zero(e1)),
            "B_generalizes_to_implied_purpose": ci_excludes_zero(e4) and e4.get("point", 0) > 0.5,
            "C_generalizes_across_perspective": ci_excludes_zero(e5) and e5.get("point", 0) > 0.5,
            "D_independent_of_payload_task_identity": ci_excludes_zero(e8) and e8.get("point", 0) > 0.5,
            "E_format_alone_induces_evaluation_state": False,
            "E6_interpretation": "projection_or_drift_only; not evidence of an evaluation-like state",
        }
    output = {"schema_version": 1, "reports": models, "claims": claim_status, "no_selective_reporting": True}
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": sorted(models), "out": str(args.out)}))


if __name__ == "__main__":
    main()
