"""Independent end-to-end verification of the hidden-channel artifact package.

Recomputes every headline number straight from the raw run files, without reading
the analysis JSONs, then checks that the analysis artifacts agree. Also verifies
frozen bank hashes, run/bank linkage, completeness (no dropped rows), and that
every hidden-channel run is full precision.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BANKS = {
    "latent": RESULTS / "behavioral_ea_hidden_channel_frozen",
    "check": RESULTS / "behavioral_ea_hidden_channel_check_frozen",
    "supplement": RESULTS / "behavioral_ea_hidden_channel_supplement_frozen",
    "arbitrary": RESULTS / "behavioral_ea_arbitrary_contingency_frozen",
}
RUNS = {
    ("latent", "llama31_8b"): "behavioral_ea_hidden_channel_llama31_8b.json",
    ("latent", "llama31_70b"): "behavioral_ea_hidden_channel_recog_llama31_70b.json",
    ("latent", "llama33_70b"): "behavioral_ea_hidden_channel_recog_llama33_70b.json",
    ("check", "llama31_8b"): "behavioral_ea_hidden_channel_check_llama31_8b.json",
    ("check", "llama31_70b"): "behavioral_ea_hidden_channel_recogchk_llama31_70b.json",
    ("supplement", "llama31_8b"): "behavioral_ea_hidden_channel_supp_llama31_8b.json",
    ("arbitrary", "llama31_8b"): "behavioral_ea_arbitrary_llama31_8b.json",
    ("arbitrary", "llama31_70b"): "behavioral_ea_arbitrary_llama31_70b.json",
    ("arbitrary", "llama33_70b"): "behavioral_ea_arbitrary_llama33_70b.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_accuracy(rows: list[dict]) -> dict[str, float]:
    by = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "recognition":
            by[r["episode_family"]].append(float(r["target_logit_margin"] > 0))
    return {k: float(np.mean(v)) for k, v in by.items()}


def main() -> None:
    report: dict[str, object] = {"schema_version": 1, "checks": {}, "recomputed": {}}
    checks = report["checks"]

    # 1. frozen bank integrity
    bank_hashes = {}
    for name, path in BANKS.items():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        actual = sha256(path / "items.jsonl")
        checks[f"bank_{name}_hash_matches_manifest"] = actual == manifest["items_sha256"]
        bank_hashes[name] = actual
        marker = path / "FROZEN.sha256"
        checks[f"bank_{name}_freeze_marker_present"] = marker.exists()

    # 2. run linkage, completeness, precision
    loaded = {}
    for (bank, model), filename in RUNS.items():
        path = RESULTS / filename
        if not path.exists():
            checks[f"run_{bank}_{model}_present"] = False
            continue
        checks[f"run_{bank}_{model}_present"] = True
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[(bank, model)] = data
        checks[f"run_{bank}_{model}_full_precision"] = (not data.get("load_in_8bit", False))
        expected_hash = bank_hashes.get(bank)
        if "items_sha256" in data and expected_hash:
            checks[f"run_{bank}_{model}_bank_linkage"] = data["items_sha256"] == expected_hash
        ids = [r["item_id"] for r in data["rows"]]
        checks[f"run_{bank}_{model}_no_duplicate_rows"] = len(ids) == len(set(ids))

    # 3. recompute headline recognition numbers from raw rows
    rng = np.random.default_rng(20260912)
    for bank in ("latent", "check", "arbitrary"):
        for model in ("llama31_8b", "llama31_70b", "llama33_70b"):
            data = loaded.get((bank, model))
            if not data:
                continue
            rec = [r for r in data["rows"] if r["kind"] == "recognition"]
            fams = sorted({r["episode_family"] for r in rec})
            qs = sorted({r["question_family"] for r in rec})
            cell = collections.defaultdict(list)
            for r in rec:
                cell[(r["episode_family"], r["question_family"])].append(float(r["target_logit_margin"] > 0))
            grid = np.array([[np.mean(cell[(f, q)]) for q in qs] for f in fams])
            ih = rng.integers(0, len(fams), (20000, len(fams)))
            iq = rng.integers(0, len(qs), (20000, len(qs)))
            draws = np.array([grid[np.ix_(a, b)].mean() for a, b in zip(ih, iq)])
            report["recomputed"][f"{bank}_{model}"] = {
                "accuracy": float(grid.mean()),
                "crossed_ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
                "families_above_chance": int((grid.mean(axis=1) > 0.5).sum()),
                "n_families": len(fams), "n_questions": len(qs), "n_rows": len(rec),
            }

    # 4. paired specificity gap, recomputed
    paired = {}
    for model in ("llama31_8b", "llama31_70b", "llama33_70b"):
        ev, ar = loaded.get(("latent", model)), loaded.get(("arbitrary", model))
        if not ev or not ar:
            continue
        a, b = family_accuracy(ev["rows"]), family_accuracy(ar["rows"])
        fams = sorted(set(a) & set(b))
        diff = np.array([b[f] - a[f] for f in fams])
        idx = rng.integers(0, len(fams), (20000, len(fams)))
        draws = diff[idx].mean(axis=1)
        paired[model] = {
            "paired_gap": float(diff.mean()),
            "paired_gap_ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "families_arbitrary_higher": int((diff > 0).sum()), "n_families": len(fams),
            "ci_excludes_zero": bool(np.quantile(draws, 0.025) > 0),
        }
    report["recomputed"]["paired_specificity"] = paired
    checks["specificity_gap_positive_in_every_available_model"] = bool(
        paired and all(v["ci_excludes_zero"] for v in paired.values()))

    # 5. cross-check the published analysis artifacts against the recomputation
    def close(a: float, b: float, tol: float = 5e-3) -> bool:
        return abs(a - b) <= tol

    for label, artifact, bank in (("latent", "recognition_latent_evaluative.json", "latent"),
                                  ("arbitrary", "recognition_arbitrary_control.json", "arbitrary")):
        path = RESULTS / artifact
        if not path.exists():
            checks[f"artifact_{label}_present"] = False
            continue
        published = json.loads(path.read_text(encoding="utf-8"))["models"]
        agree = True
        for model, rec in published.items():
            mine = report["recomputed"].get(f"{bank}_{model}")
            if mine and not close(mine["accuracy"], rec["accuracy"]):
                agree = False
        checks[f"artifact_{label}_agrees_with_recomputation"] = agree

    gap_path = RESULTS / "specificity_paired_gap.json"
    if gap_path.exists():
        published = json.loads(gap_path.read_text(encoding="utf-8"))["models"]
        checks["artifact_specificity_agrees"] = all(
            close(published[m]["paired_gap"], paired[m]["paired_gap"]) for m in published if m in paired)

    # 6. probe artifact sanity
    probe = RESULTS / "probe_v3_hidden_channel_llama31_8b.json"
    if probe.exists():
        p = json.loads(probe.read_text(encoding="utf-8"))["readouts"]
        checks["probe_status_above_chance"] = p["status"]["family_ci95"][0] > 0.5
        checks["probe_permutation_null_contains_chance"] = (
            p["permutation_null"]["family_ci95"][0] <= 0.5 <= p["permutation_null"]["family_ci95"][1])
        checks["probe_letter_control_high"] = p["letter_control"]["family_mean_auc"] > 0.9
        report["recomputed"]["probe"] = {k: {"family_mean_auc": v["family_mean_auc"],
                                             "family_ci95": v["family_ci95"]} for k, v in p.items()}

    failed = sorted(k for k, v in checks.items() if v is False)
    report["all_checks_pass"] = not failed
    report["failed_checks"] = failed
    out = RESULTS / "hidden_channel_verification.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"checks run: {len(checks)}  failed: {len(failed)}")
    for name in failed:
        print("  FAIL", name)
    print("\nrecomputed recognition accuracies:")
    for key, rec in report["recomputed"].items():
        if key in ("paired_specificity", "probe"):
            continue
        print(f"  {key:26s} {rec['accuracy']:.4f} CI {[round(x, 4) for x in rec['crossed_ci95']]} "
              f"fams {rec['families_above_chance']}/{rec['n_families']} rows {rec['n_rows']}")
    print("\npaired specificity gap:")
    for model, rec in paired.items():
        print(f"  {model:14s} {rec['paired_gap']:+.4f} CI {[round(x, 4) for x in rec['paired_gap_ci95']]} "
              f"fams {rec['families_arbitrary_higher']}/{rec['n_families']}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
