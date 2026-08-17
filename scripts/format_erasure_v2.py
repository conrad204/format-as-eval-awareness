"""Rerun development format erasure with corrected synthetic control and provenance."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from format_erasure_development import PROBE, RANKS, run_model, sha256_file, synthetic_checks

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--supersedes", type=Path, required=True)
    args = ap.parse_args()
    script = Path(__file__).resolve()
    report = {
        "schema_version": 2,
        "mode": "development",
        "model": args.model,
        "supersedes": str(args.supersedes),
        "script": {"path": str(script), "sha256": sha256_file(script)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "items": {"path": str(args.items), "sha256": sha256_file(args.items)},
        "metadata_sha256": sha256_file(args.items),
        "activations": {"path": str(args.activations), "sha256": sha256_file(args.activations)},
        "layer": args.layer,
        "probe": PROBE,
        "ranks": list(RANKS),
        "synthetic_checks": synthetic_checks(),
        "result": run_model(args.items, args.activations, args.layer),
    }
    if not report["synthetic_checks"]["pass"]:
        raise SystemExit("corrected synthetic control failed")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "script_sha256": report["script"]["sha256"], "synthetic": report["synthetic_checks"]}))

if __name__ == "__main__":
    main()
