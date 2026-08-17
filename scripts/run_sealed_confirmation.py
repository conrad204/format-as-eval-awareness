"""Execute one immutable, manifest-bound confirmatory analysis.

The wrapper is deliberately stricter than confirmation_analysis.py. It accepts
only a sealed manifest, verifies every referenced artifact before execution,
reserves a per-model one-shot sentinel with O_EXCL, and never overwrites an
output or sentinel. A failed first attempt consumes the tuple and must be
preserved rather than rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SealError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - error path is user-facing
        raise SealError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SealError(f"expected JSON object: {path}")
    return value


def resolve(manifest_path: Path, value: str) -> Path:
    root = manifest_path.parent.resolve()
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if path != root and root not in path.parents:
        raise SealError(f"path escapes manifest root: {value}")
    return path


def verify_artifacts(manifest_path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise SealError("manifest schema_version must be 1")
    if manifest.get("status") != "sealed":
        raise SealError("manifest status must be 'sealed'")
    if manifest.get("hash_policy") != "immutable_sha256":
        raise SealError("manifest hash_policy must be 'immutable_sha256'")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SealError("manifest.artifacts is required")
    required = {"bank", "frames", "rendered_items", "semantic_validation", "lexical_gate", "analysis_script"}
    missing = sorted(required - set(artifacts))
    if missing:
        raise SealError(f"manifest missing required artifacts: {missing}")
    seen_paths: set[Path] = set()
    for name, spec in artifacts.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise SealError(f"invalid artifact entry: {name}")
        path = resolve(manifest_path, spec["path"])
        if path in seen_paths:
            raise SealError(f"artifact path collision: {name}")
        seen_paths.add(path)
        expected = spec.get("sha256")
        if not path.is_file():
            raise SealError(f"missing sealed artifact {name}: {path}")
        if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
            raise SealError(f"artifact hash mismatch: {name}")

def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    except FileExistsError as exc:
        raise SealError(f"confirmation-used sentinel already exists: {path}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def update_sentinel(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        write_exclusive(temporary, value)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def validate_report(path: Path) -> None:
    report = load_json(path)
    endpoints = report.get("endpoints")
    expected = {"E1_stated_auc", "E1_ngram_auc", "E1_auc_minus_ngram", "E4_indirect_transfer", "E5_second_transfer", "E7_payload_holdout", "E8_double_holdout"}
    if not isinstance(endpoints, dict) or not expected.issubset(endpoints):
        raise SealError("analysis output is missing required confirmatory endpoints")


def _exact_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise SealError(f"{name} must be an integer")
    return value


def validate_inputs(items: Path, activations: Path) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise SealError("numpy is required for activation contract validation") from exc
    rows = []
    seen: set[str] = set()
    for line_number, line in enumerate(items.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise SealError(f"invalid item JSON at line {line_number}") from exc
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise SealError("item IDs must be nonempty and unique")
        seen.add(item_id)
        rows.append(item_id)
    try:
        with np.load(activations, allow_pickle=False) as archive:
            if set(archive.files) < {"X", "item_ids"}:
                raise SealError("activation archive requires X and item_ids")
            matrix = np.asarray(archive["X"])
            activation_ids = [str(value) for value in archive["item_ids"].tolist()]
    except SealError:
        raise
    except Exception as exc:
        raise SealError(f"invalid activation archive: {activations}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(rows):
        raise SealError("activation row count/shape mismatch")
    if activation_ids != rows:
        raise SealError("activation item IDs do not match rendered item order")
    if not np.isfinite(matrix).all():
        raise SealError("activation matrix contains non-finite values")
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    verify_artifacts(manifest_path, manifest)

    run = manifest.get("confirmatory_run")
    if not isinstance(run, dict):
        raise SealError("manifest.confirmatory_run is required")
    required = (
        "items", "activations", "output", "sentinel", "analysis_script", "layer",
        "model", "revision", "bank_sha256", "activations_sha256", "analysis_sha256",
        "wrapper_sha256", "n_layers",
    )
    missing = [key for key in required if key not in run]
    if missing:
        raise SealError(f"confirmatory_run missing fields: {missing}")
    artifacts = manifest["artifacts"]
    paths = {name: resolve(manifest_path, spec["path"]) for name, spec in artifacts.items()}
    items = resolve(manifest_path, run["items"])
    activations = resolve(manifest_path, run["activations"])
    analyzer = resolve(manifest_path, run["analysis_script"])
    output = resolve(manifest_path, run["output"])
    sentinel = resolve(manifest_path, run["sentinel"])
    if items != paths["rendered_items"] or analyzer != paths["analysis_script"]:
        raise SealError("confirmatory paths do not match sealed artifact bindings")
    if not analyzer.is_file() or not items.is_file() or not activations.is_file():
        raise SealError("analysis script, items, and activations must exist")
    if output == sentinel or output in paths.values() or sentinel in paths.values():
        raise SealError("output/sentinel collides with an immutable artifact")
    if output.exists():
        raise SealError(f"refusing to overwrite existing output: {output}")
    validate_inputs(items, activations)
    expected_bank_sha = artifacts["rendered_items"]["sha256"]
    if run["bank_sha256"] != expected_bank_sha:
        raise SealError("bank SHA does not match sealed rendered-items artifact")
    if run["activations_sha256"] != sha256_file(activations):
        raise SealError("activation SHA does not match sealed input")
    if run["analysis_sha256"] != artifacts["analysis_script"]["sha256"] or run["analysis_sha256"] != sha256_file(analyzer):
        raise SealError("analysis SHA does not match canonical executable")
    if run["wrapper_sha256"] != sha256_file(Path(__file__).resolve()):
        raise SealError("sealed wrapper SHA does not match this executable")
    if not isinstance(run["model"], str) or not isinstance(run["revision"], str) or not run["model"] or not run["revision"]:
        raise SealError("model and revision must be nonempty strings")
    layer = _exact_int(run["layer"], "layer")
    frozen_models = manifest.get("frozen_models")
    if not isinstance(frozen_models, dict) or run["model"] not in frozen_models:
        raise SealError("manifest.frozen_models must bind the confirmatory model")
    frozen = frozen_models[run["model"]]
    if not isinstance(frozen, dict) or frozen.get("revision") != run["revision"] or frozen.get("layer") != layer:
        raise SealError("model, revision, and layer do not match frozen model tuple")
    n_layers = _exact_int(run["n_layers"], "n_layers")
    if n_layers <= 0 or layer < 0 or layer >= n_layers:
        raise SealError("layer is outside frozen model bounds")
    if _exact_int(run.get("bootstrap", 10000), "bootstrap") != 10000 or _exact_int(run.get("seed", 0), "seed") != 0:
        raise SealError("confirmatory bootstrap/seed must be 10000/0")
    timeout_seconds = _exact_int(run.get("timeout_seconds", 7200), "timeout_seconds")
    if timeout_seconds <= 0:
        raise SealError("timeout_seconds must be positive")

    reservation = {
        "schema_version": 1,
        "status": "reserved",
        "reserved_at": datetime.now(timezone.utc).isoformat(),
        "model": run["model"],
        "revision": run["revision"],
        "layer": layer,
        "bank_sha256": run["bank_sha256"],
        "activations_sha256": run["activations_sha256"],
        "analysis_sha256": run["analysis_sha256"],
        "wrapper_sha256": run["wrapper_sha256"],
        "git_head": manifest.get("git_head"),
    }
    write_exclusive(sentinel, reservation)
    command = [
        sys.executable, str(analyzer), str(items), str(activations), str(output),
        "--layer", str(layer), "--bootstrap", "10000", "--seed", "0", "--confirmatory",
    ]
    try:
        completed = subprocess.run(command, check=False, timeout=timeout_seconds)
        if completed.returncode:
            raise SealError(f"confirmatory analysis exited {completed.returncode}; sentinel retained")
        if not output.is_file():
            raise SealError("analysis exited successfully but output is missing")
        validate_report(output)
        output_sha = sha256_file(output)
        reservation.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "output_sha256": output_sha})
        update_sentinel(sentinel, reservation)
    except Exception as exc:
        reservation.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": str(exc)})
        update_sentinel(sentinel, reservation)
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SealError as exc:
        print(f"SEALED CONFIRMATION REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
