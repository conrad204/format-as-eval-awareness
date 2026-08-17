from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.run_sealed_confirmation import sha256_file


ENDPOINTS = {
    "E1_stated_auc": {},
    "E1_ngram_auc": {},
    "E1_auc_minus_ngram": {},
    "E4_indirect_transfer": {},
    "E5_second_transfer": {},
    "E7_payload_holdout": {},
    "E8_double_holdout": {},
}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def make_manifest(tmp_path: Path, mutation=None):
    rows = [{"item_id": f"item-{i}", "text": "x", "label": i % 2, "purpose_family_id": "fg1", "render_slot": "stated", "cue_locus": "both", "format": "benchmark"} for i in range(5)]
    items = tmp_path / "rendered_items.jsonl"
    _write(items, "".join(json.dumps(row) + "\n" for row in rows))
    np.savez(tmp_path / "activations.npz", X=np.arange(15, dtype=np.float32).reshape(5, 3), item_ids=np.asarray([row["item_id"] for row in rows]))
    fake = tmp_path / "fake_analysis.py"
    _write(fake, "import json,sys\njson.dump({'endpoints': " + repr(ENDPOINTS) + "}, open(sys.argv[3], 'w'))\n")
    files = {
        "bank": tmp_path / "bank.json",
        "frames": tmp_path / "frames.json",
        "rendered_items": items,
        "semantic_validation": tmp_path / "semantic.jsonl",
        "lexical_gate": tmp_path / "lexical.json",
        "analysis_script": fake,
    }
    _write(files["bank"], "bank\n")
    _write(files["frames"], "[]\n")
    _write(files["semantic_validation"], "{}\n")
    _write(files["lexical_gate"], json.dumps({"pass": True}) + "\n")
    artifacts = {name: {"path": path.name, "sha256": sha256_file(path)} for name, path in files.items()}
    manifest = {
        "schema_version": 1,
        "status": "sealed",
        "hash_policy": "immutable_sha256",
        "git_head": "synthetic",
        "frozen_models": {"synthetic/model": {"revision": "rev-1", "layer": 2}},
        "artifacts": artifacts,
        "confirmatory_run": {
            "items": items.name,
            "activations": "activations.npz",
            "output": "result.json",
            "sentinel": "used.json",
            "analysis_script": fake.name,
            "layer": 2,
            "n_layers": 4,
            "model": "synthetic/model",
            "revision": "rev-1",
            "bank_sha256": artifacts["rendered_items"]["sha256"],
            "activations_sha256": sha256_file(tmp_path / "activations.npz"),
            "analysis_sha256": artifacts["analysis_script"]["sha256"],
            "wrapper_sha256": sha256_file(Path(__file__).parents[1] / "scripts" / "run_sealed_confirmation.py"),
            "bootstrap": 10000,
            "seed": 0,
            "timeout_seconds": 30,
        },
    }
    if mutation:
        mutation(manifest, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run_wrapper(manifest: Path):
    return subprocess.run([sys.executable, "scripts/run_sealed_confirmation.py", "--manifest", str(manifest)], capture_output=True, text=True)


def test_correct_tuple_allowed_once(tmp_path):
    manifest = make_manifest(tmp_path)
    first = run_wrapper(manifest)
    assert first.returncode == 0, first.stderr
    assert json.loads((tmp_path / "used.json").read_text())["status"] == "completed"
    (tmp_path / "result.json").unlink()
    second = run_wrapper(manifest)
    assert second.returncode == 2
    assert "sentinel" in second.stderr


@pytest.mark.parametrize("field,value,needle", [
    ("bank_sha256", "0" * 64, "bank SHA"),
    ("analysis_sha256", "1" * 64, "analysis SHA"),
])
def test_wrong_hashes_fail(tmp_path, field, value, needle):
    def mutate(manifest, _):
        manifest["confirmatory_run"][field] = value
    result = run_wrapper(make_manifest(tmp_path, mutate))
    assert result.returncode == 2 and needle in result.stderr


def test_wrong_model_revision_and_layer_fail(tmp_path):
    for field, value in (("model", "wrong/model"), ("revision", "wrong-rev"), ("layer", 3)):
        def mutate(manifest, _, field=field, value=value):
            manifest["confirmatory_run"][field] = value
        result = run_wrapper(make_manifest(tmp_path / field, mutate))
        assert result.returncode == 2


def test_existing_output_and_sentinel_fail(tmp_path):
    def output_exists(manifest, root):
        (root / manifest["confirmatory_run"]["output"]).write_text("existing", encoding="utf-8")
    assert run_wrapper(make_manifest(tmp_path / "output", output_exists)).returncode == 2

    def sentinel_exists(manifest, root):
        (root / manifest["confirmatory_run"]["sentinel"]).write_text("existing", encoding="utf-8")
    assert run_wrapper(make_manifest(tmp_path / "sentinel", sentinel_exists)).returncode == 2


def test_mismatched_item_ids_and_row_count_fail(tmp_path):
    def bad_ids(manifest, root):
        np.savez(root / "activations.npz", X=np.arange(15, dtype=np.float32).reshape(5, 3), item_ids=np.asarray(["wrong"] * 5))
        manifest["confirmatory_run"]["activations_sha256"] = sha256_file(root / "activations.npz")
    assert run_wrapper(make_manifest(tmp_path / "ids", bad_ids)).returncode == 2

    def bad_rows(manifest, root):
        np.savez(root / "activations.npz", X=np.arange(12, dtype=np.float32).reshape(4, 3), item_ids=np.asarray([f"item-{i}" for i in range(4)]))
        manifest["confirmatory_run"]["activations_sha256"] = sha256_file(root / "activations.npz")
    assert run_wrapper(make_manifest(tmp_path / "rows", bad_rows)).returncode == 2
