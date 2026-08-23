from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_pilot80_factorial_causal import analyze, parse_args
from scripts.run_pilot80_factorial_causal import condition_specs, sha256_file


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def make_fixture(tmp_path: Path) -> tuple[Path, list[Path], list[Path], list[Path]]:
    source_config = json.loads(Path("configs/pilot80_factorial_causal_mediation_v2_reextracted.json").read_text(encoding="utf-8"))
    families = source_config["dataset"]["purpose_families"]
    rows = []
    for family_index, family in enumerate(families):
        for payload_fold in range(5):
            purpose = "evaluation" if (family_index + payload_fold) % 2 == 0 else "deployment"
            format_name = "benchmark" if payload_fold % 2 == 0 else "casual"
            rows.append({
                "item_id": f"item-{family}-{payload_fold}", "payload_block_id": f"block-{family}-{payload_fold}",
                "purpose_pair_id": f"pair-{family}-{payload_fold}", "purpose_family_id": family,
                "topic_id": f"topic-{family_index}", "render_slot": "stated",
                "intended_purpose": purpose, "format": format_name, "text": "synthetic",
            })
    items_path = tmp_path / "meta.jsonl"
    items_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    items_sha = sha256_file(items_path)
    models = []
    for size in ("small", "large"):
        models.append({
            "model_id": f"test/{size}", "revision": f"rev-{size}",
            "checkpoint_path_cluster": f"/scratch/jppatton/langea/{size}",
            "tokenizer_path_cluster": f"/scratch/jppatton/langea/{size}",
            "activation_path": f"{size}.npz", "activation_path_cluster": f"/scratch/jppatton/langea/{size}.npz",
            "activation_sha256": hashlib.sha256(size.encode()).hexdigest(),
            "n_layers": 4, "hidden_size": 4,
            "source_layers": {"low_interaction": 1, "high_interaction": 2}, "target_layer": 3,
        })
    protocol = source_config
    protocol["dataset"].update({
        "items_path": items_path.as_posix(), "items_sha256": items_sha, "n_rows": 25,
        "n_stated_rows": 25, "n_payload_blocks": 25, "n_complete_2x2_groups": 25,
    })
    protocol["models"] = models
    protocol["uncertainty"]["bootstrap_resamples"] = 200
    config_path = tmp_path / "config.json"
    dump(config_path, protocol)
    config_sha = sha256_file(config_path)
    builder_sha = sha256_file(Path("scripts/build_pilot80_factorial_causal_plan.py"))
    runner_sha = sha256_file(Path("scripts/run_pilot80_factorial_causal.py"))
    plan_json_paths, plan_npz_paths, raw_paths = [], [], []
    specs = condition_specs(protocol)
    payload_fold_by_block = {row["payload_block_id"]: index % 5 for index, row in enumerate(rows)}
    for model in models:
        bundle = tmp_path / model["model_id"].split("/")[-1]
        bundle.mkdir()
        arrays = {}
        folds = []
        for family_index, family in enumerate(families):
            for payload_fold in range(5):
                fold_id = f"{family}_pf{payload_fold}"
                row_index = family_index * 5 + payload_fold
                source_layers = {}
                for depth_label, layer in model["source_layers"].items():
                    prefix = f"fold_{fold_id}_layer_{layer}_"
                    arrays[prefix + "A"] = np.asarray([1, 0, 0, 0], dtype=np.float32)
                    arrays[prefix + "B"] = np.asarray([0, 1, 0, 0], dtype=np.float32)
                    arrays[prefix + "C"] = np.asarray([0, 0, 1, 0], dtype=np.float32)
                    arrays[prefix + "random"] = np.tile(np.asarray([[0, 0, 0, 1]], dtype=np.float32), (32, 1))
                    source_layers[str(layer)] = {
                        "depth_label": depth_label, "n_direction_groups": 4, "common_scale": 1.0,
                        "norm_A": 1.0, "norm_B": 1.0, "norm_C": 1.0, "max_random_component_cosine": 0.0,
                    }
                readouts = {}
                for readout in ("purpose_benchmark", "purpose_casual", "format"):
                    prefix = f"fold_{fold_id}_readout_{readout}_"
                    arrays[prefix + "mean"] = np.zeros(4, dtype=np.float32)
                    arrays[prefix + "scale"] = np.ones(4, dtype=np.float32)
                    arrays[prefix + "coef"] = np.ones(4, dtype=np.float32)
                    arrays[prefix + "intercept"] = np.asarray(0.0)
                    arrays[prefix + "score_sd"] = np.asarray(1.0)
                    readouts[readout] = {"roc_auc": 1.0, "balanced_accuracy": 1.0, "n_train": 4, "n_test": 1, "train_score_sd": 1.0, "n_iter": 1}
                folds.append({
                    "fold_id": fold_id, "held_family": family, "held_payload_fold": payload_fold,
                    "test_indices": [row_index], "direction_indices": [], "readout_indices": [],
                    "source_layers": source_layers, "readouts": readouts,
                })
        plan_npz_path = bundle / "causal_plan.npz"
        np.savez_compressed(plan_npz_path, **arrays)
        plan = {
            "schema_version": 1, "protocol_id": protocol["protocol_id"], "config_path": config_path.as_posix(),
            "config_sha256": config_sha, "script_sha256": builder_sha,
            "items_path": items_path.as_posix(), "items_sha256": items_sha,
            "activation_path": model["activation_path"], "activation_sha256": model["activation_sha256"],
            "model": model, "canonical_item_ids": [row["item_id"] for row in rows],
            "payload_fold_by_block": payload_fold_by_block, "folds": folds,
            "plan_npz": plan_npz_path.name, "plan_npz_sha256": sha256_file(plan_npz_path),
        }
        plan_json_path = bundle / "causal_plan.json"
        dump(plan_json_path, plan)
        plan_json_sha = sha256_file(plan_json_path)
        raw_path = bundle / "causal_raw.jsonl"
        with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
            for fold in folds:
                row_index = fold["test_indices"][0]
                row = rows[row_index]
                purpose_base = 2.0 if row["intended_purpose"] == "evaluation" else -2.0
                format_base = 2.0 if row["format"] == "benchmark" else -2.0
                format_sign = 1.0 if row["format"] == "benchmark" else -1.0
                for layer_text, source_meta in fold["source_layers"].items():
                    layer = int(layer_text)
                    c_strength = 0.2 if source_meta["depth_label"] == "low_interaction" else 1.0
                    for spec in specs:
                        purpose_shift = 0.0
                        format_shift = 0.0
                        if spec["condition_type"] == "component":
                            if spec["component"] == "A":
                                purpose_shift = spec["sign"] * spec["alpha"]
                            elif spec["component"] == "B":
                                format_shift = spec["sign"] * spec["alpha"]
                            else:
                                purpose_shift = spec["sign"] * spec["alpha"] * format_sign * c_strength
                        elif spec["condition_type"] == "cube":
                            purpose_shift = spec["cube_signs"]["A"] + spec["cube_signs"]["C"] * format_sign * c_strength
                            format_shift = spec["cube_signs"]["B"]
                        record = {
                            "schema_version": 1, "protocol_id": protocol["protocol_id"],
                            "model_id": model["model_id"], "revision": model["revision"],
                            "config_sha256": config_sha, "plan_json_sha256": plan_json_sha,
                            "plan_npz_sha256": plan["plan_npz_sha256"], "script_sha256": runner_sha,
                            "fold_id": fold["fold_id"], "row_index": row_index, "item_id": row["item_id"],
                            "payload_block_id": row["payload_block_id"], "purpose_family_id": row["purpose_family_id"],
                            "intended_purpose": row["intended_purpose"], "format": row["format"],
                            "source_layer": layer, "source_depth_label": source_meta["depth_label"], "target_layer": 3,
                            "token_position": -1, "token_length": 3, "last_prompt_token_id": 1,
                            "inference_dtype": "bfloat16", "common_residual_scale": 1.0,
                            "intervention_norm": 0.0, **spec,
                            "hook_count": 1, "source_hook_count": 1, "target_hook_count": 1,
                            "source_baseline_max_abs": 0.0, "source_baseline_cosine": 1.0,
                            "target_baseline_max_abs": 0.0, "target_baseline_cosine": 1.0,
                            "readout_scores": {
                                "purpose_benchmark": purpose_base + purpose_shift,
                                "purpose_casual": purpose_base + purpose_shift,
                                "format": format_base + format_shift,
                            },
                            "source_finite": True, "target_finite": True,
                        }
                        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        raw_sha = sha256_file(raw_path)
        expected = len(folds) * len(model["source_layers"]) * len(specs)
        dump(bundle / "runner_manifest.json", {
            "schema_version": 1, "status": "completed_uninspected_factorial_causal_raw",
            "protocol_id": protocol["protocol_id"], "model_id": model["model_id"], "revision": model["revision"],
            "config_sha256": config_sha, "plan_json_sha256": plan_json_sha, "plan_npz_sha256": plan["plan_npz_sha256"],
            "script_sha256": sha256_file(Path("scripts/run_pilot80_factorial_causal.py")),
            "activation_sha256": model["activation_sha256"], "items_sha256": items_sha,
            "raw_path": raw_path.name, "raw_sha256": sha256_file(raw_path), "records": expected,
            "expected_records": expected, "records_written_this_invocation": expected,
            "model_file_hashes": {"checkpoint/config.json": "0" * 64},
        })
        plan_json_paths.append(plan_json_path)
        plan_npz_paths.append(plan_npz_path)
        raw_paths.append(raw_path)
    return config_path, plan_json_paths, plan_npz_paths, raw_paths


def test_analyzer_accepts_complete_positive_response_surface(tmp_path: Path) -> None:
    config, plans, archives, raws = make_fixture(tmp_path)
    out_dir = tmp_path / "analysis"
    argv = ["--config", str(config), "--out-dir", str(out_dir)]
    for flag, paths in (("--plan-json", plans), ("--plan-npz", archives), ("--raw", raws)):
        for path in paths:
            argv.extend((flag, str(path)))
    result = analyze(parse_args(argv))
    assert result["decision"] == "positive"
    assert all(result["criteria"].values())
    assert all(value["additive_r2"] == 1.0 for value in result["factorial_cube"].values())
    assert (out_dir / "run_manifest.json").is_file()


def test_analyzer_rejects_invalid_run_marker(tmp_path: Path) -> None:
    config, plans, archives, raws = make_fixture(tmp_path)
    dump(raws[0].with_name("INVALID.json"), {"status": "invalid", "use_for_inference": False})
    argv = ["--config", str(config), "--out-dir", str(tmp_path / "analysis")]
    for flag, paths in (("--plan-json", plans), ("--plan-npz", archives), ("--raw", raws)):
        for path in paths:
            argv.extend((flag, str(path)))
    with pytest.raises(ValueError, match="invalid run marker present"):
        analyze(parse_args(argv))


def test_analyzer_rejects_runner_manifest_raw_path_mismatch(tmp_path: Path) -> None:
    config, plans, archives, raws = make_fixture(tmp_path)
    manifest_path = raws[0].with_name("runner_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw_path"] = "different.jsonl"
    dump(manifest_path, manifest)
    argv = ["--config", str(config), "--out-dir", str(tmp_path / "analysis")]
    for flag, paths in (("--plan-json", plans), ("--plan-npz", archives), ("--raw", raws)):
        for path in paths:
            argv.extend((flag, str(path)))
    with pytest.raises(ValueError, match="runner manifest mismatch .*: raw_path"):
        analyze(parse_args(argv))
