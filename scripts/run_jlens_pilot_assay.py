"""Run the predeclared development-only J-lens semantic assay on pilot rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        mean_rank = (index + 1 + end) / 2.0
        rank_sum += mean_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def token_ids(tokenizer, words):
    out = {}
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            out[word] = ids[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jlens-dir", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--assay-spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.jlens_dir))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.lens import JacobianLens

    spec = json.loads(args.assay_spec.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lens_model = from_hf(model, tokenizer, compile=False, force_bos=True)
    lens = JacobianLens.load(str(args.lens))
    source_layers = [int(layer) for layer in lens.jacobians]
    groups = spec["concept_groups"]
    encoded = {name: token_ids(tokenizer, words) for name, words in groups.items()}
    scores = []
    for index, row in enumerate(rows):
        text = row["rendered_text"]
        lens_logits, _, input_ids = lens.apply(
            lens_model, text, layers=source_layers, positions=[-1],
            max_seq_len=128, use_jacobian=True,
        )
        input_set = set(input_ids[0].tolist())
        row_scores = {"index": index, "item_id": row.get("item_id"), "purpose": row.get("intended_purpose"),
                      "format": row.get("format"), "cue_style": row.get("cue_style"),
                      "cue_perspective": row.get("cue_perspective"), "layers": {}}
        for layer in source_layers:
            logits = lens_logits[layer]
            if logits.ndim == 3:
                logits = logits[0, -1]
            elif logits.ndim == 2:
                logits = logits[-1]
            layer_result = {}
            for contrast, (left, right) in {
                "purpose": ("evaluation", "deployment"),
                # The frozen spec defines format as a projection along the
                # purpose contrast, not as a benchmark/casual token lexicon.
                "format_proxy": ("evaluation", "deployment"),
            }.items():
                left_ids = [token for token in encoded.get(left, {}).values() if token not in input_set]
                right_ids = [token for token in encoded.get(right, {}).values() if token not in input_set]
                layer_result[contrast] = {
                    "score": (float(logits[left_ids].mean().item() - logits[right_ids].mean().item())
                              if left_ids and right_ids else None),
                    "left_n": len(left_ids), "right_n": len(right_ids),
                }
            row_scores["layers"][str(layer)] = layer_result
        scores.append(row_scores)
        if (index + 1) % 100 == 0:
            print(f"scored {index + 1}/{len(rows)}", flush=True)

    contrasts = {
        "stated_third_person": lambda r: r["cue_style"] == "stated" and r["cue_perspective"] == "third" and r["purpose"] in {"evaluation", "deployment"},
        "indirect_third_person": lambda r: r["cue_style"] == "indirect" and r["purpose"] in {"evaluation", "deployment"},
        "second_person": lambda r: r["cue_perspective"] == "second" and r["purpose"] in {"evaluation", "deployment"},
        "no_cue_format": lambda r: r["cue_style"] == "none" and r["format"] in {"benchmark", "casual"},
    }
    summary = {}
    for contrast, select in contrasts.items():
        selected = [r for r in scores if select(r)]
        summary[contrast] = {}
        for layer in source_layers:
            purpose_values = [r["layers"][str(layer)]["purpose"]["score"] for r in selected]
            format_values = [r["layers"][str(layer)]["format_proxy"]["score"] for r in selected]
            if contrast == "no_cue_format":
                labels = [1 if r["format"] == "benchmark" else 0 for r in selected]
                values = format_values
            else:
                labels = [1 if r["purpose"] == "evaluation" else 0 for r in selected]
                values = purpose_values
            usable = [(v, y) for v, y in zip(values, labels) if v is not None]
            summary[contrast][str(layer)] = {
                "n": len(usable), "auc": auc([v for v, _ in usable], [y for _, y in usable]),
                "positive": sum(y for _, y in usable), "negative": sum(1 - y for _, y in usable),
            }
    result = {
        "schema_version": 1, "status": "development_only", "implementation": "anthropics/jacobian-lens",
        "format_score_interpretation": "benchmark/casual AUC uses the predeclared evaluation-minus-deployment J-space projection; it is not a format-token lexicon score.",
        "model": args.model, "revision": args.revision, "lens": str(args.lens),
        "lens_sha256": sha256(args.lens), "items": str(args.items), "items_sha256": sha256(args.items),
        "assay_spec": str(args.assay_spec), "assay_spec_sha256": sha256(args.assay_spec),
        "source_layers": source_layers, "n_rows": len(rows), "summary": summary,
        "scores": scores, "environment": {"python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
