"""Run the development-only jLens assay on the frozen full320-v3 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def token_ids(tokenizer, words: list[str]) -> dict[str, int]:
    encoded: dict[str, int] = {}
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            encoded[word] = ids[0]
    return encoded


def summary_for(rows: list[dict], layer: int, label_field: str, positive: str) -> dict:
    usable = [
        (row["layers"][str(layer)]["purpose"]["score"], int(row[label_field] == positive))
        for row in rows
        if row["layers"][str(layer)]["purpose"]["score"] is not None
    ]
    scores = [score for score, _ in usable]
    labels = [label for _, label in usable]
    raw = auc(scores, labels)
    return {
        "n": len(usable),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "auc": raw,
        "oriented_auc": max(raw, 1.0 - raw) if raw is not None else None,
    }


def factorial_blocks(rows: list[dict], layer: int) -> list[dict]:
    cells: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        purpose = row.get("intended_purpose")
        fmt = row.get("format")
        if purpose not in {"evaluation", "deployment"} or fmt not in {"benchmark", "casual"}:
            continue
        score = row["layers"][str(layer)]["purpose"]["score"]
        if score is not None:
            cells[row["payload_block_id"]][(purpose, fmt)].append(score)
    required = {
        ("evaluation", "benchmark"),
        ("evaluation", "casual"),
        ("deployment", "benchmark"),
        ("deployment", "casual"),
    }
    output = []
    for block_id, block_cells in sorted(cells.items()):
        if set(block_cells) != required:
            continue
        means = {key: sum(values) / len(values) for key, values in block_cells.items()}
        eb = means[("evaluation", "benchmark")]
        ec = means[("evaluation", "casual")]
        db = means[("deployment", "benchmark")]
        dc = means[("deployment", "casual")]
        output.append({
            "payload_block_id": block_id,
            "cell_n": {f"{purpose}_{fmt}": len(block_cells[(purpose, fmt)]) for purpose, fmt in sorted(required)},
            "cell_means": {f"{purpose}_{fmt}": means[(purpose, fmt)] for purpose, fmt in sorted(required)},
            "A": (eb + ec - db - dc) / 4.0,
            "B": (eb + db - ec - dc) / 4.0,
            "C": (eb - ec - db + dc) / 4.0,
        })
    return output


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
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-blocks", type=int)
    parser.add_argument("--expected-canonical-rows", type=int)
    parser.add_argument("--expected-purpose-rows", type=int)
    parser.add_argument("--expected-format-rows", type=int)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    sys.path.insert(0, str(args.jlens_dir))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.lens import JacobianLens

    spec = json.loads(args.assay_spec.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, got {len(rows)}")
    block_ids = {row.get("payload_block_id") for row in rows}
    if None in block_ids:
        raise ValueError("every row must have payload_block_id")
    if args.expected_blocks is not None and len(block_ids) != args.expected_blocks:
        raise ValueError(f"expected {args.expected_blocks} payload blocks, got {len(block_ids)}")

    purpose_rows = [
        row for row in rows
        if not row.get("legacy", False)
        and row.get("cue_style") == "stated"
        and row.get("intended_purpose") in {"evaluation", "deployment"}
    ]
    format_rows = [
        row for row in rows
        if not row.get("legacy", False)
        and row.get("cue_style") == "none"
        and row.get("intended_purpose") == "none"
        and row.get("format") in {"benchmark", "casual"}
    ]
    canonical_rows = [row for row in rows if not row.get("legacy", False)]
    if not canonical_rows or len(canonical_rows) != len(purpose_rows) + len(format_rows):
        raise ValueError("canonical full320 rows contain an unexpected render slot")
    if len({row["item_id"] for row in canonical_rows}) != len(canonical_rows):
        raise ValueError("duplicate canonical item IDs")
    if {row["intended_purpose"] for row in purpose_rows} != {"evaluation", "deployment"}:
        raise ValueError("canonical stated purpose contrast is incomplete")
    if {row["format"] for row in format_rows} != {"benchmark", "casual"}:
        raise ValueError("canonical no-cue format contrast is incomplete")
    for expected, actual, label in (
        (args.expected_canonical_rows, len(canonical_rows), "canonical rows"),
        (args.expected_purpose_rows, len(purpose_rows), "purpose rows"),
        (args.expected_format_rows, len(format_rows), "format rows"),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"expected {expected} {label}, got {actual}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lens_model = from_hf(model, tokenizer, compile=False, force_bos=True)

    def encode_exact(self, text: str, *, max_length: int = args.max_seq_len):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        if encoded.input_ids.shape[1] > max_length:
            raise ValueError(
                f"chat-templated prompt has {encoded.input_ids.shape[1]} tokens, "
                f"exceeding max_seq_len={max_length}"
            )
        return encoded.input_ids.to(self.input_device)

    lens = JacobianLens.load(str(args.lens))
    if lens.d_model != lens_model.d_model:
        raise ValueError(f"lens/model hidden dimension mismatch: {lens.d_model} vs {lens_model.d_model}")
    source_layers = args.layers or sorted(int(layer) for layer in lens.jacobians)
    available_layers = {int(layer) for layer in lens.jacobians}
    missing = set(source_layers) - available_layers
    if missing:
        raise ValueError(f"requested layers are absent from lens: {sorted(missing)}")

    groups = spec["concept_groups"]
    encoded = {name: token_ids(tokenizer, words) for name, words in groups.items()}
    if not encoded.get("evaluation") or not encoded.get("deployment"):
        raise ValueError("predeclared evaluation/deployment concepts have no single-token entries")
    scores = []
    for index, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        lens_logits, _, input_ids = lens.apply(
            lens_model,
            prompt,
            layers=source_layers,
            positions=[-1],
            max_seq_len=args.max_seq_len,
            use_jacobian=True,
        )
        input_set = set(input_ids[0].tolist())
        record = {
            "index": index,
            "item_id": row.get("item_id"),
            "payload_block_id": row.get("payload_block_id"),
            "topic_id": row.get("topic_id"),
            "purpose_family_id": row.get("purpose_family_id"),
            "purpose_pair_id": row.get("purpose_pair_id"),
            "split_group_id": row.get("split_group_id"),
            "template_id": row.get("template_id"),
            "contract_id": row.get("contract_id"),
            "intended_purpose": row.get("intended_purpose"),
            "format": row.get("format"),
            "render_slot": row.get("render_slot"),
            "cue_style": row.get("cue_style"),
            "cue_perspective": row.get("cue_perspective"),
            "legacy": bool(row.get("legacy", False)),
            "prompt_token_count": int(input_ids.shape[1]),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "layers": {},
        }
        for layer in source_layers:
            logits = lens_logits[layer]
            if logits.ndim == 3:
                logits = logits[0, -1]
            elif logits.ndim == 2:
                logits = logits[-1]
            left_ids = [token for token in encoded["evaluation"].values() if token not in input_set]
            right_ids = [token for token in encoded["deployment"].values() if token not in input_set]
            if not left_ids or not right_ids:
                raise ValueError(f"literal-copy exclusion removed a concept side at row {index}, layer {layer}")
            record["layers"][str(layer)] = {
                "purpose": {
                    "score": float(logits[left_ids].mean().item() - logits[right_ids].mean().item()),
                    "left_n": len(left_ids),
                    "right_n": len(right_ids),
                }
            }
        scores.append(record)
        if (index + 1) % 100 == 0:
            print(f"scored {index + 1}/{len(rows)}", flush=True)

    purpose_ids = {row["item_id"] for row in purpose_rows}
    format_ids = {row["item_id"] for row in format_rows}
    purpose_scores = [row for row in scores if row["item_id"] in purpose_ids]
    format_scores = [row for row in scores if row["item_id"] in format_ids]
    summary = {"purpose": {}, "format": {}, "factorial": {}}
    for layer in source_layers:
        summary["purpose"][str(layer)] = summary_for(purpose_scores, layer, "intended_purpose", "evaluation")
        summary["format"][str(layer)] = summary_for(format_scores, layer, "format", "benchmark")
        blocks = factorial_blocks(purpose_scores, layer)
        if not blocks:
            raise ValueError(f"no complete four-cell payload blocks at layer {layer}")
        summary["factorial"][str(layer)] = {"n_complete_blocks": len(blocks), "blocks": blocks}

    result = {
        "schema_version": 2,
        "status": "development_only",
        "implementation": "anthropics/jacobian-lens",
        "model": args.model,
        "revision": args.revision,
        "lens": {"path": str(args.lens), "sha256": sha256(args.lens)},
        "jlens_dir": str(args.jlens_dir),
        "jlens_commit": spec.get("jlens_commit"),
        "items": {"path": str(args.items), "sha256": sha256(args.items)},
        "assay_spec": {"path": str(args.assay_spec), "sha256": sha256(args.assay_spec)},
        "git_commit": git_commit(args.repo_root),
        "source_layers": source_layers,
        "max_seq_len": args.max_seq_len,
        "n_rows": len(rows),
        "n_canonical_rows": len(canonical_rows),
        "n_legacy_rows": len(rows) - len(canonical_rows),
        "n_payload_blocks": len(block_ids),
        "n_purpose_rows": len(purpose_rows),
        "n_format_rows": len(format_rows),
        "concept_groups": {name: sorted(values) for name, values in encoded.items()},
        "summary": summary,
        "scores": scores,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "canonical_rows": len(canonical_rows), "blocks": len(block_ids), "out": str(args.out)}))


if __name__ == "__main__":
    main()
