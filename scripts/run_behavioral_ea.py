"""Score frozen behavioral-EA A/B choices with causal language-model logits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--load-in-8bit", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256(args.items) != manifest["items_sha256"]:
        raise RuntimeError("behavioral bank hash mismatch")
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != manifest["n_items"]:
        raise RuntimeError("behavioral bank row-count mismatch")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    candidate_ids = {}
    for letter in ("A", "B"):
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"candidate {letter!r} is not one token: {ids}")
        candidate_ids[letter] = ids[0]

    quantization_config = (
        BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        if args.load_in_8bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.eval()
    outputs = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True) for row in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        input_device = model.get_input_embeddings().weight.device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :].float()
        ab = torch.stack([logits[:, candidate_ids["A"]], logits[:, candidate_ids["B"]]], dim=1)
        probs = torch.softmax(ab, dim=1).cpu().numpy()
        ab = ab.cpu().numpy()
        for row, row_logits, row_probs in zip(batch, ab, probs):
            target_letter = row.get("target_letter", row.get("evaluator_favored_letter"))
            if target_letter not in {"A", "B"}:
                raise RuntimeError(f"missing A/B target for {row['item_id']}")
            target = 0 if target_letter == "A" else 1
            outputs.append({
                "item_id": row["item_id"],
                "block_id": row["block_id"],
                "history_family": row["history_family"],
                "decision_family": row.get("decision_family"),
                "question_family": row.get("question_family"),
                "condition": row["condition"],
                "target_letter": target_letter,
                "evaluator_favored_letter": row.get("evaluator_favored_letter"),
                "logit_a": float(row_logits[0]),
                "logit_b": float(row_logits[1]),
                "target_logit_margin": float(row_logits[target] - row_logits[1 - target]),
                "p_target": float(row_probs[target]),
                "logit_evaluator_minus_user": float(row_logits[target] - row_logits[1 - target]),
                "p_evaluator_favored": float(row_probs[target]),
            })
        print(f"scored {len(outputs)}/{len(rows)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_key": args.model_key,
        "model": args.model,
        "items_sha256": manifest["items_sha256"],
        "load_in_8bit": args.load_in_8bit,
        "manifest_sha256": sha256(args.manifest),
        "n_items": len(outputs),
        "candidate_token_ids": candidate_ids,
        "rows": outputs,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
