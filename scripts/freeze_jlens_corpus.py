"""Freeze an independent generic corpus for development J-lens fitting."""
from __future__ import annotations
import argparse, hashlib, json, urllib.parse, urllib.request
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"
SPLIT = "train"
BASE = "https://datasets-server.huggingface.co/rows"
REJECT = ("multiple choice", "benchmark", "evaluation set", "exam question", "test question")

def fetch(offset: int, length: int = 100) -> dict:
    query = urllib.parse.urlencode({"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                                    "offset": offset, "length": length})
    with urllib.request.urlopen(f"{BASE}?{query}", timeout=60) as response:
        return json.load(response)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/jlens_fineweb_1000"))
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    candidates, rejected = [], []
    offset = 0
    while len(candidates) < 5000 and offset < 200000:
        page = fetch(offset)
        for entry in page.get("rows", []):
            row = entry["row"]
            text = str(row.get("text", "")).strip()
            reason = None
            if row.get("language") != "en" or float(row.get("language_score", 0)) < 0.9:
                reason = "language_gate"
            elif int(row.get("token_count", 0)) < 128 or len(text) < 512:
                reason = "too_short"
            elif any(term in text.lower() for term in REJECT):
                reason = "obvious_eval_or_benchmark_content"
            if reason:
                rejected.append({"row_idx": int(entry["row_idx"]), "id": row.get("id"), "reason": reason})
                continue
            candidates.append({"dataset_row_idx": int(entry["row_idx"]), "id": row["id"], "text": text,
                               "dump": row.get("dump"), "url": row.get("url"), "date": row.get("date"),
                               "language": row.get("language"), "language_score": row.get("language_score"),
                               "token_count": row.get("token_count")})
        offset += 100
    rng = __import__("numpy").random.default_rng(args.seed)
    if len(candidates) < 1000:
        raise RuntimeError(f"only selected {len(candidates)} candidates")
    records = [candidates[i] for i in rng.permutation(len(candidates))[:1000]]
    raw = args.out_dir / "raw.jsonl"
    raw.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in records), encoding="utf-8")
    (args.out_dir / "rejections.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rejected), encoding="utf-8")
    manifest = {"schema_version": 1, "mode": "development", "dataset": DATASET, "revision": "2026-08-17 dataset-server snapshot",
                "config": CONFIG, "split": SPLIT, "selection_seed": args.seed,
                "selection_method": "seeded permutation of first 5000 filter-passing rows", "candidate_pool": len(candidates), "rows": 1000,
                "row_indices": [x["dataset_row_idx"] for x in records], "ids": [x["id"] for x in records],
                "filter": {"language": "en", "language_score_min": 0.9, "token_count_min": 128, "char_min": 512,
                           "reject_terms": list(REJECT)}, "raw_sha256": sha256(raw),
                "rejections_sha256": sha256(args.out_dir / "rejections.jsonl"),
                "tokenization": {"status": "pending_cluster_tokenizer", "model": "meta-llama/Llama-3.1-8B-Instruct",
                                 "max_seq_len": 128}}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(records), "raw_sha256": manifest["raw_sha256"], "out_dir": str(args.out_dir)}))

if __name__ == "__main__":
    main()
