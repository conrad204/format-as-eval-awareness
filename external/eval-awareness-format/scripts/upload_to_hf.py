#!/usr/bin/env python3
"""
Upload 2x2 matrix datasets to HuggingFace Hub.

Datasets (all sampled to 1076 for consistency):
- bench_eval: MMLU + ARC + GSM8K benchmark questions
- casual_deploy_matched: ShareGPT user queries (length-matched)
- bench_deploy_rewritten: Casual queries rewritten as MCQ (casual style)
- casual_eval_mutual: MuTual dialogue first turns
- casual_deploy_formal: Casual queries rewritten to formal style (no MCQ)
- bench_deploy_formal: Casual queries rewritten to MCQ + formal style
"""

import json
import os
import random
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
MUTUAL_DIR = BASE_DIR / "data" / "datasets" / "raw" / "mutual" / "train"

# Fixed sample size for consistency
SAMPLE_SIZE = 1076
random.seed(42)


def load_mutual_first_turns():
    """Load first turns from MuTual dialogues."""
    mutual_data = []

    for filename in sorted(os.listdir(MUTUAL_DIR)):
        if filename.endswith('.txt'):
            with open(MUTUAL_DIR / filename) as f:
                data = json.loads(f.read())
                article = data.get('article', '')
                item_id = data.get('id', filename)

                if article:
                    # Split by speakers and get first message
                    parts = article.split(' f : ')
                    if len(parts) > 0:
                        first_part = parts[0]
                        # Remove speaker prefix
                        if first_part.startswith('m : '):
                            first_turn = first_part[4:]
                        elif first_part.startswith('f : '):
                            first_turn = first_part[4:]
                        else:
                            first_turn = first_part

                        mutual_data.append({
                            'id': str(item_id),
                            'first_turn': str(first_turn.strip()),
                            'full_dialogue': str(article),
                            'length': len(first_turn.strip())
                        })

    return mutual_data


def format_bench_eval_prompt(item):
    """Format bench_eval item as prompt."""
    source = item.get("source", "unknown")

    if source == "mmlu":
        q = item.get("question", "")
        choices = item.get("choices", [])
        return f"Question: {q}\nChoices: {', '.join(choices)}\nAnswer:"
    elif source == "arc":
        q = item.get("question", "")
        choices = item.get("choices", {})
        if isinstance(choices, dict):
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choice_str = ", ".join([f"{l}) {t}" for l, t in zip(labels, texts)])
        else:
            choice_str = str(choices)
        return f"Question: {q}\nOptions: {choice_str}\nAnswer:"
    elif source == "gsm8k":
        q = item.get("question", "")
        return f"Problem: {q}\nSolution:"

    return str(item.get("question", item))


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to HuggingFace Hub")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="HuggingFace repo name (e.g., 'username/eval-awareness-2x2')")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private")
    args = parser.parse_args()

    # Login
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("⚠️ No token provided. Using cached credentials or will fail.")

    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    # 1. Bench-Eval (full + sampled)
    print("\n1. Loading Bench-Eval...")
    with open(PROCESSED_DIR / "bench_eval.json") as f:
        bench_eval_raw = json.load(f)

    # Full dataset
    bench_eval_full_data = []
    for item in bench_eval_raw:
        formatted = format_bench_eval_prompt(item)
        bench_eval_full_data.append({
            'id': str(item.get('id', '')),
            'source': str(item.get('source', '')),
            'question': str(item.get('question', '')),
            'formatted_prompt': formatted,
            'choices': json.dumps(item.get('choices', [])),
            'answer': str(item.get('answer', '')),
            'length': len(formatted)
        })

    # Sampled dataset
    bench_eval_sampled = random.sample(bench_eval_raw, min(SAMPLE_SIZE, len(bench_eval_raw)))
    bench_eval_data = []
    for item in bench_eval_sampled:
        formatted = format_bench_eval_prompt(item)
        bench_eval_data.append({
            'id': str(item.get('id', '')),
            'source': str(item.get('source', '')),
            'question': str(item.get('question', '')),
            'formatted_prompt': formatted,
            'choices': json.dumps(item.get('choices', [])),
            'answer': str(item.get('answer', '')),
            'length': len(formatted)
        })
    print(f"   Full: {len(bench_eval_full_data)}, Sampled: {len(bench_eval_data)} examples")

    # 2. Casual-Deploy Matched
    print("\n2. Loading Casual-Deploy Matched...")
    with open(PROCESSED_DIR / "casual_deploy_matched.json") as f:
        casual_deploy_raw = json.load(f)

    casual_deploy_data = []
    for item in casual_deploy_raw:
        casual_deploy_data.append({
            'id': str(item.get('id', '')),
            'first_turn': str(item.get('first_turn', '')),
            'length': int(item.get('first_turn_length', len(item.get('first_turn', ''))))
        })
    print(f"   Loaded {len(casual_deploy_data)} examples")

    # 3. Bench-Deploy Rewritten
    print("\n3. Loading Bench-Deploy Rewritten...")
    with open(PROCESSED_DIR / "bench_deploy_rewritten.json") as f:
        rewritten_raw = json.load(f)

    rewritten_data = []
    for item in rewritten_raw:
        rewritten_data.append({
            'id': str(item.get('id', '')),
            'original_query': str(item.get('first_turn', '')),
            'label': str(item.get('label', '')),
            'rewritten': str(item.get('rewritten', '')),
            'length': len(item.get('rewritten', ''))
        })
    print(f"   Loaded {len(rewritten_data)} examples")

    # 4. Casual-Eval (MuTual) - full + sampled
    print("\n4. Loading Casual-Eval (MuTual)...")
    mutual_data_full = load_mutual_first_turns()
    mutual_data = random.sample(mutual_data_full, min(SAMPLE_SIZE, len(mutual_data_full)))
    print(f"   Full: {len(mutual_data_full)}, Sampled: {len(mutual_data)} examples")

    # 5. Casual-Deploy Formal (no MCQ + formal academic style)
    print("\n5. Loading Casual-Deploy Formal...")
    casual_deploy_formal_data = []
    casual_deploy_formal_path = PROCESSED_DIR / "casual_deploy_formal.json"
    if casual_deploy_formal_path.exists():
        with open(casual_deploy_formal_path) as f:
            casual_formal_raw = json.load(f)
        for item in casual_formal_raw:
            if item.get('rewritten'):
                casual_deploy_formal_data.append({
                    'id': str(item.get('id', '')),
                    'original_query': str(item.get('first_turn', '')),
                    'label': str(item.get('label', '')),
                    'rewritten': str(item.get('rewritten', '')),
                    'length': len(item.get('rewritten', ''))
                })
        print(f"   Loaded {len(casual_deploy_formal_data)} examples")
    else:
        print("   ⚠️ File not found, skipping")

    # 6. Bench-Deploy Formal (MCQ + formal academic style)
    print("\n6. Loading Bench-Deploy Formal...")
    bench_deploy_formal_data = []
    bench_deploy_formal_path = PROCESSED_DIR / "bench_deploy_formal.json"
    if bench_deploy_formal_path.exists():
        with open(bench_deploy_formal_path) as f:
            bench_formal_raw = json.load(f)
        for item in bench_formal_raw:
            if item.get('rewritten'):
                bench_deploy_formal_data.append({
                    'id': str(item.get('id', '')),
                    'original_query': str(item.get('first_turn', '')),
                    'label': str(item.get('label', '')),
                    'rewritten': str(item.get('rewritten', '')),
                    'length': len(item.get('rewritten', ''))
                })
        print(f"   Loaded {len(bench_deploy_formal_data)} examples")
    else:
        print("   ⚠️ File not found, skipping")

    # Create datasets
    print("\n" + "=" * 60)
    print("Creating HuggingFace datasets...")
    print("=" * 60)

    datasets_to_upload = {
        # Sampled datasets (1076 each)
        'bench_eval': Dataset.from_list(bench_eval_data),
        'casual_deploy_matched': Dataset.from_list(casual_deploy_data),
        'bench_deploy_rewritten': Dataset.from_list(rewritten_data),
        'casual_eval_mutual': Dataset.from_list(mutual_data),
        # Full datasets
        'bench_eval_full': Dataset.from_list(bench_eval_full_data),
        'casual_eval_mutual_full': Dataset.from_list(mutual_data_full),
    }

    # Add new rewritten datasets if available
    if casual_deploy_formal_data:
        datasets_to_upload['casual_deploy_formal'] = Dataset.from_list(casual_deploy_formal_data)
    if bench_deploy_formal_data:
        datasets_to_upload['bench_deploy_formal'] = Dataset.from_list(bench_deploy_formal_data)

    print("\nDataset structure:")
    for name, ds in datasets_to_upload.items():
        print(f"  {name}: {len(ds)} examples, columns: {ds.column_names}")

    # Upload each dataset as a separate config
    print("\n" + "=" * 60)
    print(f"Uploading to {args.repo_name}...")
    print("=" * 60)

    for name, ds in datasets_to_upload.items():
        print(f"\nUploading {name}...")
        ds.push_to_hub(
            args.repo_name,
            config_name=name,
            private=args.private
        )
        print(f"  ✅ {name} uploaded")

    print(f"\n✅ Successfully uploaded to: https://huggingface.co/datasets/{args.repo_name}")


if __name__ == "__main__":
    main()

