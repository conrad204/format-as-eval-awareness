#!/usr/bin/env python3
"""
Prepare datasets for 2x2 evaluation-awareness research.

Matrix:
              | Evaluation      | Deployment
--------------+-----------------+------------------
  Benchmark   | bench_eval      | bench_deploy_raw
  Casual      | casual_eval     | casual_deploy

This script:
1. Downloads raw datasets (MMLU, ARC, GSM8K, CoQA, ShareGPT-English)
2. Creates processed datasets for each cell of the 2x2 matrix
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
import huggingface_hub

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)


# =============================================================================
# Download Functions
# =============================================================================

def download_mmlu():
    """Download MMLU dataset."""
    print("\n" + "=" * 60)
    print("Downloading MMLU...")
    print("=" * 60)

    output_dir = RAW_DIR / "mmlu"
    output_dir.mkdir(exist_ok=True)

    try:
        ds = load_dataset("cais/mmlu", "all")
        all_data = []
        for split in ds.keys():
            for i, item in enumerate(ds[split]):
                all_data.append({
                    "id": f"mmlu_{split}_{i}",
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "subject": item.get("subject", ""),
                    "split": split,
                    "source": "mmlu"
                })

        output_file = output_dir / "mmlu.json"
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"✓ MMLU: {len(all_data)} examples saved")
        return all_data
    except Exception as e:
        print(f"✗ MMLU error: {e}")
        return []


def download_arc():
    """Download ARC dataset."""
    print("\n" + "=" * 60)
    print("Downloading ARC...")
    print("=" * 60)

    output_dir = RAW_DIR / "arc"
    output_dir.mkdir(exist_ok=True)

    try:
        all_data = []
        for subset in ["ARC-Challenge", "ARC-Easy"]:
            ds = load_dataset("allenai/ai2_arc", subset)
            for split in ds.keys():
                for i, item in enumerate(ds[split]):
                    all_data.append({
                        "id": f"arc_{subset}_{split}_{i}",
                        "question": item["question"],
                        "choices": item["choices"],
                        "answerKey": item["answerKey"],
                        "subset": subset,
                        "split": split,
                        "source": "arc"
                    })

        output_file = output_dir / "arc.json"
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"✓ ARC: {len(all_data)} examples saved")
        return all_data
    except Exception as e:
        print(f"✗ ARC error: {e}")
        return []


def download_gsm8k():
    """Download GSM8K dataset."""
    print("\n" + "=" * 60)
    print("Downloading GSM8K...")
    print("=" * 60)

    output_dir = RAW_DIR / "gsm8k"
    output_dir.mkdir(exist_ok=True)

    try:
        ds = load_dataset("openai/gsm8k", "main")
        all_data = []
        for split in ds.keys():
            for i, item in enumerate(ds[split]):
                all_data.append({
                    "id": f"gsm8k_{split}_{i}",
                    "question": item["question"],
                    "answer": item["answer"],
                    "split": split,
                    "source": "gsm8k"
                })

        output_file = output_dir / "gsm8k.json"
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"✓ GSM8K: {len(all_data)} examples saved")
        return all_data
    except Exception as e:
        print(f"✗ GSM8K error: {e}")
        return []


def download_coqa():
    """Download CoQA dataset for casual-eval."""
    print("\n" + "=" * 60)
    print("Downloading CoQA...")
    print("=" * 60)

    output_dir = RAW_DIR / "coqa"
    output_dir.mkdir(exist_ok=True)

    try:
        ds = load_dataset("stanfordnlp/coqa")
        all_data = []
        for split in ds.keys():
            for i, item in enumerate(ds[split]):
                all_data.append({
                    "id": f"coqa_{split}_{i}",
                    "story": item["story"],
                    "questions": item["questions"],
                    "answers": item["answers"],
                    "split": split,
                    "source": "coqa"
                })

        output_file = output_dir / "coqa.json"
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"✓ CoQA: {len(all_data)} examples saved")
        return all_data
    except Exception as e:
        print(f"✗ CoQA error: {e}")
        return []


def download_sharegpt_english():
    """Download ShareGPT English dataset for casual-deploy and bench-deploy."""
    print("\n" + "=" * 60)
    print("Downloading ShareGPT English...")
    print("=" * 60)

    output_dir = RAW_DIR / "sharegpt_english"
    output_dir.mkdir(exist_ok=True)

    try:
        ds = load_dataset("theblackcat102/sharegpt-english")
        all_data = []
        for i, item in enumerate(ds["train"]):
            # Normalize format: user/text -> from/value
            convs = []
            for turn in item["conversations"]:
                convs.append({
                    "from": turn.get("user", "unknown"),
                    "value": turn.get("text", "")
                })
            all_data.append({
                "id": i,
                "conversations": convs,
                "source": "sharegpt_english"
            })

        output_file = output_dir / "sharegpt_english.json"
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"✓ ShareGPT English: {len(all_data)} conversations saved")
        return all_data
    except Exception as e:
        print(f"✗ ShareGPT English error: {e}")
        return []


# =============================================================================
# Prepare Functions (create 2x2 matrix datasets)
# =============================================================================

def prepare_bench_eval(n_per_source=1000):
    """
    Create Bench-Eval: 1k from each benchmark (MMLU, ARC, GSM8K).
    Total: 3k examples.
    """
    print("\n" + "=" * 60)
    print(f"Preparing Bench-Eval ({n_per_source} per source)...")
    print("=" * 60)

    bench_eval = []

    # MMLU (try multiple filenames)
    mmlu_paths = [RAW_DIR / "mmlu" / "mmlu.json", RAW_DIR / "mmlu" / "mmlu_sample.json"]
    for mmlu_path in mmlu_paths:
        if mmlu_path.exists():
            with open(mmlu_path) as f:
                mmlu = json.load(f)
            sample = random.sample(mmlu, min(n_per_source, len(mmlu)))
            for item in sample:
                item['source'] = 'mmlu'
            bench_eval.extend(sample)
            print(f"  ✓ MMLU: {len(sample)} examples")
            break
    else:
        print(f"  ✗ MMLU not found")

    # ARC (try multiple filenames)
    arc_paths = [RAW_DIR / "arc" / "arc.json", RAW_DIR / "arc" / "arc_test.json"]
    for arc_path in arc_paths:
        if arc_path.exists():
            with open(arc_path) as f:
                arc = json.load(f)
            sample = random.sample(arc, min(n_per_source, len(arc)))
            for item in sample:
                item['source'] = 'arc'
            bench_eval.extend(sample)
            print(f"  ✓ ARC: {len(sample)} examples")
            break
    else:
        print(f"  ✗ ARC not found")

    # GSM8K (try multiple filenames)
    gsm_paths = [RAW_DIR / "gsm8k" / "gsm8k.json", RAW_DIR / "gsm8k" / "gsm8k_test.json"]
    for gsm_path in gsm_paths:
        if gsm_path.exists():
            with open(gsm_path) as f:
                gsm = json.load(f)
            sample = random.sample(gsm, min(n_per_source, len(gsm)))
            for item in sample:
                item['source'] = 'gsm8k'
            bench_eval.extend(sample)
            print(f"  ✓ GSM8K: {len(sample)} examples")
            break
    else:
        print(f"  ✗ GSM8K not found")

    # Save
    output_file = PROCESSED_DIR / "bench_eval.json"
    with open(output_file, "w") as f:
        json.dump(bench_eval, f, indent=2)
    print(f"\n✅ Bench-Eval: {len(bench_eval)} examples saved to {output_file}")
    return bench_eval


def prepare_casual_eval(n_examples=3000):
    """
    Create Casual-Eval: CoQA wrapped in User:/Assistant: format.
    """
    print("\n" + "=" * 60)
    print(f"Preparing Casual-Eval ({n_examples} examples)...")
    print("=" * 60)

    coqa_path = RAW_DIR / "coqa" / "coqa.json"
    if not coqa_path.exists():
        print(f"  ✗ CoQA not found at {coqa_path}")
        return []

    with open(coqa_path) as f:
        coqa = json.load(f)

    # Convert to User:/Assistant: format
    formatted = []
    for item in coqa:
        story = item["story"]
        questions = item["questions"]
        answers = item["answers"]

        for i in range(len(questions)):
            q = questions[i]
            a = answers["input_text"][i] if isinstance(answers, dict) else answers[i]["input_text"]

            formatted.append({
                "id": f"{item['id']}_q{i}",
                "conversations": [
                    {"from": "user", "value": f"Context: {story}\n\nQuestion: {q}"},
                    {"from": "assistant", "value": a}
                ],
                "source": "coqa_casual_eval"
            })

    # Sample
    random.shuffle(formatted)
    selected = formatted[:n_examples]

    # Save
    output_file = PROCESSED_DIR / "casual_eval_coqa.json"
    with open(output_file, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\n✅ Casual-Eval: {len(selected)} examples saved to {output_file}")
    return selected


def prepare_sharegpt_splits(n_each=3000):
    """
    Split ShareGPT English into casual_deploy and bench_deploy_raw.
    """
    print("\n" + "=" * 60)
    print(f"Preparing ShareGPT splits ({n_each} each)...")
    print("=" * 60)

    sharegpt_path = RAW_DIR / "sharegpt_english" / "sharegpt_english.json"
    if not sharegpt_path.exists():
        print(f"  ✗ ShareGPT English not found at {sharegpt_path}")
        return [], []

    with open(sharegpt_path) as f:
        data = json.load(f)

    random.shuffle(data)

    # Split
    casual_deploy = data[:n_each]
    bench_deploy_raw = data[n_each:n_each * 2]

    # Save casual_deploy
    output_file = PROCESSED_DIR / "casual_deploy_sharegpt4.json"
    with open(output_file, "w") as f:
        json.dump(casual_deploy, f, indent=2)
    print(f"  ✓ casual_deploy: {len(casual_deploy)} conversations")

    # Save bench_deploy_raw
    output_file = PROCESSED_DIR / "bench_deploy_raw_sharegpt4.json"
    with open(output_file, "w") as f:
        json.dump(bench_deploy_raw, f, indent=2)
    print(f"  ✓ bench_deploy_raw: {len(bench_deploy_raw)} conversations")

    print(f"\n✅ ShareGPT splits saved")
    return casual_deploy, bench_deploy_raw


# =============================================================================
# Main
# =============================================================================

def download_all():
    """Download all raw datasets."""
    print("\n" + "=" * 60)
    print("DOWNLOADING RAW DATASETS")
    print("=" * 60)

    download_mmlu()
    download_arc()
    download_gsm8k()
    download_coqa()
    download_sharegpt_english()


def prepare_all():
    """Prepare all processed datasets for 2x2 matrix."""
    print("\n" + "=" * 60)
    print("PREPARING 2x2 MATRIX DATASETS")
    print("=" * 60)

    prepare_bench_eval(n_per_source=1000)      # 3k total (1k each)
    prepare_casual_eval(n_examples=3000)        # 3k
    prepare_sharegpt_splits(n_each=3000)        # 3k + 3k


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare datasets for evaluation-awareness research")
    parser.add_argument("--download", action="store_true", help="Download raw datasets")
    parser.add_argument("--prepare", action="store_true", help="Prepare processed datasets")
    parser.add_argument("--all", action="store_true", help="Download and prepare all")
    args = parser.parse_args()

    if args.all or (not args.download and not args.prepare):
        download_all()
        prepare_all()
    else:
        if args.download:
            download_all()
        if args.prepare:
            prepare_all()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    2x2 Matrix:

                  | Evaluation           | Deployment
    --------------+----------------------+----------------------
      Benchmark   | bench_eval (3k)      | bench_deploy_raw (3k)
                  | MMLU+ARC+GSM8K       | ShareGPT (to rewrite)
    --------------+----------------------+----------------------
      Casual      | casual_eval (3k)     | casual_deploy (3k)
                  | CoQA (User/Asst)     | ShareGPT
    """)


if __name__ == "__main__":
    main()

