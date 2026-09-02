#!/usr/bin/env python3
"""
Plot length distribution comparison for 3 main datasets (histograms).

Compares:
- Bench-Eval (formatted prompts, sampled to match size)
- Casual-Deploy Matched (histogram-matched to Bench-Eval)
- Bench-Deploy Rewritten (rewritten casual queries in benchmark format)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
OUTPUT_DIR = BASE_DIR / "data"

random.seed(42)


def format_bench_eval_prompt(item):
    """Format bench_eval item as prompt (same as in extract_ae_from_datasets.py)."""
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
    # Load datasets
    print("Loading datasets...")

    with open(PROCESSED_DIR / "bench_eval.json") as f:
        bench_eval = json.load(f)

    with open(PROCESSED_DIR / "casual_deploy_matched.json") as f:
        casual_matched = json.load(f)

    with open(PROCESSED_DIR / "bench_deploy_rewritten.json") as f:
        rewritten = json.load(f)

    # Get lengths
    bench_all_lengths = [len(format_bench_eval_prompt(item)) for item in bench_eval]
    n_matched = len(casual_matched)

    # Sample bench_eval to match size
    bench_sampled_indices = random.sample(range(len(bench_eval)), n_matched)
    bench_sampled_lengths = [bench_all_lengths[i] for i in bench_sampled_indices]

    # Casual matched lengths
    matched_lengths = [item['first_turn_length'] for item in casual_matched]

    # Rewritten lengths (already formatted!)
    rewritten_lengths = [len(item['rewritten']) for item in rewritten]

    # Stats
    print(f"\n{'='*70}")
    print("LENGTH STATISTICS")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<25} {'N':<8} {'Mean':<10} {'Median':<10} {'Min':<8} {'Max':<8}")
    print("-" * 70)
    print(f"{'Bench-Eval Sampled':<25} {len(bench_sampled_lengths):<8} {np.mean(bench_sampled_lengths):<10.1f} {np.median(bench_sampled_lengths):<10.1f} {min(bench_sampled_lengths):<8} {max(bench_sampled_lengths):<8}")
    print(f"{'Casual-Deploy Matched':<25} {len(matched_lengths):<8} {np.mean(matched_lengths):<10.1f} {np.median(matched_lengths):<10.1f} {min(matched_lengths):<8} {max(matched_lengths):<8}")
    print(f"{'Bench-Deploy Rewritten':<25} {len(rewritten_lengths):<8} {np.mean(rewritten_lengths):<10.1f} {np.median(rewritten_lengths):<10.1f} {min(rewritten_lengths):<8} {max(rewritten_lengths):<8}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    bins = np.linspace(0, 800, 50)

    plt.hist(bench_sampled_lengths, bins=bins, alpha=0.5,
             label=f'Bench-Eval Sampled (n={len(bench_sampled_lengths)})',
             color='#2ecc71', edgecolor='white')
    plt.hist(matched_lengths, bins=bins, alpha=0.5,
             label=f'Casual-Deploy Matched (n={len(matched_lengths)})',
             color='#3498db', edgecolor='white')
    plt.hist(rewritten_lengths, bins=bins, alpha=0.5,
             label=f'Bench-Deploy Rewritten (n={len(rewritten_lengths)})',
             color='#e74c3c', edgecolor='white')

    plt.axvline(np.mean(bench_sampled_lengths), color='#27ae60', linestyle='--', linewidth=2,
               label=f'Bench mean: {np.mean(bench_sampled_lengths):.0f}')
    plt.axvline(np.mean(matched_lengths), color='#2980b9', linestyle='--', linewidth=2,
               label=f'Matched mean: {np.mean(matched_lengths):.0f}')
    plt.axvline(np.mean(rewritten_lengths), color='#c0392b', linestyle='--', linewidth=2,
               label=f'Rewritten mean: {np.mean(rewritten_lengths):.0f}')

    plt.xlabel('Length (characters)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Length Distribution: All Datasets Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.xlim(0, 800)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "length_matched_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")


if __name__ == "__main__":
    main()
