#!/usr/bin/env python3
"""
Create length-matched Casual-Deploy dataset.

Matches the length distribution of Casual-Deploy to Bench-Eval
using histogram matching (no minimum length filter).
"""

import json
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"

random.seed(42)
np.random.seed(42)


def extract_first_turn(conversations):
    """Extract first user turn from conversation."""
    for turn in conversations:
        sender = turn.get('from', turn.get('user', '')).lower()
        if sender in ['human', 'user']:
            return turn.get('value', turn.get('text', ''))
    return None


def format_bench_eval_prompt(item):
    """Format bench_eval item as prompt (same as in extract_ae_from_datasets.py)."""
    source = item.get("source", "unknown")

    if source == "mmlu":
        q = item.get("question", "")
        choices = item.get("choices", [])
        prompt = f"Question: {q}\nChoices: {', '.join(choices)}\nAnswer:"
    elif source == "arc":
        q = item.get("question", "")
        choices = item.get("choices", {})
        if isinstance(choices, dict):
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choice_str = ", ".join([f"{l}) {t}" for l, t in zip(labels, texts)])
        else:
            choice_str = str(choices)
        prompt = f"Question: {q}\nOptions: {choice_str}\nAnswer:"
    elif source == "gsm8k":
        q = item.get("question", "")
        prompt = f"Problem: {q}\nSolution:"
    else:
        prompt = str(item.get("question", item))

    return prompt


def create_length_matched_dataset(n_bins=30, min_length=0, max_length=None):
    """
    Create Casual-Deploy dataset matched to Bench-Eval length distribution.

    Algorithm:
    1. Compute histogram of Bench-Eval lengths
    2. For each bin, sample proportionally from Casual-Deploy
    3. Result: Casual-Deploy with same length distribution as Bench-Eval
    """

    # Load Bench-Eval
    print("Loading Bench-Eval...")
    with open(PROCESSED_DIR / "bench_eval.json") as f:
        bench_eval = json.load(f)

    # Format prompts (same as in extract_ae_from_datasets.py)
    bench_prompts = [format_bench_eval_prompt(item) for item in bench_eval]
    bench_lengths = [len(p) for p in bench_prompts]

    # Apply max_length filter to bench_eval for histogram
    if max_length:
        bench_lengths = [l for l in bench_lengths if l <= max_length]
        print(f"  After max_length filter: {len(bench_lengths)} examples")
    print(f"  Loaded {len(bench_lengths)} examples")
    print(f"  Length: mean={np.mean(bench_lengths):.1f}, median={np.median(bench_lengths):.1f}")
    print(f"  Range: {min(bench_lengths)} - {max(bench_lengths)}")

    # Load Casual-Deploy
    print("\nLoading Casual-Deploy...")
    with open(PROCESSED_DIR / "casual_deploy_sharegpt4.json") as f:
        casual_deploy = json.load(f)

    # Extract first turns with lengths
    casual_data = []
    for item in casual_deploy:
        first_turn = extract_first_turn(item.get('conversations', []))
        if first_turn:
            length = len(first_turn)
            if length >= min_length and (max_length is None or length <= max_length):
                casual_data.append({
                    'id': item.get('id'),
                    'conversations': item.get('conversations'),
                    'first_turn': first_turn,
                    'first_turn_length': length
                })

    casual_lengths = [item['first_turn_length'] for item in casual_data]
    filter_str = f"length >= {min_length}"
    if max_length:
        filter_str += f" and <= {max_length}"
    print(f"  Loaded {len(casual_lengths)} examples ({filter_str})")
    print(f"  Length: mean={np.mean(casual_lengths):.1f}, median={np.median(casual_lengths):.1f}")
    print(f"  Range: {min(casual_lengths)} - {max(casual_lengths)}")

    # Create histogram bins based on Bench-Eval
    # Use percentile-based bins for better coverage
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(bench_lengths, percentiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicates

    print(f"\nUsing {len(bin_edges)-1} bins based on Bench-Eval percentiles")

    # Count Bench-Eval examples per bin
    bench_hist, _ = np.histogram(bench_lengths, bins=bin_edges)

    # Group Casual-Deploy by bins (only if length falls within bin range)
    casual_by_bin = defaultdict(list)
    for item in casual_data:
        length = item['first_turn_length']
        # Find which bin this belongs to (must be within bin range)
        for bin_idx in range(len(bin_edges) - 1):
            bin_start = bin_edges[bin_idx]
            bin_end = bin_edges[bin_idx + 1]
            if bin_start <= length < bin_end:
                casual_by_bin[bin_idx].append(item)
                break
        # Items outside all bins are not included

    # Sample from each bin proportionally
    print("\n=== Histogram Matching ===")
    print(f"{'Bin':<20} {'Bench':<8} {'Casual':<8} {'Sampled':<8}")
    print("-" * 50)

    matched_data = []
    total_bench = 0
    total_casual = 0
    total_sampled = 0

    for bin_idx in range(len(bin_edges) - 1):
        bin_start = int(bin_edges[bin_idx])
        bin_end = int(bin_edges[bin_idx + 1])
        bench_count = bench_hist[bin_idx]
        casual_available = len(casual_by_bin[bin_idx])

        # Sample min(bench_count, casual_available)
        sample_count = min(bench_count, casual_available)

        if sample_count > 0 and casual_available > 0:
            sampled = random.sample(casual_by_bin[bin_idx], sample_count)
            matched_data.extend(sampled)

        print(f"{bin_start:>4}-{bin_end:<12} {bench_count:<8} {casual_available:<8} {sample_count:<8}")

        total_bench += bench_count
        total_casual += casual_available
        total_sampled += sample_count

    print("-" * 50)
    print(f"{'TOTAL':<20} {total_bench:<8} {total_casual:<8} {total_sampled:<8}")

    # Shuffle result
    random.shuffle(matched_data)

    # Stats
    matched_lengths = [item['first_turn_length'] for item in matched_data]

    print(f"\n=== Result ===")
    print(f"Matched examples: {len(matched_data)}")
    print(f"Length: mean={np.mean(matched_lengths):.1f}, median={np.median(matched_lengths):.1f}")
    print(f"Range: {min(matched_lengths)} - {max(matched_lengths)}")

    # Compare distributions
    print(f"\n=== Distribution Comparison ===")
    print(f"{'Metric':<15} {'Bench-Eval':<15} {'Casual-Matched':<15}")
    print("-" * 45)
    print(f"{'Count':<15} {len(bench_lengths):<15} {len(matched_lengths):<15}")
    print(f"{'Mean':<15} {np.mean(bench_lengths):<15.1f} {np.mean(matched_lengths):<15.1f}")
    print(f"{'Median':<15} {np.median(bench_lengths):<15.1f} {np.median(matched_lengths):<15.1f}")
    print(f"{'Std':<15} {np.std(bench_lengths):<15.1f} {np.std(matched_lengths):<15.1f}")
    print(f"{'Min':<15} {min(bench_lengths):<15} {min(matched_lengths):<15}")
    print(f"{'Max':<15} {max(bench_lengths):<15} {max(matched_lengths):<15}")

    return matched_data, bench_lengths, matched_lengths


def save_matched_dataset(matched_data, output_path):
    """Save matched dataset to JSON."""
    with open(output_path, 'w') as f:
        json.dump(matched_data, f, indent=2)
    print(f"\n✅ Saved to {output_path}")


def plot_distributions(bench_lengths, matched_lengths, output_path):
    """Plot comparison of distributions."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        plt.style.use('seaborn-v0_8-whitegrid')

        bins = np.linspace(0, 500, 50)

        plt.hist(bench_lengths, bins=bins, alpha=0.6, label=f'Bench-Eval (n={len(bench_lengths)})',
                 color='#2ecc71', edgecolor='white')
        plt.hist(matched_lengths, bins=bins, alpha=0.6, label=f'Casual-Deploy Matched (n={len(matched_lengths)})',
                 color='#3498db', edgecolor='white')

        plt.axvline(np.mean(bench_lengths), color='#27ae60', linestyle='--', linewidth=2,
                   label=f'Bench mean: {np.mean(bench_lengths):.0f}')
        plt.axvline(np.mean(matched_lengths), color='#2980b9', linestyle='--', linewidth=2,
                   label=f'Matched mean: {np.mean(matched_lengths):.0f}')

        plt.xlabel('Length (characters)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Length Distribution: Bench-Eval vs Casual-Deploy Matched', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.xlim(0, 500)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {output_path}")

    except ImportError:
        print("⚠️ matplotlib not available, skipping plot")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create length-matched Casual-Deploy dataset")
    parser.add_argument("--n_bins", type=int, default=30, help="Number of histogram bins")
    parser.add_argument("--min_length", type=int, default=0, help="Minimum length filter (0 = no filter)")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum length filter (None = no filter)")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--plot", action="store_true", help="Generate distribution plot")
    args = parser.parse_args()

    # Create matched dataset
    matched_data, bench_lengths, matched_lengths = create_length_matched_dataset(
        n_bins=args.n_bins,
        min_length=args.min_length,
        max_length=args.max_length
    )

    # Save
    output_path = args.output or PROCESSED_DIR / "casual_deploy_matched.json"
    save_matched_dataset(matched_data, output_path)

    # Plot
    if args.plot:
        plot_path = BASE_DIR / "data" / "length_matched_comparison.png"
        plot_distributions(bench_lengths, matched_lengths, plot_path)

