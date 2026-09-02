#!/usr/bin/env python3
"""
Plot length distribution boxplots for all 4 datasets (2x2 matrix).

Datasets:
- Bench-Eval (formatted prompts)
- Casual-Deploy Matched (histogram-matched to Bench-Eval)
- Bench-Deploy Rewritten (rewritten casual queries in benchmark format)
- Casual-Eval (MuTual first turns)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
MUTUAL_DIR = BASE_DIR / "data" / "datasets" / "raw" / "mutual" / "train"
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


def load_mutual_first_turns():
    """Load first turns from MuTual dialogues."""
    mutual_first_turns = []

    for filename in os.listdir(MUTUAL_DIR):
        if filename.endswith('.txt'):
            with open(MUTUAL_DIR / filename) as f:
                data = json.loads(f.read())
                article = data.get('article', '')
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
                        mutual_first_turns.append(first_turn.strip())

    return mutual_first_turns


def main():
    # Load datasets
    print("Loading datasets...")

    # Bench-Eval
    with open(PROCESSED_DIR / "bench_eval.json") as f:
        bench_eval = json.load(f)
    bench_lengths = [len(format_bench_eval_prompt(item)) for item in bench_eval]
    # Sample 1000
    bench_sampled = random.sample(bench_lengths, 1000)

    # Casual-Deploy Matched
    with open(PROCESSED_DIR / "casual_deploy_matched.json") as f:
        casual_matched = json.load(f)
    matched_lengths = [item['first_turn_length'] for item in casual_matched]
    # Sample 1000 if more
    if len(matched_lengths) > 1000:
        matched_sampled = random.sample(matched_lengths, 1000)
    else:
        matched_sampled = matched_lengths

    # Bench-Deploy Rewritten
    with open(PROCESSED_DIR / "bench_deploy_rewritten.json") as f:
        rewritten = json.load(f)
    rewritten_lengths = [len(item['rewritten']) for item in rewritten]

    # Casual-Eval (MuTual)
    print("Loading MuTual...")
    mutual_first_turns = load_mutual_first_turns()
    mutual_sampled = random.sample(mutual_first_turns, min(1000, len(mutual_first_turns)))
    mutual_lengths = [len(t) for t in mutual_sampled]

    print(f"Loaded: Bench-Eval={len(bench_sampled)}, Casual-Deploy={len(matched_sampled)}, "
          f"Bench-Deploy={len(rewritten_lengths)}, Casual-Eval={len(mutual_lengths)}")

    # Stats
    print(f"\n{'='*80}")
    print("LENGTH STATISTICS (2x2 Matrix)")
    print(f"{'='*80}")
    print(f"\n{'Dataset':<25} {'N':<8} {'Mean':<10} {'Median':<10} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    print(f"{'Bench-Eval':<25} {len(bench_sampled):<8} {np.mean(bench_sampled):<10.1f} {np.median(bench_sampled):<10.1f} {min(bench_sampled):<8} {max(bench_sampled):<8}")
    print(f"{'Casual-Deploy Matched':<25} {len(matched_sampled):<8} {np.mean(matched_sampled):<10.1f} {np.median(matched_sampled):<10.1f} {min(matched_sampled):<8} {max(matched_sampled):<8}")
    print(f"{'Bench-Deploy Rewritten':<25} {len(rewritten_lengths):<8} {np.mean(rewritten_lengths):<10.1f} {np.median(rewritten_lengths):<10.1f} {min(rewritten_lengths):<8} {max(rewritten_lengths):<8}")
    print(f"{'Casual-Eval (MuTual)':<25} {len(mutual_lengths):<8} {np.mean(mutual_lengths):<10.1f} {np.median(mutual_lengths):<10.1f} {min(mutual_lengths):<8} {max(mutual_lengths):<8}")

    # Prepare data for boxplot
    data = [bench_sampled, matched_sampled, rewritten_lengths, mutual_lengths]
    labels = [
        f'Bench-Eval\n(n={len(bench_sampled)})',
        f'Casual-Deploy\nMatched\n(n={len(matched_sampled)})',
        f'Bench-Deploy\nRewritten\n(n={len(rewritten_lengths)})',
        f'Casual-Eval\nMuTual\n(n={len(mutual_lengths)})'
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    # Plot boxplot
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True,
                    medianprops=dict(color='black', linewidth=2))

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, 5), means, color='white', marker='D', s=100, zorder=5,
               edgecolors='black', linewidths=1.5, label='Mean')

    # Add annotations for medians and means
    for i, (d, mean) in enumerate(zip(data, means), 1):
        median = np.median(d)
        ax.annotate(f'med={median:.0f}\nmean={mean:.0f}',
                   xy=(i, max(d)),
                   xytext=(i, max(d) + 50),
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Length (characters)', fontsize=12)
    ax.set_title('Length Distribution: 2x2 Matrix Datasets (Boxplot)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(d) for d in data) + 150)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "length_boxplot_2x2.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")


if __name__ == "__main__":
    main()

