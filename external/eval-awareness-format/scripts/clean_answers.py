#!/usr/bin/env python3
"""
Remove answers after 'Answer:' in rewritten datasets.
Keeps only 'Answer:' without any text after it.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"

def clean_answer(text):
    """Remove text after Answer:, keeping only 'Answer:'"""
    if not text:
        return text

    # Replace Answer: <anything> with Answer:
    # Match Answer: followed by any text until end of line or end of string
    cleaned = re.sub(r'Answer:\s*.+?(?=\n|$)', 'Answer:', text, flags=re.IGNORECASE | re.DOTALL)

    # Also handle cases where Answer: might be at the end
    cleaned = re.sub(r'Answer:\s*$', 'Answer:', cleaned, flags=re.IGNORECASE | re.MULTILINE)

    return cleaned

def main():
    datasets_to_clean = [
        'bench_deploy_rewritten.json',
        'bench_deploy_formal.json'
    ]

    print("=" * 80)
    print("CLEANING DATASETS: Removing answers after 'Answer:'")
    print("=" * 80)

    for dataset_name in datasets_to_clean:
        input_path = PROCESSED_DIR / dataset_name

        if not input_path.exists():
            print(f"\n⚠️  Skipping {dataset_name} (not found)")
            continue

        print(f"\n📂 Processing {dataset_name}...")

        # Load dataset
        with open(input_path) as f:
            data = json.load(f)

        # Count before
        before_with_answer = 0
        for item in data:
            rewritten = item.get('rewritten', '')
            if re.search(r'Answer:\s*.+', rewritten, re.IGNORECASE):
                before_with_answer += 1

        print(f"  Before: {len(data)} items, {before_with_answer} with answers ({before_with_answer/len(data)*100:.1f}%)")

        # Clean
        cleaned_count = 0
        for item in data:
            original_rewritten = item.get('rewritten', '')
            cleaned_rewritten = clean_answer(original_rewritten)

            if original_rewritten != cleaned_rewritten:
                cleaned_count += 1
                item['rewritten'] = cleaned_rewritten

        print(f"  Cleaned: {cleaned_count} items modified")

        # Count after
        after_with_answer = 0
        for item in data:
            rewritten = item.get('rewritten', '')
            if re.search(r'Answer:\s*.+', rewritten, re.IGNORECASE):
                after_with_answer += 1

        print(f"  After: {len(data)} items, {after_with_answer} with answers ({after_with_answer/len(data)*100:.1f}%)")

        # Save cleaned version
        output_path = PROCESSED_DIR / dataset_name
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Saved cleaned version to {output_path}")

    print("\n" + "=" * 80)
    print("✅ Cleaning complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()





