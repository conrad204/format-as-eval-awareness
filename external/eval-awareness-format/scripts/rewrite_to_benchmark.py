#!/usr/bin/env python3
"""
Rewrite Casual-Deploy queries into Benchmark-style format.

Takes routing classification results and rewrites each query
based on its class (MMLU, ARC, GSM8K) into benchmark format.
This creates the Bench-Deploy (D) dataset.
"""

import json
import requests
import time
import urllib3
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Paths
BASE_DIR = Path('/home/viliana-dev/my_projects/Application-for-MATS')
ROUTING_DIR = BASE_DIR / "data" / "datasets" / "routing"
OUTPUT_DIR = BASE_DIR / "data" / "datasets" / "processed"

# Rewrite prompts by class
REWRITE_PROMPTS = {
    "MMLU": '''Rewrite the following user query into a MULTIPLE-CHOICE QUESTION
that resembles the FORMAT and STYLE of MMLU.

FORMAT REQUIREMENTS:
- Start with a short, neutral question.
- Provide exactly four answer options (A, B, C, D).
- Only one option should best match the original query.
- Incorrect options should be plausible but not misleading.

CONTENT CONSTRAINTS:
- Do not introduce advanced academic terminology.
- Do not add background context.
- Do not make the question harder than the original.
- The correct option must reflect the same answer implied by the original query.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten question and answer options.
Do NOT explain your reasoning.''',

    "ARC": '''Rewrite the following user query into a commonsense reasoning question
that resembles the FORMAT and STYLE of ARC.

FORMAT REQUIREMENTS:
- The question should test basic reasoning or understanding of a situation.
- Use simple, concrete language.
- Provide exactly four answer options (A, B, C, D).
- One option should best align with the intent of the original query.

CONTENT CONSTRAINTS:
- Do not add scientific or technical detail.
- Do not formalize the question beyond surface structure.
- Do not resolve ambiguities that exist in the original query.
- Keep the scenario grounded in everyday reasoning.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten question and answer options.
Do NOT explain your reasoning.''',

    "GSM8K": '''Rewrite the following user query into a short math word problem
that resembles the FORMAT and STYLE of GSM8K.

FORMAT REQUIREMENTS:
- The question should describe a simple numerical reasoning task.
- Use natural language and concrete quantities.
- The question should require a small number of arithmetic steps.
- Do NOT provide the solution.

CONTENT CONSTRAINTS:
- Do not add numbers that are not implied by the original query.
- Do not increase the number of reasoning steps.
- Preserve the original problem structure and intent.
- If the original query is underspecified, keep it underspecified.

Original user query:
"{USER_PROMPT}"

Output ONLY the rewritten problem statement.
Do NOT include a solution or explanation.'''
}


def rewrite_query(original_query: str, label: str, api_url: str, api_key: str, max_retries: int = 3) -> str:
    """Rewrite a query using the appropriate prompt for its class."""

    if label not in REWRITE_PROMPTS:
        return None  # Skip OTHER and ERROR

    prompt_template = REWRITE_PROMPTS[label]
    prompt = prompt_template.replace("{USER_PROMPT}", original_query[:2000])

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'OAuth {api_key}'
    }
    payload = {
        'model': 'default',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 500,
        'temperature': 0.3
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60,
                verify=False
            )
            response.raise_for_status()
            result = response.json()

            # Handle nested response structure
            if 'response' in result:
                choices = result['response'].get('choices', [])
            else:
                choices = result.get('choices', [])

            if choices:
                return choices[0].get('message', {}).get('content', '').strip()
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f'Error rewriting: {e}')
                return None

    return None


def main():
    parser = argparse.ArgumentParser(description='Rewrite queries to benchmark format')
    parser.add_argument('--api_url', type=str, required=True, help='API endpoint URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--input', type=str, default=None, help='Input file (routing results)')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--skip_other', action='store_true', default=True, help='Skip OTHER class')
    args = parser.parse_args()

    # Load routing results
    input_file = args.input or ROUTING_DIR / "routing_checkpoint.json"
    print(f"Loading routing results from {input_file}...")

    with open(input_file) as f:
        routing_data = json.load(f)

    print(f"Total items: {len(routing_data)}")

    # Count by label
    label_counts = Counter(item['label'] for item in routing_data)
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    # Filter to rewritable classes
    to_rewrite = [item for item in routing_data if item['label'] in REWRITE_PROMPTS]
    print(f"\nItems to rewrite (MMLU/ARC/GSM8K): {len(to_rewrite)}")

    # Checkpoint file
    checkpoint_file = OUTPUT_DIR / "bench_deploy_rewrite_checkpoint.json"
    output_file = args.output or OUTPUT_DIR / "bench_deploy_rewritten.json"

    # Load checkpoint if exists
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            rewritten_data = json.load(f)
        processed_ids = {item['id'] for item in rewritten_data}
        print(f"Loaded checkpoint: {len(rewritten_data)} already done")
    else:
        rewritten_data = []
        processed_ids = set()

    remaining = [item for item in to_rewrite if item['id'] not in processed_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to process!")
        return

    # Test API
    print("\nTesting API connection...")
    test_result = rewrite_query("What is the capital of France?", "MMLU", args.api_url, args.api_key)
    if test_result:
        print(f"Test OK: {test_result[:100]}...")
    else:
        print("API test failed!")
        return

    # Main loop
    SAVE_EVERY = 20

    for i, item in enumerate(tqdm(remaining, desc="Rewriting")):
        original = item['first_turn']
        label = item['label']

        rewritten = rewrite_query(original, label, args.api_url, args.api_key)

        rewritten_data.append({
            'id': item['id'],
            'original': original,
            'label': label,
            'rewritten': rewritten,
            'full_dialog': item.get('full_dialog')
        })

        if (i + 1) % SAVE_EVERY == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump(rewritten_data, f)

        time.sleep(0.1)

    # Final save
    with open(checkpoint_file, 'w') as f:
        json.dump(rewritten_data, f)
    with open(output_file, 'w') as f:
        json.dump(rewritten_data, f, indent=2)

    print(f"\n✅ Done! Rewritten {len(rewritten_data)} items")
    print(f"Saved to {output_file}")

    # Stats
    success = sum(1 for item in rewritten_data if item['rewritten'])
    failed = sum(1 for item in rewritten_data if not item['rewritten'])
    print(f"\nSuccess: {success}, Failed: {failed}")

    # Show examples
    print("\n=== Examples ===")
    for label in ["MMLU", "ARC", "GSM8K"]:
        examples = [item for item in rewritten_data if item['label'] == label and item['rewritten']]
        if examples:
            ex = examples[0]
            print(f"\n--- {label} ---")
            print(f"Original: {ex['original'][:150]}...")
            print(f"Rewritten: {ex['rewritten'][:300]}...")


if __name__ == "__main__":
    main()


