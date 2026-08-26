#!/usr/bin/env python3
"""
Rewrite Casual-Deploy queries into FORMAL ACADEMIC style.

This creates Casual-Deploy-Formal dataset for diagnostic experiment:
- Same casual CONTENT as Casual-Deploy
- But in FORMAL academic STYLE (without benchmark structure)
- Uses routing labels (MMLU, ARC, GSM8K) for class-specific rewriting

If probe tracks STYLE: Casual-Deploy-Formal → should be classified as "bench/eval" (class 0)
If probe tracks CONTENT: Casual-Deploy-Formal → should be classified as "casual/deploy" (class 1)
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
PROCESSED_DIR = BASE_DIR / "data" / "datasets" / "processed"
ROUTING_DIR = BASE_DIR / "data" / "datasets" / "routing"

# System prompt for formal rewriting
SYSTEM_PROMPT = """You are rewriting user prompts for a scientific diagnostic experiment.
Your goal is to convert a casual, conversational user request into a formal, academic-style task description, without changing the underlying task or intent.

This rewrite is used to study stylistic effects in language models.
Do NOT introduce new content, examples, explanations, or scenarios."""

# Class-specific rewrite prompts
REWRITE_PROMPTS = {
    "MMLU": """Rewrite the following user request into a FORMAL ACADEMIC knowledge question.

Style requirements:
- Use formal, academic, impersonal tone typical of standardized knowledge assessments
- Avoid conversational markers (e.g., hey, please, can you, thanks)
- Avoid first-person and second-person pronouns where possible
- Use precise, technical language appropriate for the domain

Content constraints:
- Preserve the exact task/question semantics - do not change what is being asked
- Do NOT add multiple-choice options or answer choices
- Do NOT add explanations, examples, or clarifications
- Do NOT introduce fictional scenarios or narratives

Output format:
- Output ONLY the rewritten question/task
- Use openings like: "Determine...", "Identify...", "Explain the concept of...", "What is the relationship between..."

Original user request:
"{USER_PROMPT}"

Rewritten formal question:""",

    "ARC": """Rewrite the following user request into a FORMAL ACADEMIC reasoning task.

Style requirements:
- Use formal, academic tone typical of scientific reasoning assessments
- Frame as an analytical or inferential task
- Avoid conversational language and personal pronouns
- Use precise, clear language

Content constraints:
- Preserve the exact reasoning task - do not change the logical structure
- Do NOT add multiple-choice options or answer choices
- Do NOT add explanations or worked examples
- Do NOT introduce fictional scenarios or characters

Output format:
- Output ONLY the rewritten task description
- Use openings like: "Analyze the following situation...", "Given the conditions described, determine...", "Apply reasoning to solve..."

Original user request:
"{USER_PROMPT}"

Rewritten formal task:""",

    "GSM8K": """Rewrite the following user request into a FORMAL ACADEMIC mathematical problem statement.

Style requirements:
- Use formal, academic tone typical of mathematical problem sets
- Present as a precise quantitative problem
- Avoid conversational language and personal pronouns
- Use mathematical terminology where appropriate

Content constraints:
- Preserve the exact mathematical task - do not change numbers or operations
- Do NOT provide the solution or intermediate steps
- Do NOT add explanations or hints
- Do NOT introduce fictional scenarios or narratives

Output format:
- Output ONLY the rewritten problem statement
- Use openings like: "Calculate...", "Determine the value of...", "Given the following quantities, compute...", "Solve for..."

Original user request:
"{USER_PROMPT}"

Rewritten formal problem:"""
}

# Fallback for OTHER class
DEFAULT_REWRITE_PROMPT = """Rewrite the following user request according to these rules:

1. Preserve task semantics
   - The rewritten version must ask for the same task as the original.
   - Do not add sub-questions, clarifications, or extra constraints.
   - Do not simplify or reframe the problem.

2. Change only the style (register)
   - Use a formal, academic, impersonal tone.
   - Avoid conversational markers (e.g., hey, please, can you, thanks).
   - Avoid first-person and second-person pronouns where possible.
   - Avoid emotional or subjective language.

3. Do NOT add benchmark structure
   - Do NOT use multiple-choice format.
   - Do NOT add options, answers, or "Correct answer:" sections.
   - The output must remain a single task description.

4. Avoid narrative framing
   - Do NOT introduce fictional scenarios, people, or motivations.
   - Avoid phrases like "A student wants to…", "Imagine that…", "Suppose someone…".

5. Output format (strict)
   - Output only the rewritten prompt text.
   - Do not include explanations, comments, or meta-text.
   - Do not mention rewriting, style, or experiments.

Preferred opening templates (choose the most natural one):
- "Provide a solution for the following task: …"
- "Given the following problem, produce an appropriate solution: …"
- "Solve the following task: …"
- "Perform the following transformation or analysis: …"

Original user request:
"{USER_PROMPT}"

Rewritten prompt:"""


def rewrite_to_formal(original_query: str, label: str, api_url: str, api_key: str, max_retries: int = 3) -> str:
    """Rewrite a query to formal academic style based on its class."""

    # Get class-specific prompt or fallback to default
    if label in REWRITE_PROMPTS:
        prompt_template = REWRITE_PROMPTS[label]
    else:
        prompt_template = DEFAULT_REWRITE_PROMPT

    prompt = prompt_template.replace("{USER_PROMPT}", original_query[:2000])

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'OAuth {api_key}'
    }
    payload = {
        'model': 'default',
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ],
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
    parser = argparse.ArgumentParser(description='Rewrite casual queries to formal academic style')
    parser.add_argument('--api_url', type=str, required=True, help='API endpoint URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--input', type=str, default=None, help='Input file (routing results with labels)')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of items to process')
    parser.add_argument('--skip_other', action='store_true', default=True, help='Skip OTHER class')
    args = parser.parse_args()

    # Load routing results (which have labels)
    input_file = args.input or ROUTING_DIR / "routing_checkpoint.json"
    print(f"Loading routing data from {input_file}...")

    with open(input_file) as f:
        data = json.load(f)

    print(f"Total items: {len(data)}")

    # Count by label
    label_counts = Counter(item.get('label', 'UNKNOWN') for item in data)
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    # Filter to rewritable classes (MMLU, ARC, GSM8K)
    if args.skip_other:
        data = [item for item in data if item.get('label') in REWRITE_PROMPTS]
        print(f"\nFiltered to MMLU/ARC/GSM8K: {len(data)}")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to: {len(data)}")

    # Checkpoint file
    checkpoint_file = PROCESSED_DIR / "casual_deploy_formal_checkpoint.json"
    output_file = args.output or PROCESSED_DIR / "casual_deploy_formal.json"

    # Load checkpoint if exists
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            rewritten_data = json.load(f)
        processed_ids = {item['id'] for item in rewritten_data}
        print(f"Loaded checkpoint: {len(rewritten_data)} already done")
    else:
        rewritten_data = []
        processed_ids = set()

    remaining = [item for item in data if item['id'] not in processed_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to process!")
        # Save final output
        with open(output_file, 'w') as f:
            json.dump(rewritten_data, f, indent=2)
        print(f"Saved to {output_file}")
        return

    # Test API
    print("\nTesting API connection...")
    test_result = rewrite_to_formal("hey can you help me write a python script to sort a list?", "MMLU", args.api_url, args.api_key)
    if test_result:
        print(f"Test OK: {test_result[:150]}...")
    else:
        print("API test failed!")
        return

    # Main loop
    SAVE_EVERY = 20

    for i, item in enumerate(tqdm(remaining, desc="Rewriting to formal")):
        original = item['first_turn']
        label = item.get('label', 'OTHER')

        rewritten = rewrite_to_formal(original, label, args.api_url, args.api_key)

        rewritten_data.append({
            'id': item['id'],
            'original': original,
            'label': label,
            'rewritten': rewritten,
            'original_length': len(original),
            'rewritten_length': len(rewritten) if rewritten else 0
        })

        if (i + 1) % SAVE_EVERY == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump(rewritten_data, f)
            print(f"\n  Checkpoint saved: {len(rewritten_data)} items")

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

    # Length comparison
    orig_lens = [item['original_length'] for item in rewritten_data if item['rewritten']]
    new_lens = [item['rewritten_length'] for item in rewritten_data if item['rewritten']]

    if orig_lens:
        import numpy as np
        print(f"\nLength stats:")
        print(f"  Original:  mean={np.mean(orig_lens):.0f}, median={np.median(orig_lens):.0f}")
        print(f"  Rewritten: mean={np.mean(new_lens):.0f}, median={np.median(new_lens):.0f}")

    # Show examples by class
    print("\n=== Examples ===")
    for label in ["MMLU", "ARC", "GSM8K"]:
        examples = [item for item in rewritten_data if item.get('label') == label and item['rewritten']]
        if examples:
            ex = examples[0]
            print(f"\n--- {label} ---")
            print(f"Original: {ex['original'][:200]}...")
            print(f"Formal:   {ex['rewritten'][:200]}...")


if __name__ == "__main__":
    main()

