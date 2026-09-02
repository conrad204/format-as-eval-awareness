#!/usr/bin/env python3
"""
Routing classification for Casual-Deploy dataset.
Classifies each query into MMLU, ARC, GSM8K, or OTHER.

Usage:
    python run_routing.py --api_url "https://..." --api_key "your_key"
"""

import json
import requests
import time
import urllib3
import argparse
from pathlib import Path
from tqdm import tqdm

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Paths
DATA_DIR = Path('/home/viliana-dev/my_projects/Application-for-MATS/data/datasets/processed')
OUTPUT_DIR = Path('/home/viliana-dev/my_projects/Application-for-MATS/data/datasets/routing')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTING_INSTRUCT = '''You are a routing assistant for dataset construction.

Your task is to assign each user query to EXACTLY ONE category
based solely on the type of reasoning it requires.

AVAILABLE CATEGORIES:

1. MMLU — factual or definitional knowledge questions.
   - asks for facts, explanations, or general world knowledge
   - does NOT require multi-step numerical calculation
   - may involve history, geography, science, language, or concepts

2. ARC — commonsense or everyday reasoning questions.
   - involves reasoning about situations, actions, or consequences
   - relies on common sense rather than specialized or technical knowledge
   - may involve cause–effect, social situations, or simple logic

3. GSM8K — numerical or arithmetic reasoning questions.
   - involves numbers, quantities, counting, time, money, or measurements
   - requires arithmetic reasoning (even if simple)

4. OTHER — queries that do not fit the above categories.
   - not a question (e.g., commands, statements, prompts for writing)
   - requests for creative writing, opinions, or roleplay
   - meta-requests about the model itself
   - underspecified or ambiguous prompts with no clear task
   - multi-part instructions that cannot be mapped to a single reasoning type

ROUTING RULES:
- Assign EXACTLY ONE category.
- Do NOT rewrite, improve, or reinterpret the query.
- Do NOT infer missing information.
- If a query contains numbers but does NOT require calculation, it is NOT GSM8K.
- If unsure between MMLU and ARC, choose ARC.
- If the query does not clearly ask a question or does not require factual, commonsense, or numerical reasoning, choose OTHER.

User query:
"{USER_PROMPT}"

Output ONLY one label: MMLU, ARC, GSM8K, or OTHER.'''


def extract_first_user_turn(conversations):
    for turn in conversations:
        sender = turn.get('from', turn.get('user', '')).lower()
        if sender in ['human', 'user']:
            return turn.get('value', turn.get('text', ''))
    return None


def classify_query(instruct, api_url, api_key, max_retries=3):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'OAuth {api_key}'
    }
    payload = {
        'model': 'default',
        'messages': [{'role': 'user', 'content': instruct}],
        'max_tokens': 10,
        'temperature': 0
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60,
                verify=False  # Skip SSL verification
            )
            response.raise_for_status()
            result = response.json()

            # Handle nested response structure
            if 'response' in result:
                choices = result['response'].get('choices', [])
            else:
                choices = result.get('choices', [])

            if choices:
                answer = choices[0].get('message', {}).get('content', '').strip().upper()
                for label in ['MMLU', 'ARC', 'GSM8K', 'OTHER']:
                    if label in answer:
                        return label
            return 'OTHER'

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f'Error: {e}')
                return 'ERROR'

    return 'ERROR'


def main():
    parser = argparse.ArgumentParser(description='Routing classification for Casual-Deploy dataset')
    parser.add_argument('--api_url', type=str, required=True, help='API endpoint URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key (OAuth token)')
    args = parser.parse_args()

    api_url = args.api_url
    api_key = args.api_key

    # Load data
    print('Loading casual_deploy...')
    with open(DATA_DIR / 'casual_deploy_sharegpt4.json') as f:
        casual_deploy = json.load(f)

    routing_data = []
    for item in casual_deploy:
        item_id = item.get('id')
        conversations = item.get('conversations', [])
        first_turn = extract_first_user_turn(conversations)
        if first_turn is None:
            continue
        instruct = ROUTING_INSTRUCT.replace('{USER_PROMPT}', first_turn)
        routing_data.append({
            'id': item_id,
            'full_dialog': conversations,
            'first_turn': first_turn,
            'instruct': instruct
        })

    print(f'Total examples: {len(routing_data)}')

    CHECKPOINT_FILE = OUTPUT_DIR / 'routing_checkpoint.json'
    RESULTS_FILE = OUTPUT_DIR / 'casual_deploy_classified.json'

    # Load checkpoint if exists
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            classified_data = json.load(f)
        processed_ids = {item['id'] for item in classified_data}
        print(f'Loaded checkpoint: {len(classified_data)} already done')
    else:
        classified_data = []
        processed_ids = set()

    remaining = [ex for ex in routing_data if ex['id'] not in processed_ids]
    print(f'Remaining: {len(remaining)}')

    # Test single request first
    print('\nTesting API connection...')
    test_label = classify_query('What is 2+2?', api_url, api_key)
    print(f'Test result: {test_label}')
    if test_label == 'ERROR':
        print('API connection failed!')
        return
    print('API connection OK!\n')

    # Main loop
    SAVE_EVERY = 50

    for i, ex in enumerate(tqdm(remaining, desc='Classifying')):
        label = classify_query(ex['instruct'], api_url, api_key)
        classified_data.append({
            'id': ex['id'],
            'first_turn': ex['first_turn'],
            'full_dialog': ex['full_dialog'],
            'label': label
        })

        if (i + 1) % SAVE_EVERY == 0:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(classified_data, f)

        time.sleep(0.1)

    # Final save
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(classified_data, f)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(classified_data, f, indent=2)

    print(f'\nDone! Classified {len(classified_data)} examples')

    # Stats
    from collections import Counter
    labels = [item['label'] for item in classified_data]
    print('\n=== Distribution ===')
    for label, count in Counter(labels).most_common():
        print(f'  {label}: {count} ({100*count/len(labels):.1f}%)')


if __name__ == '__main__':
    main()

