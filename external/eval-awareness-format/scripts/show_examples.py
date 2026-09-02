#!/usr/bin/env python3
import json

# Load datasets
with open('data/datasets/processed/bench_deploy_rewritten.json') as f:
    bench = {item['id']: item for item in json.load(f)}
with open('data/datasets/processed/bench_deploy_formal.json') as f:
    formal = {item['id']: item for item in json.load(f)}

# IDs from analysis
ids_bench_eval = [38806, 18738, 18542, 39731, 12775]
ids_formal_eval = [23131, 3475, 43361, 36623, 24828]

print('='*100)
print('5 EXAMPLES: bench_deploy=EVAL ✅, bench_deploy_formal=DEPLOY ❌')
print('='*100)

for i, id in enumerate(ids_bench_eval, 1):
    if id in bench and id in formal:
        b = bench[id]
        f = formal[id]
        print()
        print('-'*100)
        print(f'EXAMPLE {i} | ID: {id} | Label: {b["label"]}')
        print('-'*100)
        print()
        print('ORIGINAL:')
        print(b["first_turn"])
        print()
        print('bench_deploy (pred=EVAL):')
        print(b["rewritten"])
        print()
        print('bench_deploy_formal (pred=DEPLOY):')
        print(f["rewritten"])

print()
print()
print('='*100)
print('5 EXAMPLES: bench_deploy=DEPLOY ❌, bench_deploy_formal=EVAL ✅')
print('='*100)

for i, id in enumerate(ids_formal_eval, 1):
    if id in bench and id in formal:
        b = bench[id]
        f = formal[id]
        print()
        print('-'*100)
        print(f'EXAMPLE {i} | ID: {id} | Label: {b["label"]}')
        print('-'*100)
        print()
        print('ORIGINAL:')
        print(b["first_turn"])
        print()
        print('bench_deploy (pred=DEPLOY):')
        print(b["rewritten"])
        print()
        print('bench_deploy_formal (pred=EVAL):')
        print(f["rewritten"])

