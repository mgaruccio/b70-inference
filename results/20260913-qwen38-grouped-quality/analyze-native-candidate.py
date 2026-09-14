#!/usr/bin/env python3
"""Supplement the three-arm report with the direct kernel comparison."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ('target', 'native', 'candidate')

def load(path):
    return json.loads(path.read_text())

scores = {a: load(ROOT / f'full-{a}-01/scores/scores.json')['task_results'] for a in ARMS}
rows = {a: {r['id']: r for r in map(json.loads, (ROOT / f'full-{a}-01/generation.jsonl').read_text().splitlines())} for a in ARMS}
for a in ARMS:
    summary = load(ROOT / f'full-{a}-01/summary.json')
    assert summary['status'] == 'passed' and summary['host_unchanged']
    assert len(rows[a]) == 2466 and all(r['ok'] for r in rows[a].values())
report = {'interpretation': 'Observed quality is not neutral; no statistical equivalence or production promotion.', 'tasks': {}}
for task, key in (('ifeval', 'prompt_level_strict_acc'), ('gsm8k', 'strict_exact'), ('humanevalplus', 'plus_pass'), ('mbppplus', 'plus_pass')):
    items = {a: {r['id']: r for r in scores[a][task]['items']} for a in ARMS}
    assert items['native'].keys() == items['candidate'].keys() == items['target'].keys()
    deltas = [int(items['candidate'][i][key]) - int(items['native'][i][key]) for i in items['native']]
    mean = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    report['tasks'][task] = {'metric': key, 'n': len(deltas),
        'scores_percent': {a: 100 * statistics.mean(int(r[key]) for r in items[a].values()) for a in ARMS},
        'candidate_minus_native_pp': 100 * mean,
        'descriptive_normal_95ci_pp': [100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se)],
        'regressed_ids': [i for i in items['native'] if items['native'][i][key] and not items['candidate'][i][key]],
        'improved_ids': [i for i in items['native'] if not items['native'][i][key] and items['candidate'][i][key]]}
ids = [i for i, r in rows['target'].items() if r.get('divergence_sample') and r.get('sample_role') == 'primary']
assert len(ids) == 500
first = []
for i in ids:
    a, b = rows['native'][i], rows['candidate'][i]
    assert a['prompt_token_ids'] == b['prompt_token_ids']
    x, y = a['output_token_ids'], b['output_token_ids']
    if x != y:
        first.append(next((j + 1 for j, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)) + 1))
report['native_candidate_500_token_divergence'] = {'n': 500, 'exact': 500 - len(first), 'diverged': len(first),
    'exact_percent': (500 - len(first)) / 5, 'median_first_divergence_one_based': statistics.median(first) if first else None}
long = {a: load(ROOT / f'full-{a}-01/long-divergence/summary.json')['rows'] for a in ARMS}
assert all(len(v) == 12 for v in long.values())
report['long_64k'] = {'cross_arm': {}, 'repeatability': {}}
for a, b in (('target', 'native'), ('target', 'candidate'), ('native', 'candidate')):
    count = 0
    for x, y in zip(long[a], long[b]):
        assert (x['trial'], x['repeat']) == (y['trial'], y['repeat'])
        stem = f"prompt-{x['trial']:02}-repeat-{x['repeat']}.request.json"
        assert load(ROOT / f'full-{a}-01/long-divergence' / stem) == load(ROOT / f'full-{b}-01/long-divergence' / stem)
        count += x['token_ids'] == y['token_ids']
    report['long_64k']['cross_arm'][f'{a}_vs_{b}'] = {'exact_token_matches': count, 'n': 12}
for a, rr in long.items():
    report['long_64k']['repeatability'][a] = {'exact_token_matches': sum(rr[i]['token_ids'] == rr[i+1]['token_ids'] for i in range(0, 12, 2)), 'n': 6}
print(json.dumps(report, indent=2))
