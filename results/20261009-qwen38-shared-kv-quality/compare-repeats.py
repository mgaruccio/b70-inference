#!/usr/bin/env python3
"""Summarize retained diagnostic repetitions; never execute generated code."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

CELLS = {'a1': 'native', 'b1': 'candidate', 'b2': 'candidate', 'a2': 'native'}
IDS = ('HumanEval/1', 'HumanEval/11', 'HumanEval/19', 'HumanEval/126', 'HumanEval/130')


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def compare(root):
    samples, tokens, request_hashes = [], {}, {task: set() for task in IDS}
    for cell, arm in CELLS.items():
        state = json.loads((root / f'repeat-{cell}/cell-result.json').read_text())
        assert state['driver_exit'] == 0 and state['arm'] == arm and not state['new_guarded_error_lines']
        for repeat in range(1, 5):
            point = root / f'repeat-{cell}/native/repeat-{repeat:02d}'
            rows = [json.loads(line) for line in (point / 'generation.jsonl').read_text().splitlines()]
            scores = json.loads((point / 'scores/scores.humanevalplus.eval_results.json').read_text())['eval']
            assert len(rows) == 5 and {r['id'] for r in rows} == set(IDS) == set(scores)
            assert json.loads((point / 'scores/exit.json').read_text())['returncode'] == 0
            for row in rows:
                task = row['id']
                assert len(scores[task]) == 1 and row['http_status'] == 200
                output = row['output_token_ids']
                assert isinstance(output, list) and output and all(type(v) is int for v in output)
                score = scores[task][0]
                assert score['base_status'] in ('pass', 'fail', 'timeout') and score['plus_status'] in ('pass', 'fail', 'timeout')
                request_hashes[task].add(row['request_sha256'])
                tokens[(cell, repeat, task)] = output
                samples.append({'cell': cell, 'arm': arm, 'repeat': repeat, 'task': task,
                                'base_status': score['base_status'], 'plus_status': score['plus_status'],
                                'passed': score['base_status'] == score['plus_status'] == 'pass',
                                'fenced': row['content'].lstrip().startswith('```'),
                                'token_sha256': digest(output), 'output_tokens': len(output),
                                'finish_reason': row['finish_reason']})
    assert len(samples) == 80 and all(len(hashes) == 1 for hashes in request_hashes.values())
    per_task = {}
    for task in IDS:
        per_task[task] = {}
        for arm in ('native', 'candidate'):
            selected = [s for s in samples if s['task'] == task and s['arm'] == arm]
            per_task[task][arm] = {
                'observations': len(selected), 'passes': sum(s['passed'] for s in selected),
                'fenced': sum(s['fenced'] for s in selected),
                'distinct_token_sequences': len({s['token_sha256'] for s in selected}),
                'cells': {cell: {'passes': sum(s['passed'] for s in selected if s['cell'] == cell),
                                 'distinct_token_sequences': len({s['token_sha256'] for s in selected if s['cell'] == cell})}
                          for cell, name in CELLS.items() if name == arm}}
    pairings = {}
    for left, right in (('a1', 'a2'), ('b1', 'b2'), ('a1', 'b1'), ('a1', 'b2'), ('a2', 'b1'), ('a2', 'b2')):
        rows = [{'task': task, 'repeat': repeat,
                 'first_difference': first_difference(tokens[left, repeat, task], tokens[right, repeat, task])}
                for repeat in range(1, 5) for task in IDS]
        pairings[f'{left}/{right}'] = {'matching': sum(row['first_difference'] is None for row in rows),
                                      'comparisons': len(rows), 'details': rows}
    long_outputs = {cell: json.loads((root / f'repeat-{cell}/native/long-divergence/summary.json').read_text())['rows']
                    for cell in CELLS}
    assert all(len(rows) == 2 and all(len(r['token_ids']) == 128 for r in rows) for rows in long_outputs.values())
    return {'scope': 'Five-task diagnosis, eight repetitions per arm across two server starts; not full-suite accuracy or independent trials',
            'samples': samples, 'per_task': per_task, 'paired_token_comparisons': pairings,
            'finish_reasons': dict(Counter(s['finish_reason'] for s in samples)),
            'request_sha256': {task: next(iter(hashes)) for task, hashes in request_hashes.items()},
            'long_diagnostic_token_sha256': {cell: [digest(r['token_ids']) for r in rows] for cell, rows in long_outputs.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    print(json.dumps(compare(parser.parse_args().root), indent=2, sort_keys=True))
