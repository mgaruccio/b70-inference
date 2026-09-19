"""Aggregate the four fixed full-HumanEval+ cells without pooling into pass@2."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
labels = ('A1', 'B1', 'B2', 'A2')
expected = {f'HumanEval/{i}' for i in range(164)}
cells, answers, prompts = {}, {}, {}
for label in labels:
    evaluation = json.loads((root / f'quality-{label}-evaluation.json').read_text())
    assert evaluation['returncode'] == 0
    rows = evaluation['results']['eval']
    assert set(rows) == expected and all(len(v) == 1 for v in rows.values())
    base = {key for key, value in rows.items() if value[0]['base_status'] == 'pass'}
    plus = {key for key, value in rows.items() if value[0]['base_status'] == value[0]['plus_status'] == 'pass'}
    source = root / f'quality-{label}-output'
    generation = json.loads((source / 'summary.json').read_text())
    assert generation['requests'] == 164 and generation['skipped_long'] == 0
    answers[label], prompts[label] = {}, {}
    for directory in sorted(source.glob('request-*')):
        measurement = json.loads((directory / 'measurement.json').read_text())
        response = json.loads((directory / 'response.json').read_text())
        request = json.loads((directory / 'request.json').read_text())
        key = measurement['prompt_id']
        assert key not in answers[label]
        answers[label][key] = hashlib.sha256(response['choices'][0]['message']['content'].encode()).hexdigest()
        prompts[label][key] = {'tokens': response['prompt_token_ids'],
                              'messages': request['messages'],
                              'sampling': {k: request[k] for k in ('temperature', 'seed', 'top_p', 'top_k', 'max_tokens', 'chat_template_kwargs')}}
    assert set(answers[label]) == expected
    cells[label] = {'head': 'stock' if label.startswith('A') else 'candidate',
                    'tasks': 164, 'base_pass': len(base), 'base_plus_pass': len(plus),
                    'base_pass_at_1': len(base) / 164, 'base_plus_pass_at_1': len(plus) / 164,
                    'base_failures': sorted(expected - base), 'base_plus_failures': sorted(expected - plus),
                    'generation': generation}
assert all(prompts[label] == prompts['A1'] for label in labels)
pass_sets = {label: expected - set(cells[label]['base_plus_failures']) for label in labels}
a, b = pass_sets['A1'] & pass_sets['A2'], pass_sets['B1'] & pass_sets['B2']
result = {'tier': 'development', 'benchmark': 'HumanEval+ v0.1.10 full 164',
          'identical_prompts_and_sampling': True, 'cells': cells,
          'stock_both_pass_candidate_any_fail': sorted(a - b),
          'stock_both_pass_candidate_both_fail': sorted(a - (pass_sets['B1'] | pass_sets['B2'])),
          'candidate_both_pass_stock_any_fail': sorted(b - a),
          'candidate_both_pass_stock_both_fail': sorted(b - (pass_sets['A1'] | pass_sets['A2'])),
          'stock_pass_status_disagreements': sorted(pass_sets['A1'] ^ pass_sets['A2']),
          'candidate_pass_status_disagreements': sorted(pass_sets['B1'] ^ pass_sets['B2']),
          'exact_answer_match_counts': {f'{x}-{y}': sum(answers[x][k] == answers[y][k] for k in expected)
                                        for x, y in [('A1', 'A2'), ('B1', 'B2'), ('A1', 'B1'), ('A2', 'B2')]},
          'limitations': ['Public/pretraining overlap possible; not fresh-private generalization.',
                          'Each cell is pass@1; repeats are not pooled into pass@2.',
                          'No tool execution, broad reasoning or long-context semantic correctness coverage.']}
with (root / 'quality-summary.json').open('x') as output:
    json.dump(result, output, indent=2)
print(json.dumps(result, indent=2))
