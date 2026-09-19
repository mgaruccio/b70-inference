"""Convert native API responses using the pinned official EvalPlus sanitizer; never execute."""
import json
from pathlib import Path
import sys
from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize

root = Path(sys.argv[1])
cell = sys.argv[2]
tasks = get_human_eval_plus()
records = [json.loads(line) for line in (root / 'data/public-tasks.jsonl').read_text().splitlines()]
source = root / f'quality-{cell}-output'
summary = json.loads((source / 'summary.json').read_text())
assert summary['status'] == 'completed', summary
requests = sorted(source.glob('request-*'))
assert len(records) == len(requests) == len(tasks) == 164
samples = []
for record, directory in zip(records, requests):
    measurement = json.loads((directory / 'measurement.json').read_text())
    request = json.loads((directory / 'request.json').read_text())
    response = json.loads((directory / 'response.json').read_text())
    task_id = record['prompt_id']
    assert measurement['prompt_id'] == task_id
    assert request['messages'] == record['messages']
    assert request['max_tokens'] == 2048
    assert request['chat_template_kwargs']['enable_thinking'] is False
    assert len(response['choices']) == 1
    content = response['choices'][0]['message']['content']
    assert isinstance(content, str)
    samples.append({'task_id': task_id, 'solution': sanitize(content, entrypoint=tasks[task_id]['entry_point'])})
with (root / f'quality-{cell}-samples.jsonl').open('x') as output:
    for sample in samples:
        output.write(json.dumps(sample) + '\n')
print('SANITIZED', cell, len(samples))
