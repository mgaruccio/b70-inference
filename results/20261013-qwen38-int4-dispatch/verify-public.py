#!/usr/bin/env python3
"""Independent restored production public API check; tiny real text canary."""
import json
import math
from pathlib import Path
import time
import urllib.request

BASE = 'http://inference-host:8000'
report = {'base': BASE}
for endpoint in ('/health', '/v1/models'):
    with urllib.request.urlopen(BASE + endpoint, timeout=10) as response:
        raw = response.read()
        report[endpoint] = {'status': response.status, 'body': json.loads(raw) if raw else None}
models = report['/v1/models']['body']['data']
assert any(m['id'] == 'qwen38' and m['max_model_len'] == 212992 for m in models)
payload = {'model': 'qwen38', 'prompt': 'The sum of two and three is',
           'max_tokens': 1, 'temperature': 0, 'logprobs': 1}
request = urllib.request.Request(BASE + '/v1/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
start = time.monotonic()
with urllib.request.urlopen(request, timeout=30) as response:
    body = json.load(response)
    report['completion'] = {'status': response.status, 'elapsed_s': time.monotonic() - start,
                            'request': payload, 'response': body}
assert body['usage']['prompt_tokens'] == 7 and body['usage']['completion_tokens'] == 1
assert body['choices'][0]['text'].strip() in ('five', '5'), body['choices']
assert all(math.isfinite(v) for c in body['choices'] for v in c['logprobs']['token_logprobs'])
report['pass'] = True
Path(__file__).with_name('public-restored.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'pass': True, 'elapsed_s': report['completion']['elapsed_s'],
                  'usage': body['usage'], 'text': body['choices'][0]['text']}))
