"""Correctness probe: per-step top-k logprobs for decode and prompt (prefill) scoring."""
import http.client, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
ref = json.loads(pathlib.Path('probe-ref.json').read_text())

def post(body):
    conn = http.client.HTTPConnection('127.0.0.1', 18080, timeout=120)
    try:
        conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
        r = conn.getresponse(); raw = r.read()
        if r.status != 200:
            raise RuntimeError(f'HTTP {r.status}: {raw[:2048]!r}')
        return json.loads(raw)
    finally:
        conn.close()

base = {'model': 'flint-xpu-test', 'temperature': 0, 'seed': 42, 'ignore_eos': True,
        'return_token_ids': True}
with (root / 'probe.jsonl').open('w') as out:
    for case in ref:
        p = case['prompt_ids']
        for rep in range(2):
            r = post({**base, 'prompt': p, 'max_tokens': 48, 'logprobs': 5})
            ch, = r['choices']
            print(json.dumps({'case': case['case'], 'kind': 'decode', 'rep': rep,
                              'token_ids': ch['token_ids'], 'logprobs': ch['logprobs']}), file=out, flush=True)
        for name in ('nospec', 'spec'):
            r = post({**base, 'prompt': p + case[name], 'max_tokens': 1, 'prompt_logprobs': 5})
            ch, = r['choices']
            print(json.dumps({'case': case['case'], 'kind': 'prefill-' + name,
                              'prompt_logprobs': ch.get('prompt_logprobs') or r.get('prompt_logprobs')}), file=out, flush=True)
