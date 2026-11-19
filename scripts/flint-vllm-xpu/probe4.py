import http.client, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
prompts = json.loads(pathlib.Path('benchmark-prompts.json').read_text())
long_prompt = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
def post(body):
    conn = http.client.HTTPConnection('127.0.0.1', 18080, timeout=300)
    try:
        conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
        r = conn.getresponse(); raw = r.read()
        if r.status != 200: raise RuntimeError(f'HTTP {r.status}: {raw[:2048]!r}')
        return json.loads(raw)
    finally: conn.close()
with (root / 'probe4.jsonl').open('w') as out:
    for i, p in enumerate(prompts + [long_prompt]):
        r = post({'model': 'flint-xpu-test', 'prompt': p, 'max_tokens': 1, 'temperature': 0, 'return_token_ids': True, 'prompt_logprobs': 1})
        ch = r['choices'][0]; ids = ch['prompt_token_ids']; pl = ch.get('prompt_logprobs') or r.get('prompt_logprobs')
        print(json.dumps({'case': i, 'ids': ids, 'lp': [None] + [pl[j][str(ids[j])]['logprob'] for j in range(1, len(ids))]}), file=out, flush=True)
