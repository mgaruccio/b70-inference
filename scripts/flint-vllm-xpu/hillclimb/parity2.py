"""Greedy parity at several max_tokens (end-of-request spec truncation check)."""
import http.client, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
prompts = json.loads(pathlib.Path('benchmark-prompts.json').read_text())
long_prompt = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
cases = {0: prompts[0], 7: long_prompt, 3: prompts[3]}
def post(body):
    conn = http.client.HTTPConnection('127.0.0.1', 18080, timeout=180)
    try:
        conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
        r = conn.getresponse(); raw = r.read()
        if r.status != 200:
            raise RuntimeError(f'HTTP {r.status}: {raw[:2048]!r}')
        return json.loads(raw)
    finally:
        conn.close()
with (root / 'parity2.jsonl').open('w') as out:
    for mt in (128, 131, 140):
        for i, p in cases.items():
            r = post({'model': 'flint-xpu-test', 'prompt': p, 'temperature': 0, 'seed': 42, 'max_tokens': mt,
                      'ignore_eos': True, 'return_token_ids': True, 'logprobs': 5})
            ch, = r['choices']
            print(json.dumps({'case': f'{i}-{mt}', 'prompt_len': len(ch['prompt_token_ids']), 'token_ids': ch['token_ids'],
                              'top': ch['logprobs']['top_logprobs'], 'lp': ch['logprobs']['token_logprobs'],
                              'tokens': ch['logprobs']['tokens']}), file=out, flush=True)
