"""Prefill-score fixed token sequences: prompt + generated tokens -> prompt_logprobs per position."""
import http.client, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
prompts = json.loads(pathlib.Path('benchmark-prompts.json').read_text())
long_prompt = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
cases = prompts + [long_prompt]
ref = json.loads(pathlib.Path('divergence-ref-fused.json').read_text())
def post(body):
    conn = http.client.HTTPConnection('127.0.0.1', 18080, timeout=300)
    try:
        conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
        r = conn.getresponse(); raw = r.read()
        if r.status != 200: raise RuntimeError(f'HTTP {r.status}: {raw[:2048]!r}')
        return json.loads(raw)
    finally: conn.close()
base = {'model': 'flint-xpu-test', 'temperature': 0, 'seed': 42, 'return_token_ids': True}
with (root / 'probe3.jsonl').open('w') as out:
    for c in ref:
        pid = post({**base, 'prompt': cases[c['case']], 'max_tokens': 1})['choices'][0]['prompt_token_ids']
        seq = pid + c['prefix'] + [c['nospec_tok']]
        r = post({**base, 'prompt': seq, 'max_tokens': 1, 'prompt_logprobs': 3})
        pl = r['choices'][0].get('prompt_logprobs') or r.get('prompt_logprobs')
        lp = [pl[i][str(seq[i])]['logprob'] for i in range(len(pid), len(seq))]
        print(json.dumps({'case': c['case'], 'P': len(pid), 'lp': lp}), file=out, flush=True)
