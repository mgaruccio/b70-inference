"""Score each divergence point by pure prefill: prompt + common prefix -> next-token top-5."""
import http.client, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
prompts = json.loads(pathlib.Path('benchmark-prompts.json').read_text())
long_prompt = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
cases = prompts + [long_prompt]
ref = json.loads(pathlib.Path('divergence-ref.json').read_text())
def post(body):
    conn = http.client.HTTPConnection('127.0.0.1', 18080, timeout=180)
    try:
        conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
        r = conn.getresponse(); raw = r.read()
        if r.status != 200: raise RuntimeError(f'HTTP {r.status}: {raw[:2048]!r}')
        return json.loads(raw)
    finally: conn.close()
base = {'model': 'flint-xpu-test', 'temperature': 0, 'seed': 42, 'ignore_eos': True, 'return_token_ids': True}
with (root / 'probe2.jsonl').open('w') as out:
    for c in ref:
        pid = post({**base, 'prompt': cases[c['case']], 'max_tokens': 1})['choices'][0]['prompt_token_ids']
        ids = pid + c['prefix']
        r = post({**base, 'prompt': ids, 'max_tokens': 1, 'logprobs': 5})
        ch, = r['choices']
        # also score the 3 steps before divergence by prompt_logprobs to see where things drift
        r2 = post({**base, 'prompt': ids + [c['nospec_tok']], 'max_tokens': 1, 'prompt_logprobs': 5})
        pl = r2['choices'][0].get('prompt_logprobs') or r2.get('prompt_logprobs')
        print(json.dumps({'case': c['case'], 'k': c['k'], 'prompt_len': len(pid), 'top_next': ch['logprobs']['top_logprobs'][0],
                          'tok': ch['token_ids'][0], 'nospec_tok': c['nospec_tok'], 'spec_tok': c['spec_tok'],
                          'pl_at_k': pl[len(ids)]}), file=out, flush=True)
