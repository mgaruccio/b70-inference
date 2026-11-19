import http.client, json, math, pathlib, subprocess, sys
here = pathlib.Path('/home/mike/flint-vllm-spike')
pid = subprocess.check_output(['systemctl', 'show', 'flint-inference.service', '-p', 'MainPID', '--value']).decode().strip()
args = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
key = pathlib.Path(args[args.index('--api-key-file') + 1]).read_text().strip()
serve = {json.loads(l)['case']: json.loads(l) for l in open(sys.argv[1])}
out = open(sys.argv[2], 'w')
def post(body):
    c = http.client.HTTPConnection('127.0.0.1', 8080, timeout=120)
    c.request('POST', '/completion', json.dumps(body), {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    r = c.getresponse(); raw = r.read(); c.close()
    assert r.status == 200, raw[:300]
    return json.loads(raw)
for case, s in sorted(serve.items()):
    ids = s['ids']; rows = []
    for p in range(1, len(ids)):
        r = post({'prompt': ids[:p], 'n_predict': 1, 'n_probs': 20, 'temperature': 0, 'cache_prompt': True})
        cp = r['completion_probabilities'][0]
        probs = cp.get('top_logprobs') or cp.get('probs')
        d = {x['id']: x.get('logprob', math.log(max(x.get('prob', 1e-30), 1e-30))) for x in probs}
        rows.append({'p': p, 'lp': d.get(ids[p]), 'floor': min(d.values()), 'top1': max(d, key=d.get)})
    print(json.dumps({'case': case, 'rows': rows}), file=out, flush=True)
    print('case', case, 'done', len(rows), flush=True)
