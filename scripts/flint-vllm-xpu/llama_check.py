import http.client, json, math, pathlib, sys
from transformers import AutoTokenizer
here = pathlib.Path('/home/mike/flint-vllm-spike')
pid = int(open('/run/user/%d/flint-pid' % 0).read()) if False else None
import subprocess
pid = subprocess.check_output(['systemctl', 'show', 'flint-inference.service', '-p', 'MainPID', '--value']).decode().strip()
args = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
key = pathlib.Path(args[args.index('--api-key-file') + 1]).read_text().strip()
tk = AutoTokenizer.from_pretrained(str(here / 'gptq'))
prompts = json.loads((here / 'benchmark-prompts.json').read_text())
lp = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
ids = tk(lp)['input_ids']
def post(route, body):
    c = http.client.HTTPConnection('127.0.0.1', 8080, timeout=120)
    c.request('POST', route, json.dumps(body), {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    r = c.getresponse(); raw = r.read(); c.close()
    assert r.status == 200, raw[:500]
    return json.loads(raw)
lt = post('/tokenize', {'content': lp, 'add_special': True})['tokens']
print('tokenizer match:', lt == ids, len(lt), len(ids), flush=True)
ref = {r['case']: r for r in json.load(open(here / 'cpuref-prompts.json'))}[7]
serve = [json.loads(l) for l in open(sys.argv[1])][7]
for p in [int(x) for x in sys.argv[2].split(',')]:
    r = post('/completion', {'prompt': ids[:p], 'n_predict': 1, 'n_probs': 20, 'temperature': 0, 'cache_prompt': False, 'post_sampling_probs': False})
    probs = r['completion_probabilities'][0]['top_logprobs'] if 'top_logprobs' in r['completion_probabilities'][0] else r['completion_probabilities'][0]['probs']
    want = ids[p]
    got = [x for x in probs if x.get('id') == want]
    lpv = (got[0].get('logprob') if got and 'logprob' in got[0] else (math.log(got[0]['prob']) if got else None))
    top = [(x.get('id'), round(x.get('logprob', math.log(max(x.get('prob', 1e-30), 1e-30))), 3)) for x in probs[:3]]
    print(f"pos {p} want {want} {tk.decode([want])!r}: llama.cpp lp {lpv} top3 {top} | vllm {serve['lp'][p]:.3f} | fp32 {ref['rows'][p-1]['lp_actual']:.3f}", flush=True)
