"""Prefix-cache correctness on the live server: long needle prompts sent cold, then repeated
(cache hit), then with a shared prefix and a different tail. API key on stdin.
Usage: sudo cat /etc/b70-flint/api-key | python3 -I prefix_check.py OUT.json"""
import json, pathlib, random, sys
out = sys.argv[1]
sys.argv = sys.argv[:1] + ['/tmp/flint-validate-import', 'none']  # validate: key from stdin, no checks
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import validate as v  # noqa: E402

SPIKE = pathlib.Path.home() / 'flint-vllm-spike'
src = sorted((SPIKE / '.venv/lib/python3.12/site-packages/vllm/v1/worker').rglob('*.py'))
filler = '\n\n'.join(p.read_text() for p in src)
rng = random.Random(11)
res = []
for case in range(3):
    word = ''.join(rng.choice('abcdefghjkmnpqrstuvwxyz') for _ in range(8))
    off = case * 60000
    text = filler[off: off + 36000]
    pos = int(len(text) * (0.2 + 0.3 * case))
    head = f'Session {case}-{rng.random():.6f}. Here is a source dump:\n\n'
    doc = text[:pos] + f'\n# NOTE: the deployment passphrase is "{word}".\n' + text[pos:]
    q = '\n\nQuestion: what is the deployment passphrase mentioned in a NOTE comment? Answer with just the passphrase.'
    for kind, content in (('cold', head + doc + q), ('warm-repeat', head + doc + q),
                          ('warm-newtail', head + doc + q.replace('Answer with just', 'Reply with only'))):
        s = v.stream_chat([{'role': 'user', 'content': content}], 2048)
        res.append(dict(case=case, kind=kind, prompt_tokens=s['usage']['prompt_tokens'],
                        completion_tokens=s['usage']['completion_tokens'], finish=s['finish'],
                        found=word in s['content'], ttft_ms=round(s['ttft_ms']),
                        acceptance_length=s['acceptance_length'] and round(s['acceptance_length'], 2),
                        cache_hit_tokens=s['prefix_cache_hit_tokens'], answer=s['content'][-60:]))
        print(res[-1], flush=True)
pathlib.Path(out).write_text(json.dumps(res, indent=1))
