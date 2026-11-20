"""Development-only chat-template streaming bench at concurrency 1 (greedy and the live
service sampling), with per-request spec-decode acceptance from /metrics deltas.
Usage: chat-bench.py RUN_DIR"""
import http.client, json, pathlib, re, sys, time

root = pathlib.Path(sys.argv[1])
prompts = json.loads(pathlib.Path('benchmark-prompts.json').read_text())
prompts = [p.replace('\n\nSolution:\n', '') for p in prompts] + [
    'A train leaves at 09:40 and arrives at 13:05. It stops 3 times for 7 minutes each. '
    'What is its average speed in km/h over 260 km of track, counting only time in motion?',
    'Prove that the sum of the first n odd numbers is n^2, then use it to compute 1+3+...+199.',
    'Explain the difference between a mutex and a semaphore, with a short example of each in C.',
    'Write a SQL query that returns the top 3 customers by total order value per country.',
]
N = 256
SAMPLING = {'greedy': dict(temperature=0), 'service': dict(temperature=1, top_p=0.95, top_k=20)}


def spec_counters():
    conn = http.client.HTTPConnection('127.0.0.1', 18080, timeout=15)
    try:
        conn.request('GET', '/metrics')
        txt = conn.getresponse().read().decode()
    finally:
        conn.close()
    out = {}
    for name in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens'):
        m = re.search(rf'^vllm:spec_decode_{name}_total{{[^}}]*}} ([0-9.e+]+)$', txt, re.M)
        out[name] = float(m.group(1)) if m else None
    return out


for mode, samp in SAMPLING.items():
    for case, prompt in enumerate(prompts):
        body = dict(model='flint-xpu-test', messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=N, seed=42 + case, ignore_eos=True, stream=True,
                    stream_options={'include_usage': True}, **samp)
        before = spec_counters()
        conn = http.client.HTTPConnection('127.0.0.1', 18080, timeout=120)
        first = last = usage = None
        nchunks = 0
        try:
            start = time.monotonic()
            conn.request('POST', '/v1/chat/completions', json.dumps(body),
                         {'Content-Type': 'application/json'})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f'HTTP {resp.status}: {resp.read(4096)!r}')
            while line := resp.readline():
                line = line.strip()
                if line == b'data: [DONE]':
                    break
                if not line.startswith(b'data: '):
                    continue
                frame = json.loads(line[6:])
                if frame.get('usage'):
                    usage = frame['usage']
                for ch in frame.get('choices', ()):
                    d = ch.get('delta') or {}
                    if d.get('content') or d.get('reasoning_content') or d.get('reasoning'):
                        now = time.monotonic()
                        first = now if first is None else first
                        last = now
                        nchunks += 1
        finally:
            conn.close()
        after = spec_counters()
        acc = None
        if before['num_drafts'] is not None and after['num_drafts'] is not None:
            nd = after['num_drafts'] - before['num_drafts']
            na = after['num_accepted_tokens'] - before['num_accepted_tokens']
            acc = dict(drafts=nd, accepted=na, mean_acceptance_length=1 + na / nd if nd else None)
        assert usage and usage['completion_tokens'] == N, usage
        print(json.dumps(dict(sampling=mode, case=case, warmup=case == 0 and mode == 'greedy',
                              prompt=prompt[:80], usage=usage, ttft_ms=(first - start) * 1e3,
                              decode_tps=(N - 1) / (last - first), spec=acc)), flush=True)
