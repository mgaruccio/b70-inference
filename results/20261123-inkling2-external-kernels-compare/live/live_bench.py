"""Development-tier C1 greedy chat decode bench + parity capture (vLLM public API).

Usage: decode_bench.py ART [--port 18080] [--tokens 256] [--depth 0]
Per case: chat completion, temperature 0, ignore_eos, streaming with logprobs.
Decode tok/s = (completion_tokens - 1) / (t_last_token - t_first_token).
With --depth N, each prompt is preceded by ~N tokens of filler (decode at context depth).
Writes ART/decode-<depth>-<case>.json and ART/decode-<depth>-summary.json.
"""
import argparse
import os
import http.client
import json
import pathlib
import re
import statistics
import time

PROMPTS = [
    'Python coding exercise: merge overlapping intervals. Implement merge(intervals) with a doc comment and three unit tests.',
    'Go coding exercise: implement a bounded worker pool that honors context cancellation. Implement Process(ctx, jobs, workers).',
    'Rust coding exercise: parse timestamps from rows, keep good rows and report invalid lines. Implement parse_rows(input).',
    'Explain the difference between a mutex and a semaphore, with a short example of each in C.',
    'A train leaves at 09:40 and arrives at 13:05. It stops 3 times for 7 minutes each. What is its average speed in km/h '
    'over 260 km of track, counting only time in motion?',
    'Write a SQL query that returns the top 3 customers by total order value per country, and explain it.',
]


def spec_counters(port):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
    try:
        conn.request('GET', '/metrics', headers={'Authorization': 'Bearer ' + os.environ['LIVE_KEY']})
        txt = conn.getresponse().read().decode()
    finally:
        conn.close()
    out = {}
    for name in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens'):
        m = re.search(rf'^vllm:spec_decode_{name}_total{{[^}}]*}} ([0-9.e+]+)$', txt, re.M)
        out[name] = float(m.group(1)) if m else None
    return out


def filler(depth):
    if depth <= 0:
        return ''
    # ~20 tokens per line with this tokenizer family; trimmed by line count, not exact.
    lines = [f'Log {i:05d}: sensor array {i % 17} reported nominal voltage and the operator signed off.'
             for i in range(depth // 20)]
    return 'Background log (ignore unless asked):\n' + '\n'.join(lines) + '\n\nTask: '


def run_case(port, case, prompt, tokens, depth, art):
    body = dict(model='inkling-2-mini', messages=[{'role': 'user', 'content': filler(depth) + prompt}],
                max_tokens=tokens, temperature=0, ignore_eos=True, stream=True, logprobs=True, return_tokens_as_token_ids=True,
                stream_options={'include_usage': True})
    before = spec_counters(port)
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=900)
    times, toks, lps, usage = [], [], [], None
    try:
        start = time.monotonic()
        conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['LIVE_KEY']})
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
            now = time.monotonic() - start
            for ch in frame.get('choices', ()):
                content = (ch.get('logprobs') or {}).get('content') or []
                for item in content:
                    times.append(now)
                    toks.append(item['token'])
                    lps.append(item['logprob'])
    finally:
        conn.close()
    after = spec_counters(port)
    n = usage['completion_tokens']
    # Several tokens can share one SSE frame (spec decode); use frame times.
    decode_s = times[-1] - times[0]
    row = dict(case=case, depth=depth, prompt_tokens=usage['prompt_tokens'], completion_tokens=n,
               logprob_tokens=len(toks), ttft_s=times[0], decode_s=decode_s,
               decode_tok_s=(n - 1) / decode_s if decode_s > 0 else None,
               spec={k: (after[k] - before[k]) if after[k] is not None else None for k in after},
               tokens=toks, logprobs=lps)
    (art / f'decode-{depth}-{case}.json').write_text(json.dumps(row) + '\n')
    print(json.dumps({k: v for k, v in row.items() if k not in ('tokens', 'logprobs')}), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('art', type=pathlib.Path)
    ap.add_argument('--port', type=int, default=18080)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--depth', type=int, default=0)
    ap.add_argument('--cases', type=int, default=len(PROMPTS))
    a = ap.parse_args()
    # Warm-up (not scored): same shapes as the cases.
    run_case(a.port, 'warmup', 'Say hello in five words.', 32, a.depth, a.art)
    rows = [run_case(a.port, i, p, a.tokens, a.depth, a.art) for i, p in enumerate(PROMPTS[:a.cases])]
    spec = {k: sum(r['spec'][k] or 0 for r in rows) for k in rows[0]['spec']}
    summary = dict(tier='development', depth=a.depth, tokens=a.tokens,
                   median_decode_tok_s=statistics.median(r['decode_tok_s'] for r in rows),
                   median_ttft_s=statistics.median(r['ttft_s'] for r in rows),
                   per_case=[round(r['decode_tok_s'], 2) for r in rows], spec=spec)
    if spec.get('num_drafts'):
        summary['mean_acceptance_length'] = 1 + spec['num_accepted_tokens'] / spec['num_drafts']
    (a.art / f'decode-{a.depth}-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
