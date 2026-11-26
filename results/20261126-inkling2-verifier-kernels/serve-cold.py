"""Cold prefill TTFT on loopback. Same three distinct ID fixtures per arm.

A 131K warmup compiles the long-context path; no measured prefix may be cached.
Timestamp the first token-bearing SSE frame, not the usage frame or HTTP headers.
"""
import json
import statistics
import sys
import time
from pathlib import Path

from checks import MODEL, delta, fixture, metrics, post

root = Path(sys.argv[1])
depth = 131072
rows = []
for trial in range(4):
    body = dict(model=MODEL, prompt=fixture(depth, 301902 + trial), max_tokens=1,
                temperature=0, ignore_eos=True, seed=42, return_token_ids=True,
                stream=True, stream_options={'include_usage': True})
    before = metrics()
    conn, resp, start = post('/v1/completions', body, stream=True)
    ttft, usage, ids = None, None, []
    try:
        while line := resp.readline():
            line = line.strip()
            if line == b'data: [DONE]':
                break
            if not line.startswith(b'data: '):
                continue
            frame = json.loads(line[6:])
            usage = frame.get('usage') or usage
            for ch in frame.get('choices', []):
                new = ch.get('token_ids') or []
                if new and ttft is None:
                    ttft = time.monotonic() - start
                ids.extend(new)
    finally:
        conn.close()
    assert ttft and usage and usage['prompt_tokens'] == depth and usage['completion_tokens'] == len(ids) == 1
    assert usage['prompt_tokens_details']['cached_tokens'] == 0
    rows.append(dict(trial=trial, warmup=trial == 0, ttft_s=ttft, token_ids=ids,
                     spec=delta(before, metrics()), usage=usage))
    (root / 'cold-131072.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(trial, 'TTFT seconds', ttft, flush=True)
print('median scored TTFT seconds', statistics.median(r['ttft_s'] for r in rows[1:]), flush=True)
