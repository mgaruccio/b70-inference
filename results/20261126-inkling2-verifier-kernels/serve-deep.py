"""Compare warmed loopback greedy completions at exact deep context lengths."""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from checks import MODEL, delta, fixture, metrics, post

root = Path(sys.argv[1])
for depth in (131072, 375000):
    body = dict(model=MODEL, prompt=fixture(depth, depth), temperature=0, seed=42,
                ignore_eos=True, return_token_ids=True)
    warm, _ = post('/v1/completions', dict(body, max_tokens=16))
    assert warm['usage']['prompt_tokens'] == depth
    rows = []
    for i in range(5):
        before = metrics()
        conn, resp, start = post('/v1/completions', dict(body, max_tokens=128, stream=True,
                                 stream_options={'include_usage': True}), stream=True)
        ids, times, usage, first_count = [], [], None, None
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
                    tok = ch.get('token_ids') or []
                    if tok:
                        times.append(time.monotonic() - start)
                        first_count = len(tok) if first_count is None else first_count
                        ids.extend(tok)
        finally:
            conn.close()
        assert usage and usage['prompt_tokens'] == depth and usage['completion_tokens'] == len(ids) == 128
        assert len(times) >= 2 and usage['prompt_tokens_details']['cached_tokens'] >= depth - 256
        rows.append(dict(trial=i+1, usage=usage, digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                         first_chunk_tokens=first_count, ttft_s=times[0],
                         decode_tps=(128-first_count)/(times[-1]-times[0]), spec=delta(before, metrics())))
    (root/f'deep-{depth}.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(depth, 'median decode tok/s', statistics.median(r['decode_tps'] for r in rows),
          'distinct greedy outputs', len({r['digest'] for r in rows}), flush=True)
