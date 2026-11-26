"""Bounded XPU traces on a loopback engine. Warm the identical prompt(s) first.
Run alongside checks.py and the hill-climb harness (see README).
"""
import http.client
import json
import sys
from pathlib import Path

from checks import HOST, PORT, MODEL, fixture, post


def control(path, body=None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=120)
    try:
        conn.request('POST', path, json.dumps(body) if body is not None else None,
                     {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        text = resp.read()
        if resp.status != 200:
            raise RuntimeError((resp.status, text[:1024]))
    finally:
        conn.close()


def main(root):
    cases = ([(1, 131072), (1, 375000)] if root.name.endswith('-profile-deep') else
             [(1, 128), (1, 30080), (8, 128), (32, 128)])
    for n, depth in cases:
        name = f'c{n}-p{depth}'
        prompts = [fixture(depth, depth + i) for i in range(n)]
        def run(tokens):
            data, _ = post('/v1/completions', dict(model=MODEL, prompt=prompts,
                max_tokens=tokens, ignore_eos=True, temperature=0, seed=42))
            if data['usage']['completion_tokens'] != n * tokens:
                raise RuntimeError(data['usage'])
            return data['usage']
        run(16)
        control('/start_profile', dict(profile_prefix=name, delay_iterations=8, max_iterations=4))
        try:
            usage = run(128)
        finally:
            control('/stop_profile')
        print(name, usage, flush=True)
        (root / (name + '.json')).write_text(json.dumps(usage) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
