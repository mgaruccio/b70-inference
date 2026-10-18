#!/usr/bin/env python3
"""Development-only first-row comparison; execute on inference-host after restoration."""
import array
import json
import math
import runpy
import socket
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase')
OUT = ROOT / 'trace-short55-r2'
PARSER = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-route-trace/analyze-trace.py'
BATCH_FIELDS = ('ctx', 'off', 'n', 'tokens', 'pos', 'outputs', 'seq')


def load(path):
    return json.loads(path.read_text())


def first_frame(parser, directory, columns):
    stdout = (directory / 'short-diagnostic-log-fresh.stdout.log').read_text()
    assert 'BOLT_VERIFY' not in stdout and 'BOLT_ROUTE' not in stdout, 'split trace streams'
    log = (directory / 'short-diagnostic-log-fresh.stderr.log').read_text()
    events = parser['records'](log)
    frames = parser['route_frames'](log)
    prompt = load(directory / 'public/short/diagnostic.prompt-comparison.json')['actual_ids']
    emitted = load(directory / 'short/diagnostic.diagnostic-view.json')['ids']
    assert len(prompt) == 55 and len(emitted) == 128
    frame = next(f for f in frames if f['batch']['pos'][0] >= len(prompt))
    batch = frame['batch']
    assert batch['n'] == columns and batch['off'] == 0
    assert batch['pos'] == list(range(55, 55 + columns))
    assert batch['outputs'] == [1] * columns and batch['tokens'][0] == emitted[0]
    observed_prompt = {}
    for prior in frames:
        if prior is frame:
            break
        assert prior['batch']['ctx'] == batch['ctx']
        for position, token in zip(prior['batch']['pos'], prior['batch']['tokens']):
            assert 0 <= position < 55
            assert position not in observed_prompt or observed_prompt[position] == token
            observed_prompt[position] = token
    assert sorted(observed_prompt) == list(range(55))
    assert [observed_prompt[i] for i in range(55)] == prompt

    start = next(i for i, e in enumerate(events) if e['event'] == 'batch'
                 and all(e[k] == batch[k] for k in BATCH_FIELDS))
    stop = next((i for i in range(start + 1, len(events))
                 if events[i]['event'] == 'batch'), len(events))
    block = events[start + 1:stop]
    assert block[0]['event'] == 'decoded' and block[0]['ret'] == 0
    assert block[0]['ctx'] == batch['ctx']
    sample = next(e for e in block if e['event'] == 'sample' and e['ctx'] == batch['ctx'])
    row = sample['idx'] if sample['idx'] >= 0 else columns + sample['idx']
    assert row == 0 and sample['selected'] == emitted[1]
    prior_samples = [e for e in events[:start]
                     if e['event'] == 'sample' and e['ctx'] == batch['ctx']]
    assert len(prior_samples) == 1 and prior_samples[0]['selected'] == emitted[0]
    prefill = prior_samples[0]

    contexts = {r['backend_ctx'] for r in frame['routes']
                if r['weight'] in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight')}
    assert len(contexts) == 1, 'ambiguous target backend'
    context = next(iter(contexts))
    routes = [r for r in frame['routes'] if r['backend_ctx'] == context]
    assert any(r['weight'] == 'output.weight' for r in routes)
    data = {key: parser['raw_bytes'](directory / 'capture', value)
            for key, value in (('prefill', prefill), ('verification', sample))}
    report = dict(batch=batch, prompt_ids=prompt, first_emitted_ids=emitted[:2],
                  target_backend_ctx=context, routes=routes,
                  other_backend_routes=[r for r in frame['routes'] if r['backend_ctx'] != context],
                  samples={key: dict(value, **parser['score_summary'](data[key]))
                           for key, value in (('prefill', prefill), ('verification', sample))})
    return report, data


def compare_rows(left, right):
    assert len(left) == len(right) and len(left) % 4 == 0
    a, b = array.array('f'), array.array('f')
    assert a.itemsize == 4
    a.frombytes(left)
    b.frombytes(right)
    assert not any(math.isnan(x) for x in a) and not any(math.isnan(x) for x in b)
    changed = sum(left[i:i + 4] != right[i:i + 4] for i in range(0, len(left), 4))
    delta = max((abs(x - y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)), default=0.0)
    return dict(equal_bytes=left == right, changed_f32_words=changed,
                max_abs_finite_delta=delta,
                nonfinite_value_differences=sum(x != y for x, y in zip(a, b)
                                               if not (math.isfinite(x) and math.isfinite(y))))


def main():
    assert __debug__ and socket.gethostname() == 'inference-host'
    assert Path(__file__).resolve().parent == ROOT
    restored = load(OUT / 'independent-restoration.json')
    assert restored['passed'] and not restored['errors']
    analysis = load(OUT / 'analysis.json')
    assert analysis['reproduced_cells'] == ['TRACE-A', 'TRACE-W3']
    parser = runpy.run_path(str(PARSER))
    baseline, a = first_frame(parser, OUT / 'TRACE-A', 1)
    candidate, b = first_frame(parser, OUT / 'TRACE-W3', 4)
    assert baseline['prompt_ids'] == candidate['prompt_ids']
    assert baseline['first_emitted_ids'] == candidate['first_emitted_ids']
    comparisons = {key: compare_rows(a[key], b[key]) for key in a}
    result = dict(status='analyzed', baseline=baseline, candidate=candidate,
                  comparisons=comparisons, same_logical_prefix=True,
                  operator_input_bitwise_equality_claimed=False,
                  operator_causality_claimed=False, timing_permitted=False, promotion_permitted=False)
    (OUT / 'first-frame-comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(comparisons, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
