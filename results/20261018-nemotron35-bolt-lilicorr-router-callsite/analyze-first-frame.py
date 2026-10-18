#!/usr/bin/env python3
"""Development-only actual callsite route/F32 comparison; remote, after restoration."""
import array
import itertools
import json
import math
import runpy
import socket
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite')
OUT = ROOT / 'trace-short55-r1'
PARSER = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-route-trace/analyze-trace.py'
NATIVE = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-verify-phase/analyze-trace.py'
BATCH_FIELDS = ('ctx', 'off', 'n', 'tokens', 'pos', 'outputs', 'seq')
ROUTERS = ('blk.1.ffn_gate_inp.weight', 'blk.8.ffn_gate_inp.weight',
           'blk.21.ffn_gate_inp.weight', 'blk.37.ffn_gate_inp.weight', 'blk.48.ffn_gate_inp.weight')


def load(path):
    return json.loads(path.read_text())


def check_batch(batch, positions, outputs):
    assert batch['off'] == 0 and batch['n'] == len(positions)
    assert batch['pos'] == positions and batch['outputs'] == outputs
    assert len(batch['tokens']) == len(positions)
    assert batch['seq'] == ','.join('0' for _ in positions)


def check_router_frame(native, frame, marked):
    """Check REAL target routes, using the unchanged native parser's operation semantics.

    The enclosing BOLT_VERIFY batch supplies the original column count. Split
    records contain singleton views: ordered MUL_MAT_SYCL/library pairs, NOT a
    flattened list whose every src1_ne1 must equal the enclosing batch's n.
    """
    batch, all_routes = frame['batch'], frame['routes']
    assert frame['decoded']['ctx'] == batch['ctx'] and frame['decoded']['ret'] == 0
    anchors = [r for r in all_routes if r['weight'] in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight')]
    assert {r['weight'] for r in anchors} == {'blk.0.ssm_in.weight', 'blk.0.ssm_out.weight'}
    contexts = {r['backend_ctx'] for r in anchors}
    assert len(contexts) == 1, 'ambiguous target backend'
    context = next(iter(contexts))
    routes = [r for r in all_routes if r['backend_ctx'] == context]
    if any(batch['outputs']):
        assert any(r['weight'] == 'output.weight' for r in routes)
    router_rows = [r for r in routes if r['weight'].endswith('.ffn_gate_inp.weight')]
    groups = [(weight, list(rows)) for weight, rows in itertools.groupby(router_rows, key=lambda r: r['weight'])]
    assert [weight for weight, _ in groups] == list(ROUTERS), 'missing, extra, or reordered target routers'
    n = batch['n']
    if marked:
        assert n == 4 and batch['outputs'] == [1] * n
    else:
        assert n == 1 or batch['outputs'] != [1] * n
    for weight, rows in groups:
        block = weight.split('.')[1]
        assert all(r['dst_name'] == 'ffn_moe_logits-' + block for r in rows)
        # Native semantics check full shapes/strides, precision, ordered outer/inner
        # pairs, default N1/prefill routing, and four singleton pairs when marked.
        native['assert_router_operations'](
            rows, (weight, native['ROUTER_WIDTHS'][weight], 0, n),
            'router_columns8_r3' if marked else 'router_prefill')
    return dict(batch=batch, target_backend_ctx=context, routes=routes,
                other_backend_routes=[r for r in all_routes if r['backend_ctx'] != context],
                router_accum_prec=10 if marked else 0,
                router_operation_pairs_per_weight=n if marked else 1,
                router_view_columns=1 if marked else n)


def check_prompt_frames(native, frames, prompt, target_ctx):
    assert len(prompt) == 55 and len(frames) == 3, 'complete three-chunk target prefix required'
    reports = []
    for frame, start, n, outputs in zip(frames, (0, 5, 51), (5, 46, 4),
                                       ([0] * 5, [0] * 46, [0, 0, 0, 1])):
        batch = frame['batch']
        assert batch['ctx'] == target_ctx
        check_batch(batch, list(range(start, start + n)), outputs)
        assert batch['tokens'] == prompt[start:start + n]
        reports.append(check_router_frame(native, frame, marked=False))
    return reports


def first_frame(parser, native, directory, columns):
    assert columns in (1, 4)
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
    check_batch(batch, list(range(55, 55 + columns)), [1] * columns)
    assert batch['tokens'][0] == emitted[0]
    prefills = check_prompt_frames(native, frames[:frames.index(frame)], prompt, batch['ctx'])
    route_report = check_router_frame(native, frame, marked=(columns == 4))
    assert all(p['target_backend_ctx'] == route_report['target_backend_ctx'] for p in prefills)

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
    data = {key: parser['raw_bytes'](directory / 'capture', value)
            for key, value in (('prefill', prefill), ('verification', sample))}
    report = dict(route_report, prompt_ids=prompt, first_emitted_ids=emitted[:2],
                  prefill_frames=prefills,
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
    return dict(equal_bytes=left == right, bytes=len(left), changed_f32_words=changed,
                max_abs_finite_delta=delta,
                nonfinite_value_differences=sum(x != y for x, y in zip(a, b)
                                               if not (math.isfinite(x) and math.isfinite(y))))


def main():
    assert __debug__ and socket.gethostname() == 'inference-host'
    assert Path(__file__).resolve().parent == ROOT
    restored = load(OUT / 'independent-restoration.json')
    assert restored['passed'] and not restored['errors']
    analysis = load(OUT / 'analysis.json')
    assert analysis['reproduced_cells'] == ['RCT-A', 'RCT-W3']
    parser, native = runpy.run_path(str(PARSER)), runpy.run_path(str(NATIVE))
    baseline, a = first_frame(parser, native, OUT / 'RCT-A', 1)
    candidate, b = first_frame(parser, native, OUT / 'RCT-W3', 4)
    assert baseline['prompt_ids'] == candidate['prompt_ids']
    assert baseline['first_emitted_ids'][0] == candidate['first_emitted_ids'][0]
    comparisons = {key: compare_rows(a[key], b[key]) for key in a}
    result = dict(status='analyzed' if comparisons['prefill']['equal_bytes'] else 'failed',
                  baseline=baseline, candidate=candidate, comparisons=comparisons, same_logical_prefix=True,
                  actual_callsite_routes_passed=True, operator_input_bitwise_equality_claimed=False,
                  operator_causality_claimed=False, timing_permitted=False, promotion_permitted=False)
    (OUT / 'first-frame-comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(comparisons, allow_nan=False), flush=True)
    assert comparisons['prefill']['equal_bytes'], 'prefill raw F32 bytes changed; retain evidence, do not qualify'


if __name__ == '__main__':
    main()
