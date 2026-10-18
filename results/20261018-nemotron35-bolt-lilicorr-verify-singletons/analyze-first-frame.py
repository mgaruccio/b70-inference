#!/usr/bin/env python3
"""Development-only actual-route and raw-F32 comparison for singleton verification."""
import array
import itertools
import json
import math
import runpy
import socket
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons')
OUT = ROOT / 'trace-short55-r1'
PARSER = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-route-trace/analyze-trace.py'
NATIVE = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-verify-phase/analyze-trace.py'
BATCH_FIELDS = ('ctx', 'off', 'n', 'tokens', 'pos', 'outputs', 'seq')
ROUTERS = ('blk.1.ffn_gate_inp.weight', 'blk.8.ffn_gate_inp.weight',
           'blk.21.ffn_gate_inp.weight', 'blk.37.ffn_gate_inp.weight',
           'blk.48.ffn_gate_inp.weight')
SPECIAL_ORDER = ('blk.1.ffn_latent_down.weight', 'blk.1.ffn_up_exps.weight',
                 'blk.1.ffn_down_exps.weight', 'blk.1.ffn_latent_up.weight')

# These are the exact eligible first-block weights and backing layouts observed in
# the immutable parent-A frame.  Candidate rows may expose an outer n-column view or
# a singleton view of that tensor, but the backing dimensions/strides remain fixed.
SPECIAL_SPECS = {
    'blk.1.ffn_latent_down.weight': dict(
        family='q8', route='DMMV_ESIMD', op='MUL_MAT', op_id=29,
        weight_shape=(4096, 1024, 1, 1),
        weight_strides=(34, 4352, 4456448, 4456448),
        src1_ne0=4096, dst_ne0=1024, src1_nb1=16384, dst_nb1=4096),
    'blk.1.ffn_latent_up.weight': dict(
        family='q8', route='DMMV_ESIMD', op='MUL_MAT', op_id=29,
        weight_shape=(1024, 4096, 1, 1),
        weight_strides=(34, 1088, 4456448, 4456448),
        src1_ne0=1024, dst_ne0=4096, src1_nb1=4096, dst_nb1=16384),
    'blk.1.ffn_up_exps.weight': dict(
        family='iq2', route='MOE_SINGLE_MMVQ_PLAIN_Q8_1', op='MUL_MAT_ID', op_id=30,
        weight_shape=(1024, 2048, 240, 1),
        weight_strides=(66, 264, 540672, 129761280),
        src1_ne0=1024, src1_ne1=1, src1_nb1=4096,
        dst_ne0=2048, dst_ne1=9, dst_nb1=8192),
    'blk.1.ffn_down_exps.weight': dict(
        family='iq2', route='MOE_SINGLE_MMVQ_PLAIN_Q8_1', op='MUL_MAT_ID', op_id=30,
        weight_shape=(2048, 1024, 240, 1),
        weight_strides=(66, 528, 540672, 129761280),
        src1_ne0=2048, src1_ne1=9, src1_nb1=8192,
        dst_ne0=1024, dst_ne1=9, dst_nb1=4096),
}
SPECIAL_WEIGHTS = frozenset(SPECIAL_SPECS)
# Exact (weight, route) sequences and cardinalities observed in immutable parent A.
# These are keyed by the actual n values used by the prefix and first post-prefix
# frames; unmarked validation must not widen them to route-family allowlists.
UNMARKED_SPECIAL_ROUTES = {
    1: (
        ('blk.1.ffn_latent_down.weight', 'DMMV_ESIMD'),
        ('blk.1.ffn_up_exps.weight', 'MOE_SINGLE_MMVQ_PLAIN_Q8_1'),
        ('blk.1.ffn_down_exps.weight', 'MOE_SINGLE_MMVQ_PLAIN_Q8_1'),
        ('blk.1.ffn_latent_up.weight', 'DMMV_ESIMD'),
    ),
    5: (
        ('blk.1.ffn_latent_down.weight', 'MMVQ_REORDERED'),
        ('blk.1.ffn_up_exps.weight', 'MOE_GROUPED_GEMM_ATTEMPT'),
        ('blk.1.ffn_up_exps.weight', 'MOE_GROUPED_GEMM_RESULT'),
        ('blk.1.ffn_down_exps.weight', 'MOE_GROUPED_GEMM_ATTEMPT'),
        ('blk.1.ffn_down_exps.weight', 'MOE_GROUPED_GEMM_RESULT'),
        ('blk.1.ffn_latent_up.weight', 'MMVQ_REORDERED'),
    ),
    46: (
        ('blk.1.ffn_latent_down.weight', 'MUL_MAT_SYCL'),
        ('blk.1.ffn_latent_down.weight', 'LIBRARY_DNNL_F16'),
        ('blk.1.ffn_up_exps.weight', 'MOE_GROUPED_GEMM_ATTEMPT'),
        ('blk.1.ffn_up_exps.weight', 'MOE_GROUPED_GEMM_RESULT'),
        ('blk.1.ffn_down_exps.weight', 'MOE_GROUPED_GEMM_ATTEMPT'),
        ('blk.1.ffn_down_exps.weight', 'MOE_GROUPED_GEMM_RESULT'),
        ('blk.1.ffn_latent_up.weight', 'MUL_MAT_SYCL'),
        ('blk.1.ffn_latent_up.weight', 'LIBRARY_DNNL_F16'),
    ),
    4: (
        ('blk.1.ffn_latent_down.weight', 'MMVQ_REORDERED'),
        ('blk.1.ffn_up_exps.weight', 'MOE_GROUPED_GEMM_ATTEMPT'),
        ('blk.1.ffn_up_exps.weight', 'MOE_GROUPED_GEMM_RESULT'),
        ('blk.1.ffn_down_exps.weight', 'MOE_GROUPED_GEMM_ATTEMPT'),
        ('blk.1.ffn_down_exps.weight', 'MOE_GROUPED_GEMM_RESULT'),
        ('blk.1.ffn_latent_up.weight', 'MMVQ_REORDERED'),
    ),
}


def load(path):
    return json.loads(path.read_text())


def check_batch(batch, positions, outputs):
    assert batch['off'] == 0 and batch['n'] == len(positions)
    assert batch['pos'] == positions and batch['outputs'] == outputs
    assert len(batch['tokens']) == len(positions)
    assert batch['seq'] == ','.join('0' for _ in positions)


def _assert_tensor(row, prefix, shape, strides):
    assert tuple(row[f'{prefix}ne{i}'] for i in range(4)) == shape, row
    assert tuple(row[f'{prefix}nb{i}'] for i in range(4)) == strides, row


def _assert_common_special(row, weight, context, n, marked, expected_route=None):
    spec = SPECIAL_SPECS[weight]
    assert row['weight'] == weight and row['backend_ctx'] == context
    route = spec['route'] if marked else expected_route
    assert route is not None and row['route'] == route, row
    assert row['op'] == spec['op'] and row['op_id'] == spec['op_id'], row
    assert row['aux_weight'] == row['aux_type'] == '-', row
    assert row['src0_type'] == ('q8_0' if spec['family'] == 'q8' else 'iq2_xxs')
    assert row['src1_type'] == row['dst_type'] == 'f32', row
    assert row['reorder'] == (1 if spec['family'] == 'q8' else 0), row
    assert row['src1_prec'] == row['dynamic_prec'] == 0, row
    assert row['accum_prec'] == (10 if marked else 0), row
    assert row['device'] == 0 and row['expert'] == -1, row
    if marked:
        assert row['result'] == -1, row
    elif row['route'] == 'MOE_GROUPED_GEMM_RESULT':
        assert row['result'] == 1, row
    else:
        assert row['result'] == -1, row
    _assert_tensor(row, '', spec['weight_shape'], spec['weight_strides'])

    if spec['family'] == 'q8':
        src1_ne1 = row['src1_ne1']
        dst_ne1 = row['dst_ne1']
        assert src1_ne1 in (1, n) and dst_ne1 in (1, n), row
        assert src1_ne1 == dst_ne1, row
        _assert_tensor(row, 'src1_',
                       (spec['src1_ne0'], src1_ne1, 1, 1),
                       (4, spec['src1_nb1'], spec['src1_nb1'] * n,
                        spec['src1_nb1'] * n))
        _assert_tensor(row, 'dst_',
                       (spec['dst_ne0'], dst_ne1, 1, 1),
                       (4, spec['dst_nb1'], spec['dst_nb1'] * n,
                        spec['dst_nb1'] * n))
    else:
        # MUL_MAT_ID has one outer row. Its token batch and IDs must all retain
        # the complete n-column view; singleton IQ2 helper views are invalid.
        src1_ne2 = row['src1_ne2']
        dst_ne2 = row['dst_ne2']
        ids_ne1 = row['ids_ne1']
        assert src1_ne2 == dst_ne2 == ids_ne1 == n, row
        _assert_tensor(row, 'src1_',
                       (spec['src1_ne0'], spec['src1_ne1'], src1_ne2, 1),
                       (4, spec['src1_nb1'], spec['src1_nb1'] * spec['src1_ne1'],
                        spec['src1_nb1'] * spec['src1_ne1'] * n))
        _assert_tensor(row, 'dst_',
                       (spec['dst_ne0'], spec['dst_ne1'], dst_ne2, 1),
                       (4, spec['dst_nb1'], spec['dst_nb1'] * spec['dst_ne1'],
                        spec['dst_nb1'] * spec['dst_ne1'] * n))
        _assert_tensor(row, 'ids_', (9, ids_ne1, 1, 1),
                       (4, 960, 960 * n, 960 * n))

def check_special_routes(routes, n, context, marked):
    """Require the exact eligible first-block route contract.

    Marked Q8 accepts one coherent outer n-column row or exactly n coherent
    singleton rows. Marked IQ2 remains one outer MUL_MAT_ID row with a complete
    n-column token/ID view. Unmarked frames use the exact parent-A sequence for
    their observed n value, rather than a broad route-family allowlist.
    """
    assert n >= 1
    eligible_names = ('.ffn_latent_down.weight', '.ffn_latent_up.weight',
                      '.ffn_up_exps.weight', '.ffn_down_exps.weight')
    named_rows = [row for row in routes if row['weight'].endswith(eligible_names)]
    assert {row['weight'] for row in named_rows} == SPECIAL_WEIGHTS, (
        'unexpected or missing eligible weight names', {row['weight'] for row in named_rows})
    rows = [row for row in named_rows if row['weight'] in SPECIAL_WEIGHTS]
    assert rows, 'missing eligible Q8/IQ2 route records'
    by_weight = {weight: [] for weight in SPECIAL_ORDER}
    for row in rows:
        by_weight.setdefault(row['weight'], []).append(row)
    assert set(by_weight) == SPECIAL_WEIGHTS and all(by_weight.values()), (
        'missing eligible Q8/IQ2 weight', {k: len(v) for k, v in by_weight.items()})
    first_seen = []
    for row in rows:
        if not first_seen or first_seen[-1] != row['weight']:
            first_seen.append(row['weight'])
    assert first_seen == list(SPECIAL_ORDER), ('eligible route order', first_seen)

    if marked:
        assert 3 <= n <= 8
        for weight in SPECIAL_ORDER:
            weight_rows = by_weight[weight]
            spec = SPECIAL_SPECS[weight]
            if spec['family'] == 'q8':
                assert len(weight_rows) in (1, n), (weight, len(weight_rows))
                if len(weight_rows) == 1:
                    assert (weight_rows[0]['src1_ne1'] ==
                            weight_rows[0]['dst_ne1'] == n), weight_rows[0]
                else:
                    assert all(row['src1_ne1'] == row['dst_ne1'] == 1
                               for row in weight_rows), (weight, weight_rows)
            else:
                assert len(weight_rows) == 1, (weight, len(weight_rows))
            for row in weight_rows:
                _assert_common_special(row, weight, context, n, True)
    else:
        expected = UNMARKED_SPECIAL_ROUTES.get(n)
        assert expected is not None, ('unrecognized unmarked frame width', n)
        actual = tuple((row['weight'], row['route']) for row in rows)
        assert actual == expected, ('unmarked eligible route sequence', actual, expected)
        for row, (weight, route) in zip(rows, expected):
            _assert_common_special(row, weight, context, n, False, route)
    return dict(
        special_route_weights=list(SPECIAL_ORDER),
        special_route_counts={weight: len(by_weight[weight]) for weight in SPECIAL_ORDER},
        special_routes_marked=bool(marked),
        special_accum_prec=10 if marked else 0,
    )


def check_router_frame(native, frame, marked):
    """Check real target routes and the bounded Q8/IQ2 intervention contract."""
    batch, all_routes = frame['batch'], frame['routes']
    assert frame['decoded']['ctx'] == batch['ctx'] and frame['decoded']['ret'] == 0
    anchors = [r for r in all_routes if r['weight'] in
               ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight')]
    assert {r['weight'] for r in anchors} == {
        'blk.0.ssm_in.weight', 'blk.0.ssm_out.weight'}
    contexts = {r['backend_ctx'] for r in anchors}
    assert len(contexts) == 1, 'ambiguous target backend'
    context = next(iter(contexts))
    routes = [r for r in all_routes if r['backend_ctx'] == context]
    if any(batch['outputs']):
        assert any(r['weight'] == 'output.weight' for r in routes)
    router_rows = [r for r in routes if r['weight'].endswith('.ffn_gate_inp.weight')]
    groups = [(weight, list(rows)) for weight, rows in
              itertools.groupby(router_rows, key=lambda r: r['weight'])]
    assert [weight for weight, _ in groups] == list(ROUTERS), (
        'missing, extra, or reordered target routers')
    n = batch['n']
    if marked:
        assert 3 <= n <= 8 and batch['outputs'] == [1] * n
    else:
        assert n == 1 or batch['outputs'] != [1] * n
    for weight, rows in groups:
        block = weight.split('.')[1]
        assert all(r['dst_name'] == 'ffn_moe_logits-' + block for r in rows)
        native['assert_router_operations'](
            rows, (weight, native['ROUTER_WIDTHS'][weight], 0, n),
            'router_columns8_r3' if marked else 'router_prefill')
    special = check_special_routes(routes, n, context, marked)
    return dict(batch=batch, target_backend_ctx=context, routes=routes,
                other_backend_routes=[r for r in all_routes if r['backend_ctx'] != context],
                router_accum_prec=10 if marked else 0,
                router_operation_pairs_per_weight=n if marked else 1,
                router_view_columns=1 if marked else n,
                **special)


def check_prompt_frames(native, frames, prompt, target_ctx):
    assert len(prompt) == 55 and len(frames) == 3, 'complete three-chunk target prefix required'
    reports = []
    for frame, start, n, outputs in zip(
            frames, (0, 5, 51), (5, 46, 4),
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
    delta = max((abs(x - y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)),
                default=0.0)
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
    assert analysis['reproduced_cells'] == ['VST-A', 'VST-W3']
    parser, native = runpy.run_path(str(PARSER)), runpy.run_path(str(NATIVE))
    baseline, a = first_frame(parser, native, OUT / 'VST-A', 1)
    candidate, b = first_frame(parser, native, OUT / 'VST-W3', 4)
    assert baseline['prompt_ids'] == candidate['prompt_ids']
    assert baseline['first_emitted_ids'][0] == candidate['first_emitted_ids'][0]
    comparisons = {key: compare_rows(a[key], b[key]) for key in a}
    result = dict(status='analyzed' if comparisons['prefill']['equal_bytes'] else 'failed',
                  baseline=baseline, candidate=candidate, comparisons=comparisons,
                  same_logical_prefix=True, actual_callsite_routes_passed=True,
                  singleton_intervention_routes_checked=True,
                  operator_input_bitwise_equality_claimed=False,
                  operator_causality_claimed=False, timing_permitted=False,
                  promotion_permitted=False)
    (OUT / 'first-frame-comparison.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(comparisons, allow_nan=False), flush=True)
    assert comparisons['prefill']['equal_bytes'], 'prefill raw F32 bytes changed; retain evidence, do not qualify'


if __name__ == '__main__':
    main()
