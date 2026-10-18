#!/usr/bin/env python3
"""CPU-only A0/S0 and A0/C0 fidelity analysis, strictly after restoration."""
import array
import json
import math
import re
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('run-a0.py')))
ROOT, REFERENCE, OLD = C['ROOT'], C['REFERENCE'], C['OLD']
A = runpy.run_path(str(REFERENCE / 'analyze-trace.py'))
load, save = C['load'], C['save']
ROUTERS = A['ROUTER_WIDTHS']
# Model dimensions/types are unchanged. None of these selectors is last layer 87.
DENSE = {
    'blk.0.ssm_in.weight': ('nvfp4', 4096, 18560, None),
    'blk.0.ssm_out.weight': ('nvfp4', 8192, 4096, None),
    'blk.1.ffn_latent_down.weight': ('q8_0', 4096, 1024, None),
    'blk.1.ffn_latent_up.weight': ('q8_0', 1024, 4096, None),
    'blk.1.ffn_up_shexp.weight': ('iq2_xxs', 4096, 5376, 'ffn_up-1'),
    'blk.1.ffn_down_shexp.weight': ('iq2_xxs', 5376, 4096, 'ffn_shexp-1'),
    'blk.7.attn_q.weight': ('q8_0', 4096, 4096, 'Qcur-7'),
    'blk.7.attn_v.weight': ('q8_0', 4096, 256, 'Vcur-7'),
    'blk.7.attn_k.weight': ('q8_0', 4096, 256, 'Kcur-7'),
    'blk.7.attn_output.weight': ('q8_0', 4096, 4096, 'attn_out-7'),
    'output.weight': ('q8_0', 4096, 131072, 'result_output'),
}
ORDER = ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight', 'blk.1.ffn_gate_inp.weight',
         'blk.1.ffn_latent_down.weight', 'blk.1.ffn_up_exps.weight', 'blk.1.ffn_down_exps.weight',
         'blk.1.ffn_latent_up.weight', 'blk.1.ffn_up_shexp.weight', 'blk.1.ffn_down_shexp.weight',
         'blk.7.attn_q.weight', 'blk.7.attn_v.weight', 'blk.7.attn_k.weight', 'blk.7.attn_output.weight',
         *[w for w in ROUTERS if w != 'blk.1.ffn_gate_inp.weight'])


def dispatch(weight, n):
    """Pinned SYCL eligibility, not a whitelist inferred from whatever A0 emits."""
    assert 1 <= n <= 55
    if weight in ROUTERS:
        # Router two-column singleton views preserve original strides.
        m, width = ROUTERS[weight], 1 if n == 2 else n
        inner = 'LIBRARY_MKL_F32' if m * width * 4096 < 256**3 else 'LIBRARY_DNNL_F32'
        return [('MUL_MAT_SYCL', -1), (inner, 10)] * (2 if n == 2 else 1)
    if weight in A['IQ2_PROJECTIONS']:
        if n <= 2:
            return [('MOE_SINGLE_MMVQ_PLAIN_Q8_1', -1)] * n
        # Saved target grouped routes succeed. A fallback is a parser blocker,
        # never silently waived; preserve the fresh capture for inspection.
        return [('MOE_GROUPED_GEMM_ATTEMPT', -1), ('MOE_GROUPED_GEMM_RESULT', -1)]
    typ, k, m, _ = DENSE[weight]
    if weight == 'output.weight':
        assert n == 1, 'A0 has one output column'
        return [('MMVQ_REORDERED', -1)]
    if typ == 'q8_0' and n <= 2:
        return [('DMMV_ESIMD', -1)]
    if n <= 8:
        return [('MMVQ_PLAIN' if typ == 'iq2_xxs' else 'MMVQ_REORDERED', -1)]
    if typ == 'nvfp4':
        inner, prec = ('LIBRARY_MKL_F32' if k*m*n < 256**3 else 'LIBRARY_DNNL_F32'), 10
    elif typ == 'q8_0':
        inner, prec = 'LIBRARY_DNNL_F16', 20
    else:
        inner, prec = 'LIBRARY_FUSED_DEQUANT_GEMM', -1
    return [('MUL_MAT_SYCL', -1), (inner, prec)]


def assert_routes(weight, rows, n):
    expected = dispatch(weight, n)
    assert [(r['route'], r['operand_prec']) for r in rows] == expected, (weight, n, rows)
    tensor = A['assert_route_tensor']
    expert = weight in A['IQ2_PROJECTIONS']
    router = weight in ROUTERS
    if expert:
        m, k, b = A['IQ2_PROJECTIONS'][weight]
        typ, name = 'iq2_xxs', 'ffn_moe_' + ('up-1' if b == 1 else 'down-1')
    elif router:
        typ, k, m, name = 'f32', 4096, ROUTERS[weight], 'ffn_moe_logits-' + weight.split('.')[1]
    else:
        typ, k, m, name = DENSE[weight]
    block, size = {'f32': (1, 4), 'q8_0': (32, 34), 'nvfp4': (64, 36), 'iq2_xxs': (256, 66)}[typ]
    row_bytes = k // block * size
    for i, r in enumerate(rows):
        assert r['weight'] == weight
        assert r['op'] == ('MUL_MAT_ID' if expert else 'MUL_MAT') and r['op_id'] == (30 if expert else 29)
        assert r['src0_type'] == typ and r['src1_type'] == r['dst_type'] == 'f32'
        assert r['aux_type'] == r['aux_weight'] == '-'
        assert r['device'] == 0 and r['expert'] == -1
        assert r['reorder'] == int(typ in ('q8_0', 'nvfp4'))
        assert r['src1_prec'] == r['dynamic_prec'] == 0
        assert r['accum_prec'] == (15 if typ == 'nvfp4' else 0)
        assert r['result'] == (1 if expert and n > 2 and i == 1 else -1)
        assert r['dst_name'] == name if name is not None else re.fullmatch(r'node_\d+', r['dst_name'])
        if expert:
            if n <= 2:
                A['assert_iq2_singleton_route'](r, n, 240)
            else:
                tensor(r, '', (k, m, 240, 1), (size, row_bytes, row_bytes*m, row_bytes*m*240))
                tensor(r, 'src1_', (k, b, n, 1), (4, k*4, k*b*4, k*b*n*4))
                tensor(r, 'dst_', (m, 9, n, 1), (4, m*4, m*9*4, m*9*n*4))
                tensor(r, 'ids_', (9, n, 1, 1), (4, 960, 960*n, 960*n))
        else:
            view = 1 if router and n == 2 else n
            tensor(r, '', (k, m, 1, 1), (size, row_bytes, row_bytes*m, row_bytes*m))
            tensor(r, 'src1_', (k, view, 1, 1), (4, k*4, k*n*4, k*n*4))
            tensor(r, 'dst_', (m, view, 1, 1), (4, m*4, m*n*4, m*n*4))
            tensor(r, 'ids_', (0, 0, 0, 0), (0, 0, 0, 0))
    if expected[0][0] == 'MUL_MAT_SYCL':
        for outer, inner in zip(rows[::2], rows[1::2]):
            assert {k: v for k, v in outer.items() if k not in ('route', 'operand_prec')} == {
                k: v for k, v in inner.items() if k not in ('route', 'operand_prec')}


def route_frames(log):
    """Two independent FIFO streams; never use relative stderr/common-log positions."""
    frames = A['_verify_frames'](log)
    routes = [r['route'] for r in A['_route_stream'](log)]
    contexts = {r['backend_ctx'] for r in routes if r['weight'] in ORDER[:2]}
    assert len(contexts) == 1, 'missing/ambiguous target backend'
    ctx = next(iter(contexts))
    assert all(r['backend_ctx'] == ctx for r in routes), 'foreign/draft backend route in A0'
    starts = [i for i, r in enumerate(routes) if r['weight'] == ORDER[0]
              and r['route'] not in ('LIBRARY_MKL_F32', 'LIBRARY_DNNL_F32')]
    assert starts and starts[0] == 0, 'unknown leading route'
    assert len(starts) == len(frames), 'route/VERIFY frame cardinality'
    checked = []
    for i, frame in enumerate(frames):
        batch = frame['batch']
        n = batch['n']
        assert 1 <= n <= 55
        group = routes[starts[i]:starts[i+1] if i+1 < len(starts) else len(routes)]
        weights = list(ORDER) + (['output.weight'] if any(batch['outputs']) else [])
        expected_weights = [w for w in weights for _ in dispatch(w, 1 if w == 'output.weight' else n)]
        assert [r['weight'] for r in group] == expected_weights, 'missing/extra/reordered route'
        for weight in weights:
            assert_routes(weight, [r for r in group if r['weight'] == weight],
                          1 if weight == 'output.weight' else n)
        checked.append(dict(batch=A['_route_batch_metadata'](batch), routes=group, target_backend_ctx=ctx))
    return checked


def check_batches(batches, prompt, emitted):
    assert len(prompt) == 55 and len(emitted) == 128
    end, prefill, decode = 0, [], []
    for batch in batches:
        n = batch['n']
        assert batch['off'] == 0 and batch['seq'] == ','.join(['0'] * n)
        assert batch['pos'] == list(range(end, end + n)), 'gap/overlap/replay in A0 history'
        if end < 55:
            assert end + n <= 55, 'prompt/decode mixing'
            assert batch['tokens'] == prompt[end:end+n], 'frozen prompt IDs changed'
            assert batch['outputs'] == [int(p == 54) for p in batch['pos']], 'unexpected prompt output flags'
            prefill.append(batch)
        else:
            assert n == 1 and batch['outputs'] == [1], 'non-singleton A0 decode'
            assert batch['tokens'] == [emitted[end-55]], 'decode history differs'
            decode.append(batch)
        end += n
    assert end == 182 and len(decode) == 127 and prefill
    return dict(prefill_widths=[b['n'] for b in prefill], prefill_batches=prefill,
                post_prompt_frames=len(decode), decode_widths=[1], prompt_coverage=55)


def align_a0(log, tokens, capture, prompt):
    events = A['records'](log)
    assert all(e['event'] not in ('verify', 'checkpoint') for e in events), 'draft verify/checkpoint in A0'
    assert all(e['kind'] == 'prompt' and e['rollback'] == 0 for e in events if e['event'] == 'remove')
    assert sum(e['event'] == 'single' for e in events) == 128
    batches = [e for e in events if e['event'] == 'batch']
    metadata = check_batches(batches, prompt, tokens)
    result = A['align'](events, tokens, capture)
    assert result['prompt_ids'] == prompt and not result['verifications']
    for sample in result['samples']:
        assert sample['vocab'] == 131072
        values = array.array('f')
        values.frombytes(A['raw_bytes'](capture, sample))
        assert all(math.isfinite(v) for v in values), 'non-finite raw F32 capture'
    result.update(label='A0', capture_path=str(capture), batch_metadata=metadata,
                  proposed=0, accepted=0, rejection_events=0, parent_invariance_required=False)
    return result


def compare(left, right, label):
    assert left['label'] == 'A0' and label in ('S0', 'C0')
    assert left['within_arm_text_passed'] and right['within_arm_text_passed']
    assert left['prompt_ids'] == right['prompt_ids']
    a, b = left['emitted_ids'], right['emitted_ids']
    assert len(a) == len(b) == len(left['samples']) == len(right['samples']) == 128
    first_id = next((i+1 for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    cutoff = first_id or 128
    rows, first_score = [], None
    for i in range(cutoff):
        assert a[:i] == b[:i], 'raw comparison beyond shared prior history'
        x, y = left['samples'][i], right['samples'][i]
        assert x['position'] == y['position'] == 54+i and x['vocab'] == y['vocab'] == 131072
        assert x['input_id'] == y['input_id'], 'actual input row differs before comparison cutoff'
        xb = A['raw_bytes'](Path(left['capture_path']), x)
        yb = A['raw_bytes'](Path(right['capture_path']), y)
        row = dict(ordinal=i+1, raw_bytes_equal=(xb == yb), position=x['position'],
                   samples={'A0': x, label: y})
        if xb != yb and first_score is None:
            first_score = dict(row, first_differing_float_id=next(j for j in range(len(xb)) if xb[j] != yb[j])//4,
                               batches={'A0': left['batches'][x['batch_index']], label: right['batches'][y['batch_index']]})
        rows.append(row)
    return dict(arms=['A0', label], first_output_difference_ordinal=first_id,
                compared_through_ordinal=cutoff, first_score_difference=first_score, comparisons=rows,
                all_emitted_ids_match=(a == b), all_compared_raw_vectors_equal=all(r['raw_bytes_equal'] for r in rows),
                actual_batch_inputs_match=A['_parent_batch_inputs'](left) == A['_parent_batch_inputs'](right),
                selected_row_metadata_match=A['_parent_sample_metadata'](left) == A['_parent_sample_metadata'](right),
                parent_invariance_required=False, timing_permitted=False, promotion_permitted=False)


def quality_compare(cell, reference, label):
    summaries = [load(p / 'quality/summary.json') for p in (cell, reference)]
    names = [[c['name'] for c in s['checks']] for s in summaries]
    assert names[0] == names[1] and len(names[0]) == len(set(names[0])) == 20
    rows = []
    for name in names[0]:
        a, b = [load(p / 'quality' / (name + '.result.json')) for p in (cell, reference)]
        requests_equal = load(cell / 'quality' / (name + '.request.json')) == load(reference / 'quality' / (name + '.request.json'))
        rows.append(dict(name=name, requests_equal=requests_equal, text_equal=a['text'] == b['text'],
                         finish_equal=a['finish_reason'] == b['finish_reason'],
                         completion_count_equal=a['usage']['completion_tokens'] == b['usage']['completion_tokens'],
                         actual={arm: {k: result[k] for k in ('text', 'finish_reason', 'usage')}
                                 for arm, result in (('A0', a), (label, b))}))
    return dict(arms=['A0', label], cases=20, comparisons=rows, raw_token_ids_claimed=False,
                functional_gates={arm: s['pass'] and all(c['pass'] for c in s['checks'])
                                  for arm, s in zip(('A0', label), summaries)},
                all_requests_equal=all(r['requests_equal'] for r in rows),
                all_outputs_finish_counts_equal=all(r['text_equal'] and r['finish_equal'] and r['completion_count_equal'] for r in rows))


def analyze_arm(cell):
    payload, diagnostic_request = C['frozen_requests']()
    for stem in ('warmup', 'ordinary'):
        assert load(cell / (stem + '.request.json')) == payload
        OLD['check_stream_result'](load(cell / (stem + '.result.json')))
    assert load(cell / 'diagnostic.request.json') == diagnostic_request
    diagnostic = load(cell / 'diagnostic.response.json')
    tokens = OLD['check_diagnostic_result'](diagnostic)
    A['within_arm_text_consistency'](load(cell / 'ordinary.result.json'), diagnostic)
    assert all(diagnostic['timings'].get(k, 0) == 0 for k in ('draft_n', 'draft_n_accepted'))
    refs = [load(REFERENCE / 'run-r1' / label / 'alignment.json') for label in ('S0', 'C0')]
    assert refs[0]['prompt_ids'] == refs[1]['prompt_ids']
    log = (cell / 'diagnostic-window.log').read_text()
    result = align_a0(log, tokens, cell / 'capture', refs[0]['prompt_ids'])
    result.update(within_arm_text_passed=True, route_analysis=route_frames(log))
    return result


def main():
    assert socket.gethostname() == 'inference-host'
    out, cell = ROOT / 'run-r1', ROOT / 'run-r1/A0'
    result = dict(status='failed', timing_permitted=False, promotion_permitted=False,
                  parent_invariance_required=False, raw_capture_remote_only=True)
    # Never overwrite a failed analysis summary. A parser blocker does not authorize a launch retry.
    output = out / 'analysis.json'
    assert not output.exists()
    try:
        summary = load(out / 'summary.json')
        assert summary['status'] == 'captured' and summary['restored'] and summary['order'] == ['A0']
        assert summary['timing_permitted'] is summary['promotion_permitted'] is False
        assert load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
        C['P']['check_restored'](load(out / 'original-container.json'), load(out / 'restored-container.json'))
        assert load(out / 'hash-checks-before.json') == load(out / 'hash-checks-after.json') == C['check_sources_and_binaries']()
        argv = load(cell / 'create-argv.json')
        assert argv == C['make_create'](load(REFERENCE / 'run-r1/S0/create-argv.json'), cell / 'capture')
        container = load(cell / 'container.json')
        assert container['Name'] == '/' + C['NAME'] and not container['State']['Running']
        assert all(m['Destination'] != '/draft' for m in container['Mounts'])
        assert container['Config']['Cmd'][-1] == argv[-1]
        log = (cell / 'server.log').read_text()
        C['check_startup'](log)
        assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
        result['quality_comparisons'] = {label: quality_compare(cell, REFERENCE / 'run-r1' / label, label)
                                         for label in ('S0', 'C0')}
        a0 = analyze_arm(cell)
        save(cell / 'alignment.json', a0)
        result['comparisons'] = {label: compare(a0, load(REFERENCE / 'run-r1' / label / 'alignment.json'), label)
                                 for label in ('S0', 'C0')}
        result.update(status='analyzed', attribution_gates_passed=True, restoration_verified=True,
                      batch_metadata=a0['batch_metadata'])
        assert all(q['all_requests_equal'] and all(q['functional_gates'].values())
                   for q in result['quality_comparisons'].values()), 'quality gate failed (differences retained)'
    except BaseException as exc:
        result.update(status='failed', attribution_gates_passed=False, error=repr(exc))
        raise
    finally:
        save(output, result)
    print('BOLT_A0_ANALYZED_NO_TIMING')


if __name__ == '__main__':
    main()
