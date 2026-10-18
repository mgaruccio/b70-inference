#!/usr/bin/env python3
"""Independent post-restoration analyzer. No inference, build, arithmetic patch, or retry."""
import argparse
import array
import hashlib
import json
import math
import re
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
ROOT, OLD, A, NA = C['ROOT'], C['OLD'], C['A'], C['NA']
load, save = C['load'], C['save']
ROUTERS, DENSE, ORDER = NA['ROUTERS'], NA['DENSE'], NA['ORDER']


def dispatch(weight, n):
    """Pinned eligibility, including >64-column shared IQ2 (not an observed whitelist)."""
    assert type(n) is int and 1 <= n <= 1024
    if weight in ROUTERS:
        width = 1 if n == 2 else n
        inner = 'LIBRARY_MKL_F32' if ROUTERS[weight] * width * 4096 < 256**3 else 'LIBRARY_DNNL_F32'
        return [('MUL_MAT_SYCL', -1), (inner, 10)] * (2 if n == 2 else 1)
    if weight in A['IQ2_PROJECTIONS']:
        if n <= 2:
            return [('MOE_SINGLE_MMVQ_PLAIN_Q8_1', -1)] * n
        # total_rows=9*n <= 240*64 is possible throughout this bounded range.
        # Actual n_active must satisfy 9*n <= n_active*64. Only a successful
        # grouped result proves that gate; per-expert fallback is NOT waived.
        assert 9 * n <= 240 * 64
        return [('MOE_GROUPED_GEMM_ATTEMPT', -1), ('MOE_GROUPED_GEMM_RESULT', -1)]
    typ, k, m, _ = DENSE[weight]
    if weight == 'output.weight':
        assert n in (1, 2)
        return [('MMVQ_REORDERED', -1)]
    if typ == 'q8_0' and n <= 2:
        return [('DMMV_ESIMD', -1)]
    if n <= 8:
        return [('MMVQ_PLAIN' if typ == 'iq2_xxs' else 'MMVQ_REORDERED', -1)]
    # supports_mmq() is unconditionally false in this pinned runtime.
    if typ == 'nvfp4':
        # Accumulation hint BF16=15 blocks the default F16 library branch.
        inner, prec = ('LIBRARY_MKL_F32' if k*m*n < 256**3 else 'LIBRARY_DNNL_F32'), 10
    elif typ == 'iq2_xxs' and n <= 64:
        inner, prec = 'LIBRARY_FUSED_DEQUANT_GEMM', -1
    else:
        # fg_plain_ok excludes Q8. Shared IQ2 fails FG_MAX_N=64 above 64.
        inner, prec = 'LIBRARY_DNNL_F16', 20
    return [('MUL_MAT_SYCL', -1), (inner, prec)]


def assert_routes(weight, rows, n):
    expected = dispatch(weight, n)
    assert [(r['route'], r['operand_prec']) for r in rows] == expected, (weight, n, rows)
    tensor = A['assert_route_tensor']
    expert, router = weight in A['IQ2_PROJECTIONS'], weight in ROUTERS
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


def route_frames(log, label):
    """Independent FIFO streams anchored by target-only block0 SSM; no cross-stream ordering."""
    assert label in C['ARMS']
    frames = A['_verify_frames'](log)
    all_routes = [r['route'] for r in A['_route_stream'](log)]
    contexts = {r['backend_ctx'] for r in all_routes if r['weight'] in ORDER[:2]}
    assert len(contexts) == 1, 'missing/ambiguous target SSM backend'
    ctx = next(iter(contexts))
    routes = [r for r in all_routes if r['backend_ctx'] == ctx]
    foreign = [r for r in all_routes if r['backend_ctx'] != ctx]
    if label == 'A0':
        assert not foreign, 'draft/foreign backend in true no-draft arm'
    else:
        assert len({r['backend_ctx'] for r in foreign}) <= 1, 'multiple foreign/draft backends'
    starts = [i for i, r in enumerate(routes) if r['weight'] == ORDER[0]
              and r['route'] not in ('LIBRARY_MKL_F32', 'LIBRARY_DNNL_F32')]
    assert starts and starts[0] == 0 and len(starts) == len(frames), 'route/VERIFY cardinality'
    checked = []
    for i, frame in enumerate(frames):
        batch = frame['batch']
        n, outputs = batch['n'], sum(batch['outputs'])
        assert 1 <= n <= 1024 and 0 <= outputs <= (1 if label == 'A0' else 2)
        group = routes[starts[i]:starts[i+1] if i+1 < len(starts) else len(routes)]
        weights = list(ORDER) + (['output.weight'] if outputs else [])
        expected_weights = [w for w in weights for _ in dispatch(w, outputs if w == 'output.weight' else n)]
        assert [r['weight'] for r in group] == expected_weights, 'missing/extra/reordered all-frame weights'
        for w in weights:
            assert_routes(w, [r for r in group if r['weight'] == w], outputs if w == 'output.weight' else n)
        checked.append(dict(batch=A['_route_batch_metadata'](batch), routes=group, target_backend_ctx=ctx))
    return dict(frames=checked, other_backend_routes=foreign,
                other_backend_routes_not_paired_by_print_order=True)


def check_batches(batches, prompt, emitted, label):
    """Actual contiguous prefill chunks, then actual singleton/width1 verify history."""
    assert len(emitted) == 128 and prompt
    end, split = 0, 0
    for batch in batches:
        n = batch['n']
        assert batch['off'] == 0 and batch['seq'] == ','.join(['0'] * n)
        assert 1 <= n <= 1024
        if end >= len(prompt):
            break
        assert batch['pos'] == list(range(end, end + n)), 'prefill gap/overlap/replay'
        assert end + n <= len(prompt), 'mixed prefill/decode'
        assert batch['tokens'] == prompt[end:end+n], 'public/frozen prompt IDs differ'
        assert batch['outputs'] == [int(p == len(prompt)-1) for p in batch['pos']]
        end += n
        split += 1
    assert end == len(prompt) and split
    decode = batches[split:]
    if label == 'A0':
        for i, b in enumerate(decode):
            assert b['n'] == 1 and b['outputs'] == [1]
            assert b['pos'] == [len(prompt) + i] and b['tokens'] == [emitted[i]]
        assert len(decode) == 127
    else:
        assert label == 'C0'
        for b in decode:
            assert b['n'] in (1, 2) and b['outputs'] == [1] * b['n']
            assert b['pos'][0] >= len(prompt)
        # align() validates every selected row, accepted draft and real rollback.
    return dict(prompt_coverage=end, prefill_widths=[b['n'] for b in batches[:split]],
                post_prompt_frames=len(decode), decode_widths=sorted({b['n'] for b in decode}))


def align(log, tokens, capture, prompt, label):
    events = A['records'](log)
    if label == 'A0':
        assert all(e['event'] not in ('verify', 'checkpoint') for e in events)
        assert sum(e['event'] == 'single' for e in events) == 128
        assert all(e['kind'] == 'prompt' and e['rollback'] == 0 for e in events if e['event'] == 'remove')
    # This legacy function ALREADY accepts actual prompt_tokens and bounds n<=1024.
    # Pass the argument; do not mutate a runpy dictionary expecting function globals to change.
    result = A['align'](events, tokens, capture, prompt_tokens=len(prompt))
    assert result['prompt_ids'] == prompt
    metadata = check_batches(result['batches'], prompt, tokens, label)
    for s in result['samples']:
        assert s['vocab'] == 131072
        values = array.array('f')
        values.frombytes(A['raw_bytes'](capture, s))
        assert values.itemsize == 4 and all(math.isfinite(v) for v in values), 'non-finite raw F32'
    proposed = sum(len(v['drafts']) for v in result['verifications'])
    accepted = sum(v['accepted'] for v in result['verifications'])
    result.update(label=label, capture_path=str(capture), batch_metadata=metadata, proposed=proposed,
                  accepted=accepted, rejection_events=sum(v['rollback'] > 0 for v in result['verifications']),
                  parent_invariance_required=False)
    return result


def normalize(stem, payload):
    assert load(stem.with_suffix('.request.json')) == payload
    assert load(stem.with_suffix('.http.json'))['status'] == 200
    raw = stem.with_suffix('.http-body').read_bytes()
    if payload['stream']:
        events, chunks, usage, timings, finish, done = [], [], {}, {}, None, False
        for line in raw.splitlines():
            if not line.startswith(b'data: '):
                continue
            assert not done, 'SSE data after DONE'
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                continue
            body = json.loads(data)
            assert 'error' not in body
            events.append(body)
            for choice in body.get('choices', []):
                chunks.append(choice.get('delta', {}).get('content') or '')
                if choice.get('finish_reason') is not None:
                    assert finish is None, 'repeated stream finish'
                    finish = choice['finish_reason']
            usage = body.get('usage') or usage
            timings = body.get('timings') or timings
        assert done and events and finish is not None
        result = dict(text=''.join(chunks), usage=usage, timings=timings, finish_reason=finish)
        save(stem.with_suffix('.events.json'), events)
    else:
        body = json.loads(raw)
        save(stem.with_suffix('.response.json'), body)
        result = dict(text=body['choices'][0]['message'].get('content') or '', usage=body['usage'],
                      timings=body.get('timings', {}), finish_reason=body['choices'][0]['finish_reason'])
    save(stem.with_suffix('.result.json'), result)
    return result


def analyze_responses(cell, jobs):
    results, public, checks, errors = {}, {}, [], []
    for i, job in enumerate(jobs):
        stem = job['stem']
        try:
            result = normalize(cell / stem, job['payload'])
            results[stem] = result
            public[stem] = C['check_public'](job['payload'], cell / 'public' / f'{i:02}')
            assert result['usage']['prompt_tokens'] == len(public[stem]), 'public/usage prompt count'
            passed = True
            if stem.startswith('quality/'):
                passed = C['functional_quality'](stem.split('/')[1], result)
            elif stem == 'anchor':
                passed = C['functional_anchor'](result)
            checks.append(dict(stem=stem, functional_pass=passed, actual=result,
                               public_prompt_tokens=len(public[stem]), historical_expected=job.get('historical_prompt_tokens'),
                               raw_token_ids_claimed=(stem == 'diagnostic')))
        except BaseException as exc:
            errors.append(dict(stem=stem, error=repr(exc)))
    report = dict(checks=checks, errors=errors, pass_all=not errors and all(c['functional_pass'] for c in checks))
    save(cell / 'response-checks.json', report)
    return results, public, report


def analyze_arm(cell, label, results, public):
    prompt = public['diagnostic']
    assert prompt == public['warmup'] == public['ordinary']
    for stem in ('warmup', 'ordinary'):
        OLD['check_stream_result'](results[stem], prompt_tokens=len(prompt))
    diagnostic = load(cell / 'diagnostic.response.json')
    tokens = OLD['check_diagnostic_result'](diagnostic, prompt_tokens=len(prompt))
    A['within_arm_text_consistency'](results['ordinary'], diagnostic)
    assert results['ordinary']['finish_reason'] == diagnostic['choices'][0]['finish_reason']
    assert results['ordinary']['usage'] == diagnostic['usage']
    if label == 'A0':
        assert all(diagnostic['timings'].get(k, 0) == 0 for k in ('draft_n', 'draft_n_accepted'))
    boundary = load(cell / 'diagnostic-boundary.json')
    assert boundary['prefix_ok']
    full = (cell / 'server.log').read_bytes()
    start, end = boundary['before_bytes'], boundary['through_bytes']
    assert hashlib.sha256(full[:start]).hexdigest() == boundary['before_sha256']
    assert (cell / 'diagnostic-window.log').read_bytes() == full[start:end]
    log = (cell / 'diagnostic-window.log').read_text()
    result = align(log, tokens, cell / 'capture', prompt, label)
    if label == 'C0':
        assert diagnostic['timings']['draft_n'] == result['proposed']
        assert diagnostic['timings']['draft_n_accepted'] == result['accepted']
    result.update(within_arm_text_passed=True, route_analysis=route_frames(log, label))
    return result


def compare(left, right):
    assert left['label'] == 'A0' and right['label'] == 'C0'
    assert left['within_arm_text_passed'] and right['within_arm_text_passed']
    assert left['prompt_ids'] == right['prompt_ids']
    a, b = left['emitted_ids'], right['emitted_ids']
    assert len(a) == len(b) == len(left['samples']) == len(right['samples']) == 128
    first_id = next((i+1 for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    cutoff, first_score, rows = first_id or 128, None, []
    for i in range(cutoff):
        assert a[:i] == b[:i], 'not shared prior history'
        x, y = left['samples'][i], right['samples'][i]
        assert x['position'] == y['position'] == len(left['prompt_ids']) - 1 + i
        assert x['input_id'] == y['input_id'] and x['vocab'] == y['vocab'] == 131072
        xb, yb = [A['raw_bytes'](Path(r['capture_path']), s) for r, s in ((left, x), (right, y))]
        row = dict(ordinal=i+1, raw_bytes_equal=(xb == yb), position=x['position'], samples={'A0': x, 'C0': y})
        if xb != yb and first_score is None:
            first_score = dict(row, first_differing_float_id=next(j for j in range(len(xb)) if xb[j] != yb[j])//4,
                               batches={'A0': left['batches'][x['batch_index']], 'C0': right['batches'][y['batch_index']]})
        rows.append(row)
    return dict(first_output_difference_ordinal=first_id, compared_through_ordinal=cutoff,
                first_score_difference=first_score, comparisons=rows, all_emitted_ids_match=(a == b),
                all_compared_raw_vectors_equal=all(r['raw_bytes_equal'] for r in rows),
                actual_batch_inputs_match=A['_parent_batch_inputs'](left) == A['_parent_batch_inputs'](right),
                selected_row_metadata_match=A['_parent_sample_metadata'](left) == A['_parent_sample_metadata'](right),
                parent_invariance_required=False, timing_permitted=False, promotion_permitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-after-restore', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host'
    out = ROOT / 'run-r1'
    report = dict(status='failed', errors=[], timing_permitted=False, promotion_permitted=False,
                  raw_capture_remote_only=True, parent_invariance_required=False)
    # Independent analysis is exclusive too: no overwriting failures to make a run pass.
    with (out / 'analysis.json').open('x') as output:
        try:
            summary = load(out / 'summary.json')
            assert summary['restored'] and load(out / 'restoration-checks.json')['passed']
            assert summary['order'] == list(C['ARMS'])
            assert summary['timing_permitted'] is summary['promotion_permitted'] is False
            C['check_tested_code']()
            C['source_contracts']()
            assert load(out / 'hash-checks-before.json') == load(out / 'hash-checks-after.json') == C['N']['check_sources_and_binaries']()
            assert load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
            C['P']['check_restored'](load(out / 'original-container.json'), load(out / 'restored-container.json'))
            # A stored snapshot is NOT a live check. Explicitly query the restored original.
            helper = runpy.run_path(str(OLD['GROUPED'] / 'run-grouped-diagnostic.py'))
            api = runpy.run_path(str(OLD['OLD'] / 'probe-api.py'))
            command = helper['command']
            live = json.loads(command('docker', 'inspect', OLD['ORIGINAL']))[0]
            C['P']['check_restored'](load(out / 'original-container.json'), live)
            assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {OLD['ORIGINAL']}
            assert api['request'](OLD['URL'], '/health')['status'] == 'ok'
            save(out / 'analysis-live-restoration.json', dict(container=live, health_ok=True))
            jobs = C['frozen_jobs']()
            assert load(out / 'frozen-requests.json') == jobs
            responses, public, aligned = {}, {}, {}
            report['response_checks'] = {}
            # Normalize ALL quality/anchor responses in BOTH arms before fidelity parsing.
            for label in summary['completed']:
                cell = out / label
                responses[label], public[label], report['response_checks'][label] = analyze_responses(cell, jobs)
            if all(label in responses for label in C['ARMS']):
                rows = []
                for j in jobs:
                    stem = j['stem']
                    if all(stem in responses[label] and stem in public[label] for label in C['ARMS']):
                        a, b = [responses[label][stem] for label in C['ARMS']]
                        rows.append(dict(stem=stem, requests_equal=load(out / 'A0' / (stem + '.request.json')) == load(out / 'C0' / (stem + '.request.json')),
                                         prompt_ids_equal=public['A0'][stem] == public['C0'][stem],
                                         text_equal=a['text'] == b['text'], finish_equal=a['finish_reason'] == b['finish_reason'],
                                         counts_equal=a['usage'] == b['usage'], raw_token_ids_claimed=(stem == 'diagnostic')))
                report['public_comparisons'] = rows
            for label in summary['completed']:
                try:
                    cell = out / label
                    assert load(cell / 'create-argv.json') == C['make_create'](label, cell / 'capture')
                    container = load(cell / 'container.json')
                    assert container['Name'] == '/' + C['PREFIX'] + label.lower() and not container['State']['Running']
                    assert container['Config']['Cmd'][-1] == load(cell / 'create-argv.json')[-1]
                    assert any(m['Destination'] == '/draft' for m in container['Mounts']) == (label == 'C0')
                    log = (cell / 'server.log').read_text()
                    C['check_startup'](log, label)
                    assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
                    assert C['capture_inventory'](cell / 'capture') == load(cell / 'capture-inventory.json')
                    aligned[label] = analyze_arm(cell, label, responses[label], public[label])
                    save(cell / 'alignment.json', aligned[label])
                except BaseException as exc:
                    report['errors'].append(dict(arm=label, error=repr(exc)))
            if all(label in aligned for label in C['ARMS']):
                report['fidelity'] = compare(aligned['A0'], aligned['C0'])
            assert summary['status'] == 'captured' and summary['completed'] == list(C['ARMS'])
            assert not report['errors'] and 'fidelity' in report
            assert all(r['pass_all'] for r in report['response_checks'].values()), 'functional/public mechanics failed'
            assert len(report['public_comparisons']) == 24 and all(r['requests_equal'] and r['prompt_ids_equal'] for r in report['public_comparisons'])
            report.update(status='analyzed', restoration_verified=True, attribution_gates_passed=True)
        except BaseException as exc:
            report['errors'].append(dict(error=repr(exc)))
            raise
        finally:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
    print('BOLT_LONG_PAIR_ANALYZED_NO_TIMING_NO_PROMOTION')


if __name__ == '__main__':
    main()
