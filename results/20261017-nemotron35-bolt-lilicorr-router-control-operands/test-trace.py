#!/usr/bin/env python3
"""Remote CPU-only regression tests. No Docker launch, API calls, or model load."""
import array
import ast
import copy
import json
import re
import runpy
import shlex
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

assert socket.gethostname() == 'inference-host'
M = runpy.run_path('run-trace.py')
A = runpy.run_path('analyze-trace.py')
OLD = runpy.run_path(str(M['WIDTH1'] / 'run-controls.py'))
NATIVE = runpy.run_path(str(M['WIDTH1'] / 'analyze-controls.py'))
BASE = json.loads((OLD['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
IMAGE = 'intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04'
PARENT_TOP = M['PARENT_RESULTS']
PARENT_CONTROL = runpy.run_path(str(PARENT_TOP/'run-trace.py'))


def iq2_route_line(weight, n, n_mats=16, route='MOE_SINGLE_MMVQ_PLAIN_Q8_1', **extra):
    m, k, b = A['IQ2_PROJECTIONS'][weight]
    row = k // 256 * 66
    view_n = 1 if route == 'MOE_SINGLE_MMVQ_PLAIN_Q8_1' else n
    fields = dict(src0_type='iq2_xxs', op='MUL_MAT_ID', op_id=30,
                  ne0=k, ne1=m, ne2=n_mats, ne3=1, nb0=66, nb1=row, nb2=row*m, nb3=row*m*n_mats,
                  src1_ne0=k, src1_ne1=b, src1_ne2=view_n, src1_ne3=1,
                  src1_nb0=4, src1_nb1=k*4, src1_nb2=k*b*4, src1_nb3=k*b*n*4,
                  dst_ne0=m, dst_ne1=9, dst_ne2=view_n, dst_ne3=1,
                  dst_nb0=4, dst_nb1=m*4, dst_nb2=m*9*4, dst_nb3=m*9*n*4,
                  ids_ne0=9, ids_ne1=view_n, ids_ne2=1, ids_ne3=1,
                  ids_nb0=4, ids_nb1=n_mats*4, ids_nb2=n_mats*n*4, ids_nb3=n_mats*n*4)
    fields.update(extra)
    return 'BOLT_ROUTE ' + route_line(route=route, weight=weight, **fields).split('BOLT_ROUTE ', 1)[1]


def iq2_native_output(grouped_success=True):
    lines = ['Backend 1/2: SYCL0']
    for n in (1, 2, 3):
        for b, m, k in ((1, 2048, 1024), (0, 1024, 2048)):
            weight = 'blk.1.ffn_' + ('up' if b else 'down') + '_exps.weight'
            if n <= 2:
                lines.extend([iq2_route_line(weight, n)] * n)
            else:
                lines.append(iq2_route_line(weight, n, route='MOE_GROUPED_GEMM_ATTEMPT'))
                lines.append(iq2_route_line(weight, n, route='MOE_GROUPED_GEMM_RESULT', result=int(grouped_success)))
                if not grouped_success:
                    row = k // 256 * 66
                    for expert in range(9):
                        extra = dict(ne2=1, nb3=row*m, src1_ne1=3, src1_ne2=1,
                                     src1_nb2=k*3*4, src1_nb3=k*3*4,
                                     dst_ne1=3, dst_ne2=1, dst_nb2=m*3*4, dst_nb3=m*3*4)
                        lines.append(iq2_route_line(weight, n, route='MOE_PER_EXPERT_FALLBACK', expert=expert, **extra))
                        extra.update({f'ids_{field}{i}': 0 for field in ('ne', 'nb') for i in range(4)})
                        lines.append(iq2_route_line(weight, n, route='MMVQ_PLAIN', **extra))
            lines.append(f'  MUL_MAT_ID(type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,b={b},m={m},n={n},k={k},amax=1.000000,m_v=0,ids_offset=0): \x1b[1;32mOK\x1b[0m')
    return ('\n'.join(lines) + '\n  6/6 tests passed\n'
            '  Backend SYCL0: OK\nBackend 2/2: CPU\n  Skipping\n2/2 backends passed\nOK\n')


def fixture(directory, speculative=False, diverge=False, chunked_prompt=False):
    directory.mkdir()
    tokens = list(range(1000, 1128))
    tokens[115] = 4618 if diverge else 6833
    vectors = array.array('f', [-1] * 8192)
    vectors[6833], vectors[4618] = 10, 9
    for i in range(128):
        (directory / f'row-{i:06d}').write_bytes(vectors.tobytes())
    rows = []
    def log(event, **kwargs):
        fields = []
        for key, value in kwargs.items():
            if isinstance(value, list):
                value = ','.join(map(str, value)) or '-'
            fields.append(f'{key}={value}')
        rows.append('2026-10-17T00:00:00Z 0.0.1 I BOLT_VERIFY ' + event + ' ' + ' '.join(fields))
    def batch(ids, pos, outputs):
        log('batch', ctx='0xabc', off=0, n=len(ids), tokens=ids, pos=pos,
            outputs=outputs, seq=','.join('0' for _ in ids))
        log('decoded', ctx='0xabc', ret=0)
    def sample(i, idx):
        log('sample', ctx='0xabc', smpl='0xdef', idx=idx, selected=tokens[i], vocab=8192,
            status='cpu', file=f'row-{i:06d}')
    def emit(i):
        log('emit', ctx='0xabc', slot=0, ordinal=i + 1, token=tokens[i])
    log('remove', ctx='0xabc', slot=0, kind='prompt', p0=0, p1=-1, rollback=0)
    if chunked_prompt:
        for start, end in ((0, 5), (5, 51), (51, 55)):
            if start:
                log('remove', ctx='0xabc', slot=0, kind='prompt', p0=start, p1=-1, rollback=0)
            batch(list(range(start, end)), list(range(start, end)),
                  [int(p == 54) for p in range(start, end)])
        first_row = 3
    else:
        batch(list(range(55)), list(range(55)), [0] * 54 + [1])
        first_row = 54
    sample(0, first_row)
    log('single', ctx='0xabc', smpl='0xdef', slot=0, row=first_row, id=tokens[0])
    emit(0)
    i = 1
    while i < 128:
        if not speculative or i == 127:
            batch([tokens[i - 1]], [54 + i], [1])
            sample(i, 0)
            log('single', ctx='0xabc', smpl='0xdef', slot=0, row=0, id=tokens[i])
            emit(i)
            i += 1
            continue
        rejected = i == 1
        draft = 777 if rejected else tokens[i]
        batch([tokens[i - 1], draft], [54 + i, 55 + i], [1, 1])
        sample(i, 0)
        emitted = [tokens[i]]
        if not rejected:
            sample(i + 1, 1)
            emitted.append(tokens[i + 1])
        log('verify', ctx='0xabc', smpl='0xdef', slot=0, rows=[0, 1], drafts=[draft],
            emitted=emitted, accepted=len(emitted) - 1, rollback=int(rejected), replay=0, rejection=0, synth=0)
        log('remove', ctx='0xabc', slot=0, kind='verify', p0=54 + i + len(emitted), p1=-1, rollback=int(rejected))
        for j in range(len(emitted)):
            emit(i + j)
        i += len(emitted)
    return '\n'.join(rows) + '\n', tokens

def route_line(route='MMVQ_PLAIN', weight='blk.0.attn_q.weight', backend='0xdef', aux='-',
               result=-1, expert=-1, op='MUL_MAT', op_id=0, ids=False, omit=(), **extra):
    fields = dict(route=route, weight=weight, aux_weight=aux, op=op, op_id=op_id,
                  dst_name='dst.tensor', src0_type='nvfp4', aux_type='-', src1_type='f32', dst_type='f32',
                  ne0=8, ne1=4, ne2=1, ne3=1, nb0=1, nb1=8, nb2=32, nb3=32,
                  src1_ne0=8, src1_ne1=1, src1_ne2=1, src1_ne3=1,
                  src1_nb0=4, src1_nb1=32, src1_nb2=32, src1_nb3=32,
                  dst_ne0=8, dst_ne1=1, dst_ne2=1, dst_ne3=1,
                  dst_nb0=4, dst_nb1=32, dst_nb2=32, dst_nb3=32,
                  ids_ne0=2 if ids else 0, ids_ne1=1 if ids else 0,
                  ids_ne2=1 if ids else 0, ids_ne3=1 if ids else 0,
                  ids_nb0=4 if ids else 0, ids_nb1=8 if ids else 0,
                  ids_nb2=8 if ids else 0, ids_nb3=8 if ids else 0,
                  reorder=0, src1_prec=0, operand_prec=-1, accum_prec=0, dynamic_prec=3,
                  device=0, backend_ctx=backend, expert=expert, result=result)
    fields.update(extra)
    if 'src1_ne1' in extra and 'src1_nb2' not in extra and 'src1_nb3' not in extra:
        fields['src1_nb2'] = fields['src1_nb1'] * fields['src1_ne1']
        fields['src1_nb3'] = fields['src1_nb2']
    if 'dst_ne1' in extra and 'dst_nb2' not in extra and 'dst_nb3' not in extra:
        fields['dst_nb2'] = fields['dst_nb1'] * fields['dst_ne1']
        fields['dst_nb3'] = fields['dst_nb2']
    for key in omit:
        fields.pop(key)
    return '2026-10-17T00:00:00Z 0.0.1 I BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in fields.items())


def route_batch(pos, tokens):
    outputs = [1] * len(tokens)
    seq = ','.join('0' for _ in tokens)
    return ('2026-10-17T00:00:00Z 0.0.1 I BOLT_VERIFY batch ctx=0xabc off=0 '
            f'n={len(tokens)} tokens={",".join(map(str, tokens))} '
            f'pos={",".join(map(str, pos))} outputs={",".join(map(str, outputs))} seq={seq}')


def route_decoded(ctx='0xabc', ret=0):
    return f'2026-10-17T00:00:00Z 0.0.1 I BOLT_VERIFY decoded ctx={ctx} ret={ret}'


def head_route(backend='0xabc', head_ne1=1):
    return route_line(weight='output.weight', backend=backend, src0_type='q8_0',
                      route='MMVQ_REORDERED', reorder=1, src1_ne1=head_ne1)


def ssm_route(weight, backend='0xabc', head_ne1=1):
    return route_line(weight=weight, backend=backend, src0_type='nvfp4',
                      route='MMVQ_REORDERED', reorder=1, src1_ne1=head_ne1)


def router_lines(n=2, weight='blk.1.ffn_gate_inp.weight', layout=0, backend='0xabc', native=False):
    split = weight.endswith('.ffn_gate_inp.weight') and n == 2 and layout == 0
    view_n = 1 if split else n
    stride = (4096 + (16 if layout == 2 else 0)) * 4
    fields = dict(op_id=29, src0_type='f32', dynamic_prec=0,
                  ne0=4096, ne1=240, ne2=1, ne3=1, nb0=4, nb1=16384, nb2=3932160, nb3=3932160,
                  src1_ne0=4096, src1_ne1=view_n, src1_ne2=1, src1_ne3=1,
                  src1_nb0=4, src1_nb1=stride, src1_nb2=stride*n, src1_nb3=stride*n,
                  dst_ne0=240, dst_ne1=view_n, dst_ne2=1, dst_ne3=1,
                  dst_nb0=4, dst_nb1=960, dst_nb2=960*n, dst_nb3=960*n,
                  dst_name=f'router_single2-l{layout}-n{n}' if native else 'ffn_moe_logits-1')
    return [route_line(route=route, weight=weight, backend=backend, operand_prec=prec, **fields)
            for _ in range(2 if split else 1) for route, prec in (('MUL_MAT_SYCL', -1), ('LIBRARY_MKL_F32', 10))]


def router_native_output():
    lines = ['Backend 1/2: SYCL0']
    for weight, layout, n in sorted(M['ROUTER_NATIVE_CASES']):
        lines.extend(router_lines(n, weight, layout, native=True))
        lines.append(f'  MUL_MAT(router_single2=1,weight={weight},layout={layout},type_a=f32,type_b=f32,'
                     f'm=240,n={n},k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],'
                     'k_v=0,o=1,src_overlap=0,m_v=0,pad=0): \x1b[1;32mOK\x1b[0m')
    return '\n'.join(lines) + '\n  6/6 tests passed\n  Backend SYCL0: OK\nBackend 2/2: CPU\n  Skipping\n2/2 backends passed\nOK\n'


def router_candidate_log(log):
    """Test-only transformation of saved parent routes, never used for real analysis."""
    contexts = {r['backend_ctx'] for f in A['route_frames'](log) for r in f['routes']
                if r['weight'] in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight')}
    assert len(contexts) == 1
    target = next(iter(contexts))
    lines, pending = [], None
    for line in log.splitlines():
        if 'BOLT_ROUTE ' in line:
            r = A['route_records'](line)[0]
            if r['backend_ctx'] == target and r['weight'].endswith('.ffn_gate_inp.weight') and r['src1_ne1'] == 2:
                r['src1_ne1'] = r['dst_ne1'] = 1
                changed = 'BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in r.items())
                if r['route'] == 'MUL_MAT_SYCL':
                    assert pending is None
                    pending = changed
                else:
                    assert pending is not None and r['route'] == 'LIBRARY_MKL_F32'
                    lines.extend([pending, changed, pending, changed])
                    pending = None
                continue
        assert pending is None, 'non-adjacent parent router records'
        lines.append(line)
    assert pending is None
    return '\n'.join(lines) + '\n'


def block1_route_lines(label='S0', backend='0xabc', head_ne1=1):
    assert label in ('S0', 'C0')
    rows = [
        route_line(weight='blk.1.ffn_latent_down.weight', backend=backend, src0_type='q8_0',
                   route='DMMV_ESIMD', reorder=1, src1_ne1=head_ne1),
    ]
    for weight in A['IQ2_PROJECTIONS']:
        rows.extend([iq2_route_line(weight, head_ne1, 240, backend=backend)] * head_ne1)
    rows.append(route_line(weight='blk.1.ffn_latent_up.weight', backend=backend, src0_type='q8_0',
                           route='DMMV_ESIMD', reorder=1, src1_ne1=head_ne1))
    for weight in ('blk.1.ffn_up_shexp.weight', 'blk.1.ffn_down_shexp.weight'):
        rows.append(route_line(weight=weight, backend=backend, src0_type='q8_0',
                               route='MMVQ_PLAIN', reorder=1, src1_ne1=head_ne1))
    return rows

def block7_route_lines(backend='0xabc', head_ne1=1):
    rows = []
    for weight, k, m in (
            ('blk.7.attn_q.weight', 4096, 4096),
            ('blk.7.attn_v.weight', 4096, 256),
            ('blk.7.attn_k.weight', 4096, 256),
            ('blk.7.attn_output.weight', 4096, 4096)):
        rows.append(route_line(weight=weight, backend=backend, src0_type='q8_0',
                               route='DMMV_ESIMD', reorder=1, src1_ne1=head_ne1,
                               ne0=k, ne1=m, nb0=34, nb1=k // 32 * 34,
                               nb2=k // 32 * 34 * m, nb3=k // 32 * 34 * m,
                               src1_ne0=k, src1_nb0=4, src1_nb1=k * 4,
                               src1_nb2=k * head_ne1 * 4, src1_nb3=k * head_ne1 * 4,
                               dst_ne0=m, dst_ne1=head_ne1, dst_nb0=4, dst_nb1=m * 4,
                               dst_nb2=m * head_ne1 * 4, dst_nb3=m * head_ne1 * 4))
    return rows


def inject_block7_routes(metadata):
    block7 = {weight for weight, _, _ in (
        ('blk.7.attn_q.weight', 4096, 4096),
        ('blk.7.attn_k.weight', 4096, 256),
        ('blk.7.attn_v.weight', 4096, 256),
        ('blk.7.attn_output.weight', 4096, 4096),
    )}
    for batch in metadata['selected_batches']:
        batch['routes'] = [route for route in batch['routes'] if route['weight'] not in block7]
        head = next(route for route in batch['routes'] if route['weight'] == 'output.weight')
        for line in block7_route_lines(backend=head['backend_ctx'], head_ne1=head['src1_ne1']):
            batch['routes'].append(A['route_records'](line)[0])



def route_log(label='S0', missing_output=False, missing_block1=False, missing_block7=False):
    assert label in ('S0', 'C0')
    lines = [route_line(weight='blk.0.outside.weight')]

    def target_routes(head_ne1=1):
        rows = [ssm_route('blk.0.ssm_in.weight', head_ne1=head_ne1),
                ssm_route('blk.0.ssm_out.weight', head_ne1=head_ne1)]
        if not missing_block1:
            rows.extend(block1_route_lines(label, head_ne1=head_ne1))
        if not missing_block7:
            rows.extend(block7_route_lines(head_ne1=head_ne1))
        return rows

    lines += [route_batch([55], [1055])] + target_routes()
    if not missing_output:
        lines.append(head_route())
    lines.append(route_decoded())
    lines += [route_line(weight='blk.0.after.weight'), route_batch([56], [1056])]
    lines += target_routes()
    lines += [head_route(), route_decoded()]
    lines += [route_batch([55, 56], [1055, 1056])]
    lines += target_routes(head_ne1=2)
    lines += [head_route(head_ne1=2), route_decoded(), route_line(weight='blk.0.after.weight')]
    return '\n'.join(lines) + '\n'



def router_model_log(label='S0'):
    # Unlike the legacy selection fixture, this contains only this arm's frames.
    lines = []
    for frame in A['select_route_batches'](route_log(label), label)['selected_batches']:
        batch = frame['batch']
        lines.append(route_batch(batch['pos'], batch['tokens']))
        inserted = False
        for route in frame['routes']:
            if not inserted and route['weight'] == 'blk.1.ffn_latent_down.weight':
                lines.extend(router_lines(batch['n']))
                inserted = True
            lines.append('BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in route.items()))
        assert inserted
        lines.append(route_decoded())
    return '\n'.join(lines) + '\n'


def q8_native_output():
    """Synthetic console fixture, never used as evidence of a native run."""
    rows = ['Backend 1/2: SYCL0']
    cases = [(weight, m, n, k)
             for weight, m, k in (('blk.1.ffn_latent_down.weight', 1024, 4096),
                                  ('blk.1.ffn_latent_up.weight', 4096, 1024))
             for n in (1, 2, 3)] + [('output.weight', 1024, 2, 4096)]
    for weight, m, n, k in cases:
        route = 'DMMV_ESIMD' if weight != 'output.weight' and n < 3 else 'MMVQ_REORDERED'
        line = route_line(route=route, weight=weight, src0_type='q8_0', reorder=1,
                          ne0=k, ne1=m, nb0=34, nb1=k // 32 * 34,
                          nb2=k // 32 * 34 * m, nb3=k // 32 * 34 * m,
                          src1_ne0=k, src1_ne1=n, src1_nb1=k * 4, src1_nb2=k * n * 4, src1_nb3=k * n * 4,
                          dst_ne0=m, dst_ne1=n, dst_nb1=m * 4, dst_nb2=m * n * 4, dst_nb3=m * n * 4)
        rows.append('BOLT_ROUTE ' + line.split('BOLT_ROUTE ', 1)[1])
        rows.append(f'  MUL_MAT(latent_dmmv2=1,weight={weight},type_a=q8_0,type_b=f32,m={m},n={n},k={k},'
                    'bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0): \x1b[1;32mOK\x1b[0m')
    rows += ['  7/7 tests passed', '  Backend SYCL0: OK', 'Backend 2/2: CPU',
             '  Skipping', '2/2 backends passed', 'OK']
    return '\n'.join(rows) + '\n'


def attention_native_output():
    rows = ['Backend 1/2: SYCL0']
    cases = []
    for weight, m, k in (
            ('blk.7.attn_q.weight', 4096, 4096),
            ('blk.7.attn_k.weight', 256, 4096),
            ('blk.7.attn_v.weight', 256, 4096),
            ('blk.7.attn_output.weight', 4096, 4096)):
        for n in (1, 2, 3):
            cases.append((weight, m, n, k))
    cases.append(('blk.7.unrelated.weight', 256, 2, 4096))
    for weight, m, n, k in cases:
        route = M['ATTENTION_NATIVE_CASES'][(weight, m, n, k)]
        line = route_line(route=route, weight=weight, src0_type='q8_0', reorder=1, operand_prec=-1,
                          ne0=k, ne1=m, nb0=34, nb1=k // 32 * 34,
                          nb2=k // 32 * 34 * m, nb3=k // 32 * 34 * m,
                          src1_ne0=k, src1_ne1=n, src1_nb1=k * 4, src1_nb2=k * n * 4, src1_nb3=k * n * 4,
                          dst_ne0=m, dst_ne1=n, dst_nb1=m * 4, dst_nb2=m * n * 4, dst_nb3=m * n * 4)
        rows.append('BOLT_ROUTE ' + line.split('BOLT_ROUTE ', 1)[1])
        rows.append(f'  MUL_MAT(attention_dmmv2=1,weight={weight},type_a=q8_0,type_b=f32,m={m},n={n},k={k},'
                    'bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0): \x1b[1;32mOK\x1b[0m')
    rows += ['  13/13 tests passed', '  Backend SYCL0: OK', 'Backend 2/2: CPU',
             '  Skipping', '2/2 backends passed', 'OK']
    return '\n'.join(rows) + '\n'



def attention_view_native_output():
    rows = ['Backend 1/2: SYCL0']
    cases = [
        ('blk.7.attn_output.weight', 4096, 1, 4096),
        ('blk.7.attn_output.weight', 4096, 2, 4096),
        ('blk.7.attn_output.weight', 4096, 3, 4096),
        ('blk.7.prefix.attn_output.weight', 4096, 2, 4096),
        ('blk.7.offset.attn_output.weight', 4096, 2, 4096),
        ('blk.7.unrelated.weight', 4096, 2, 4096),
    ]
    for weight, m, n, k in cases:
        route = M['ATTENTION_VIEW_NATIVE_CASES'][(weight, m, n, k)]
        line = route_line(route=route, weight=weight, src0_type='q8_0', reorder=1, operand_prec=-1,
                          ne0=k, ne1=m, nb0=34, nb1=k // 32 * 34,
                          nb2=k // 32 * 34 * m, nb3=k // 32 * 34 * m,
                          src1_ne0=k, src1_ne1=n, src1_nb1=k * 4, src1_nb2=k * n * 4, src1_nb3=k * n * 4,
                          dst_ne0=m, dst_ne1=n, dst_nb1=m * 4, dst_nb2=m * n * 4, dst_nb3=m * n * 4)
        rows.append('BOLT_ROUTE ' + line.split('BOLT_ROUTE ', 1)[1])
        rows.append(f'  MUL_MAT(attention_view_dmmv2=1,weight={weight},type_a=q8_0,type_b=f32,m={m},n={n},k={k},'
                    'bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0): \x1b[1;32mOK\x1b[0m')
    rows += ['  6/6 tests passed', '  Backend SYCL0: OK', 'Backend 2/2: CPU',
             '  Skipping', '2/2 backends passed', 'OK']
    return '\n'.join(rows) + '\n'


def cpp_body(source, name):
    start = re.search(r'^static [\w:<> *]+ ' + re.escape(name) + r'\(', source, re.M).start()
    start = source.index('{', start)
    depth = 0
    for end in range(start, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError('unterminated function ' + name)


def operand_fixture_row(weight, role, n, backend='0xabc'):
    k, m = A['OPERAND_SHAPES'][weight]
    rows, cols = (k, m) if role == 'weight' else ((k, n) if role == 'src1' else (m, n))
    if role == 'weight':
        type_name = 'f32' if weight == 'blk.1.ffn_gate_inp.weight' else 'q8_0'
        block, stride = ((4, rows * 4) if type_name == 'f32' else (34, rows // 32 * 34))
    else:
        type_name, block, stride = 'f32', 4, rows * 4
    reorder = int(weight == 'blk.1.ffn_latent_down.weight')
    size = stride * cols
    stem = 'router' if weight == 'blk.1.ffn_gate_inp.weight' else 'latent_down'
    return dict(phase='after' if role == 'dst' else 'before', weight=weight, role=role,
                backend_ctx=backend, queue='0x2000', device=0, n=n,
                type=type_name, reorder=reorder,
                ne0=rows, ne1=cols, ne2=1, ne3=1, nb0=block, nb1=stride, nb2=size, nb3=size,
                data='0x1000', base='0x1000', buffer_bytes=size, offset=0, bytes=size,
                file=f'{stem}.{role}.bin')

def operand_line(row):
    return 'BOLT_MOE_OPERAND ' + ' '.join(f'{k}={v}' for k,v in row.items())


def operand_cell(cell, label):
    # Real saved router-single2 routes/VERIFY/strides; ONLY six operand records
    # and their zero-filled buffers are synthetic. No synthetic route conversion.
    cell.mkdir(parents=True)
    directory = cell/'moe-operands'; directory.mkdir(mode=0o700)
    source, alignment_path = A['parent_reference'](label)
    alignment = json.loads(alignment_path.read_text())
    log = (source/'diagnostic-window.log').read_text()
    metadata = A['select_route_batches'](log, label)
    alignment['route_analysis'] = metadata
    backend = metadata['selected_batches'][0]['target_backend_ctx']
    n = 1 if label == 'S0' else 2
    operands = []
    for weight in A['OPERAND_SHAPES']:
        for role in ('weight','src1','dst'):
            row = operand_fixture_row(weight, role, n, backend)
            with (directory/row['file']).open('xb') as stream: stream.truncate(row['bytes'])
            operands.append(operand_line(row))
    lines, active, inserted = [], False, 0
    for line in log.splitlines():
        if 'BOLT_VERIFY batch ' in line:
            batch = A['records'](line)[0]
            active = batch['n'] == n and batch['pos'] == list(range(55,55+n))
            lines.append(line)
            if active:
                lines.extend(operands)
                inserted += 1
            continue
        lines.append(line)
    assert inserted == 1
    (cell/'diagnostic-window.log').write_text('\n'.join(lines)+'\n')
    (cell/'pre-diagnostic.log').write_text('')
    return cell, alignment


def full_operand_cell(cell, label):
    cell, alignment = operand_cell(cell, label)
    source, _ = A['parent_reference'](label)
    for name in ('ordinary.result.json','warmup.result.json','diagnostic.response.json',
                 'diagnostic.request.json','ordinary.request.json','warmup.request.json'):
        shutil.copy2(source/name, cell/name)
    shutil.copytree(source/'capture', cell/'capture')
    return cell, alignment


class Trace(unittest.TestCase):
    def setUp(self):
        (M['ROOT'] / 'worker-scratch' / 'tests').mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=M['ROOT'] / 'worker-scratch' / 'tests')
        self.root = Path(self.temp.name)
        self.log, self.tokens = fixture(self.root / 'S0')
        self.events = A['records'](self.log)

    def tearDown(self):
        self.temp.cleanup()

    def aligned(self, directory='S0', events=None, tokens=None):
        result = A['align'](events or self.events, tokens or self.tokens, self.root / directory)
        result.update(
            within_arm_text_passed=True,
            capture_path=str(self.root / directory),
            prior_reference=dict(
                reference_equivalence_required=True,
                ordinary_text_matches_prior_reference=True,
                diagnostic_text_matches_prior_reference=True,
                diagnostic_token_ids_match_prior_reference=True,
            ),
        )
        return result

    def patched_sources(self):
        scratch = self.root / 'patched-source'
        for name in M['SOURCE_SCOPE']:
            path = scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((M['PARENT'] / name).read_bytes())
        subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '-p1'], cwd=scratch,
                       input=Path('moe-operands.patch').read_bytes(), check=True, capture_output=True)
        self.assertEqual({str(p.relative_to(scratch)) for p in scratch.rglob('*') if p.is_file()}, set(M['SOURCE_SCOPE']))
        return [(scratch / M['SOURCE_SCOPE'][0]).read_text(), (M['PARENT']/'tests/test-backend-ops.cpp').read_text()]


    def test_build_controller_paths_match(self):
        script = Path('build-candidate.sh').read_text()
        assignments = dict(re.findall(r'^(R|PARENT|D)=(.+)$', script, re.M))
        self.assertEqual(assignments, {'R': str(M['ROOT']), 'PARENT': str(M['PARENT']),
                                       'D': str(M['CANDIDATE'])})
        self.assertNotEqual(M['PARENT'], M['CANDIDATE'])

    def test_syntax_and_patch_scope(self):
        for path in Path('.').glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
        subprocess.run(['bash', '-n', 'build-candidate.sh'], check=True)
        source, native = self.patched_sources()
        parent = (M['PARENT'] / M['SOURCE_SCOPE'][0]).read_text()
        self.assertEqual(native, (M['PARENT'] / 'tests/test-backend-ops.cpp').read_text())
        candidate_patch = Path('moe-operands.patch').read_text()
        self.assertEqual([s[6:] for s in candidate_patch.splitlines() if s.startswith('+++ b/')], M['SOURCE_SCOPE'])
        for name in ('ggml_sycl_op_mul_mat', 'ggml_sycl_mul_mat', 'opt_for_reorder', 'reorder_qw_nvfp4',
                     'bolt_route_record', 'ggml_backend_sycl_graph_compute_impl',
                     'ggml_sycl_mul_mat_id', 'ggml_sycl_mul_mat_id_iq2_single2'):
            self.assertEqual(cpp_body(source, name), cpp_body(parent, name), name)
        body = cpp_body(source, 'ggml_sycl_compute_forward')
        hook = body.split('case GGML_OP_MUL_MAT: {', 1)[1].split('case GGML_OP_MUL_MAT_ID:', 1)[0]
        self.assertLess(hook.index('ne[3]'), hook.index('bolt_moe_before'))
        self.assertLess(hook.index('bolt_moe_before'), hook.index('ggml_sycl_mul_mat(ctx'))
        self.assertLess(hook.index('ggml_sycl_mul_mat(ctx'), hook.index('bolt_moe_after'))
        self.assertEqual(hook.count('ggml_sycl_mul_mat(ctx'), 1)
        self.assertNotIn('BOLT_ROUTE', '\n'.join(s for s in candidate_patch.splitlines() if s.startswith('+')))
        tree = ast.parse(Path('test-trace.py').read_text())
        names = [n.name for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(len(names), len(set(names)))
        # Preserve the corrected independent-FIFO parser, not old containment.
        current = ast.parse(Path('analyze-trace.py').read_text())
        parent_parser = ast.parse((PARENT_TOP/'analyze-trace.py').read_text())
        for name in ('_verify_frames', '_route_stream', '_route_original_width',
                     '_target_group_width', '_expected_target_weights', '_assert_prefill_frame',
                     'assert_frozen_prefill_batches', 'route_frames', '_assert_target_frame_width',
                     'select_route_batches', 'assert_router_operations', 'assert_router_model_routes'):
            a = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in parent_parser.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b), name)

    def test_latent_source_eligibility_and_reorder_failure_guards(self):
        source, _ = self.patched_sources()
        helper = source.split('static bool ggml_sycl_latent_dmmv2_eligible(', 1)[1].split(
            'template <template <int> typename quantize_f>', 1)[0]
        for required in (
                '#ifdef GGML_SYCL_DMMV_HAS_ESIMD', 'dst->op != GGML_OP_MUL_MAT',
                'dst->src[0] != src0', 'dst->src[1] != src1', 'src0->type != GGML_TYPE_Q8_0',
                'src1->type != GGML_TYPE_F32', 'dst->type != GGML_TYPE_F32',
                'src1->ne[1] != 2', 'dst->ne[1] != 2', 'src0->ne[0] <= 0', 'src0->ne[1] <= 0',
                'src0->ne[0] % GGML_SYCL_DMMV_X != 0', 'src0->ne[0] % QK8_0 != 0',
                'ggml_nelements(src0) > INT_MAX', 'src1->ne[0] != src0->ne[0]', 'dst->ne[0] != src0->ne[1]',
                'src0->op != GGML_OP_NONE', 'GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_COMPUTE',
                '!src0->buffer', 'ggml_backend_buffer_is_sycl_split(src0->buffer)',
                '!ggml_backend_buffer_is_sycl(src0->buffer)', '!src0->buffer->context', '!src0->extra',
                '!src0->data', '(uintptr_t) src0->data % 16 != 0', '->device != ctx.device',
                '!g_ggml_sycl_enable_esimd', '!g_ggml_sycl_enable_optimize', '!ctx.opt_feature.reorder',
                '!should_reorder_tensor(ctx, dst)', '#ifdef GGML_SYCL_F16', '!ggml_sycl_src1_f16_ok(dst)',
                'constexpr char down[] = ".ffn_latent_down.weight"', 'constexpr char up[] = ".ffn_latent_up.weight"',
                'std::strcmp(src0->name + len - (sizeof(down) - 1), down) == 0',
                'std::strcmp(src0->name + len - (sizeof(up) - 1), up) == 0',
                'constexpr char attn_q[] = ".attn_q.weight"',
                'constexpr char attn_out[] = ".attn_output.weight"', 'if (src1->view_src)',
                'src1->op != GGML_OP_RESHAPE', 'src1->data != src1_base->data',
                'src1->buffer != src1_base->buffer', '!src1->buffer',
                'ggml_backend_buffer_is_sycl_split(src1->buffer)', '!ggml_backend_buffer_is_sycl(src1->buffer)',
                'src1_base->view_src', 'src1_base->view_offs', 'src1_base->type != GGML_TYPE_F32',
                '!ggml_is_contiguous(src1_base)', 'ggml_nelements(src1) != ggml_nelements(src1_base)',
                'ggml_nbytes(src1) != ggml_nbytes(src1_base)'):
            self.assertIn(required, helper)
        for tensor in ('src0', 'src1', 'dst'):
            for guard in (f'{tensor}->ne[2] != 1', f'{tensor}->ne[3] != 1', f'{tensor}->view_src',
                          f'{tensor}->view_offs', f'!ggml_is_contiguous({tensor})'):
                self.assertIn(guard, helper)
        first_guard = helper.split('if (dst->op != GGML_OP_MUL_MAT', 1)[1].split(') {', 1)[0]
        self.assertNotIn('src1->view_src', first_guard)
        self.assertIn('src1->view_offs', first_guard)
        self.assertIn('return false;\n#endif\n}', helper)  # ESIMD not compiled: fail closed
        selector = source.split('static void ggml_sycl_mul_mat(', 1)[1].split('// {mul_mat(gate)', 1)[0]
        bootstrap = selector.split('bool use_latent_dmmv2 = ', 1)[1].split('#ifdef GGML_SYCL_F16', 1)[0]
        self.assertIn('ggml_sycl_latent_dmmv2_eligible(ctx, src0, src1, dst)', bootstrap)
        self.assertIn('if (use_latent_dmmv2)', bootstrap)
        self.assertLess(bootstrap.index('opt_for_reorder('), bootstrap.index('use_latent_dmmv2 = extra && extra->optimized_feature.reorder'))
        self.assertIn('can_use_dequantize_mul_mat_vec(src0, src1, dst) || use_latent_dmmv2', bootstrap)
        self.assertIn('use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);', selector)
        old = (M['PARENT'] / M['SOURCE_SCOPE'][0]).read_text()
        for start, end in (('static bool can_use_dequantize_mul_mat_vec(', 'static bool can_use_mul_mat_vec_q('),
                           ('// Head-only counterfactual:', '    if (!split && src0->type == GGML_TYPE_F16')):
            self.assertEqual(source.split(start, 1)[1].split(end, 1)[0], old.split(start, 1)[1].split(end, 1)[0])

    def test_wrapper_column_strides_and_unchanged_scalar_copy_events(self):
        source, _ops = self.patched_sources()
        old = (M['PARENT'] / M['SOURCE_SCOPE'][0]).read_text()
        def wrapper(text):
            return text.split('static void ggml_sycl_op_mul_mat(', 1)[1].split('\ncatch (sycl::exception', 1)[0]
        before, after = wrapper(old), wrapper(source)
        self.assertEqual(before, after)
        block = after.split('// do the computation', 1)[1].split('// copy dst to host', 1)[0]
        for required in ('op == ggml_sycl_op_dequantize_mul_mat_vec && src1_ncols == 2',
                         'ggml_sycl_latent_dmmv2_eligible(ctx, src0, src1, dst)',
                         'GGML_ASSERT(src0_extra && src0_extra->optimized_feature.reorder)',
                         'for (int64_t col = 0; col < src1_ncols; ++col)',
                         'src1_ddf_i + col * ne10, src1_ddq_i',
                         'dst_dd_i + col * (dst_on_device ? ne0 : row_diff)',
                         'dev[i].row_low, dev[i].row_high, 1, src1_padded_col_size, stream'):
            self.assertIn(required, block)
        # Element offsets (not bytes); both destination layouts retain an existing row-low offset.
        for on_device, stride in ((True, 1024), (False, 256)):
            base, row_low, ne10 = 11, 17, 4096
            destinations = [base + row_low + col * stride for col in range(2)]
            self.assertEqual(destinations, [28, 1052 if on_device else 284])
            self.assertEqual([base + col * ne10 for col in range(2)], [11, 4107])

    def test_native_fixtures_real_names_shapes_fresh_weights_and_tolerance(self):
        _, source = self.patched_sources()
        latent_fixture = source.split('struct test_mul_mat_latent_dmmv2', 1)[1].split('struct test_mul_mat_attention_dmmv2', 1)[0]
        attn_fixture = source.split('struct test_mul_mat_attention_dmmv2', 1)[1].split('struct test_mul_mat_attention_view_dmmv2', 1)[0]
        view_fixture = source.split('struct test_mul_mat_attention_view_dmmv2', 1)[1].split('// Six isolated F32 CPU-reference cases', 1)[0]
        for fixture_source, tag in ((latent_fixture, 'latent_dmmv2=1,weight='), (attn_fixture, 'attention_dmmv2=1,weight=')):
            self.assertIn('test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, m, n, k, {1, 1}, {1, 1})', fixture_source)
            self.assertIn('ggml_set_name(a, weight)', fixture_source)
            self.assertIn('ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m)', fixture_source)
            self.assertIn('return ggml_mul_mat(ctx, a, b)', fixture_source)
            self.assertIn(tag, fixture_source)
            self.assertNotIn('max_nmse_err', fixture_source)
        self.assertIn('attention_view_dmmv2=1,weight=', view_fixture)
        for required in ('ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 32, partial ? 3 : n)',
                         'ggml_view_2d(ctx, base, 4096, 2, base->nb[2], offset)',
                         'ggml_reshape_2d(ctx, view, 4096, 2)',
                         'ggml_reshape_2d(ctx, base, 4096, n)',
                         'blk.7.prefix.attn_output.weight', 'blk.7.offset.attn_output.weight'):
            self.assertIn(required, source)
        self.assertNotIn('max_nmse_err', view_fixture)
        self.assertIn('return 5e-4;', source.split('struct test_mul_mat :', 1)[1].split('struct test_mul_mat_nvfp4_reorder', 1)[0])
        latent_cases = source.split('// Exactly seven cases', 1)[1].split('// Exactly thirteen cases', 1)[0]
        attn_cases = source.split('// Exactly thirteen cases', 1)[1].split('// Experimental NVFP4 reorder', 1)[0]
        for invocation in ('("blk.1.ffn_latent_down.weight", 1024, n, 4096)',
                           '("blk.1.ffn_latent_up.weight", 4096, n, 1024)',
                           '("output.weight", 1024, 2, 4096)'):
            self.assertIn('new test_mul_mat_latent_dmmv2' + invocation, latent_cases)
        for invocation in ('("blk.7.attn_q.weight", 4096, n, 4096)',
                           '("blk.7.attn_k.weight", 256, n, 4096)',
                           '("blk.7.attn_v.weight", 256, n, 4096)',
                           '("blk.7.attn_output.weight", 4096, n, 4096)'):
            self.assertIn('new test_mul_mat_attention_dmmv2' + invocation, attn_cases)
        self.assertIn('new test_mul_mat_attention_dmmv2("blk.7.unrelated.weight", 256, 2, 4096)', attn_cases)
        view_cases = source.split('// Exactly six cases', 1)[1].split('// Experimental NVFP4 reorder', 1)[0]
        self.assertEqual(view_cases.count('new test_mul_mat_attention_view_dmmv2'), 4)
        for invocation in ('("blk.7.attn_output.weight", n)',
                           '("blk.7.prefix.attn_output.weight", 2)',
                           '("blk.7.offset.attn_output.weight", 2)',
                           '("blk.7.unrelated.weight", 2)'):
            self.assertIn('new test_mul_mat_attention_view_dmmv2' + invocation, view_cases)

    def test_capture_reserve_and_parent_results(self):
        self.assertEqual(M['CAPTURE_RESERVE_BYTES'], 4 * 1024**3)
        self.assertEqual(M['PARENT_RESULTS'].name, '20261017-nemotron35-bolt-lilicorr-router-single2')
        self.assertEqual(len(M['ARTIFACTS']), 5)

    def test_exact128_rows_real_positions_and_raw_argmax(self):
        result = self.aligned()
        self.assertEqual(len(result['samples']), 128)
        self.assertEqual(result['samples'][0]['batch_row'], 54)
        self.assertEqual(result['samples'][115]['position'], 169)
        self.assertEqual(result['samples'][115]['raw_argmax'], 6833)
        self.assertEqual(result['samples'][0]['selected'], 1000)  # raw != post-sampler is allowed

    def test_prompt_chunk_tail_boundaries(self):
        log, tokens = fixture(self.root / 'chunked', chunked_prompt=True)
        events = A['records'](log)
        result = self.aligned('chunked', events, tokens)
        self.assertEqual([b['n'] for b in result['batches'][:3]], [5, 46, 4])
        self.assertEqual(result['prompt_ids'], list(range(55)))
        self.assertEqual(result['samples'][0]['batch_row'], 3)
        self.assertEqual([r['p0'] for r in result['removals']], [0, 5, 51])
        second = [i for i, e in enumerate(events) if e['event'] == 'remove'][1]
        for key, value in (('p0', 4), ('p0', 6), ('p1', 9), ('rollback', 1),
                           ('ctx', '0xbad'), ('slot', 1)):
            bad = copy.deepcopy(events)
            bad[second][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                self.aligned('chunked', bad, tokens)
        bad = copy.deepcopy(events)
        boundary = copy.deepcopy(bad[second])
        boundary['p0'] = 55
        first_emit = next(i for i, e in enumerate(bad) if e['event'] == 'emit')
        bad.insert(first_emit + 1, boundary)
        with self.assertRaises(AssertionError):
            self.aligned('chunked', bad, tokens)

    def test_greedy_verification_acceptance_and_actual_removal(self):
        log, tokens = fixture(self.root / 'C0', speculative=True)
        result = self.aligned('C0', A['records'](log), tokens)
        rejected = [r for r in result['removals'] if r['rollback']]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['removed_batch_positions'], [56])
        self.assertEqual(result['samples'][3]['batch_row'], 1)

    def test_framing_errors_fail_closed(self):
        for bad in ('BOLT_VERIFY error op=write_close errno=28\n', 'BOLT_VERIFY\n',
                    'BOLT_VERIFY decoded ctx=0x1 ret=0 ret=0\n',
                    'BOLT_VERIFY decoded ctx=0x1 ret=0 extra=3\n',
                    'BOLT_VERIFY decoded ctx=0x1 ret=0 BOLT_VERIFY sample\n'):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                A['records'](bad)
        self.assertEqual(A['records']('unrelated\n' + self.log), self.events)

    def test_capture_path_bounds_symlinks_and_sizes(self):
        sample = next(e for e in self.events if e['event'] == 'sample')
        for name in ('../row-000000', '/tmp/row-000000', 'row-00000', 'row-000000/extra'):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                A['raw_bytes'](self.root / 'S0', dict(sample, file=name))
        link = self.root / 'S0/row-999999'
        link.symlink_to(self.root / 'S0/row-000000')
        with self.assertRaises(AssertionError):
            A['raw_bytes'](self.root / 'S0', dict(sample, file=link.name))
        path = self.root / 'S0/row-000000'
        for size in (0, 32767, 32769):
            path.write_bytes(b'\0' * size)
            with self.assertRaises(AssertionError):
                A['raw_bytes'](self.root / 'S0', sample)
        for n in (0, -1, 1_000_001):
            with self.assertRaises(AssertionError):
                A['raw_bytes'](self.root / 'S0', dict(sample, vocab=n))

    def test_backend_uncaptured_and_duplicate_block_attribution(self):
        for status in ('backend', 'uncaptured'):
            events = copy.deepcopy(self.events)
            next(e for e in events if e['event'] == 'sample')['status'] = status
            with self.assertRaises(AssertionError):
                self.aligned(events=events)
        events = copy.deepcopy(self.events)
        samples = [e for e in events if e['event'] == 'sample']
        samples[1]['file'] = samples[0]['file']
        with self.assertRaises(AssertionError):
            self.aligned(events=events)

    def test_alignment_failures(self):
        for event, key, value in (('sample', 'idx', 0), ('sample', 'selected', 999),
                                  ('sample', 'ctx', '0x999'), ('decoded', 'ret', -1),
                                  ('emit', 'ordinal', 2), ('single', 'row', 0)):
            events = copy.deepcopy(self.events)
            next(e for e in events if e['event'] == event)[key] = value
            with self.subTest(event=event, key=key), self.assertRaises(AssertionError):
                self.aligned(events=events)
        with self.assertRaises(AssertionError):
            self.aligned(tokens=self.tokens[:-1])

    def test_batch_history_and_output_flag_failures(self):
        for key, value in (('tokens', [999]), ('pos', [123]), ('outputs', [0])):
            events = copy.deepcopy(self.events)
            [e for e in events if e['event'] == 'batch'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.aligned(events=events)

    def test_negative_row_index_resolves_actual_last_output(self):
        events = copy.deepcopy(self.events)
        next(e for e in events if e['event'] == 'sample')['idx'] = -1
        next(e for e in events if e['event'] == 'single')['row'] = -1
        self.assertEqual(self.aligned(events=events)['samples'][0]['batch_row'], 54)

    def test_rollback_and_verifier_guards(self):
        log, tokens = fixture(self.root / 'C0', speculative=True)
        for kind, key, value in (('verify', 'accepted', 1), ('verify', 'rows', [1, 0]),
                                  ('verify', 'replay', 1), ('remove', 'p0', 100)):
            events = A['records'](log)
            target = next(e for e in events if e['event'] == kind and (kind != 'remove' or e['kind'] == 'verify'))
            target[key] = value
            with self.subTest(kind=kind, key=key), self.assertRaises(AssertionError):
                self.aligned('C0', events, tokens)

    def test_first_difference_and_history_cutoff(self):
        log, tokens = fixture(self.root / 'C0', speculative=True, diverge=True)
        left, right = self.aligned(), self.aligned('C0', A['records'](log), tokens)
        file = self.root / 'C0/row-000116'  # ONLY after the differing output116; must not compare
        file.write_bytes(b'\0' * 32768)
        result = A['compare'](left, right)
        self.assertEqual(result['compared_through_ordinal'], 116)
        self.assertIsNone(result['first_score_difference'])
        self.assertTrue(result['reference_equivalence_required'])
        path = self.root / 'C0/row-000115'
        data = bytearray(path.read_bytes()); data[40] ^= 1; path.write_bytes(data)
        result = A['compare'](left, right)
        self.assertEqual(result['first_score_difference']['ordinal'], 116)
        self.assertEqual(result['first_score_difference']['first_differing_float_id'], 10)
        self.assertTrue(result['first_score_difference']['C0_rollback_preceded'])
        path = self.root / 'C0/row-000000'
        data = bytearray(path.read_bytes()); data[80] ^= 1; path.write_bytes(data)
        result = A['compare'](left, right)
        self.assertEqual(result['first_score_difference']['ordinal'], 1)
        self.assertFalse(result['first_score_difference']['C0_rollback_preceded'])
        self.assertFalse(result['near_token116'][-1]['identical_prior_history'])

    def test_within_arm_text_and_prior_reference_reporting(self):
        ref = M['WIDTH1'] / 'run-r1/A0'
        ordinary = json.loads((ref / 'repeat-00.result.json').read_text())
        diagnostic = json.loads((ref / 'diagnostic.response.json').read_text())
        diagnostic['choices'][0]['message']['content'] = ordinary['text']
        A['within_arm_text_consistency'](ordinary, diagnostic)
        flags = A['prior_reference_flags'](ordinary, diagnostic, ordinary, diagnostic)
        self.assertTrue(flags['reference_equivalence_required'])
        self.assertTrue(flags['ordinary_text_matches_prior_reference'])
        changed = copy.deepcopy(diagnostic)
        changed['choices'][0]['message']['content'] += '!'
        with self.assertRaises(AssertionError):
            A['within_arm_text_consistency'](ordinary, changed)
        drifted = dict(ordinary, text='intended backend change')
        flags = A['prior_reference_flags'](drifted, diagnostic, ordinary, diagnostic)
        self.assertFalse(flags['ordinary_text_matches_prior_reference'])
        left = self.aligned()
        with self.assertRaises(AssertionError):
            A['compare'](dict(left, within_arm_text_passed=False), left)

    def test_parameters_mounts_frozen_payload_and_order(self):
        before = copy.deepcopy(BASE)
        self.assertEqual(M['ORDER'], ('S0', 'C0'))
        self.assertEqual(M['PREFIX'], 'bolt-lilicorr-router-control-operands-r1-')
        for label in M['ORDER']:
            capture = M['ROOT'] / 'run-r1' / label / 'capture'
            argv = M['make_create'](BASE, label, IMAGE, capture, OLD)
            self.assertIn(str(M['CANDIDATE']) + ':/src:ro', argv)
            self.assertIn(str(capture) + ':/capture', argv)
            for key, value in M['ENV']:
                self.assertEqual(argv.count(key + '=' + value), 1)
            for script in ('run-trace.py', 'analyze-trace.py'):
                self.assertIn(str(M['ROOT'] / script) + ':/iq2-single2-gate/' + script + ':ro', argv)
            self.assertIn('bash /native-outprod.sh;', argv[-1])
            self.assertIn("test -b SYCL0 -o MUL_MAT_ID -p 'type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,'", argv[-1])
            self.assertIn('BOLT_VERIFY_IQ2_NATIVE_PASSED', argv[-1])
            self.assertIn("test -b SYCL0 -o MUL_MAT -p '^latent_dmmv2=1,'", argv[-1])
            self.assertIn('set -eo pipefail', argv[-1])
            self.assertLess(argv[-1].index('BOLT_VERIFY_IQ2_NATIVE_PASSED'), argv[-1].index(M['Q8_NATIVE_MARKER']))
            self.assertIn("test -b SYCL0 -o MUL_MAT -p '^attention_dmmv2=1,'", argv[-1])
            self.assertLess(argv[-1].index(M['Q8_NATIVE_MARKER']), argv[-1].index(M['ATTENTION_NATIVE_MARKER']))
            self.assertIn("test -b SYCL0 -o MUL_MAT -p '^attention_view_dmmv2=1,'", argv[-1])
            self.assertLess(argv[-1].index(M['ATTENTION_NATIVE_MARKER']), argv[-1].index(M['ATTENTION_VIEW_NATIVE_MARKER']))
            self.assertLess(argv[-1].index(M['ATTENTION_VIEW_NATIVE_MARKER']), argv[-1].index(M['MARKER']))
            self.assertEqual(argv[-1].rsplit('exec ', 1)[1], BASE[-1].rsplit('exec ', 1)[1] + ' ' + M['shlex'].join(M['spec_for_arm'](label)))
        self.assertEqual(before, BASE)
        self.assertEqual(M['spec_for_arm']('S0')[3:6], ['1', '--spec-draft-n-min', '2'])
        self.assertEqual(M['spec_for_arm']('C0')[3:6], ['1', '--spec-draft-n-min', '0'])
        with self.assertRaises(AssertionError):
            M['make_create'](BASE, 'S0', IMAGE, Path('/tmp/escape'), OLD)
        payload = json.loads((M['WIDTH1'] / 'run-r1/frozen-payload.json').read_text())
        diag = json.loads((M['WIDTH1'] / 'run-r1/C0/diagnostic.request.json').read_text())
        self.assertEqual(OLD['diagnostic_payload'](payload), diag)
        self.assertFalse(diag['backend_sampling'])

    def test_legacy_native_dispatch_guard_only(self):
        prefix = BASE[-1].rsplit('exec ', 1)[0]
        changed = M['legacy_native_prefix'](prefix)
        self.assertEqual(prefix.count('test "$count" -eq 23'), 1)
        self.assertEqual(changed, prefix.replace('test "$count" -eq 23', 'test "$count" -eq 0', 1))
        for malformed in (changed, prefix + prefix):
            with self.assertRaises(AssertionError):
                M['legacy_native_prefix'](malformed)

    def test_native_and_startup_guards_on_saved_real_log(self):
        raw = (M['WIDTH1'] / 'run-r1/C0/native-and-startup.log').read_text()
        raw = '\n'.join(line for line in raw.split('\n') if 'call ggml_sycl_ssm_scan_prefill:' not in line)
        log = raw.replace(
            OLD['SERVER_MARKER'],
            iq2_native_output() + 'BOLT_VERIFY_IQ2_NATIVE_PASSED\n' + q8_native_output()
            + M['Q8_NATIVE_MARKER'] + '\n' + attention_native_output()
            + M['ATTENTION_NATIVE_MARKER'] + '\n' + attention_view_native_output()
            + M['ATTENTION_VIEW_NATIVE_MARKER'] + '\n' + router_native_output()
            + M['ROUTER_NATIVE_MARKER'] + '\n' + M['MARKER'])
        M['check_startup'](log, 'C0', OLD, NATIVE)
        suppressed = log.replace('n_max=1, n_min=0,', 'n_max=1, n_min=2,')
        M['check_startup'](suppressed, 'S0', OLD, NATIVE)
        for source, label in ((log, 'C0'), (suppressed, 'S0')):
            stamped = '\n'.join('2026-10-17T20:10:00.123456789Z ' + line for line in source.splitlines())
            M['check_startup'](stamped, label, OLD, NATIVE)
        for broken in (
                log.replace('BOLT_VERIFY_IQ2_NATIVE_PASSED', 'missing'),
                log.replace('  6/6 tests passed', '  6/6 tests passed\n  6/6 tests passed'),
                log.replace('Backend 2/2: CPU', 'Backend 2/2: UNEXPECTED_GPU'),
                log.replace('  6/6 tests passed', '  5/6 tests passed'),
                log.replace('  6/6 tests passed', 'SKIPPED\n  6/6 tests passed'),
                log.replace('BOLT_VERIFY_IQ2_NATIVE_PASSED', 'not supported\nBOLT_VERIFY_IQ2_NATIVE_PASSED'),
                log.replace('ids_nb1=64 ', 'ids_nb1=36 ', 1),
                log.replace('n_max=1, n_min=0,', 'n_max=4, n_min=0,'),
                log + '\nBOLT_VERIFY error op=mkstemp errno=28\n'):
            with self.assertRaises(AssertionError):
                M['check_startup'](broken, 'C0', OLD, NATIVE)

    def test_iq2_native_shell_gate_counts_once_and_fails_closed(self):
        transcript = self.root / 'iq2-native-input.log'
        captured = self.root / 'iq2-native-captured.log'
        suffix = M['IQ2_NATIVE_COMMAND'].split(' 2>&1 | ', 1)[1]
        suffix = suffix.replace('/tmp/iq2-xxs-native.log', shlex.quote(str(captured)))
        suffix = suffix.replace('/iq2-single2-gate/run-trace.py', str(Path('run-trace.py').resolve()))
        for output, status, succeeds in (
            (iq2_native_output(), 0, True),
            (iq2_native_output() + '  6/6 tests passed\n', 0, False),
            (iq2_native_output().replace('\x1b[1;32mOK', 'SKIPPED', 1), 0, False),
            (iq2_native_output() + 'not supported\n', 0, False),
            (iq2_native_output(), 1, False),
        ):
            transcript.write_text(output)
            producer = f'(cat {shlex.quote(str(transcript))}; exit {status})'
            command = 'set -euo pipefail; ' + producer + ' 2>&1 | ' + suffix
            result = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
            if succeeds:
                self.assertEqual(result.stdout.count('6/6 tests passed'), 1)
                M['check_iq2_native_gate']('BOLT_LILICORR_OUTPROD_NATIVE_PASSED\n' + result.stdout)
            else:
                self.assertNotIn('BOLT_VERIFY_IQ2_NATIVE_PASSED', result.stdout)

    def test_q8_native_gate_raw_and_docker_timestamps(self):
        output = q8_native_output()
        M['check_q8_native_output'](output)
        log = 'BOLT_VERIFY_IQ2_NATIVE_PASSED\n' + output + M['Q8_NATIVE_MARKER'] + '\n'
        M['check_q8_native_gate'](log)
        for stamp in ('2026-10-17T20:10:00Z ', '2026-10-17T20:10:00.123456789Z '):
            stamped = ''.join(stamp + line + '\n' for line in log.splitlines())
            M['check_q8_native_gate'](stamped)
        for broken in (log.replace(M['Q8_NATIVE_MARKER'], 'missing'), log + M['Q8_NATIVE_MARKER'],
                       log.replace('BOLT_VERIFY_IQ2_NATIVE_PASSED', 'missing')):
            with self.assertRaises(AssertionError):
                M['check_q8_native_gate'](broken)

    def test_q8_native_gate_counts_failures_and_backend_skips(self):
        good = q8_native_output()
        M['check_q8_native_output'](good)
        first_case = next(line for line in good.splitlines(True) if '  MUL_MAT(' in line)
        for bad in (good.replace(first_case, ''), good + first_case,
                    good + '  7/7 tests passed\n', good.replace('7/7 tests passed', '6/7 tests passed'),
                    good.replace('\x1b[1;32mOK', 'FAIL', 1),
                    good.replace('\x1b[1;32mOK', 'not supported [SYCL0] (expected)', 1),
                    good + 'skipping large tensors for speed\n', good + 'UNSUPPORTED\n',
                    good + 'UNEXPECTED\n', good + 'ERR = 0.1\n', good + 'failed\n',
                    good.replace('Backend 2/2: CPU', 'Backend 2/2: OTHER_GPU'),
                    good + 'Backend 3/3: CPU\n  Skipping\n',
                    good.replace('Backend SYCL0: OK', 'Backend SYCL0: FAIL'),
                    good.replace('2/2 backends passed', '1/2 backends passed'),
                    good.replace('type_a=q8_0', 'type_a=q4_0', 1),
                    good.replace('m=1024,n=2,k=4096,', 'm=1024,n=4,k=4096,', 1)):
            with self.subTest(bad=bad[-160:]), self.assertRaises(AssertionError):
                M['check_q8_native_output'](bad)

    def test_q8_native_actual_route_type_column_reorder_and_coverage(self):
        good = q8_native_output()
        M['check_q8_native_output'](good)
        native_routes = [line for line in good.splitlines(True) if line.startswith('BOLT_ROUTE ')]
        self.assertEqual(len(native_routes), 7)
        for line in native_routes:
            with self.subTest(missing=line[:100]), self.assertRaises(AssertionError):
                M['check_q8_native_output'](good.replace(line, ''))
            # Start each negative from a validated positive (not an already broken fixture).
            for key, value in (('src0_type', 'q4_0'), ('src1_type', 'f16'), ('dst_type', 'f16'),
                               ('src1_ne1', '4'), ('reorder', '0'), ('op', 'MUL_MAT_ID'),
                               ('src1_ne2', '2'), ('src1_nb1', '1'), ('dst_nb1', '1'), ('device', '1')):
                fields = line.split()
                changed = ' '.join(f'{key}={value}' if field.startswith(key + '=') else field
                                   for field in fields) + '\n'
                with self.subTest(key=key, route=line[:100]), self.assertRaises(AssertionError):
                    M['check_q8_native_output'](good.replace(line, changed))
            wrong_route = line.replace('route=DMMV_ESIMD', 'route=MMVQ_REORDERED') if 'route=DMMV_ESIMD' in line else line.replace('route=MMVQ_REORDERED', 'route=DMMV_ESIMD')
            with self.assertRaises(AssertionError):
                M['check_q8_native_output'](good.replace(line, wrong_route))
        self.assertRaises(AssertionError, M['check_q8_native_output'],
                          ''.join(line for line in good.splitlines(True) if not line.startswith('BOLT_ROUTE ')))
        self.assertRaises(AssertionError, M['check_q8_native_output'], good.replace('src1_ne1=2 ', '', 1))

    def test_q8_native_shell_pipeline_blocks_server_on_failure(self):
        transcript = self.root / 'q8-native-input.log'
        captured = self.root / 'q8-native-captured.log'
        stub = self.root / 'test-backend-ops-stub'
        good = q8_native_output()
        first_case = next(line for line in good.splitlines(True) if '  MUL_MAT(' in line)
        first_route = next(line for line in good.splitlines(True) if line.startswith('BOLT_ROUTE '))
        command = M['Q8_NATIVE_COMMAND'].replace('/src/build-sycl/bin/test-backend-ops', shlex.quote(str(stub)))
        command = command.replace('/tmp/q8-latent-dmmv2-native.log', str(captured))
        command = command.replace('/iq2-single2-gate/run-trace.py', str(Path('run-trace.py').resolve()))
        for output, status, succeeds in (
                (good, 0, True), (good, 1, False),
                (good.replace(first_case, ''), 0, False), (good + first_case, 0, False),
                (good + '  7/7 tests passed\n', 0, False),
                (good.replace('7/7 tests passed', '6/7 tests passed'), 0, False),
                (good + 'not supported\n', 0, False), (good + 'SKIPPED\n', 0, False),
                (good.replace('Backend 2/2: CPU', 'Backend 2/2: GPU1'), 0, False),
                (good.replace(first_route, ''), 0, False),
                (good.replace('src1_type=f32', 'src1_type=f16', 1), 0, False),
                (good.replace('src1_ne1=2 ', 'src1_ne1=4 ', 1), 0, False)):
            transcript.write_text(output)
            stub.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                            'test "$#" -eq 7; test "$1" = test; test "$3" = SYCL0; '
                            'test "$5" = MUL_MAT; test "$7" = "^latent_dmmv2=1,"\n'
                            f'cat {shlex.quote(str(transcript))}\nexit {status}\n')
            stub.chmod(0o700)
            result = subprocess.run(['bash', '-c', 'set -euo pipefail; ' + command + 'echo SERVER_WOULD_START'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
            if succeeds:
                self.assertEqual(result.stdout.count('7/7 tests passed'), 1)
                self.assertIn('SERVER_WOULD_START', result.stdout)
                M['check_q8_native_gate']('BOLT_VERIFY_IQ2_NATIVE_PASSED\n' + result.stdout)
            else:
                self.assertNotIn(M['Q8_NATIVE_MARKER'], result.stdout)
                self.assertNotIn('SERVER_WOULD_START', result.stdout)


    def test_attention_native_gate_raw_and_docker_timestamps(self):
        output = attention_native_output()
        M['check_attention_native_output'](output)
        log = M['Q8_NATIVE_MARKER'] + '\n' + output + M['ATTENTION_NATIVE_MARKER'] + '\n'
        M['check_attention_native_gate']('BOLT_VERIFY_IQ2_NATIVE_PASSED\n' + log)
        for stamp in ('2026-10-17T20:10:00Z ', '2026-10-17T20:10:00.123456789Z '):
            stamped = ''.join(stamp + line + '\n' for line in log.splitlines())
            M['check_attention_native_gate']('BOLT_VERIFY_IQ2_NATIVE_PASSED\n' + stamped)
        for broken in (log.replace(M['ATTENTION_NATIVE_MARKER'], 'missing'), log + M['ATTENTION_NATIVE_MARKER'],
                       log.replace(M['Q8_NATIVE_MARKER'], 'missing')):
            with self.assertRaises(AssertionError):
                M['check_attention_native_gate']('BOLT_VERIFY_IQ2_NATIVE_PASSED\n' + broken)

    def test_attention_native_gate_counts_failures_and_backend_skips(self):
        good = attention_native_output()
        M['check_attention_native_output'](good)
        first_case = next(line for line in good.splitlines(True) if '  MUL_MAT(' in line)
        for bad in (good.replace(first_case, ''), good + first_case,
                    good + '  13/13 tests passed\n', good.replace('13/13 tests passed', '12/13 tests passed'),
                    good.replace('\x1b[1;32mOK', 'FAIL', 1),
                    good.replace('\x1b[1;32mOK', 'not supported [SYCL0] (expected)', 1),
                    good + 'skipping large tensors for speed\n', good + 'UNSUPPORTED\n',
                    good + 'UNEXPECTED\n', good + 'ERR = 0.1\n', good + 'failed\n',
                    good.replace('Backend 2/2: CPU', 'Backend 2/2: OTHER_GPU'),
                    good + 'Backend 3/3: CPU\n  Skipping\n',
                    good.replace('Backend SYCL0: OK', 'Backend SYCL0: FAIL'),
                    good.replace('2/2 backends passed', '1/2 backends passed'),
                    good.replace('type_a=q8_0', 'type_a=q4_0', 1),
                    good.replace('m=4096,n=1,k=4096', 'm=4096,n=4,k=4096', 1)):
            with self.subTest(bad=bad[-160:]), self.assertRaises(AssertionError):
                M['check_attention_native_output'](bad)

    def test_attention_native_actual_route_type_column_reorder_and_coverage(self):
        good = attention_native_output()
        M['check_attention_native_output'](good)
        native_routes = [line for line in good.splitlines(True) if line.startswith('BOLT_ROUTE ')]
        self.assertEqual(len(native_routes), 13)
        for line in native_routes:
            with self.subTest(missing=line[:100]), self.assertRaises(AssertionError):
                M['check_attention_native_output'](good.replace(line, ''))
            for key, value in (('src0_type', 'q4_0'), ('src1_type', 'f16'), ('dst_type', 'f16'),
                               ('src1_ne1', '4'), ('reorder', '0'), ('op', 'MUL_MAT_ID'),
                               ('operand_prec', '0'), ('src1_ne2', '2'), ('src1_nb1', '1'),
                               ('dst_nb1', '1'), ('device', '1')):
                fields = line.split()
                changed = ' '.join(f'{key}={value}' if field.startswith(key + '=') else field
                                   for field in fields) + '\n'
                with self.subTest(key=key, route=line[:100]), self.assertRaises(AssertionError):
                    M['check_attention_native_output'](good.replace(line, changed))
            wrong_route = (line.replace('route=DMMV_ESIMD', 'route=MMVQ_REORDERED')
                           if 'route=DMMV_ESIMD' in line else line.replace('route=MMVQ_REORDERED', 'route=DMMV_ESIMD'))
            with self.assertRaises(AssertionError):
                M['check_attention_native_output'](good.replace(line, wrong_route))
        self.assertRaises(AssertionError, M['check_attention_native_output'],
                          ''.join(line for line in good.splitlines(True) if not line.startswith('BOLT_ROUTE ')))
        n3_route = next(line for line in native_routes if 'blk.7.attn_q.weight' in line and 'src1_ne1=3' in line)
        with self.assertRaises(AssertionError):
            M['check_attention_native_output'](good.replace(n3_route, n3_route.replace('route=MMVQ_REORDERED', 'route=DMMV_ESIMD')))
        n2_attn_route = next(line for line in native_routes
                             if 'blk.7.attn_k.weight' in line and 'src1_ne1=2' in line)
        with self.assertRaises(AssertionError):
            M['check_attention_native_output'](good.replace(n2_attn_route,
                                                             n2_attn_route.replace('route=DMMV_ESIMD', 'route=MMVQ_REORDERED')))
        unrelated_route = next(line for line in native_routes if 'blk.7.unrelated.weight' in line)
        with self.assertRaises(AssertionError):
            M['check_attention_native_output'](good.replace(unrelated_route,
                                                             unrelated_route.replace('route=MMVQ_REORDERED', 'route=DMMV_ESIMD')))

    def test_attention_native_shell_pipeline_blocks_server_on_failure(self):
        transcript = self.root / 'attn-native-input.log'
        captured = self.root / 'attn-native-captured.log'
        stub = self.root / 'test-backend-ops-attn-stub'
        good = attention_native_output()
        first_case = next(line for line in good.splitlines(True) if '  MUL_MAT(' in line)
        first_route = next(line for line in good.splitlines(True) if line.startswith('BOLT_ROUTE '))
        command = M['ATTENTION_NATIVE_COMMAND'].replace('/src/build-sycl/bin/test-backend-ops', shlex.quote(str(stub)))
        command = command.replace('/tmp/q8-attention-dmmv2-native.log', str(captured))
        command = command.replace('/iq2-single2-gate/run-trace.py', str(Path('run-trace.py').resolve()))
        for output, status, succeeds in (
                (good, 0, True), (good, 1, False),
                (good.replace(first_case, ''), 0, False), (good + first_case, 0, False),
                (good + '  13/13 tests passed\n', 0, False),
                (good.replace('13/13 tests passed', '12/13 tests passed'), 0, False),
                (good + 'not supported\n', 0, False), (good + 'SKIPPED\n', 0, False),
                (good.replace('Backend 2/2: CPU', 'Backend 2/2: GPU1'), 0, False),
                (good.replace(first_route, ''), 0, False),
                (good.replace('src1_type=f32', 'src1_type=f16', 1), 0, False),
                (good.replace('src1_ne1=2 ', 'src1_ne1=4 ', 1), 0, False)):
            transcript.write_text(output)
            stub.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                            'test "$#" -eq 7; test "$1" = test; test "$3" = SYCL0; '
                            'test "$5" = MUL_MAT; test "$7" = "^attention_dmmv2=1,"\n'
                            f'cat {shlex.quote(str(transcript))}\nexit {status}\n')
            stub.chmod(0o700)
            result = subprocess.run(['bash', '-c', 'set -euo pipefail; ' + command + 'echo SERVER_WOULD_START'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
            if succeeds:
                self.assertEqual(result.stdout.count('13/13 tests passed'), 1)
                self.assertIn('SERVER_WOULD_START', result.stdout)
                M['check_attention_native_gate'](M['Q8_NATIVE_MARKER'] + '\n' + result.stdout)
            else:
                self.assertNotIn(M['ATTENTION_NATIVE_MARKER'], result.stdout)
                self.assertNotIn('SERVER_WOULD_START', result.stdout)

    def test_attention_view_native_gate_raw_and_docker_timestamps(self):
        output = attention_view_native_output()
        M['check_attention_view_native_output'](output)
        log = M['ATTENTION_NATIVE_MARKER'] + '\n' + output + M['ATTENTION_VIEW_NATIVE_MARKER'] + '\n'
        M['check_attention_view_native_gate'](log)
        for stamp in ('2026-10-17T20:10:00Z ', '2026-10-17T20:10:00.123456789Z '):
            stamped = ''.join(stamp + line + '\n' for line in log.splitlines())
            M['check_attention_view_native_gate'](stamped)
        for broken in (log.replace(M['ATTENTION_VIEW_NATIVE_MARKER'], 'missing'),
                       log + M['ATTENTION_VIEW_NATIVE_MARKER'],
                       log.replace(M['ATTENTION_NATIVE_MARKER'], 'missing')):
            with self.assertRaises(AssertionError):
                M['check_attention_view_native_gate'](broken)

    def test_attention_view_native_gate_counts_failures_and_route_exclusions(self):
        good = attention_view_native_output()
        M['check_attention_view_native_output'](good)
        first_case = next(line for line in good.splitlines(True) if '  MUL_MAT(' in line)
        routes = [line for line in good.splitlines(True) if line.startswith('BOLT_ROUTE ')]
        n2_whole = next(line for line in routes if 'blk.7.attn_output.weight' in line and 'src1_ne1=2' in line)
        n3_whole = next(line for line in routes if 'blk.7.attn_output.weight' in line and 'src1_ne1=3' in line)
        prefix_route = next(line for line in routes if 'blk.7.prefix.attn_output.weight' in line)
        offset_route = next(line for line in routes if 'blk.7.offset.attn_output.weight' in line)
        unrelated_route = next(line for line in routes if 'blk.7.unrelated.weight' in line)
        for bad in (good.replace(first_case, ''), good + first_case,
                    good + '  6/6 tests passed\n', good.replace('6/6 tests passed', '5/6 tests passed'),
                    good.replace('\x1b[1;32mOK', 'FAIL', 1),
                    good.replace('\x1b[1;32mOK', 'not supported [SYCL0] (expected)', 1),
                    good + 'SKIPPED\n', good + 'UNSUPPORTED\n', good + 'failed\n',
                    good.replace('Backend 2/2: CPU', 'Backend 2/2: OTHER_GPU'),
                    good.replace('Backend SYCL0: OK', 'Backend SYCL0: FAIL'),
                    good.replace('2/2 backends passed', '1/2 backends passed'),
                    good.replace('type_a=q8_0', 'type_a=q4_0', 1),
                    good.replace('m=4096,n=1,k=4096', 'm=4096,n=4,k=4096', 1),
                    good.replace(n2_whole, n2_whole.replace('route=DMMV_ESIMD', 'route=MMVQ_REORDERED')),
                    good.replace(n3_whole, n3_whole.replace('route=MMVQ_REORDERED', 'route=DMMV_ESIMD')),
                    good.replace(prefix_route, prefix_route.replace('route=MMVQ_REORDERED', 'route=DMMV_ESIMD')),
                    good.replace(offset_route, offset_route.replace('route=MMVQ_REORDERED', 'route=DMMV_ESIMD')),
                    good.replace(unrelated_route, unrelated_route.replace('route=MMVQ_REORDERED', 'route=DMMV_ESIMD')),
                    good.replace('reorder=1', 'reorder=0', 1)):
            with self.subTest(bad=bad[-160:]), self.assertRaises(AssertionError):
                M['check_attention_view_native_output'](bad)

    def test_attention_view_native_actual_route_coverage(self):
        good = attention_view_native_output()
        M['check_attention_view_native_output'](good)
        native_routes = [line for line in good.splitlines(True) if line.startswith('BOLT_ROUTE ')]
        self.assertEqual(len(native_routes), 6)
        for line in native_routes:
            with self.subTest(missing=line[:100]), self.assertRaises(AssertionError):
                M['check_attention_view_native_output'](good.replace(line, ''))
            for key, value in (('src0_type', 'q4_0'), ('src1_type', 'f16'), ('dst_type', 'f16'),
                               ('src1_ne1', '4'), ('reorder', '0'), ('op', 'MUL_MAT_ID'),
                               ('operand_prec', '0'), ('src1_ne2', '2'), ('src1_nb1', '1'),
                               ('dst_nb1', '1'), ('device', '1')):
                fields = line.split()
                changed = ' '.join(f'{key}={value}' if field.startswith(key + '=') else field
                                   for field in fields) + '\n'
                with self.subTest(key=key, route=line[:100]), self.assertRaises(AssertionError):
                    M['check_attention_view_native_output'](good.replace(line, changed))

    def test_attention_view_native_shell_pipeline_blocks_server_on_failure(self):
        transcript = self.root / 'attn-view-native-input.log'
        captured = self.root / 'attn-view-native-captured.log'
        stub = self.root / 'test-backend-ops-attn-view-stub'
        good = attention_view_native_output()
        first_case = next(line for line in good.splitlines(True) if '  MUL_MAT(' in line)
        first_route = next(line for line in good.splitlines(True) if line.startswith('BOLT_ROUTE '))
        command = M['ATTENTION_VIEW_NATIVE_COMMAND'].replace('/src/build-sycl/bin/test-backend-ops', shlex.quote(str(stub)))
        command = command.replace('/tmp/q8-attention-view-dmmv2-native.log', str(captured))
        command = command.replace('/iq2-single2-gate/run-trace.py', str(Path('run-trace.py').resolve()))
        for output, status, succeeds in (
                (good, 0, True), (good, 1, False),
                (good.replace(first_case, ''), 0, False), (good + first_case, 0, False),
                (good + '  6/6 tests passed\n', 0, False),
                (good.replace('6/6 tests passed', '5/6 tests passed'), 0, False),
                (good + 'not supported\n', 0, False), (good + 'SKIPPED\n', 0, False),
                (good.replace('Backend 2/2: CPU', 'Backend 2/2: GPU1'), 0, False),
                (good.replace(first_route, ''), 0, False),
                (good.replace('src1_type=f32', 'src1_type=f16', 1), 0, False),
                (good.replace('src1_ne1=2 ', 'src1_ne1=4 ', 1), 0, False)):
            transcript.write_text(output)
            stub.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                            'test "$#" -eq 7; test "$1" = test; test "$3" = SYCL0; '
                            'test "$5" = MUL_MAT; test "$7" = "^attention_view_dmmv2=1,"\n'
                            f'cat {shlex.quote(str(transcript))}\nexit {status}\n')
            stub.chmod(0o700)
            result = subprocess.run(['bash', '-c', 'set -euo pipefail; ' + command + 'echo SERVER_WOULD_START'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
            if succeeds:
                self.assertEqual(result.stdout.count('6/6 tests passed'), 1)
                self.assertIn('SERVER_WOULD_START', result.stdout)
                M['check_attention_view_native_gate'](M['ATTENTION_NATIVE_MARKER'] + '\n' + result.stdout)
            else:
                self.assertNotIn(M['ATTENTION_VIEW_NATIVE_MARKER'], result.stdout)
                self.assertNotIn('SERVER_WOULD_START', result.stdout)

    def test_restoration_identity_config_all_mounts_and_hostconfig(self):
        original = dict(Id='original-id', State=dict(Running=True), Config=dict(Env=['x=1']),
                        HostConfig=dict(PortBindings={'a': 'b'}), Mounts=[dict(Destination='/src', Source='/parent', RW=False)])
        M['check_restored'](original, copy.deepcopy(original))
        for key, value in (('Id', 'replacement'), ('State', dict(Running=False)),
                           ('Config', {}), ('HostConfig', {}), ('Mounts', [])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                M['check_restored'](original, dict(original, **{key: value}))
        tree = ast.parse(Path('run-trace.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        self.assertTrue(any(isinstance(n, ast.Try) and n.finalbody and any(
            isinstance(inner, ast.Try) and inner.finalbody for inner in ast.walk(ast.Module(body=n.finalbody, type_ignores=[])))
            for n in ast.walk(main)), 'restoration must remain in nested finally')

    def test_stop_flushes_logs_before_post_snapshot_and_alignment(self):
        source = Path('run-trace.py').read_text()
        saved = source.index("save(cell / 'diagnostic.response.json', body)")
        validated = source.index("old['check_diagnostic_result'](body)", saved)
        stopped = source.index("command('docker', 'stop', '-t', '90', name)", validated)
        post = source.index("post = command('docker', 'logs', '--timestamps', name)", saved)
        aligned = source.index("checked = analyze['analyze_arm'](cell, label, old)", saved)
        self.assertLess(validated, stopped)
        self.assertLess(stopped, post)
        self.assertLess(post, aligned)

    def test_log_window_framing(self):
        self.assertEqual(M['diagnostic_window']('prefix\n', 'prefix\n' + self.log), self.log)
        with self.assertRaises(AssertionError):
            M['diagnostic_window']('prefix\n', 'other\n' + self.log)


    def test_bounded_route_batch_selection_excludes_outside(self):
        s0 = A['select_route_batches'](route_log('S0'), 'S0')
        c0 = A['select_route_batches'](route_log('C0'), 'C0')
        self.assertEqual([b['batch']['pos'] for b in s0['selected_batches']], [[55], [56]])
        self.assertEqual([b['batch']['pos'] for b in c0['selected_batches']], [[55, 56]])
        self.assertTrue(s0['actual_route_presence'] and c0['actual_route_presence'])
        for summary in (s0, c0):
            for batch in summary['selected_batches']:
                weights = [route['weight'] for route in batch['routes']]
                self.assertIn('output.weight', weights)
                self.assertTrue(any(weight.startswith('blk.0.') for weight in weights))
            weights = [route['weight'] for batch in summary['selected_batches'] for route in batch['routes']]
            self.assertNotIn('blk.0.outside.weight', weights)
        self.assertTrue(any(route['backend_ctx'] == '0xabc'
                            for batch in s0['selected_batches'] for route in batch['routes']))
        with self.assertRaises(AssertionError):
            A['select_route_batches'](route_log(missing_output=True), 'S0')

    def test_nested_other_backend_routes_are_preserved_not_attributed(self):
        marker = route_batch([55], [1055])
        foreign = route_line(weight='blk.0.attn_k.weight', backend='0x999')
        log = route_log().replace(marker, marker + '\n' + foreign)
        first = A['select_route_batches'](log, 'S0')['selected_batches'][0]
        self.assertEqual(first['target_backend_ctx'], '0xabc')
        self.assertGreaterEqual(len(first['other_backend_routes']), 1)
        self.assertIn('0x999', {row['backend_ctx'] for row in first['other_backend_routes']})
        self.assertTrue(all(row['backend_ctx'] == '0xabc' for row in first['routes']))
        ambiguous = route_log().replace(marker, marker + '\n' + route_line(weight='blk.0.ssm_in.weight', backend='0x999'))
        with self.assertRaisesRegex(AssertionError, 'ambiguous target'):
            A['select_route_batches'](ambiguous, 'S0')
        foreign_output = route_log().replace(head_route(), head_route(backend='0x999'))
        with self.assertRaisesRegex(AssertionError, 'output projection route missing in target'):
            A['select_route_batches'](foreign_output, 'S0')

    def test_route_framing_malformed_and_empty_fail_closed(self):
        with self.assertRaises(AssertionError):
            A['route_records']('')
        with self.assertRaises(AssertionError):
            A['route_frames']('')
        with self.assertRaises(AssertionError):
            A['route_records']('BOLT_ROUTE\n')
        with self.assertRaises(AssertionError):
            A['route_records'](route_line() + ' route=MMQ\n')
        with self.assertRaises(AssertionError):
            A['route_records'](route_line(route='UNKNOWN_ROUTE'))
        with self.assertRaises(AssertionError):
            A['route_records'](route_line(omit={'src1_ne0'}))
        with self.assertRaises(AssertionError):
            A['route_records'](route_line(omit={'dst_ne0'}))
        with self.assertRaises(AssertionError):
            A['route_records'](route_line(omit={'src1_prec'}))
        ids = A['route_records'](route_line(op='MUL_MAT_ID', op_id=7, ids=True))[0]
        self.assertEqual((ids['op_id'], ids['ids_ne0'], ids['ids_nb0']), (7, 2, 4))
        with self.assertRaises(AssertionError):
            A['select_route_batches'](route_line(), 'S0')
        mixed = route_batch([55], [1055]) + ' ' + route_line()
        with self.assertRaises(AssertionError):
            A['route_frames'](mixed)
        with self.assertRaises(AssertionError):
            A['select_route_batches'](route_batch([55], [1055]) + '\n' + route_batch([56], [1056]) + '\n' + route_decoded() + '\n', 'S0')
        with self.assertRaises(AssertionError):
            A['select_route_batches'](route_batch([55], [1055]) + '\n', 'S0')
        for decoded in (route_decoded(ctx='0x999'), route_decoded(ret=1)):
            log = '\n'.join((route_batch([55], [1055]), route_line(), decoded)) + '\n'
            with self.assertRaises(AssertionError):
                A['select_route_batches'](log, 'S0')

    def test_parent_comparison_reports_changed_raw_bytes(self):
        candidate = self.aligned()
        parent_results = self.root / 'parent-results'
        parent_cell = parent_results / 'run-r1' / 'S0'
        parent_capture = parent_cell / 'capture'
        parent_capture.mkdir(parents=True)
        for sample in candidate['samples']:
            shutil.copy2(Path(candidate['capture_path']) / sample['file'], parent_capture / sample['file'])
        parent = copy.deepcopy(candidate)
        parent['capture_path'] = str(parent_capture)
        (parent_results / 'reanalysis-S0.json').write_text(json.dumps(parent))
        ordinary = {'text': 'same'}
        diagnostic = {'choices': [{'message': {'content': 'same'}}], '__verbose': {'tokens': [1]}}
        (parent_cell / 'ordinary.result.json').write_text(json.dumps(ordinary))
        (parent_cell / 'diagnostic.response.json').write_text(json.dumps(diagnostic))
        with patch.dict(A['report_parent_comparison'].__globals__, PARENT_RESULTS=parent_results):
            report = A['report_parent_comparison'](candidate, 'S0', ordinary, diagnostic)
            self.assertTrue(report['prompt_ids_match'])
            self.assertTrue(all(row['raw_bytes_equal'] for row in report['comparisons']))
            broken = bytearray((parent_capture / parent['samples'][0]['file']).read_bytes())
            broken[0] ^= 1
            (parent_capture / parent['samples'][0]['file']).write_bytes(broken)
            report = A['report_parent_comparison'](candidate, 'S0', ordinary, diagnostic)
            self.assertFalse(report['comparisons'][0]['raw_bytes_equal'])
            self.assertEqual(report['first_raw_difference']['ordinal'], 1)

    def test_parent_comparison_history_cutoff(self):
        candidate = self.aligned()
        parent_results = self.root / 'parent-results'
        parent_cell = parent_results / 'run-r1' / 'S0'
        parent_capture = parent_cell / 'capture'
        parent_capture.mkdir(parents=True)
        for sample in candidate['samples']:
            shutil.copy2(Path(candidate['capture_path']) / sample['file'], parent_capture / sample['file'])
        parent = copy.deepcopy(candidate)
        parent['emitted_ids'] = list(candidate['emitted_ids'])
        parent['emitted_ids'][115] ^= 1
        parent['capture_path'] = str(parent_capture)
        (parent_results / 'reanalysis-S0.json').write_text(json.dumps(parent))
        ordinary = {'text': 'same'}
        diagnostic = {'choices': [{'message': {'content': 'same'}}], '__verbose': {'tokens': [1]}}
        (parent_cell / 'ordinary.result.json').write_text(json.dumps(ordinary))
        (parent_cell / 'diagnostic.response.json').write_text(json.dumps(diagnostic))
        with patch.dict(A['report_parent_comparison'].__globals__, PARENT_RESULTS=parent_results):
            report = A['report_parent_comparison'](candidate, 'S0', ordinary, diagnostic)
            self.assertEqual(report['first_output_difference_ordinal'], 116)
            self.assertEqual(report['compared_through_ordinal'], 116)
            self.assertEqual(len(report['comparisons']), 116)

    def test_analyze_arm_saved_parent_capture_wiring(self):
        for label in ('S0','C0'):
            cell, _ = full_operand_cell(self.root/('wiring-'+label),label)
            result=A['analyze_arm'](cell,label,OLD)
            self.assertTrue(result['observer_parent_invariant_passed'])
            self.assertTrue(result['operand_analysis']['complete'])
            self.assertEqual(result['parent_comparison']['raw_vector_count'],128)
            self.assertEqual(len(result['route_analysis']['router_batches']),127 if label=='S0' else 67)
            self.assertEqual(result['accepted'],0 if label=='S0' else 60)

    def test_route_gate_with_synthetic_runtime_metadata(self):
        latent = ('blk.1.ffn_latent_down.weight', 'blk.1.ffn_latent_up.weight')
        for label in ('S0', 'C0'):
            cell = A['parent_reference'](label)[0]
            metadata = json.loads(A['parent_reference'](label)[1].read_text())['route_analysis']
            A['assert_actual_route_gates'](metadata, label)  # IQ2 computational parent is already aligned
            inject_block7_routes(metadata)
            if label == 'C0':
                for batch in metadata['selected_batches']:
                    backend = next(r['backend_ctx'] for r in batch['routes'] if r['weight'] == 'output.weight')
                    batch['routes'] = [r for r in batch['routes'] if r['weight'] not in A['IQ2_PROJECTIONS']]
                    batch['routes'] += [A['route_records'](iq2_route_line(w, 2, 240, backend=backend))[0]
                                        for w in A['IQ2_PROJECTIONS'] for _ in range(2)]
            A['assert_actual_route_gates'](metadata, label)
            for weight in latent + ('blk.7.attn_q.weight',):
                for key, value in (('route', 'MMVQ_REORDERED'), ('route', 'DMMV_REORDERED'),
                                   ('src0_type', 'q4_0'), ('src1_type', 'f16'), ('dst_type', 'f16'),
                                   ('reorder', 0), ('src1_ne1', 3)):
                    bad = copy.deepcopy(metadata)
                    route = next(route for route in bad['selected_batches'][0]['routes'] if route['weight'] == weight)
                    route[key] = value
                    with self.subTest(label=label, weight=weight, key=key), self.assertRaises(AssertionError):
                        A['assert_actual_route_gates'](bad, label)
                bad = copy.deepcopy(metadata)
                bad['selected_batches'][0]['routes'] = [r for r in bad['selected_batches'][0]['routes'] if r['weight'] != weight]
                with self.assertRaises(AssertionError):
                    A['assert_actual_route_gates'](bad, label)

            bad = copy.deepcopy(metadata)
            route = next(route for route in bad['selected_batches'][0]['routes']
                         if route['weight'] == 'blk.1.ffn_down_exps.weight')
            route['src0_type'] = 'nvfp4'
            with self.assertRaises(AssertionError):
                A['assert_actual_route_gates'](bad, label)
            if label == 'C0':
                bad = copy.deepcopy(metadata)
                route = next(route for route in bad['selected_batches'][0]['routes']
                             if route['weight'] == 'blk.1.ffn_up_exps.weight' and route['route'] == 'MOE_SINGLE_MMVQ_PLAIN_Q8_1')
                route['route'] = 'MOE_GROUPED_GEMM_RESULT'
                with self.assertRaises(AssertionError):
                    A['assert_actual_route_gates'](bad, label)

    def test_actual_route_gate_head_and_ssm(self):
        for label in ('S0', 'C0'):
            metadata = A['select_route_batches'](route_log(label), label)
            A['assert_actual_route_gates'](metadata, label)

        bad_head = A['select_route_batches'](route_log('S0'), 'S0')
        route = next(route for route in bad_head['selected_batches'][0]['routes']
                     if route['weight'] == 'output.weight')
        route['route'] = 'DMMV_ESIMD'
        with self.assertRaises(AssertionError):
            A['assert_actual_route_gates'](bad_head, 'S0')

        bad_ssm = A['select_route_batches'](route_log('C0'), 'C0')
        route = next(route for route in bad_ssm['selected_batches'][0]['routes']
                     if route['weight'] == 'blk.0.ssm_in.weight')
        route['route'] = 'MMVQ_PLAIN'
        route['reorder'] = 0
        with self.assertRaises(AssertionError):
            A['assert_actual_route_gates'](bad_ssm, 'C0')

        bad_shape = A['select_route_batches'](route_log('C0'), 'C0')
        route = next(route for route in bad_shape['selected_batches'][0]['routes']
                     if route['weight'] == 'blk.0.ssm_out.weight')
        route['src1_ne1'] = 1
        with self.assertRaises(AssertionError):
            A['assert_actual_route_gates'](bad_shape, 'C0')
    def test_both_parent_observer_reports(self):
        for label in ('S0','C0'):
            cell, alignment = full_operand_cell(self.root/('parity-'+label), label)
            result = A['analyze_arm'](cell, label, OLD)
            report = result['parent_comparison']
            A['assert_parent_observer_invariant'](report)
            self.assertEqual(report['parent_alignment'], str(A['parent_reference'](label)[1]))
            self.assertEqual(report['compared_through_ordinal'], 128)
            self.assertTrue(all(row['raw_bytes_equal'] for row in report['comparisons']))
            path = cell/'capture'/result['samples'][0]['file']
            data=bytearray(path.read_bytes()); data[0]^=1; path.write_bytes(data)
            with self.assertRaises(AssertionError): A['analyze_arm'](cell,label,OLD)


    def test_attention_shapes_match_actual_gguf_header(self):
        header = json.loads((M['ROOT'].parent / '20261017-nemotron35-bolt-lilicorr-attention-route/attention-header.json').read_text())
        tensors = {row['name']: row for row in header['tensors']}
        for weight, k, m in A['BLOCK7_ATTENTION_WEIGHTS']:
            self.assertEqual(tensors[weight]['type'], 8)
            self.assertEqual(tensors[weight]['dimensions'], [k, m])
        for label in ('S0', 'C0'):
            A['assert_actual_route_gates'](A['select_route_batches'](route_log(label), label), label)

    def test_block7_route_presence_and_type_fail_closed(self):
        log = route_log(missing_block7=True)
        with self.assertRaisesRegex(AssertionError, 'blk.7.attn_q.weight'):
            A['assert_actual_route_gates'](A['select_route_batches'](log, 'S0'), 'S0')
        metadata = A['select_route_batches'](route_log(), 'S0')
        bad = copy.deepcopy(metadata)
        for route in bad['selected_batches'][0]['routes']:
            if route['weight'] == 'blk.7.attn_k.weight':
                route['src0_type'] = 'nvfp4'
        with self.assertRaises(AssertionError):
            A['assert_actual_route_gates'](bad, 'S0')

    def test_block1_route_presence_and_type_fail_closed(self):
        log = route_log(missing_block1=True)
        with self.assertRaisesRegex(AssertionError, 'first MoE block route missing'):
            A['select_route_batches'](log, 'S0')
        metadata = A['select_route_batches'](route_log(), 'S0')
        bad = copy.deepcopy(metadata)
        bad['selected_batches'][0]['routes'] = [
            route for route in bad['selected_batches'][0]['routes']
            if route['weight'] != 'blk.1.ffn_up_exps.weight'
        ]
        with self.assertRaises(AssertionError):
            A['assert_actual_route_gates'](bad, 'S0')
        bad = copy.deepcopy(metadata)
        for route in bad['selected_batches'][0]['routes']:
            if route['weight'] == 'blk.1.ffn_down_exps.weight':
                route['src0_type'] = 'nvfp4'
        with self.assertRaises(AssertionError):
            A['assert_actual_route_gates'](bad, 'S0')

    def test_analyze_arm_parent_gates_through_full_wiring(self):
        for label in ('S0','C0'):
            cell, _ = full_operand_cell(self.root/('gates-'+label),label)
            good=A['analyze_arm'](cell,label,OLD)
            original=A['report_parent_comparison']
            for field in ('prompt_ids_match','emitted_ids_match','ordinary_text_matches_parent',
                          'diagnostic_text_matches_parent','diagnostic_token_ids_match_parent',
                          'actual_batch_inputs_match','selected_positions_rows_match'):
                def changed(*args):
                    report=original(*args); report[field]=False; return report
                with patch.dict(A['analyze_arm'].__globals__,report_parent_comparison=changed,
                                analyze_operands=lambda *args: self.fail('operand interpretation before parent parity')):
                    with self.subTest(label=label,field=field), self.assertRaises(AssertionError):
                        A['analyze_arm'](cell,label,OLD)




    def test_iq2_preflight_purity_ranges_and_two_views_before_any_dispatch(self):
        source, _ = self.patched_sources()
        eligible = cpp_body(source, 'ggml_sycl_iq2_single2_eligible')
        layout = cpp_body(source, 'ggml_sycl_mul_mat_id_mmvq_layout')
        bounds = cpp_body(source, 'ggml_sycl_iq2_single2_range')
        dispatch = cpp_body(source, 'ggml_sycl_mul_mat_id_iq2_single2')
        for pure in (eligible, layout, bounds):
            for side_effect in ('ctx.stream(', 'ctx.pool(', 'opt_for_reorder_id(',
                                'quantize_row_', 'malloc(', 'memcpy(', 'submit(', 'mmvq_fused('):
                self.assertNotIn(side_effect, pure)
        for guard in ('src0->type != GGML_TYPE_IQ2_XXS', 'ids->type != GGML_TYPE_I32',
                      'src1->type != GGML_TYPE_F32', 'dst->type != GGML_TYPE_F32',
                      'dst->op != GGML_OP_MUL_MAT_ID', 'dst->src[0] != src0',
                      'dst->src[1] != src1', 'dst->src[2] != ids',
                      'src1->ne[2] != 2', 'ids->ne[1] != 2', 'dst->ne[2] != 2',
                      'src0->ne[3] != 1', 'src1->ne[3] != 1', 'dst->ne[3] != 1',
                      'ids->ne[2] != 1', 'ids->ne[3] != 1',
                      't->ne[i] <= 0', 't->ne[i] > INT_MAX / elements',
                      'src0->ne[0] > INT_MAX - MATRIX_ROW_PADDING', 'src0->ne[0] % QK_K != 0',
                      'src1->ne[0] != src0->ne[0]', 'dst->ne[0] != src0->ne[1]',
                      'dst->ne[1] != ids->ne[0]', 'ids->ne[0] > src0->ne[2]',
                      'src1->ne[1] != 1 && src1->ne[1] != ids->ne[0]',
                      'extra && extra->optimized_feature.reorder',
                      'src0->nb[0] != sizeof(block_iq2_xxs)',
                      'src0->nb[1] != src0->ne[0] / QK_K * sizeof(block_iq2_xxs)',
                      'src0->nb[2] != src0->nb[1] * src0->ne[1]',
                      'src0->nb[3] != src0->nb[2] * src0->ne[2]',
                      't->nb[0] != sizeof(float)', 't->nb[1] != sizeof(float) * t->ne[0]',
                      't->nb[2] != t->nb[1] * t->ne[1]', 't->nb[3] != t->nb[2] * t->ne[2]',
                      'ids->nb[0] != sizeof(int32_t)', 'ids->nb[1] < sizeof(int32_t) * ids->ne[0]',
                      'ids->nb[1] % sizeof(int32_t) != 0', 'ids->nb[1] > SIZE_MAX / 2',
                      'ids->nb[2] != 2 * ids->nb[1]', 'ids->nb[3] != ids->nb[2]'):
            self.assertIn(guard, eligible)
        for guard in ('!t->data', '!t->buffer', '!ggml_backend_buffer_is_sycl(t->buffer)',
                      '!t->buffer->context', '(uintptr_t) t->data % 4 != 0', '->device != ctx.device',
                      'ggml_backend_buffer_get_base(t->buffer)', 'ggml_backend_buffer_get_size(t->buffer)',
                      'data >= base && data - base <= size && ggml_nbytes(t) <= size - (data - base)'):
            self.assertIn(guard, bounds)
        self.assertLess(eligible.index('INT_MAX / elements'), eligible.index('src0->nb[0]'))
        self.assertLess(eligible.index('ids->nb[3]'), eligible.index('ggml_sycl_iq2_single2_range'))
        loops = [m.start() for m in re.finditer(re.escape('for (int t = 0; t < 2; ++t)'), dispatch)]
        self.assertEqual(len(loops), 2)
        for check in ('ggml_sycl_mul_mat_id_mmvq_layout(src0, &src1_token[t], &ids_token[t], &dst_token[t])',
                      'ggml_sycl_iq2_single2_range(ctx, &src1_token[t])',
                      'ggml_sycl_iq2_single2_range(ctx, &ids_token[t])',
                      'ggml_sycl_iq2_single2_range(ctx, &dst_token[t])'):
            self.assertTrue(loops[0] < dispatch.index(check) < loops[1])
        self.assertLess(dispatch.index('ggml_sycl_iq2_single2_eligible'), loops[0])
        self.assertGreater(dispatch.index('ggml_sycl_mul_mat_id_mmvq_fused'), loops[1])
        self.assertNotIn('return false', dispatch[loops[1]:])
        self.assertIn('GGML_ABORT("preflighted IQ2 singleton dispatch failed")', dispatch[loops[1]:])
        self.assertEqual(dispatch.count('ggml_sycl_mul_mat_id_mmvq_fused('), 1)
        for name, axis, stride in (('src1', 2, 2), ('ids', 1, 1), ('dst', 2, 2)):
            self.assertIn(f'ggml_tensor {name}_token[2] = {{*{name}, *{name}}};', dispatch)
            self.assertRegex(dispatch, rf'{name}_token\[t\]\.data\s*= \(char \*\) {name}->data\s*\+ t \* {name}->nb\[{stride}\]')
            self.assertIn(f'{name}_token[t].ne[{axis}] = 1;', dispatch)
        assigned = re.findall(r'(\w+_token\[t\]\.[\w\[\]]+)\s*=(?!=)', dispatch)
        self.assertEqual(set(assigned), {f'{name}_token[t].{field}' for name, axis in
                         (('src1', 2), ('ids', 1), ('dst', 2)) for field in ('data', f'ne[{axis}]')})

    def test_iq2_singleton_body_plain_support_and_width3_fallback_unchanged(self):
        source, _ = self.patched_sources()
        parent = (M['PARENT'] / M['SOURCE_SCOPE'][0]).read_text()
        for name in ('ggml_sycl_mul_mat_id_mmvq_layout', 'ggml_sycl_mul_mat_id_mmvq_fused',
                     'ggml_sycl_mul_mat_id_iq2_single2', 'ggml_sycl_iq2_single2_range',
                     'ggml_sycl_iq2_single2_eligible', 'ggml_sycl_mul_mat_id'):
            self.assertEqual(cpp_body(source, name), cpp_body(parent, name))

    def test_iq2_native_only_six_names_change_no_numerical_fixture_change(self):
        _, native = self.patched_sources()
        self.assertEqual(native, (M['PARENT'] / 'tests/test-backend-ops.cpp').read_text())
        self.assertEqual(native.count('new test_mul_mat_id_iq2_single2('), 6)
        for command in ('IQ2_NATIVE_COMMAND', 'Q8_NATIVE_COMMAND', 'ATTENTION_NATIVE_COMMAND', 'ATTENTION_VIEW_NATIVE_COMMAND', 'ROUTER_NATIVE_COMMAND'):
            self.assertEqual(M[command], PARENT_CONTROL[command])
        for name in ('check_startup', 'check_iq2_native_output', 'check_q8_native_output',
                     'check_attention_native_output', 'check_attention_view_native_output', 'legacy_native_prefix',
                     'check_restored', 'spec_for_arm'):
            a = next(n for n in ast.parse(Path('run-trace.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in ast.parse((PARENT_TOP/'run-trace.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b), name)

    def test_protected_source_and_all_five_artifact_gates(self):
        self.assertEqual(set(M['PARENT_EXPECTED']), set(M['ARTIFACTS']))
        self.assertEqual(M['binaries'](M['PARENT']), M['PARENT_EXPECTED'])
        build = Path('build-candidate.sh').read_text()
        self.assertIn("assert m['binaries'](m['PARENT']) == m['PARENT_EXPECTED']", build)
        self.assertIn("== {'libggml-sycl.so.0'}", build)
        self.assertIn("m['check_protected_sources']()", build)
        self.assertIn("assert original['Id'] == ORIGINAL_ID", Path('run-trace.py').read_text())
        expected = {'accepted': {'a': 'x'}, 'parent': {'a': 'y'}}
        actual = {'accepted': expected['accepted'].copy(), 'parent': expected['parent'].copy(),
                  'candidate': dict(expected['parent'], **{n: 'new' for n in M['SOURCE_SCOPE']})}
        roots = {M['ACCEPTED']: 'accepted', M['PARENT']: 'parent', M['CANDIDATE']: 'candidate'}
        with patch.dict(M['check_protected_sources'].__globals__, load=lambda p: expected,
                        source_hashes=lambda p: actual[roots[p]]):
            M['check_protected_sources']()
            for label in actual:
                saved = actual[label].copy()
                actual[label]['unexpected.cpp'] = 'changed'
                self.assertRaises(AssertionError, M['check_protected_sources'])
                actual[label] = saved

    def test_iq2_actual_saved_numeric_format(self):
        saved = (M['PARENT_RESULTS'] / 'run-r1/S0/server.log').read_text()
        output = saved.split('BOLT_LILICORR_OUTPROD_NATIVE_PASSED', 1)[1].split(M['IQ2_NATIVE_MARKER'], 1)[0]
        self.assertIn('amax=1.000000,', output)
        M['check_iq2_native_output'](output)
        M['check_iq2_native_output'](output.replace('amax=1.000000,', 'amax=1,'))
        for value in ('1.000001', '0.999999', 'nan', 'inf'):
            bad = output.replace('amax=1.000000,', f'amax={value},')
            self.assertNotEqual(bad, output)
            with self.subTest(value=value), self.assertRaises(AssertionError):
                M['check_iq2_native_output'](bad)

    def test_iq2_native_routes_grouped_and_fallback_with_valid_timestamps(self):
        for grouped in (True, False):
            output = iq2_native_output(grouped)
            M['check_iq2_native_output'](output)
            log = 'BOLT_LILICORR_OUTPROD_NATIVE_PASSED\n' + output + M['IQ2_NATIVE_MARKER'] + '\n'
            for stamp in ('', '2026-10-17T20:10:00Z ', '2026-10-17T20:10:00.123456789Z '):
                stamped = ''.join(stamp + line + '\n' for line in log.splitlines())
                M['check_iq2_native_gate'](stamped)
            for bad in (log.replace(M['IQ2_NATIVE_MARKER'], 'missing'), log + M['IQ2_NATIVE_MARKER'],
                        log.replace('BOLT_LILICORR_OUTPROD_NATIVE_PASSED', 'missing'),
                        M['Q8_NATIVE_MARKER'] + '\n' + log):
                self.assertRaises(AssertionError, M['check_iq2_native_gate'], bad)

    def test_iq2_native_actual_two_view_count_types_strides_shapes_and_failures(self):
        good = iq2_native_output()
        M['check_iq2_native_output'](good)
        native = [line for line in good.splitlines(True) if line.startswith('BOLT_ROUTE ')]
        self.assertEqual(len(native), 10)
        for line in native:
            self.assertRaises(AssertionError, M['check_iq2_native_output'], good.replace(line, '', 1))
            self.assertRaises(AssertionError, M['check_iq2_native_output'], good + line)
            for key, value in (('src0_type', 'q8_0'), ('src1_type', 'f16'), ('dst_type', 'f16'),
                               ('op', 'MUL_MAT'), ('op_id', 0), ('reorder', 1), ('device', 1),
                               ('ne0', 17), ('ne1', 17), ('ne2', 17), ('src1_ne0', 17),
                               ('src1_ne2', 4), ('dst_ne1', 8), ('dst_ne2', 4), ('ids_ne0', 8),
                               ('ids_ne1', 4), ('nb1', 1), ('src1_nb1', 1), ('src1_nb2', 1),
                               ('src1_nb3', 1), ('dst_nb1', 1), ('dst_nb2', 1), ('dst_nb3', 1),
                               ('ids_nb0', 8), ('ids_nb1', 36), ('ids_nb2', 1), ('ids_nb3', 1)):
                changed = re.sub(r'\b' + key + r'=[^\s]+', f'{key}={value}', line)
                with self.subTest(key=key, route=line[:90]):
                    self.assertRaises(AssertionError, M['check_iq2_native_output'], good.replace(line, changed, 1))
        n2 = next(line for line in native if 'dst_ne0=2048 ' in line and 'dst_nb3=147456 ' in line)
        n3 = next(line for line in native if 'route=MOE_GROUPED_GEMM_ATTEMPT' in line)
        for bad in (good.replace(n2, n2.replace('MOE_SINGLE_MMVQ_PLAIN_Q8_1', 'MOE_PER_EXPERT_FALLBACK'), 1),
                    good.replace(n3, iq2_route_line('blk.1.ffn_up_exps.weight', 3) + '\n', 1),
                    good.replace('src1_ne2=1 ', '', 1), good.replace('src1_ne2=1 ', 'src1_ne2=NaN ', 1),
                    good.replace('BOLT_ROUTE ', 'BOLT_ROUTE BOLT_ROUTE ', 1),
                    good.replace('route=MOE_SINGLE_MMVQ_PLAIN_Q8_1', 'route=UNKNOWN', 1)):
            self.assertRaises(AssertionError, M['check_iq2_native_output'], bad)
        fallback = iq2_native_output(False)
        pair = next(line for line in fallback.splitlines(True) if 'route=MOE_PER_EXPERT_FALLBACK' in line)
        self.assertRaises(AssertionError, M['check_iq2_native_output'], fallback.replace(pair, '', 1))
        self.assertRaises(AssertionError, M['check_iq2_native_output'], fallback.replace('route=MMVQ_PLAIN', 'route=MOE_SINGLE_MMVQ_PLAIN_Q8_1', 1))

    def test_iq2_native_numerical_counts_skips_malformed_and_duplicate_summary(self):
        good = iq2_native_output()
        case = next(line for line in good.splitlines(True) if '  MUL_MAT_ID(' in line)
        for bad in (good.replace(case, ''), good + case,
                    good + '  6/6 tests passed\n', good.replace('6/6 tests passed', '5/6 tests passed'),
                    good.replace('\x1b[1;32mOK', 'FAIL', 1), good.replace('\x1b[1;32mOK', 'SKIPPED', 1),
                    good + 'not supported\n', good + 'skipping large tensors\n', good + 'ERR = 1\n', good + 'failed\n',
                    good.replace('n=2,k=1024', 'n=1,k=1024', 1), good.replace('amax=1', 'amax=2', 1),
                    good.replace('Backend 2/2: CPU', 'Backend 2/2: GPU1'),
                    good + 'Backend 3/3: CPU\n  Skipping\n',
                    good.replace('Backend SYCL0: OK', 'Backend SYCL0: FAIL'),
                    good.replace('2/2 backends passed', '1/2 backends passed'),
                    '2026-10-17TmalformedZ ' + good):
            self.assertRaises(AssertionError, M['check_iq2_native_output'], bad)

    def test_iq2_native_shell_actual_route_gate_precedes_pass_and_server(self):
        transcript, captured, stub = (self.root / name for name in ('iq2-input', 'iq2-captured', 'iq2-stub'))
        good = iq2_native_output()
        n2 = next(line for line in good.splitlines(True) if 'dst_nb3=147456 ' in line)
        command = M['IQ2_NATIVE_COMMAND'].replace('/src/build-sycl/bin/test-backend-ops', shlex.quote(str(stub)))
        command = command.replace('/tmp/iq2-xxs-native.log', str(captured))
        command = command.replace('/iq2-single2-gate/run-trace.py', str(Path('run-trace.py').resolve()))
        for output, status, succeeds in ((good, 0, True), (good, 1, False),
                (good.replace(n2, '', 1), 0, False),
                (good.replace(n2, n2.replace('MOE_SINGLE_MMVQ_PLAIN_Q8_1', 'MOE_PER_EXPERT_FALLBACK'), 1), 0, False),
                (good.replace('ids_nb1=64', 'ids_nb1=36', 1), 0, False),
                (good.replace('src1_type=f32', 'src1_type=f16', 1), 0, False),
                (good.replace('src1_ne2=1 ', '', 1), 0, False),
                (good.replace('\x1b[1;32mOK', 'SKIPPED', 1), 0, False),
                (good + '  6/6 tests passed\n', 0, False),
                (good.replace('Backend 2/2: CPU', 'Backend 2/2: GPU1'), 0, False)):
            transcript.write_text(output)
            stub.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                            'test "$#" -eq 7; test "$1" = test; test "$3" = SYCL0; '
                            'test "$5" = MUL_MAT_ID; test "$7" = "type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,"\n'
                            f'cat {shlex.quote(str(transcript))}\nexit {status}\n')
            stub.chmod(0o700)
            result = subprocess.run(['bash', '-c', 'set -euo pipefail; ' + command + 'echo SERVER_WOULD_START'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
            self.assertEqual('SERVER_WOULD_START' in result.stdout, succeeds)
            self.assertEqual(M['IQ2_NATIVE_MARKER'] in result.stdout, succeeds)

    def test_iq2_model_two_views_through_real_analyze_arm(self):
        for label in ('S0','C0'):
            cell,_=full_operand_cell(self.root/('iq2-'+label),label)
            result=A['analyze_arm'](cell,label,OLD)
            for batch in result['route_analysis']['selected_batches']:
                for weight in A['IQ2_PROJECTIONS']:
                    routes=[r for r in batch['routes'] if r['weight']==weight]
                    self.assertEqual(len(routes),1 if label=='S0' else 2)
            log=(cell/'diagnostic-window.log').read_text()
            one=next(line for line in log.splitlines(True) if 'BOLT_ROUTE ' in line and 'weight=blk.1.ffn_up_exps.weight ' in line)
            (cell/'diagnostic-window.log').write_text(log.replace(one,'',1))
            self.assertRaises(AssertionError,A['analyze_arm'],cell,label,OLD)


    def test_router_native_numeric_and_actual_operation_gates(self):
        good = router_native_output()
        M['check_router_native_output'](good)
        log = M['ATTENTION_VIEW_NATIVE_MARKER'] + '\n' + good + M['ROUTER_NATIVE_MARKER'] + '\n' + M['MARKER']
        M['check_router_native_gate'](log)
        stamped = ''.join('2026-10-18T01:00:00.123456789Z ' + line + '\n' for line in good.splitlines())
        M['check_router_native_output'](stamped)
        native = [line for line in good.splitlines(True) if 'BOLT_ROUTE ' in line]
        self.assertEqual(len(native), 14)  # seven operations, not fourteen
        for line in native:
            self.assertRaises(AssertionError, M['check_router_native_output'], good.replace(line, '', 1))
            self.assertRaises(AssertionError, M['check_router_native_output'], good + line)
            for key, value in (('src0_type', 'f16'), ('src1_type', 'f16'), ('dst_type', 'f16'),
                               ('ne0', 2048), ('ne1', 239), ('src1_ne1', 17), ('src1_nb1', 1),
                               ('src1_nb2', 1), ('dst_ne1', 17), ('dst_nb2', 1), ('nb3', 1),
                               ('op_id', 30), ('reorder', 1), ('device', 1), ('src1_prec', 10),
                               ('accum_prec', 10), ('operand_prec', 99), ('ids_ne0', 1)):
                changed = re.sub(r'\b' + key + r'=[^\s]+', f'{key}={value}', line)
                with self.subTest(key=key, line=line[:90]):
                    self.assertRaises(AssertionError, M['check_router_native_output'], good.replace(line, changed, 1))
        case = next(line for line in good.splitlines(True) if '  MUL_MAT(' in line)
        for bad in (good.replace(case, ''), good + case, good + '  6/6 tests passed\n',
                    good.replace('6/6 tests passed', '5/6 tests passed'),
                    good.replace('\x1b[1;32mOK', 'SKIPPED', 1), good + 'ERR = 1\n',
                    good.replace('Backend 2/2: CPU', 'Backend 2/2: GPU'),
                    good.replace('route=LIBRARY_MKL_F32', 'route=LIBRARY_DNNL_F32', 1),
                    good.replace('BOLT_ROUTE ', 'BOLT_ROUTE BOLT_ROUTE ', 1)):
            self.assertRaises(AssertionError, M['check_router_native_output'], bad)
        for bad in (log.replace(M['ROUTER_NATIVE_MARKER'], 'missing'), log + M['ROUTER_NATIVE_MARKER'],
                    log.replace(M['ATTENTION_VIEW_NATIVE_MARKER'], 'missing')):
            self.assertRaises(AssertionError, M['check_router_native_gate'], bad)

    def test_router_shell_gate_blocks_partial_or_failed_native_before_server(self):
        transcript, captured, stub = (self.root / name for name in ('router-input', 'router-output', 'router-stub'))
        command = M['ROUTER_NATIVE_COMMAND'].replace('/src/build-sycl/bin/test-backend-ops', shlex.quote(str(stub)))
        command = command.replace('/tmp/router-single2-native.log', str(captured))
        command = command.replace('/iq2-single2-gate/run-trace.py', str(Path('run-trace.py').resolve()))
        good = router_native_output()
        one = next(line for line in good.splitlines(True) if 'BOLT_ROUTE ' in line and 'dst_name=router_single2-l0-n2 ' in line)
        for output, status, succeeds in ((good, 0, True), (good, 1, False),
                (good.replace(one, '', 1), 0, False),
                (good.replace('route=LIBRARY_MKL_F32', 'route=LIBRARY_DNNL_F32', 1), 0, False)):
            transcript.write_text(output)
            stub.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                            'test "$#" -eq 7; test "$1" = test; test "$3" = SYCL0; '
                            'test "$5" = MUL_MAT; test "$7" = "^router_single2=1,"\n'
                            f'cat {shlex.quote(str(transcript))}\nexit {status}\n')
            stub.chmod(0o700)
            result = subprocess.run(['bash', '-c', 'set -euo pipefail; ' + command + 'echo SERVER_WOULD_START'], capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
            self.assertEqual('SERVER_WOULD_START' in result.stdout, succeeds)
            self.assertEqual(M['ROUTER_NATIVE_MARKER'] in result.stdout, succeeds)
        argv = M['make_create'](BASE, 'C0', IMAGE, M['ROOT'] / 'run-r1/C0/capture', OLD)
        self.assertLess(argv[-1].index(M['ROUTER_NATIVE_COMMAND']), argv[-1].index('echo ' + M['MARKER']))

    def test_router_every_target_frame_and_nested_drafter_separation(self):
        for label in ('S0', 'C0'):
            log = router_model_log(label)
            checked = A['assert_router_model_routes'](log, label)
            self.assertEqual(len(checked), 2 if label == 'S0' else 1)
            foreign = '\n'.join(router_lines(1, backend='0x999'))
            log = log.replace(route_decoded(), foreign + '\n' + route_decoded(), 1)
            checked = A['assert_router_model_routes'](log, label)
            self.assertTrue(any(x['other_backend_routes'] for x in checked))
            for x in checked:
                self.assertTrue(all(r['backend_ctx'] == x['target_backend_ctx'] for r in x['routes']))
        log = router_model_log('C0')
        # Two outer/inner labels from ONE n2 operation may not masquerade as two n1 operations.
        original = '\n'.join(router_lines(2))
        single_pair = original.splitlines()[:2]
        self.assertIn(original, log)
        self.assertRaises(AssertionError, A['assert_router_model_routes'], log.replace(original, '\n'.join(single_pair)), 'C0')
        later = router_model_log('C0').replace(
            route_batch([55, 56], [1055, 1056]), route_batch([100, 101], [1000, 1001]), 1)
        A['assert_router_model_routes'](log + '\n' + later, 'C0')
        self.assertRaises(AssertionError, A['assert_router_model_routes'],
                          log + '\n' + later.replace('src1_ne1=1 ', 'src1_ne1=2 '), 'C0')

    def test_saved_real_router_fifo_order_and_position70_shift(self):
        log_path = M['PARENT_RESULTS'] / 'run-r1/S0/diagnostic-window.log'
        log = log_path.read_text()
        lines = log.splitlines()
        batch70 = next(i for i, line in enumerate(lines)
                       if 'BOLT_VERIFY batch ' in line and A['records'](line)[0]['pos'] == [70])
        previous_decoded = max(i for i in range(batch70)
                               if 'BOLT_VERIFY decoded ' in lines[i])
        shifted = [line for line in lines[previous_decoded + 1:batch70]
                   if 'BOLT_ROUTE ' in line and any(
                       weight in line for weight in (
                           'weight=blk.0.ssm_in.weight ', 'weight=blk.0.ssm_out.weight ',
                           'weight=blk.1.ffn_gate_inp.weight ',
                           'weight=blk.1.ffn_latent_down.weight ',
                           'weight=blk.1.ffn_up_exps.weight '))]
        self.assertEqual(len(shifted), 6)
        self.assertEqual([A['route_records'](line)[0]['weight'] for line in shifted], [
            'blk.0.ssm_in.weight', 'blk.0.ssm_out.weight',
            'blk.1.ffn_gate_inp.weight', 'blk.1.ffn_gate_inp.weight',
            'blk.1.ffn_latent_down.weight', 'blk.1.ffn_up_exps.weight'])
        frames = A['route_frames'](log)
        self.assertEqual(len(frames), 130)
        self.assertEqual(len(next(frame for frame in frames if frame['batch']['pos'] == [70])['routes']), 15)
        self.assertEqual(len(A['assert_router_model_routes'](log, 'S0')), 127)

        model = router_model_log('S0')
        model_lines = model.splitlines()
        batches = [i for i, line in enumerate(model_lines) if 'BOLT_VERIFY batch ' in line]
        second = batches[1]
        decoded = next(i for i in range(second + 1, len(model_lines))
                       if 'BOLT_VERIFY decoded ' in model_lines[i])
        direct = model_lines[second + 1:decoded]
        self.assertTrue(direct and all('BOLT_ROUTE ' in line for line in direct))
        shifted_model = (model_lines[:second] + direct + [model_lines[second], model_lines[decoded]]
                         + model_lines[decoded + 1:])
        shifted_model = '\n'.join(shifted_model) + '\n'
        self.assertEqual([A['records'](line)[0]['event'] for line in shifted_model.splitlines()
                          if 'BOLT_VERIFY ' in line],
                         [A['records'](line)[0]['event'] for line in model.splitlines()
                          if 'BOLT_VERIFY ' in line])
        self.assertEqual(len(A['assert_router_model_routes'](shifted_model, 'S0')), 2)

    def test_router_fifo_rejects_dropped_extra_bad_width_and_reordered_groups(self):
        log = (M['PARENT_RESULTS'] / 'run-r1/S0/diagnostic-window.log').read_text()
        lines = log.splitlines()
        batch70 = next(i for i, line in enumerate(lines)
                       if 'BOLT_VERIFY batch ' in line and A['records'](line)[0]['pos'] == [70])
        previous_decoded = max(i for i in range(batch70)
                               if 'BOLT_VERIFY decoded ' in lines[i])
        outer = next(line for line in lines[previous_decoded + 1:batch70]
                     if 'BOLT_ROUTE ' in line and 'weight=blk.1.ffn_gate_inp.weight ' in line
                     and 'route=MUL_MAT_SYCL' in line)
        for bad in (
                log.replace(outer, '', 1),
                log.replace(outer, outer + '\n' + outer, 1),
                log.replace('src1_nb2=16384', 'src1_nb2=32768', 1)):
            with self.assertRaises(AssertionError):
                A['assert_router_model_routes'](bad, 'S0')

        route_rows = [(line, A['route_records'](line)[0])
                      for line in lines if 'BOLT_ROUTE ' in line]
        target = next(route['backend_ctx'] for _, route in route_rows
                      if route['weight'] == 'blk.0.ssm_in.weight')
        starts = [i for i, (_, route) in enumerate(route_rows)
                  if route['backend_ctx'] == target
                  and route['weight'] == 'blk.0.ssm_in.weight'
                  and route['route'] not in ('LIBRARY_DNNL_F32', 'LIBRARY_MKL_F32')]
        self.assertGreaterEqual(len(starts), 3)
        first_end, second_end = starts[1], starts[2]
        reordered = (route_rows[starts[1]:second_end] + route_rows[starts[0]:first_end]
                     + route_rows[second_end:])
        non_routes = [line for line in lines if 'BOLT_ROUTE ' not in line]
        reordered_log = '\n'.join(non_routes + [line for line, _ in reordered]) + '\n'
        with self.assertRaises(AssertionError):
            A['route_frames'](reordered_log)
        first_ssm = next(line for line, route in route_rows
                         if route['backend_ctx'] == target
                         and route['weight'] == 'blk.0.ssm_in.weight')
        bad_prefix = log.replace(first_ssm, head_route(backend=target) + '\n' + first_ssm, 1)
        with self.assertRaises(AssertionError):
            A['route_frames'](bad_prefix)

        first_batch = next(i for i, line in enumerate(lines) if 'BOLT_VERIFY batch ' in line)
        unknown = (route_batch(list(range(0, 55)), list(range(0, 55))) + '\n' + route_decoded()).splitlines()
        bad_leading = '\n'.join(lines[:first_batch] + unknown + lines[first_batch:]) + '\n'
        with self.assertRaises(AssertionError):
            A['route_frames'](bad_leading)

    def test_analyze_arm_rejects_changed_reordered_or_missing_prefill(self):
        source = M['PARENT_RESULTS'] / 'run-r1/S0'
        base_log = (source / 'diagnostic-window.log').read_text()
        base_lines = base_log.splitlines()
        verify = A['_verify_frames'](base_log)
        self.assertEqual([(frame['batch']['n'], frame['batch']['pos']) for frame in verify[:3]], [
            (5, list(range(0, 5))), (46, list(range(5, 51))), (4, list(range(51, 55)))])
        names = ('ordinary.result.json', 'warmup.result.json', 'diagnostic.response.json',
                 'diagnostic.request.json', 'ordinary.request.json', 'warmup.request.json')
        for kind in ('changed', 'reordered', 'missing'):
            with self.subTest(kind=kind):
                cell = self.root / ('prefill-' + kind)
                cell.mkdir()
                for name in names:
                    shutil.copy2(source / name, cell / name)
                lines = list(base_lines)
                if kind == 'changed':
                    index = verify[0]['verify_line']
                    lines[index] = lines[index].replace('n=5 ', 'n=6 ', 1)
                elif kind == 'reordered':
                    first, second = verify[0]['verify_line'], verify[1]['verify_line']
                    lines[first], lines[second] = lines[second], lines[first]
                else:
                    removed = {verify[0]['verify_line'], verify[0]['decoded_line']}
                    lines = [line for index, line in enumerate(lines) if index not in removed]
                (cell / 'diagnostic-window.log').write_text('\n'.join(lines) + '\n')
                with self.assertRaisesRegex(AssertionError, 'frozen prefill batches changed'):
                    A['analyze_arm'](cell, 'S0', OLD)

    def test_asymmetric_parent_mapping_preserves_failed_s0(self):
        self.assertEqual(A['PARENT_RESULTS'],M['PARENT_RESULTS'])
        s0, s0_alignment=A['parent_reference']('S0')
        c0, c0_alignment=A['parent_reference']('C0')
        self.assertEqual(s0,M['PARENT_RESULTS']/'run-r1/S0')
        self.assertEqual(s0_alignment,M['PARENT_RESULTS']/'reanalysis-S0.json')
        self.assertFalse((s0/'alignment.json').exists())
        self.assertEqual(c0,M['PARENT_RESULTS']/'continuation-c0/run-r1/C0')
        self.assertEqual(c0_alignment,c0/'alignment.json')
        self.assertFalse(Path('continue-c0.py').exists())
        with self.assertRaises(AssertionError): A['parent_reference']('A0')

    def test_original_controller_remains_full_pair(self):
        self.assertEqual(M['ORDER'],('S0','C0'))
        self.assertIn('BOLT_VERIFY_PAIR_COMPLETED_NO_TIMING',Path('run-trace.py').read_text())
        self.assertNotIn('c0_parent_report_only',Path('analyze-trace.py').read_text())
        for label in ('S0','C0'):
            cell,_=full_operand_cell(self.root/('all-frames-'+label),label)
            result=A['analyze_arm'](cell,label,OLD)
            log=(cell/'diagnostic-window.log').read_text()
            last=next(line for line in reversed(log.splitlines(True)) if 'weight=blk.1.ffn_gate_inp.weight ' in line)
            (cell/'diagnostic-window.log').write_text(log[:log.rindex(last)]+log[log.rindex(last)+len(last):])
            self.assertRaises(AssertionError,A['analyze_arm'],cell,label,OLD)

    def test_observer_preserves_router_control_and_native_source_exactly(self):
        source, tests=self.patched_sources()
        parent=(M['PARENT']/M['SOURCE_SCOPE'][0]).read_text()
        self.assertEqual(tests,(M['PARENT']/'tests/test-backend-ops.cpp').read_text())
        for name in ('ggml_sycl_router_single2_range','ggml_sycl_router_single2_views','ggml_sycl_mul_mat'):
            self.assertEqual(cpp_body(source,name),cpp_body(parent,name))
        # Reversing the unchanged observer patch must restore the complete parent,
        # not just a subset of functions. No router-single2 source is replaced.
        scratch=self.root/'patched-source'
        subprocess.run(['patch','--reverse','--batch','--fuzz=0','-p1'],cwd=scratch,
                       input=Path('moe-operands.patch').read_bytes(),check=True,capture_output=True)
        self.assertEqual((scratch/M['SOURCE_SCOPE'][0]).read_bytes(),(M['PARENT']/M['SOURCE_SCOPE'][0]).read_bytes())

    def test_router_compiled_actual_preflight_offsets_metadata_and_no_partial_launch(self):
        source, _ = self.patched_sources()
        helper = source[source.index('// Bounded router control:'):source.index('static void ggml_sycl_mul_mat(')]
        dispatch = cpp_body(source, 'ggml_sycl_mul_mat').split('    scope_op_debug_print', 1)[0]
        # Real ggml public structs and actual new source; mock backend range queries
        # and a CPU spy at the unchanged singleton-dispatch boundary. NOT a MKL test.
        prelude = r"""
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdio>
struct ggml_backend_sycl_context { int device; };
struct ggml_backend_sycl_buffer_context { int device; void * base; };
struct ggml_tensor_extra_gpu { struct { bool reorder; } optimized_feature; };
static ggml_backend_buffer_type sycl_type{}, split_type{}, cpu_type{};
bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t b) { return b->buft == &sycl_type; }
bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t b) { return b->buft == &split_type; }
void * ggml_backend_buffer_get_base(ggml_backend_buffer_t b) { return ((ggml_backend_sycl_buffer_context *) b->context)->base; }
size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t b) { return b->size; }
size_t ggml_nbytes(const ggml_tensor * t) {
    size_t n = 4; for (int i = 0; i < 4; ++i) n += (t->ne[i] - 1) * t->nb[i]; return n;
}
static int contiguous_calls = 0, reject_contiguous_at = 0;
bool ggml_is_contiguous(const ggml_tensor * t) {
    if (++contiguous_calls == reject_contiguous_at) return false;
    size_t n = 4;
    for (int i = 0; i < 4; ++i) { if (t->ne[i] != 1 && t->nb[i] != n) return false; n *= t->ne[i]; }
    return true;
}
int32_t ggml_get_op_params_i32(const ggml_tensor * t, uint32_t i) { return t->op_params[i]; }
struct Call { const void * in; void * out; int64_t n; };
static std::vector<Call> calls;
static bool calculate = false;
static ggml_backend_sycl_context * expected_ctx;
static ggml_tensor * original_dst;
"""
        spy = r"""
    assert(&ctx == expected_ctx);
    calls.push_back({src1->data, dst->data, src1->ne[1]});
    assert(dst->ne[1] == src1->ne[1]);
    assert(memcmp(dst->op_params, original_dst->op_params, sizeof(dst->op_params)) == 0);
    if (!calculate) return;
    assert(dst->src[0] == src0 && dst->src[1] == src1);
    for (int64_t col = 0; col < src1->ne[1]; ++col) {
        auto * x = (float *) ((char *) src1->data + col * src1->nb[1]);
        auto * y = (float *) ((char *) dst->data + col * dst->nb[1]);
        for (int64_t row = 0; row < src0->ne[1]; ++row) {
            const float * w = (const float *) ((const char *) src0->data + row * src0->nb[1]);
            float v = 0; for (int64_t k = 0; k < src0->ne[0]; ++k) v += w[k] * x[k]; y[row] = v;
        }
    }
}
int main() {
    ggml_backend_sycl_context ctx{0}; expected_ctx = &ctx;
    std::vector<float> av(4096*240+32), bv(4096*3+32), dv(240*3+32, -99);
    for (size_t i = 0; i < av.size(); ++i) av[i] = float(int(i%13)-6) / 32;
    for (int c = 0; c < 3; ++c) for (int i = 0; i < 4096; ++i)
        bv[16+c*4096+i] = (c+1) * ((i%17)-8) / 16.0f + (c%2 ? 0.25f : -0.125f);
    ggml_backend_sycl_buffer_context ac{0, av.data()}, bc{0, bv.data()}, dc{0, dv.data()};
    ggml_backend_buffer ab{}, bb{}, db{};
    ab.buft = bb.buft = db.buft = &sycl_type;
    ab.context=&ac; bb.context=&bc; db.context=&dc;
    ab.size=av.size()*4; bb.size=bv.size()*4; db.size=dv.size()*4;
    ggml_tensor a{}, b{}, d{};
    auto tensor = [](ggml_tensor & t, int64_t k, int64_t n, void * data, ggml_backend_buffer * buf) {
        t.type=GGML_TYPE_F32; t.ne[0]=k; t.ne[1]=n; t.ne[2]=t.ne[3]=1;
        t.nb[0]=4; t.nb[1]=k*4; t.nb[2]=t.nb[3]=k*n*4; t.data=data; t.buffer=buf;
    };
    tensor(a,4096,240,av.data()+16,&ab); tensor(b,4096,2,bv.data()+16,&bb); tensor(d,240,2,dv.data()+16,&db);
    strcpy(a.name,"blk.1.ffn_gate_inp.weight");
    d.op=GGML_OP_MUL_MAT; d.src[0]=&a; d.src[1]=&b;
    d.op_params[0]=10; d.op_params[3]=10; d.op_params[4]=0x12345678; original_dst=&d;
    const auto a0=a, b0=b, d0=d; const auto ab0=ab, bb0=bb, db0=db;
    calculate=true; ggml_sycl_mul_mat(ctx,&a,&b,&d); calculate=false;
    assert(calls.size()==2 && calls[0].n==1 && calls[1].n==1);
    assert(calls[0].in==b.data && calls[1].in==(char *)b.data+16384);
    assert(calls[0].out==d.data && calls[1].out==(char *)d.data+960);
    for (int c=0;c<2;++c) for (int r=0;r<240;++r) {
        float expected=0; for (int k=0;k<4096;++k) expected+=av[16+r*4096+k]*bv[16+c*4096+k];
        assert(dv[16+c*240+r]==expected);
    }
    assert(!std::equal(dv.begin()+16,dv.begin()+256,dv.begin()+256));
    for (int i=0;i<16;++i) assert(dv[i]==-99);
    for (size_t i=496;i<dv.size();++i) assert(dv[i]==-99);
    assert(memcmp(&a,&a0,sizeof(a))==0 && memcmp(&b,&b0,sizeof(b))==0 && memcmp(&d,&d0,sizeof(d))==0);
    ggml_tensor x[2], y[2];
    assert(ggml_sycl_router_single2_views(ctx,&a,&b,&d,x,y));
    for (int c=0;c<2;++c) {
        auto ex=b; ex.data=(char *) b.data+c*b.nb[1]; ex.ne[1]=1;
        auto ey=d; ey.data=(char *) d.data+c*d.nb[1]; ey.ne[1]=1; ey.src[1]=&x[c];
        assert(memcmp(&ex,&x[c],sizeof(ex))==0 && memcmp(&ey,&y[c],sizeof(ey))==0);
    }
    int rejected=0;
    auto reset = [&] { a=a0;b=b0;d=d0;ab=ab0;bb=bb0;db=db0;ac.device=bc.device=dc.device=0;
                      calls.clear(); contiguous_calls=0; reject_contiguous_at=0; };
    auto reject = [&](auto mutate) {
        reset(); mutate(); ggml_sycl_mul_mat(ctx,&a,&b,&d);
        assert(calls.size()==1 && calls[0].n==b.ne[1]); ++rejected;
    };
    reject([&] { reject_contiguous_at=3; }); // second-column preflight fails: ZERO singleton calls
    reject([&] { strcpy(a.name,"blk.1.unrelated.weight"); });
    reject([&] { strcpy(a.name,"blk.1.ffn_gate_inp.weight.extra"); });
    reject([&] { d.op=GGML_OP_ADD; });
    reject([&] { d.src[1]=&a; });
    reject([&] { d.op_params[1]=1; });
    reject([&] { a.op=GGML_OP_VIEW; });
    reject([&] { a.flags=GGML_TENSOR_FLAG_INPUT; });
    for (int n : {1,3,4}) reject([&] { b.ne[1]=d.ne[1]=n; });
    for (int index=0;index<3;++index) {
        auto t=[&]() { return index==0 ? &a : index==1 ? &b : &d; };
        reject([&] { t()->type=GGML_TYPE_F16; });
        reject([&] { t()->ne[0]=2048; });
        reject([&] { t()->ne[2]=2; }); reject([&] { t()->ne[3]=2; });
        reject([&] { t()->view_src=&a; }); reject([&] { t()->view_offs=4; });
        for (int axis=0;axis<4;++axis) reject([&] { t()->nb[axis]=SIZE_MAX; });
        reject([&] { std::swap(t()->nb[0],t()->nb[1]); });
        reject([&] { t()->data=nullptr; }); reject([&] { t()->data=(char *)t()->data+1; });
        reject([&] { t()->buffer=nullptr; });
        reject([&] { t()->buffer->buft=&split_type; }); reject([&] { t()->buffer->buft=&cpu_type; });
        reject([&] { t()->buffer->buft=&cpu_type; t()->extra=(void *)1; });
        reject([&] { t()->buffer->buft=&split_type; t()->extra=(void *)1; });
        reject([&] { t()->buffer->context=nullptr; });
        reject([&] { ((ggml_backend_sycl_buffer_context *)t()->buffer->context)->device=1; });
        reject([&] { t()->buffer->size=4; });
        reject([&] { t()->buffer->size=SIZE_MAX; });
        reject([&] { t()->data=(char *)((ggml_backend_sycl_buffer_context *)t()->buffer->context)->base+t()->buffer->size; });
        ggml_tensor_extra_gpu extra{{true}};
        reject([&] { t()->extra=&extra; });
    }
    reject([&] { d.buffer=&bb; d.data=b.data; });
    reject([&] { d.buffer=&ab; d.data=a.data; });
    reset(); ggml_tensor_extra_gpu plain{{false}}; a.extra=&plain;
    ggml_sycl_mul_mat(ctx,&a,&b,&d); assert(calls.size()==2);
    printf("router preflight: %d exclusions; distinct columns, metadata, canaries, late-preflight no-partial-launch OK\n", rejected);
}
"""
        program = prelude + helper + '\nstatic void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {\n' + dispatch + spy
        cpp, executable = self.root / 'router-preflight.cpp', self.root / 'router-preflight'
        cpp.write_text(program)
        command = ['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                   '-I' + str(M['PARENT'] / 'ggml/include'), '-I' + str(M['PARENT'] / 'ggml/src'),
                   str(cpp), '-o', str(executable)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('late-preflight no-partial-launch OK', result.stdout)
        self.assertNotIn('runtime error:', result.stderr)
        print(result.stdout.strip())

    def test_observer_source_guards_copy_lifetime_once_and_queue(self):
        source, _ = self.patched_sources()
        before = cpp_body(source, 'bolt_moe_before')
        span = cpp_body(source, 'bolt_moe_span')
        copy_body = cpp_body(source, 'bolt_moe_copy')
        after = cpp_body(source, 'bolt_moe_after')
        for required in ('BOLT_MOE_OPERAND_DIR', 'if (!directory || !*directory) return capture;',
                         'blk.1.ffn_gate_inp.weight', 'blk.1.ffn_latent_down.weight',
                         'x->ne[1] < 1 || x->ne[1] > 2', 'errno == ENOENT', 'O_NOFOLLOW',
                         'st.st_size == 2', 'S_ISREG(st.st_mode)', 'if (seen[slot]) return capture;',
                         '"armed_width"', '"backend_changed"', '"projection_order"',
                         '"reorder_expectation"', 'capture.queue = ctx.stream()',
                         'capture.queue->is_in_order()'):
            self.assertIn(required, before)
        self.assertLess(before.index('if (!directory'), before.index('open('))
        self.assertLess(before.index('x->ne[1] > 2'), before.index('open('))
        self.assertLess(before.index('if (seen[slot])'), before.index('"armed_width"'))
        self.assertLess(before.index('seen[slot] = true'), before.index('bolt_moe_copy'))
        self.assertLess(before.index('bolt_moe_span(ctx, dst'), before.index('bolt_moe_copy'))
        for required in ('GGML_TYPE_Q8_0', 'GGML_TYPE_F32', 'rows % QK8_0 == 0',
                         't->ne[2] == 1', 't->ne[3] == 1', 't->nb[3] == bytes',
                         'ggml_is_contiguous(t)', 'ggml_backend_buffer_is_sycl',
                         '->device == ctx.device', 'data >= base', 'data - base <= size',
                         'bytes <= size - (data - base)', 'q8_weight ? 16 : 4'):
            self.assertIn(required, span)
        self.assertLess(span.index('"shape"'), span.index('const size_t row'))
        self.assertIn('O_EXCL | O_NOFOLLOW | O_CLOEXEC', copy_body)
        self.assertIn('std::vector<uint8_t> host(bytes);', copy_body)
        self.assertIn('memcpy(host.data(), capture.data[index], bytes).wait_and_throw();', copy_body)
        self.assertIn('capture.queue->wait();\n            throw;', copy_body)
        self.assertLess(copy_body.index('.wait_and_throw()'), copy_body.index('write(fd'))
        self.assertLess(copy_body.index('close(fd) == 0'), copy_body.index('GGML_LOG_INFO'))
        self.assertIn('capture.queue == ctx.stream()', after)
        self.assertIn('capture.tensors[i]->data == capture.data[i]', after)
        self.assertIn('bolt_moe_copy(capture, ctx, 2)', after)
        added = source[source.index('// Temporary, default-off block1 MoE observer.'):source.index('static bool ggml_sycl_compute_forward')]
        for forbidden in ('parallel_for(', 'submit(', 'quantize_row_', 'opt_for_reorder(', 'new sycl::queue', 'ctx.pool(', 'memset('):
            self.assertNotIn(forbidden, added)
        common = (M['PARENT'] / 'ggml/src/ggml-sycl/common.hpp').read_text()
        dpct = (M['PARENT'] / 'ggml/src/ggml-sycl/dpct/helper.hpp').read_text()
        self.assertIn('return stream(device, 0);', common)
        self.assertIn('default_queue() { return in_order_queue(); }', dpct)
        # Actual pointer/span equations in the unchanged consumer, not a guessed allocation.
        wrapper = cpp_body(source, 'ggml_sycl_op_mul_mat')
        for required in ('dev[i].src0_dd = (char *) src0->data;', 'dev[i].src1_ddf = (float *) src1->data;',
                         'dev[i].dst_dd = (float *) dst->data;', 'const int64_t is = split ?',
                         'queue_ptr stream = ctx.stream(i, is);'):
            self.assertIn(required, wrapper)

    def test_reorder_guard_null_safe_for_f32_and_q8(self):
        source, _ = self.patched_sources()
        helper = cpp_body(source, 'bolt_route_reordered')
        before = cpp_body(source, 'bolt_moe_before')
        after = cpp_body(source, 'bolt_moe_after')
        self.assertIn('weight ? static_cast<const ggml_tensor_extra_gpu *>(weight->extra) : nullptr', helper)
        self.assertIn('return extra && extra->optimized_feature.reorder;', helper)
        self.assertIn('bolt_route_reordered(w) == reordered', before)
        self.assertIn('bolt_route_reordered(capture.tensors[0]) == capture.reordered', after)
        self.assertNotIn('extra && extra->optimized_feature.reorder ==', before + after)

        def helper_result(extra):
            return bool(extra is not None and extra.get('reorder', False))

        cases = (
            ('F32 null extra', None, False, False, True),
            ('F32 installed reorder', {'reorder': True}, True, False, False),
            ('Q8 null extra', None, False, True, False),
            ('Q8 unset reorder', {'reorder': False}, False, True, False),
            ('Q8 reordered', {'reorder': True}, True, True, True),
        )
        for label, extra, helper_expected, guard_expected, accepted in cases:
            actual = helper_result(extra)
            self.assertEqual(actual, helper_expected, label)
            self.assertEqual(actual == guard_expected, accepted, label)

    def test_operand_marker_cleanup_and_separate_mount(self):
        directory = self.root / 'operands'
        directory.mkdir(mode=0o700)
        for label, width in (('S0', '1\n'), ('C0', '2\n')):
            def request(url, path, payload):
                self.assertEqual((directory/'armed').read_text(), width)
                self.assertEqual(path, '/v1/chat/completions')
                return {'ok': True}
            self.assertEqual(M['operand_diagnostic'](request, 'url', {}, directory, label), {'ok': True})
            self.assertEqual(list(directory.iterdir()), [])
            for error in (RuntimeError('http'), KeyboardInterrupt(), SystemExit()):
                def failed(*args):
                    self.assertTrue((directory/'armed').exists())
                    raise error
                with self.assertRaises(type(error)):
                    M['operand_diagnostic'](failed, 'url', {}, directory, label)
                self.assertEqual(list(directory.iterdir()), [])
        (directory/'stale').touch()
        with self.assertRaises(AssertionError):
            M['operand_diagnostic'](request, 'url', {}, directory, 'S0')
        for label in M['ORDER']:
            capture = M['ROOT']/'run-r1'/label/'capture'
            argv = M['make_create'](BASE, label, IMAGE, capture, OLD)
            self.assertIn(str(capture)+':/capture', argv)
            self.assertIn(str(capture.parent/'moe-operands')+':/moe-operands', argv)
            self.assertIn('BOLT_MOE_OPERAND_DIR=/moe-operands', argv)
        body = Path('run-trace.py').read_text()
        self.assertLess(body.index("for stem in ('warmup', 'ordinary')"), body.index('body = operand_diagnostic('))
        self.assertIn('operands.mkdir(mode=0o700)', body)

    def test_operand_strict_records_bounds_and_malformed(self):
        row = operand_fixture_row('blk.1.ffn_gate_inp.weight', 'src1', 2)
        line = operand_line(row)
        self.assertEqual(A['operand_record'](line), row)
        for key, value in [('weight', 'blk.1.unknown.weight'), ('role', 'unknown'), ('phase', 'after'),
                           ('backend_ctx', 'nil'), ('queue', '0x0'), ('device', 1), ('n', 3), ('reorder', 1),
                           ('type', 'f16'), ('ne0', 1), ('ne2', 2), ('nb0', 8), ('nb1', 4), ('nb3', 0),
                           ('data', '0x1001'), ('base', '0x2000'), ('offset', 1), ('buffer_bytes', 1),
                           ('bytes', row['bytes']-1), ('file', '../escape')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                A['operand_record'](operand_line(dict(row, **{key:value})))
        for bad in (line+' n=2', line.replace(' n=2', ''), line+' unknown=1', line.replace(' n=2',' n=-1'),
                    line.replace(' n=2', ' n=2.0'), line+' BOLT_ROUTE route=BAD', 'BOLT_MOE_OPERAND error=span'):
            with self.assertRaises(AssertionError):
                A['operand_record'](bad)

    def test_operand_file_truncated_nonfinite_symlink_and_strided_column(self):
        directory = self.root/'operands'
        directory.mkdir()
        row = operand_fixture_row('blk.1.ffn_latent_down.weight', 'src1', 2)
        path = directory/row['file']
        good = array.array('f', [1.] * row['ne0'] + [2.] * row['ne0']).tobytes()
        path.write_bytes(good)
        self.assertEqual(A['first_operand_column'](A['operand_bytes'](directory, row), row), good[:row['nb1']])
        for bad in (good[:-1], good+b'x', array.array('f', [float('nan')]).tobytes()+good[4:],
                    array.array('f', [float('inf')]).tobytes()+good[4:]):
            path.write_bytes(bad)
            with self.assertRaises(AssertionError):
                A['operand_bytes'](directory, row)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            A['operand_bytes'](directory, row)
        target = directory/'target'; target.write_bytes(good); path.symlink_to(target)
        with self.assertRaises(OSError):
            A['operand_bytes'](directory, row)
        strided = dict(row, ne0=2, nb0=8)
        self.assertEqual(A['first_operand_column'](b'abcd----efgh----', strided), b'abcdefgh')
        with self.assertRaises(AssertionError):
            A['first_operand_column'](b'abcd', strided)

    def test_historical_ssm_capture_mixed_logger_order_is_not_moe_evidence(self):
        # This is preserved parser evidence from the prior SSM observer, not a MoE capture.
        cell = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-ssm-operands/run-r1/S0')
        lines = (cell/'diagnostic-window.log').read_text().splitlines()
        first = next(i for i, line in enumerate(lines) if 'BOLT_SSM_OPERAND' in line)
        src = next(i for i in range(first, len(lines)) if 'BOLT_SSM_OPERAND' in lines[i] and 'role=src1 ' in lines[i])
        route = next(i for i in range(first, len(lines)) if 'BOLT_ROUTE' in lines[i] and 'weight=blk.0.ssm_in.weight ' in lines[i])
        self.assertLess(route, src)  # Direct-stderr route position is not common-logger copy order.
        self.assertFalse(any('BOLT_MOE_OPERAND' in line for line in lines))

    def test_both_arm_parent_parity_and_controller_return_fields(self):
        for label in ('S0', 'C0'):
            source = A['parent_reference'](label)[0]
            # Real original requests, text, emitted tokens, raw vectors and route logs.
            # Only the new six operand records/files are synthetic CPU fixtures.
            cell, _ = operand_cell(self.root/label/'full-arm', label)
            for name in ('ordinary.result.json', 'warmup.result.json', 'diagnostic.response.json',
                         'diagnostic.request.json', 'ordinary.request.json', 'warmup.request.json'):
                shutil.copy2(source/name, cell/name)
            shutil.copytree(source/'capture', cell/'capture')
            result = A['analyze_arm'](cell, label, OLD)
            self.assertTrue(result['observer_parent_invariant_passed'])
            self.assertTrue(result['operand_analysis']['complete'])
            controller = ast.parse(Path('run-trace.py').read_text())
            keys = {node.slice.value for node in ast.walk(controller) if isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name) and node.value.id == 'checked'
                    and isinstance(node.slice, ast.Constant)}
            self.assertTrue(keys <= result.keys())
            report = result['parent_comparison']
            for field in ('prompt_ids_match', 'emitted_ids_match', 'ordinary_text_matches_parent',
                          'diagnostic_text_matches_parent', 'diagnostic_token_ids_match_parent',
                          'actual_batch_inputs_match', 'selected_positions_rows_match'):
                with self.subTest(label=label, field=field), self.assertRaises(AssertionError):
                    A['assert_parent_observer_invariant'](dict(report, **{field: False}))
            # The last vector is mandatory too, not only through the cross-arm divergence.
            path = cell/'capture'/result['samples'][-1]['file']
            data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
            with patch.dict(A['analyze_arm'].__globals__, analyze_operands=lambda *args: self.fail('capture interpretation before parity')):
                with self.assertRaises(AssertionError):
                    A['analyze_arm'](cell, label, OLD)

    def test_operand_compare_only_after_both_parity_and_localization(self):
        arms = {}
        for label in ('S0', 'C0'):
            cell, alignment = operand_cell(self.root/label/'compare', label)
            metadata = A['analyze_operands'](cell, label, alignment)
            comparison = dict(compared_through_ordinal=128, comparisons=[{'raw_bytes_equal': True}]*128,
                              first_output_difference_ordinal=None, first_raw_difference=None,
                              **{k: True for k in ('prompt_ids_match', 'emitted_ids_match', 'ordinary_text_matches_parent',
                              'diagnostic_text_matches_parent', 'diagnostic_token_ids_match_parent',
                              'actual_batch_inputs_match', 'selected_positions_rows_match')})
            arms[label] = dict(observer_parent_invariant_passed=True, parent_comparison=comparison, operand_analysis=metadata)
        result = A['compare_operands'](arms)
        self.assertEqual(result['localization'], 'block1_router_and_latent_down_excluded_in_this_fixture')
        for label in ('S0', 'C0'):
            arms[label]['observer_parent_invariant_passed'] = False
            with patch.dict(A['compare_operands'].__globals__, operand_bytes=lambda *args: self.fail('read before both parity')):
                with self.assertRaises(AssertionError):
                    A['compare_operands'](arms)
            arms[label]['observer_parent_invariant_passed'] = True
        for weight, role, expected in [('blk.1.ffn_gate_inp.weight','src1','before_block1_router'),
                                       ('blk.1.ffn_gate_inp.weight','weight','router_weight_bytes_differ'),
                                       ('blk.1.ffn_gate_inp.weight','dst','block1_router_projection'),
                                       ('blk.1.ffn_latent_down.weight','src1','between_router_and_latent_down'),
                                       ('blk.1.ffn_latent_down.weight','weight','latent_down_weight_bytes_differ'),
                                       ('blk.1.ffn_latent_down.weight','dst','latent_down_projection')]:
            meta = arms['C0']['operand_analysis']; row = meta['projections'][weight][role]
            path = Path(meta['directory'])/row['file']
            with path.open('r+b') as stream: stream.write(b'\x01')
            self.assertEqual(A['compare_operands'](arms)['localization'], expected)
            with path.open('r+b') as stream: stream.write(b'\x00')

    def test_build_bootstrap_and_protected_source_fail_closed(self):
        script = Path('build-candidate.sh').read_text()
        parent = (PARENT_TOP/'build-candidate.sh').read_text()
        bootstrap = 'apt-get update -qq\nDEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config'
        self.assertIn(bootstrap, script)
        self.assertIn(bootstrap, parent)
        self.assertEqual(script.count('apt-get '), 2)
        self.assertIn('--config Release -j6 --target ggml-sycl', script)
        self.assertNotIn('--target llama-server', script)
        self.assertIn("== {'libggml-sycl.so.0'}", script)
        self.assertIn('test ! -e "$D"', script)
        roots = {name: self.root/name for name in ('accepted','parent','candidate')}
        for path in roots.values():
            path.mkdir()
            (path/'untouched').write_text('same')
            target = path/M['SOURCE_SCOPE'][0]
            target.parent.mkdir(parents=True)
            target.write_text('parent')
        (roots['candidate']/M['SOURCE_SCOPE'][0]).write_text('observer')
        expected = {name: M['source_hashes'](roots[name]) for name in ('accepted','parent')}
        with patch.dict(M['check_protected_sources'].__globals__, ROOT=self.root,
                        ACCEPTED=roots['accepted'], PARENT=roots['parent'], CANDIDATE=roots['candidate']):
            (self.root/'protected-source-hashes.json').write_text(json.dumps(expected))
            M['check_protected_sources']()
            for name in roots:
                path = roots[name]/'untouched'
                path.write_text('bad')
                with self.assertRaises(AssertionError):
                    M['check_protected_sources']()
                path.write_text('same')

    def test_actual_saved_projection_reorder_for_every_role(self):
        root=M['ROOT'].parent/'20261017-nemotron35-bolt-lilicorr-moe-operands'
        for cell in (root/'run-r1/S0',root/'continuation-c0/run-r1/C0'):
            lines=[line for line in (cell/'diagnostic-window.log').read_text().splitlines() if 'BOLT_MOE_OPERAND' in line]
            self.assertEqual(len(lines),6)
            for line in lines:
                row=A['operand_record'](line)
                self.assertEqual(row['reorder'],int(row['weight']=='blk.1.ffn_latent_down.weight'))
                A['operand_bytes'](cell/'moe-operands',row)
                with self.assertRaises(AssertionError): A['operand_record'](operand_line(dict(row,reorder=1-row['reorder'])))

    def test_operand_attribution_order_duplicates_and_outside(self):
        for label in ('S0','C0'):
            cell,alignment=operand_cell(self.root/('operand-'+label),label)
            good=(cell/'diagnostic-window.log').read_text()
            self.assertTrue(A['analyze_operands'](cell,label,alignment)['complete'])
            lines=good.splitlines()
            records=[line for line in lines if 'BOLT_MOE_OPERAND' in line]
            first=records[0]
            mutations=[good.replace(first,'',1),good.replace(first,first+'\n'+first,1),
                       first+'\n'+good, good.replace(first,first.replace('queue=0x2000','queue=0x3000'),1),
                       good.replace(first,operand_line(dict(A['operand_record'](first),backend_ctx='0xbeef')),1),
                       good.replace(records[0]+'\n'+records[1],records[1]+'\n'+records[0],1)]
            outer=next(line for line in lines if 'weight=blk.1.ffn_gate_inp.weight ' in line and 'route=MUL_MAT_SYCL' in line)
            inner=next(line for line in lines if 'weight=blk.1.ffn_gate_inp.weight ' in line and 'route=LIBRARY_MKL_F32' in line)
            mutations += [good.replace(outer,'',1),good.replace(inner,inner+'\n'+inner,1),
                          good.replace(inner,inner.replace('route=LIBRARY_MKL_F32','route=LIBRARY_DNNL_F32'),1)]
            for bad in mutations:
                (cell/'diagnostic-window.log').write_text(bad)
                with self.assertRaises(AssertionError): A['analyze_operands'](cell,label,alignment)
            # Both FIFOs stay intact even if ALL direct routes print before or
            # after the complete common logger stream, outside every batch/copy.
            direct=[line for line in lines if 'BOLT_ROUTE ' in line]
            common=[line for line in lines if 'BOLT_ROUTE ' not in line]
            for shifted in (direct+common,common+direct):
                (cell/'diagnostic-window.log').write_text('\n'.join(shifted)+'\n')
                self.assertTrue(A['analyze_operands'](cell,label,alignment)['complete'])
            (cell/'diagnostic-window.log').write_text(good)
            (cell/'pre-diagnostic.log').write_text(first)
            self.assertRaises(AssertionError,A['analyze_operands'],cell,label,alignment)
            (cell/'pre-diagnostic.log').write_text('')
            (cell/'moe-operands/stale').touch()
            self.assertRaises(AssertionError,A['analyze_operands'],cell,label,alignment)

    def test_real_c0_singleton_pairs_correlate_with_full_width2_capture(self):
        source,_=A['parent_reference']('C0')
        log=(source/'diagnostic-window.log').read_text()
        frames=A['assert_router_model_routes'](log,'C0')
        self.assertEqual(len(frames),67)
        self.assertTrue(all(frame['batch']['n']==2 for frame in frames))
        for frame in frames:
            self.assertEqual([r['route'] for r in frame['routes']],['MUL_MAT_SYCL','LIBRARY_MKL_F32']*2)
            self.assertTrue(all(r['src1_ne1']==r['dst_ne1']==1 for r in frame['routes']))
            self.assertTrue(all(r['src1_nb2']==r['src1_nb3']==32768 and r['dst_nb2']==r['dst_nb3']==1920 for r in frame['routes']))
        cell,alignment=operand_cell(self.root/'c0-full-width','C0')
        metadata=A['analyze_operands'](cell,'C0',alignment)
        for role in ('src1','dst'):
            self.assertEqual(metadata['projections']['blk.1.ffn_gate_inp.weight'][role]['ne1'],2)
        good=(cell/'diagnostic-window.log').read_text()
        lines=good.splitlines()
        target=frames[0]['target_backend_ctx']
        # Mutate one REAL width2 singleton pair, not a fabricated expected record.
        indexes=[i for i,line in enumerate(lines) if 'BOLT_ROUTE ' in line and
                 (lambda r: r['weight']=='blk.1.ffn_gate_inp.weight' and r['backend_ctx']==target and r['src1_nb2']==32768)(A['route_records'](line)[0])][:4]
        self.assertEqual(len(indexes),4)
        for mutation in ('drop_second','packed_strides','unsplit_shape','wrong_inner'):
            bad=list(lines)
            if mutation=='drop_second':
                for i in reversed(indexes[2:]): bad.pop(i)
            else:
                for i in indexes:
                    row=A['route_records'](bad[i])[0]
                    if mutation=='packed_strides': row.update(src1_nb2=16384,src1_nb3=16384,dst_nb2=960,dst_nb3=960)
                    elif mutation=='unsplit_shape': row.update(src1_ne1=2,dst_ne1=2)
                    elif i==indexes[-1]: row['route']='LIBRARY_DNNL_F32'
                    bad[i]='BOLT_ROUTE '+' '.join(f'{k}={v}' for k,v in row.items())
            (cell/'diagnostic-window.log').write_text('\n'.join(bad)+'\n')
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                A['analyze_operands'](cell,'C0',alignment)

if __name__ == '__main__':
    unittest.main(verbosity=2)
