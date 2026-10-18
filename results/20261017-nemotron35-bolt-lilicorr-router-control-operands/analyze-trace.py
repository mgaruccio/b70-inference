#!/usr/bin/env python3
"""Fail-closed analysis of this development-only router-control operand observer; no timing or promotion."""
import array
import json
import math
import os
import re
import runpy
import socket
import stat
import sys
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-control-operands')
WIDTH1 = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1'
PARENT_RESULTS = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-router-single2'
FIELDS = {
    'batch': 'ctx off n tokens pos outputs seq', 'decoded': 'ctx ret',
    'sample': 'ctx smpl idx selected vocab status file', 'single': 'ctx smpl slot row id',
    'verify': 'ctx smpl slot rows drafts emitted accepted rollback replay rejection synth',
    'emit': 'ctx slot ordinal token', 'remove': 'ctx slot kind p0 p1 rollback',
    'checkpoint': 'ctx slot p0 p1 rollback',
}
LISTS = {'tokens', 'pos', 'outputs', 'rows', 'drafts', 'emitted'}
STRINGS = {'ctx', 'smpl', 'status', 'file', 'kind', 'seq'}
ROUTE_FIELDS = {
    'route', 'weight', 'aux_weight', 'op', 'op_id', 'dst_name',
    'src0_type', 'aux_type', 'src1_type', 'dst_type',
    'ne0', 'ne1', 'ne2', 'ne3', 'nb0', 'nb1', 'nb2', 'nb3',
    'src1_ne0', 'src1_ne1', 'src1_ne2', 'src1_ne3',
    'src1_nb0', 'src1_nb1', 'src1_nb2', 'src1_nb3',
    'dst_ne0', 'dst_ne1', 'dst_ne2', 'dst_ne3',
    'dst_nb0', 'dst_nb1', 'dst_nb2', 'dst_nb3',
    'ids_ne0', 'ids_ne1', 'ids_ne2', 'ids_ne3',
    'ids_nb0', 'ids_nb1', 'ids_nb2', 'ids_nb3',
    'reorder', 'src1_prec', 'operand_prec', 'accum_prec', 'dynamic_prec',
    'device', 'backend_ctx', 'expert', 'result',
}
ROUTE_STRINGS = {
    'route', 'weight', 'aux_weight', 'op', 'dst_name',
    'src0_type', 'aux_type', 'src1_type', 'dst_type', 'backend_ctx',
}
ROUTE_INTS = ROUTE_FIELDS - ROUTE_STRINGS
ROUTE_NAMES = {
    'LIBRARY_DNNL_BF16', 'LIBRARY_FUSED_DEQUANT_GEMM', 'LIBRARY_DNNL_F16',
    'LIBRARY_MKL_F16', 'LIBRARY_DNNL_F32', 'LIBRARY_MKL_F32',
    'F16_P021', 'F16_BATCHED_SYCL', 'F16_NC',
    'DMMV_ESIMD', 'DMMV_REORDERED', 'DMMV_PLAIN',
    'MMVQ_REORDERED', 'MMVQ_PLAIN', 'MMQ', 'MUL_MAT_SYCL',
    'GATEUP_GLU_PLAIN', 'GATEUP_GLU_REORDERED',
    'MOE_SINGLE_MMVQ_REORDERED_Q8_1', 'MOE_SINGLE_MMVQ_PLAIN_Q8_1',
    'MOE_PER_EXPERT_FALLBACK', 'MOE_GROUPED_GEMM_ATTEMPT', 'MOE_GROUPED_GEMM_RESULT',
}

def load(path):
    return json.loads(path.read_text())


def records(log):
    result = []
    for line in log.splitlines():
        if 'BOLT_VERIFY' not in line:
            continue
        assert line.count('BOLT_VERIFY ') == 1, 'bad trace framing'
        tail = line.split('BOLT_VERIFY ', 1)[1]
        words = tail.split()
        assert words and words[0] in FIELDS, ('capture error or unknown record', tail)
        kind, pairs = words[0], words[1:]
        assert all(re.fullmatch(r'[a-z_][a-z_0-9]*=[^\s=]+', p) for p in pairs), tail
        fields = [p.split('=', 1) for p in pairs]
        assert len({k for k, _ in fields}) == len(fields), 'duplicate field'
        assert {k for k, _ in fields} == set(FIELDS[kind].split()), tail
        row = dict(event=kind, event_index=len(result))
        for key, value in fields:
            if key in LISTS:
                assert value == '-' or re.fullmatch(r'-?\d+(,-?\d+)*', value), tail
                row[key] = [] if value == '-' else list(map(int, value.split(',')))
            elif key in STRINGS:
                if key in ('ctx', 'smpl'):
                    assert re.fullmatch(r'0x[0-9a-fA-F]+', value), tail
                row[key] = value
            else:
                assert re.fullmatch(r'-?\d+', value), tail
                row[key] = int(value)
        result.append(row)
    assert result, 'empty diagnostic trace'
    return result

def route_records(log):
    result = []
    for line in log.splitlines():
        if 'BOLT_ROUTE' not in line:
            continue
        assert 'BOLT_VERIFY' not in line, 'mixed trace records'
        assert line.count('BOLT_ROUTE ') == 1, 'bad route trace framing'
        tail = line.split('BOLT_ROUTE ', 1)[1]
        pairs = tail.split()
        assert pairs and all(re.fullmatch(r'[a-z_][a-z_0-9]*=[^\s=]+', p) for p in pairs), tail
        fields = [p.split('=', 1) for p in pairs]
        assert len({k for k, _ in fields}) == len(fields), 'duplicate route field'
        assert {k for k, _ in fields} == ROUTE_FIELDS, tail
        row = {}
        for key, value in fields:
            if key in ROUTE_STRINGS:
                if key == 'backend_ctx':
                    assert re.fullmatch(r'0x[0-9a-fA-F]+', value), tail
                elif key == 'route':
                    assert re.fullmatch(r'[A-Z0-9_]+', value), tail
                    assert value in ROUTE_NAMES, value
                row[key] = value
            else:
                assert re.fullmatch(r'-?\d+', value), tail
                row[key] = int(value)
        result.append(row)
    assert result, 'empty route trace'
    return result


def _verify_frames(log):
    """Parse the FIFO common logger without using direct-stderr positions."""
    frames = []
    active = None
    for line_index, line in enumerate(log.splitlines()):
        if 'BOLT_VERIFY' not in line:
            continue
        assert not ('BOLT_VERIFY' in line and 'BOLT_ROUTE' in line), 'mixed trace records'
        assert 'BOLT_VERIFY ' in line, 'malformed BOLT_VERIFY framing'
        events = records(line)
        assert len(events) == 1
        event = events[0]
        if event['event'] == 'batch':
            assert active is None, 'nested BOLT_VERIFY batch'
            active = dict(batch=event, verify_line=line_index)
        elif event['event'] == 'decoded':
            assert active is not None, 'decoded without batch'
            assert event['ctx'] == active['batch']['ctx'], 'decoded context mismatch'
            assert event['ret'] == 0, 'decoded failed'
            active['decoded'] = event
            active['decoded_line'] = line_index
            frames.append(active)
            active = None
    assert active is None, 'unterminated BOLT_VERIFY batch'
    assert frames, 'no BOLT_VERIFY batch frames'
    return frames


def _route_stream(log):
    """Return the direct route logger in its own FIFO order."""
    result = []
    for line_index, line in enumerate(log.splitlines()):
        if 'BOLT_ROUTE' not in line:
            continue
        assert 'BOLT_VERIFY' not in line, 'mixed trace records'
        assert line.count('BOLT_ROUTE ') == 1, 'bad route trace framing'
        result.append(dict(route=route_records(line)[0], line_index=line_index))
    assert result, 'empty route trace'
    return result


def _route_original_width(route):
    """Recover the original column extent from retained strides, not labels."""
    if route['weight'].endswith(('.ffn_up_exps.weight', '.ffn_down_exps.weight')):
        numerator, denominator = route['src1_nb3'], route['src1_nb2']
    else:
        numerator, denominator = route['src1_nb2'], route['src1_nb1']
    assert denominator > 0 and numerator > 0 and numerator % denominator == 0, route
    width = numerator // denominator
    assert 1 <= width <= 1024, route
    return width


def _target_group_width(routes):
    widths = [_route_original_width(route) for route in routes
              if route['weight'] != 'output.weight']
    assert widths and len(set(widths)) == 1, ('inconsistent target route widths', widths)
    return widths[0]


def _expected_target_weights(n, label=None, output=False):
    assert n in (1, 2, 3, 4, 5, 46)
    pair = 2 if n == 46 else 1
    router_pairs = 2 if label == 'C0' and n == 2 else 1
    expert_pairs = 2 if n >= 2 else 1
    weights = (['blk.0.ssm_in.weight'] * pair
               + ['blk.0.ssm_out.weight'] * pair
               + ['blk.1.ffn_gate_inp.weight'] * (2 * router_pairs)
               + ['blk.1.ffn_latent_down.weight'] * pair
               + ['blk.1.ffn_up_exps.weight'] * expert_pairs
               + ['blk.1.ffn_down_exps.weight'] * expert_pairs
               + ['blk.1.ffn_latent_up.weight'] * pair
               + ['blk.1.ffn_up_shexp.weight'] * pair
               + ['blk.1.ffn_down_shexp.weight'] * pair
               + ['blk.7.attn_q.weight'] * pair
               + ['blk.7.attn_v.weight'] * pair
               + ['blk.7.attn_k.weight'] * pair
               + ['blk.7.attn_output.weight'] * pair)
    if output:
        weights.append('output.weight')
    return weights


def _assert_route_pair(rows, inner_route, inner_operand_prec=None):
    assert len(rows) == 2, rows
    outer, inner = rows
    assert outer['route'] == 'MUL_MAT_SYCL' and inner['route'] == inner_route, rows
    assert outer['operand_prec'] == -1, rows
    if inner_operand_prec is not None:
        assert inner['operand_prec'] == inner_operand_prec, rows
    assert {k: v for k, v in outer.items() if k not in ('route', 'operand_prec')} == {
        k: v for k, v in inner.items() if k not in ('route', 'operand_prec')}, rows


def _assert_prefill_frame(frame):
    """Account for the three real prompt chunks before post-prompt gates."""
    batch = frame['batch']
    n = batch['n']
    if batch['pos'][0] >= 55 or n not in (4, 5, 46):
        return
    routes = frame['routes']
    expected = _expected_target_weights(n, output=any(batch['outputs']))
    assert [route['weight'] for route in routes] == expected, (batch, routes)
    router = [route for route in routes if route['weight'] == 'blk.1.ffn_gate_inp.weight']
    assert_router_operations(router, n, split=False,
                             inner_route='LIBRARY_DNNL_F32' if n == 46 else 'LIBRARY_MKL_F32')
    if n == 46:
        for weight in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight'):
            _assert_route_pair([route for route in routes if route['weight'] == weight],
                               'LIBRARY_DNNL_F32', inner_operand_prec=10)
        for weight in ('blk.1.ffn_latent_down.weight', 'blk.1.ffn_latent_up.weight'):
            _assert_route_pair([route for route in routes if route['weight'] == weight],
                               'LIBRARY_DNNL_F16', inner_operand_prec=20)
        for weight in ('blk.1.ffn_up_shexp.weight', 'blk.1.ffn_down_shexp.weight'):
            _assert_route_pair([route for route in routes if route['weight'] == weight],
                               'LIBRARY_FUSED_DEQUANT_GEMM', inner_operand_prec=-1)
        for weight in ('blk.7.attn_q.weight', 'blk.7.attn_v.weight',
                       'blk.7.attn_k.weight', 'blk.7.attn_output.weight'):
            _assert_route_pair([route for route in routes if route['weight'] == weight],
                               'LIBRARY_DNNL_F16', inner_operand_prec=20)
    else:
        assert all(route['route'] == 'MMVQ_REORDERED'
                   for route in routes
                   if route['weight'] in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight',
                                          'blk.1.ffn_latent_down.weight',
                                          'blk.1.ffn_latent_up.weight'))
        assert all(route['route'] == 'MMVQ_PLAIN'
                   for route in routes
                   if route['weight'] in ('blk.1.ffn_up_shexp.weight',
                                          'blk.1.ffn_down_shexp.weight'))
        for weight in ('blk.1.ffn_up_exps.weight', 'blk.1.ffn_down_exps.weight'):
            expert_routes = [route for route in routes if route['weight'] == weight]
            assert len(expert_routes) == 2, routes
            assert [route['route'] for route in expert_routes] == [
                'MOE_GROUPED_GEMM_ATTEMPT', 'MOE_GROUPED_GEMM_RESULT'], routes


def assert_frozen_prefill_batches(log):
    """Require the immutable three-batch prompt reservation before decode."""
    frames = _verify_frames(log)
    expected = [
        (5, list(range(0, 5))),
        (46, list(range(5, 51))),
        (4, list(range(51, 55))),
    ]
    actual = [(frame['batch']['n'], frame['batch']['pos']) for frame in frames[:3]]
    assert actual == expected, ('frozen prefill batches changed', actual)
    assert len(frames) >= 4 and all(frame['batch']['pos'][0] >= 55 for frame in frames[3:]), (
        'unexpected batch before post-prompt decode', frames[3:])


def route_frames(log):
    """Pair direct target-call order with FIFO BOLT_VERIFY batch order.

    BOLT_ROUTE is direct stderr while BOLT_VERIFY is an asynchronous common
    logger.  Their textual positions are deliberately not used to associate a
    target operation with a batch.  The unique block-0 SSM backend anchors
    target groups in the direct stream; groups and batches are then consumed in
    exact serial order, with foreign backend routes retained separately.
    """
    verify_frames = _verify_frames(log)
    stream = _route_stream(log)
    routes = [row['route'] for row in stream]
    anchors = [i for i, route in enumerate(routes)
               if route['weight'] in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight')]
    contexts = {routes[i]['backend_ctx'] for i in anchors}
    assert len(contexts) == 1, 'missing or ambiguous target SSM backend context'
    target_context = next(iter(contexts))
    group_starts = [i for i, route in enumerate(routes)
                    if route['backend_ctx'] == target_context
                    and route['weight'] == 'blk.0.ssm_in.weight'
                    and route['route'] not in ('LIBRARY_DNNL_F32', 'LIBRARY_MKL_F32')]
    assert group_starts, 'missing target SSM route anchors'
    assert not any(route['backend_ctx'] == target_context
                   for route in routes[:group_starts[0]]), (
        'target operation precedes first SSM anchor')
    groups = []
    for index, start in enumerate(group_starts):
        end = group_starts[index + 1] if index + 1 < len(group_starts) else len(routes)
        target_routes = [route for route in routes[start:end]
                         if route['backend_ctx'] == target_context]
        assert target_routes and target_routes[0] == routes[start], 'malformed target route group'
        groups.append(dict(routes=target_routes,
                           other_backend_routes=[route for route in routes[start:end]
                                                 if route['backend_ctx'] != target_context]))
    group_widths = [_target_group_width(group['routes']) for group in groups]
    assert len(groups) == len(verify_frames), (
        'target route/verify frame count mismatch', len(groups), len(verify_frames))
    frames = []
    for group_index, group in enumerate(groups):
        width = group_widths[group_index]
        verify = verify_frames[group_index]
        assert verify['batch']['n'] == width, (
            'target route/verify width mismatch', group_index, width, verify['batch'])
        frame = dict(batch=verify['batch'], decoded=verify['decoded'],
                     routes=group['routes'],
                     other_backend_routes=group['other_backend_routes'],
                     target_backend_ctx=target_context,
                     verify_line=verify['verify_line'],
                     decoded_line=verify['decoded_line'])
        _assert_target_frame_width(frame)
        _assert_prefill_frame(frame)
        frames.append(frame)
    assert len(frames) == len(verify_frames), 'target route/verify frame mismatch'
    if frames:
        prefix_other = [row['route'] for row in stream[:group_starts[0]]
                        if row['route']['backend_ctx'] != target_context
                        and row['line_index'] >= frames[0]['verify_line']]
        frames[0]['other_backend_routes'] = prefix_other + frames[0]['other_backend_routes']
        frames[0]['unframed_other_routes'] = [row['route'] for row in stream[:group_starts[0]]
                                              if row['route']['backend_ctx'] != target_context
                                              and row['line_index'] < frames[0]['verify_line']]
        frames[0]['unframed_target_routes'] = [row['route'] for row in stream[:group_starts[0]]
                                               if row['route']['backend_ctx'] == target_context]
    return frames


def _assert_target_frame_width(frame):
    batch = frame['batch']
    expected = batch['n']
    output_width = sum(batch['outputs'])
    for route in frame['routes']:
        width = _route_original_width(route)
        if route['weight'] == 'output.weight':
            assert width == output_width, (batch, route)
        else:
            assert width == expected, (batch, route)

def _route_batch_metadata(batch):
    return {key: batch[key] for key in ('off', 'n', 'tokens', 'pos', 'outputs', 'seq')}


def select_route_batches(log, label):
    assert label in ('S0', 'C0')
    frames = route_frames(log)
    if label == 'S0':
        chosen = [frame for frame in frames
                  if frame['batch']['n'] == 1 and frame['batch']['pos'][0] >= 55][:2]
        assert len(chosen) == 2, 'S0 route window missing first two singleton decode batches'
        assert [frame['batch']['pos'] for frame in chosen] == [[55], [56]]
    else:
        chosen = [frame for frame in frames
                  if frame['batch']['n'] == 2 and frame['batch']['pos'] == [55, 56]][:1]
        assert len(chosen) == 1, 'C0 route window missing first two-row verification batch'
    selected = []
    for frame in chosen:
        assert frame['routes'], 'selected target batch has no filtered route records'
        # Draft feature work may run inside the target decode call. Anchor the
        # target backend to Bolt's first-block SSM projections, not pointer types.
        target_context = frame['target_backend_ctx']
        target_routes = frame['routes']
        other_routes = frame['other_backend_routes']
        assert any(route['weight'] == 'output.weight' for route in target_routes), (
            'output projection route missing in target backend')
        selected.append(dict(batch=_route_batch_metadata(frame['batch']), routes=target_routes,
                             target_backend_ctx=target_context, other_backend_routes=other_routes))
    routes = [route for frame in selected for route in frame['routes']]
    assert any(route['weight'] == 'output.weight' for route in routes), 'output projection route missing'
    assert any(route['weight'].startswith('blk.0.') for route in routes), 'first-block route missing'
    assert any(route['weight'].startswith('blk.1.') for route in routes), 'first MoE block route missing'
    return dict(label=label, selected_batches=selected, frame_count=len(frames),
                selected_route_count=len(routes), actual_route_presence=True)



BLOCK7_ATTENTION_WEIGHTS = (
    ('blk.7.attn_q.weight', 4096, 4096),
    ('blk.7.attn_k.weight', 4096, 256),
    ('blk.7.attn_v.weight', 4096, 256),
    ('blk.7.attn_output.weight', 4096, 4096),
)


def _q8_weight_strides(k, m):
    return (34, k // 32 * 34, k // 32 * 34 * m, k // 32 * 34 * m)


def _f32_activation_strides(k, n):
    return (4, k * 4, k * n * 4, k * n * 4)

IQ2_PROJECTIONS = {
    'blk.1.ffn_up_exps.weight': (2048, 1024, 1),
    'blk.1.ffn_down_exps.weight': (1024, 2048, 9),
}


def assert_route_tensor(route, prefix, shape, strides):
    assert tuple(route[f'{prefix}ne{i}'] for i in range(4)) == shape, route
    assert tuple(route[f'{prefix}nb{i}'] for i in range(4)) == strides, route


def assert_iq2_route_common(route):
    assert route['weight'] in IQ2_PROJECTIONS, route
    assert route['op'] == 'MUL_MAT_ID' and route['op_id'] == 30, route
    assert route['src0_type'] == 'iq2_xxs' and route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
    assert route['reorder'] == 0 and route['device'] == 0 and route['operand_prec'] == -1, route
    assert route['aux_weight'] == '-' and route['aux_type'] == '-', route


def assert_iq2_singleton_route(route, original_n, n_mats):
    """Existing success-only logger; nb3 still describes the original token extent."""
    assert original_n in (1, 2)
    assert_iq2_route_common(route)
    assert route['route'] == 'MOE_SINGLE_MMVQ_PLAIN_Q8_1', route
    assert route['expert'] == -1 and route['result'] == -1, route
    m, k, b = IQ2_PROJECTIONS[route['weight']]
    row = k // 256 * 66
    assert_route_tensor(route, '', (k, m, n_mats, 1), (66, row, row * m, row * m * n_mats))
    assert_route_tensor(route, 'src1_', (k, b, 1, 1), (4, k * 4, k * b * 4, k * b * original_n * 4))
    assert_route_tensor(route, 'dst_', (m, 9, 1, 1), (4, m * 4, m * 9 * 4, m * 9 * original_n * 4))
    assert_route_tensor(route, 'ids_', (9, 1, 1, 1), (4, n_mats * 4, n_mats * original_n * 4, n_mats * original_n * 4))


def assert_actual_route_gates(route_analysis, label, expected_head_ne1=None):
    assert label in ('S0', 'C0')
    if expected_head_ne1 is None:
        expected_head_ne1 = 1 if label == 'S0' else 2
    latent_route = 'DMMV_ESIMD'
    expert_weights = ('blk.1.ffn_up_exps.weight', 'blk.1.ffn_down_exps.weight')
    for batch in route_analysis['selected_batches']:
        head_routes = [route for route in batch['routes'] if route['weight'] == 'output.weight']
        assert head_routes, 'output.weight missing from target backend routes'
        for route in head_routes:
            assert route['src0_type'] == 'q8_0', route
            assert route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
            assert route['route'] == 'MMVQ_REORDERED', route
            assert route['reorder'] == 1, route
            assert route['src1_ne1'] == expected_head_ne1, route
        for weight in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight'):
            ssm_routes = [route for route in batch['routes'] if route['weight'] == weight]
            assert ssm_routes, weight
            for route in ssm_routes:
                assert route['src0_type'] == 'nvfp4', route
                assert route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
                assert route['src1_ne1'] == expected_head_ne1, route
                assert route['route'] == 'MMVQ_REORDERED' and route['reorder'] == 1, route
        for weight in ('blk.1.ffn_latent_down.weight', 'blk.1.ffn_latent_up.weight'):
            latent_routes = [route for route in batch['routes'] if route['weight'] == weight]
            assert latent_routes, weight
            for route in latent_routes:
                assert route['src0_type'] == 'q8_0', route
                assert route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
                assert route['src1_ne1'] == expected_head_ne1, route
                assert route['route'] == latent_route and route['reorder'] == 1, route
        for weight in expert_weights:
            block_routes = [route for route in batch['routes'] if route['weight'] == weight]
            assert len(block_routes) == expected_head_ne1, (weight, block_routes)
            for route in block_routes:
                assert_iq2_singleton_route(route, expected_head_ne1, 240)

        for weight, k, m in BLOCK7_ATTENTION_WEIGHTS:
            attn_routes = [route for route in batch['routes'] if route['weight'] == weight]
            assert attn_routes, weight
            for route in attn_routes:
                assert route['src0_type'] == 'q8_0', route
                assert route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
                assert route['src1_ne1'] == expected_head_ne1, route
                assert route['expert'] == -1 and route['result'] == -1, route
                assert route['op'] == 'MUL_MAT', route
                assert tuple(route[f'ne{i}'] for i in range(4)) == (k, m, 1, 1), route
                assert tuple(route[f'nb{i}'] for i in range(4)) == _q8_weight_strides(k, m), route
                n = expected_head_ne1
                assert tuple(route[f'src1_ne{i}'] for i in range(4)) == (k, n, 1, 1), route
                assert tuple(route[f'src1_nb{i}'] for i in range(4)) == _f32_activation_strides(k, n), route
                assert tuple(route[f'dst_ne{i}'] for i in range(4)) == (m, n, 1, 1), route
                assert tuple(route[f'dst_nb{i}'] for i in range(4)) == _f32_activation_strides(m, n), route
                assert route['route'] == 'DMMV_ESIMD' and route['reorder'] == 1, route
                assert route['operand_prec'] == -1, route


def assert_router_operations(routes, n, split, layout=0, inner_route='LIBRARY_MKL_F32'):
    """An outer dispatch plus its inner library record is ONE operation, not two."""
    assert n in (1, 2, 3, 4, 5, 46) and layout in (0, 1, 2)
    assert not split or (n == 2 and layout == 0)
    calls = 2 if split else 1
    assert len(routes) == 2 * calls, ('router operation count', n, split, routes)
    view_n = 1 if split else n
    input_stride = (4096 + (16 if layout == 2 else 0)) * 4
    for i in range(calls):
        outer, inner = routes[2*i:2*i+2]
        assert (outer['route'], inner['route']) == ('MUL_MAT_SYCL', inner_route), (outer, inner)
        assert outer['operand_prec'] == -1 and inner['operand_prec'] == 10
        assert {k: v for k, v in outer.items() if k not in ('route', 'operand_prec')} == {
            k: v for k, v in inner.items() if k not in ('route', 'operand_prec')}, (outer, inner)
        for route in (outer, inner):
            assert route['op'] == 'MUL_MAT' and route['op_id'] == 29, route
            assert all(route[k] == 'f32' for k in ('src0_type', 'src1_type', 'dst_type')), route
            assert route['aux_weight'] == route['aux_type'] == '-', route
            assert route['reorder'] == route['device'] == 0, route
            assert route['src1_prec'] == route['accum_prec'] == route['dynamic_prec'] == 0, route
            assert route['expert'] == route['result'] == -1, route
            assert_route_tensor(route, '', (4096, 240, 1, 1), (4, 16384, 3932160, 3932160))
            assert_route_tensor(route, 'src1_', (4096, view_n, 1, 1),
                                (4, input_stride, input_stride*n, input_stride*n))
            assert_route_tensor(route, 'dst_', (240, view_n, 1, 1), (4, 960, 960*n, 960*n))
            assert all(route[f'ids_{field}{j}'] == 0 for field in ('ne', 'nb') for j in range(4)), route
    assert len({r['backend_ctx'] for r in routes}) == 1
    assert len({r['weight'] for r in routes}) == len({r['dst_name'] for r in routes}) == 1

def assert_router_model_routes(log, label):
    """Require activation on EVERY post-prompt target frame; keep nested drafter routes separate."""
    assert label in ('S0', 'C0')
    checked = []
    for frame in route_frames(log):
        batch = frame['batch']
        if batch['pos'][0] < 55:
            continue
        n = batch['n']
        assert n in ((1,) if label == 'S0' else (1, 2)), batch
        target = frame['target_backend_ctx']
        assert all(route['backend_ctx'] == target for route in frame['routes'])
        expected = _expected_target_weights(n, label, output=any(batch['outputs']))
        assert [route['weight'] for route in frame['routes']] == expected, (
            batch, frame['routes'])
        routers = [route for route in frame['routes']
                   if route['weight'] == 'blk.1.ffn_gate_inp.weight']
        assert routers, 'missing actual target router route'
        assert_router_operations(routers, n, split=(label == 'C0' and n == 2))
        assert_actual_route_gates(dict(selected_batches=[dict(batch=batch, routes=frame['routes'])]),
                                  label, expected_head_ne1=n)
        checked.append(dict(batch=_route_batch_metadata(batch), target_backend_ctx=target,
                            routes=routers,
                            other_backend_routes=frame['other_backend_routes']))
    assert checked, 'missing post-prompt router frames'
    if label == 'C0':
        assert any(x['batch']['n'] == 2 and x['batch']['pos'] == [55, 56] for x in checked), 'missing first two-column router control'
    return checked


def raw_bytes(directory, sample):
    """Read exactly one regular, bounded, non-symlink mkstemp file."""
    name, vocab = sample['file'], sample['vocab']
    assert sample['status'] == 'cpu', 'uncaptured/backend sample blocks attribution'
    assert re.fullmatch(r'row-[a-zA-Z0-9]{6}', name), 'capture path outside expected basename'
    assert 6833 < vocab <= 1_000_000, 'invalid vocabulary bound'
    directory = directory.resolve(strict=True)
    path = directory / name
    assert path.resolve(strict=True).parent == directory and not path.is_symlink(), 'capture escapes directory'
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        assert stat.S_ISREG(info.st_mode) and info.st_size == 4 * vocab, 'wrong F32 capture size/type'
        data = stream.read(4 * vocab + 1)
    assert len(data) == 4 * vocab
    return data


def score_summary(data):
    assert sys.byteorder == 'little', 'experiment captures are little-endian F32'
    values = array.array('f')
    assert values.itemsize == 4
    values.frombytes(data)
    assert not any(math.isnan(x) for x in values), 'NaN raw logits'
    argmax = max(range(len(values)), key=values.__getitem__)
    def number(value):
        return value if math.isfinite(value) else str(value)
    return dict(raw_argmax=argmax, raw_max=number(values[argmax]),
                score_6833=number(values[6833]), score_4618=number(values[4618]))


def within_arm_text_consistency(ordinary, diagnostic):
    assert ordinary['text'] == diagnostic['choices'][0]['message']['content'], (
        'ordinary/diagnostic text mismatch within arm')


def prior_reference_flags(ordinary, diagnostic, reference_ordinary, reference_diagnostic):
    return dict(
        reference_equivalence_required=True,
        ordinary_text_matches_prior_reference=(ordinary['text'] == reference_ordinary['text']),
        diagnostic_text_matches_prior_reference=(
            diagnostic['choices'][0]['message']['content']
            == reference_diagnostic['choices'][0]['message']['content']),
        diagnostic_token_ids_match_prior_reference=(
            diagnostic['__verbose']['tokens'] == reference_diagnostic['__verbose']['tokens']),
    )


def align(events, tokens, capture, prompt_tokens=55):
    assert len(tokens) == 128 and all(type(t) is int for t in tokens)
    batches, samples, removals, verifications, emissions = [], [], [], [], []
    pending, to_emit, current, last_verify = [], [], None, None
    context = sampler = slot_id = None
    prompt = {}
    files = set()
    for e in events:
        kind = e['event']
        if kind == 'batch':
            assert not pending and not to_emit, 'unmatched sample/emission at next decode'
            assert current is None or current['decoded'], 'decode result missing'
            if context is None:
                context = e['ctx']
            assert e['ctx'] == context and e['off'] == 0, 'unexpected context/sub-batch'
            assert 0 < e['n'] <= 1024
            assert all(len(e[k]) == e['n'] for k in ('tokens', 'pos', 'outputs'))
            seq = e['seq'].split(',')
            assert len(seq) == e['n'] and all(re.fullmatch(r'\d+', s) for s in seq)
            assert len(set(seq)) == 1, 'multi-slot batch outside frozen journey'
            if slot_id is None:
                slot_id = int(seq[0])
            assert int(seq[0]) == slot_id
            assert all(o in (0, 1) for o in e['outputs']) and all(t >= 0 for t in e['tokens'])
            assert e['pos'] == list(range(e['pos'][0], e['pos'][0] + e['n']))
            current = dict(e, decoded=False, batch_index=len(batches))
            batches.append(current)
            for pos, token in zip(e['pos'], e['tokens']):
                if pos < prompt_tokens:
                    assert pos >= 0 and (pos not in prompt or prompt[pos] == token)
                    prompt[pos] = token
            continue
        if kind == 'remove' and e['kind'] == 'prompt':
            # Server clears the unprocessed tail between prompt chunks too.
            # Permit only the boundary immediately after the decoded prefix,
            # before any sample; this is not speculative rollback.
            assert not samples and not pending and not to_emit
            assert e['rollback'] == 0 and e['p1'] == -1
            assert sorted(prompt) == list(range(len(prompt)))
            assert e['p0'] == len(prompt) < prompt_tokens
            if context is None:
                assert current is None and e['p0'] == 0
            else:
                assert e['ctx'] == context and e['slot'] == slot_id
                assert current['decoded'] and not any(current['outputs'])
                assert e['p0'] == current['pos'][-1] + 1
            removals.append(e)
            continue
        assert current is not None and e['ctx'] == context, 'unmapped context/sample'
        if 'slot' in e:
            assert e['slot'] == slot_id
        if kind == 'decoded':
            assert not current['decoded'] and e['ret'] == 0, 'failed/repeated decode'
            current['decoded'] = True
        elif kind == 'sample':
            assert current['decoded'] and not to_emit
            if sampler is None:
                sampler = e['smpl']
            assert e['smpl'] == sampler, 'sampler changed within frozen request'
            outputs = [i for i, o in enumerate(current['outputs']) if o]
            idx = e['idx']
            if idx < 0:
                assert -idx <= len(outputs)
                row = outputs[idx]
            else:
                row = idx
            assert row in outputs, 'selected row is not a real output row'
            ordinal = len(samples) + 1
            assert ordinal <= 128 and e['selected'] == tokens[ordinal - 1], 'sample does not align to emission'
            pos = current['pos'][row]
            assert pos == prompt_tokens + ordinal - 2, 'selected row position/history mismatch'
            for p, token in zip(current['pos'][:row + 1], current['tokens'][:row + 1]):
                if p >= prompt_tokens:
                    assert token == tokens[p - prompt_tokens], 'batch draft differs from prior emitted history'
            assert e['file'] not in files, 'duplicate raw capture'
            files.add(e['file'])
            data = raw_bytes(capture, e)
            item = dict(e, ordinal=ordinal, batch_index=current['batch_index'], batch_n=current['n'],
                        batch_row=row, position=pos, input_id=current['tokens'][row], **score_summary(data))
            assert not samples or item['vocab'] == samples[0]['vocab']
            samples.append(item)
            pending.append(item)
        elif kind in ('single', 'verify'):
            assert e['smpl'] == sampler and pending and not to_emit
            if kind == 'single':
                assert len(pending) == 1 and e['row'] == pending[0]['idx'] and e['id'] == pending[0]['selected']
                to_emit = [e['id']]
                last_verify = None
            else:
                assert not e['replay'] and not e['rejection'] and not e['synth'], 'not the greedy verification path'
                assert len(e['drafts']) == 1 and len(e['rows']) == 2
                assert e['rows'] == [i for i, o in enumerate(current['outputs']) if o]
                assert e['rows'][:len(pending)] == [s['idx'] for s in pending]
                assert e['emitted'] == [s['selected'] for s in pending]
                assert e['accepted'] == len(e['emitted']) - 1
                assert e['rollback'] == len(e['drafts']) - e['accepted']
                assert e['emitted'][:e['accepted']] == e['drafts'][:e['accepted']]
                if e['rollback']:
                    assert e['emitted'][-1] != e['drafts'][e['accepted']]
                assert current['tokens'][e['rows'][1]] == e['drafts'][0]
                last_verify = dict(e, first_position=pending[0]['position'], removed=False)
                verifications.append(last_verify)
                to_emit = list(e['emitted'])
            pending = []
        elif kind == 'emit':
            assert to_emit and e['token'] == to_emit.pop(0)
            assert e['ordinal'] == len(emissions) + 1 and e['token'] == tokens[len(emissions)]
            if last_verify is not None:
                assert last_verify['removed'], 'missing real removal boundary'
            emissions.append(e)
        elif kind == 'remove':
            assert e['kind'] == 'verify' and last_verify is not None, 'unexpected history mutation'
            assert not last_verify['removed'] and e['rollback'] == last_verify['rollback']
            assert e['p1'] == -1 and e['p0'] == last_verify['first_position'] + len(last_verify['emitted'])
            removed_positions = [p for p in current['pos'] if p >= e['p0']]
            assert len(removed_positions) == e['rollback']
            removals.append(dict(e, removed_batch_positions=removed_positions))
            last_verify['removed'] = True
        elif kind == 'checkpoint':
            raise AssertionError('checkpoint/replay outside bounded width1 journey; attribution blocked')
    assert not pending and not to_emit and len(samples) == len(emissions) == 128
    assert current and current['decoded']
    assert sorted(prompt) == list(range(prompt_tokens)), 'incomplete actual prompt batch'
    assert all(e['ctx'] == context and e['slot'] == slot_id for e in removals)
    return dict(samples=samples, batches=batches, removals=removals, verifications=verifications,
                emitted_ids=tokens, prompt_ids=[prompt[i] for i in range(prompt_tokens)])


def _parent_batch_inputs(alignment):
    keys = ('off', 'n', 'tokens', 'pos', 'outputs', 'seq')
    return [{key: batch[key] for key in keys} for batch in alignment['batches']]


def _parent_sample_metadata(alignment):
    keys = ('idx', 'selected', 'batch_index', 'batch_n', 'batch_row', 'position', 'input_id', 'vocab')
    return [{key: sample[key] for key in keys} for sample in alignment['samples']]


def parent_reference(label):
    # The failed S0 cell is immutable: its validated alignment lives at the root.
    assert label in ('S0', 'C0')
    if label == 'S0':
        return PARENT_RESULTS / 'run-r1/S0', PARENT_RESULTS / 'reanalysis-S0.json'
    cell = PARENT_RESULTS / 'continuation-c0/run-r1/C0'
    return cell, cell / 'alignment.json'


def report_parent_comparison(candidate, label, ordinary, diagnostic):
    parent_cell, parent_alignment = parent_reference(label)
    parent = load(parent_alignment)
    parent_capture = parent_cell / 'capture'
    candidate_capture = Path(candidate['capture_path'])
    parent_ordinary = load(parent_cell / 'ordinary.result.json')
    parent_diagnostic = load(parent_cell / 'diagnostic.response.json')
    assert parent_capture.is_dir()
    assert len(candidate['samples']) == len(parent['samples']) == 128
    prompt_ids_match = (candidate['prompt_ids'] == parent['prompt_ids'])
    assert prompt_ids_match, f'{label} prompt IDs differ from router-single2 computational parent'
    a_ids, b_ids = candidate['emitted_ids'], parent['emitted_ids']
    first_output = next((i + 1 for i, pair in enumerate(zip(a_ids, b_ids)) if pair[0] != pair[1]), None)
    cutoff = first_output or 128
    comparisons, first_raw = [], None
    for i in range(cutoff):
        assert a_ids[:i] == b_ids[:i], 'raw comparison beyond identical prior output history'
        candidate_sample, parent_sample = candidate['samples'][i], parent['samples'][i]
        candidate_bytes = raw_bytes(candidate_capture, candidate_sample)
        parent_bytes = raw_bytes(parent_capture, parent_sample)
        equal = candidate_bytes == parent_bytes
        row = dict(ordinal=i + 1, raw_bytes_equal=equal,
                   candidate_batch_index=candidate_sample['batch_index'],
                   parent_batch_index=parent_sample['batch_index'],
                   candidate_batch_row=candidate_sample['batch_row'],
                   parent_batch_row=parent_sample['batch_row'],
                   candidate_position=candidate_sample['position'],
                   parent_position=parent_sample['position'],
                   candidate_selected=candidate_sample['selected'],
                   parent_selected=parent_sample['selected'])
        if not equal and first_raw is None:
            first_byte = next(j for j in range(len(candidate_bytes)) if candidate_bytes[j] != parent_bytes[j])
            first_raw = dict(row, first_differing_float_id=first_byte // 4,
                             candidate_sample=candidate_sample, parent_sample=parent_sample)
        comparisons.append(row)
    return dict(reference_equivalence_required=True,
                parent_alignment=str(parent_alignment),
                parent_capture=str(parent_capture),
                prompt_ids_match=prompt_ids_match,
                emitted_ids_match=(a_ids == b_ids),
                ordinary_text_matches_parent=(ordinary['text'] == parent_ordinary['text']),
                diagnostic_text_matches_parent=(
                    diagnostic['choices'][0]['message']['content']
                    == parent_diagnostic['choices'][0]['message']['content']),
                diagnostic_token_ids_match_parent=(
                    diagnostic['__verbose']['tokens'] == parent_diagnostic['__verbose']['tokens']),
                actual_batch_inputs_match=(_parent_batch_inputs(candidate) == _parent_batch_inputs(parent)),
                selected_positions_rows_match=(
                    _parent_sample_metadata(candidate) == _parent_sample_metadata(parent)),
                first_output_difference_ordinal=first_output,
                compared_through_ordinal=cutoff,
                first_raw_difference=first_raw,
                raw_vector_count=128,
                comparisons=comparisons)


def assert_parent_observer_invariant(parent_comparison):
    """BOTH arms must match their immediate router-single2 parent on all 128 raw vectors and identity fields."""
    assert parent_comparison['compared_through_ordinal'] == 128
    assert len(parent_comparison['comparisons']) == 128
    assert parent_comparison['first_output_difference_ordinal'] is None
    assert parent_comparison['first_raw_difference'] is None
    assert parent_comparison['prompt_ids_match']
    assert parent_comparison['emitted_ids_match']
    assert parent_comparison['ordinary_text_matches_parent']
    assert parent_comparison['diagnostic_text_matches_parent']
    assert parent_comparison['diagnostic_token_ids_match_parent']
    assert parent_comparison['actual_batch_inputs_match']
    assert parent_comparison['selected_positions_rows_match']
    assert all(row['raw_bytes_equal'] for row in parent_comparison['comparisons'])


def analyze_arm(cell, label, old):
    assert label in ('S0', 'C0')
    reference, _ = parent_reference(label)
    ordinary, diagnostic = load(cell / 'ordinary.result.json'), load(cell / 'diagnostic.response.json')
    old['check_stream_result'](ordinary)
    old['check_stream_result'](load(cell / 'warmup.result.json'))
    tokens = old['check_diagnostic_result'](diagnostic)
    assert load(cell / 'diagnostic.request.json') == load(reference / 'diagnostic.request.json')
    for stem in ('warmup', 'ordinary'):
        assert load(cell / (stem + '.request.json')) == load(reference / (stem + '.request.json')) == load(WIDTH1 / 'run-r1/frozen-payload.json')
    within_arm_text_consistency(ordinary, diagnostic)
    prior_reference = prior_reference_flags(
        ordinary, diagnostic,
        load(reference / 'ordinary.result.json'),
        load(reference / 'diagnostic.response.json'))
    diagnostic_log = (cell / 'diagnostic-window.log').read_text()
    assert_frozen_prefill_batches(diagnostic_log)
    result = align(records(diagnostic_log), tokens, cell / 'capture')
    result['capture_path'] = str(cell / 'capture')
    route_analysis = select_route_batches(diagnostic_log, label)
    assert_actual_route_gates(route_analysis, label)
    route_analysis['router_batches'] = assert_router_model_routes(diagnostic_log, label)
    parent_comparison = report_parent_comparison(result, label, ordinary, diagnostic)
    assert_parent_observer_invariant(parent_comparison)
    verifications = result['verifications']
    accepted, proposed = sum(v['accepted'] for v in verifications), sum(len(v['drafts']) for v in verifications)
    assert (accepted, proposed) == (diagnostic['timings'].get('draft_n_accepted', 0), diagnostic['timings'].get('draft_n', 0))
    assert (not verifications) if label == 'S0' else (proposed > 0)
    result.update(within_arm_text_passed=True, prior_reference=prior_reference,
                  accepted=accepted, proposed=proposed,
                  rejection_events=sum(v['rollback'] > 0 for v in verifications), capture_path=str(cell / 'capture'),
                  route_analysis=route_analysis, parent_comparison=parent_comparison,
                  observer_parent_invariant_passed=True)
    result['operand_analysis'] = analyze_operands(cell, label, result)
    return result


def compare(left, right):
    assert left['within_arm_text_passed'] and right['within_arm_text_passed']
    assert left['prompt_ids'] == right['prompt_ids'], 'actual input prompts differ'
    a, b = left['emitted_ids'], right['emitted_ids']
    assert len(a) == len(b) == len(left['samples']) == len(right['samples']) == 128
    first_output = next((i + 1 for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
    cutoff = first_output or 128
    comparisons, first_score = [], None
    for i in range(cutoff):
        assert a[:i] == b[:i], 'raw-vector comparison beyond identical prior history'
        x, y = left['samples'][i], right['samples'][i]
        assert x['position'] == y['position'] and x['vocab'] == y['vocab']
        xb, yb = raw_bytes(Path(left['capture_path']), x), raw_bytes(Path(right['capture_path']), y)
        equal = xb == yb  # actual bytes, not hashes, top-five probabilities, or rounded scores
        row = dict(ordinal=i + 1, equal=equal, S0_batch=x['batch_index'], C0_batch=y['batch_index'],
                   S0_row=x['batch_row'], C0_row=y['batch_row'], position=x['position'])
        if not equal and first_score is None:
            first_byte = next(j for j in range(len(xb)) if xb[j] != yb[j])
            first_score = dict(row, first_differing_float_id=first_byte // 4,
                               S0=x, C0=y, S0_batch=left['batches'][x['batch_index']],
                               C0_batch=right['batches'][y['batch_index']])
            for label, arm, sample in (('S0', left, x), ('C0', right, y)):
                preceding = [r for r in arm['removals'] if r['rollback'] > 0 and r['event_index'] < sample['event_index']]
                first_score[label + '_rollback_preceded'] = bool(preceding)
                first_score[label + '_preceding_rollbacks'] = preceding
        comparisons.append(row)
    near = [{ 'ordinal': i + 1, 'identical_prior_history': i < cutoff,
              'S0': left['samples'][i], 'C0': right['samples'][i]} for i in range(113, 118)]
    return dict(first_output_difference_ordinal=first_output, compared_through_ordinal=cutoff,
                first_score_difference=first_score, comparisons=comparisons, near_token116=near,
                reference_equivalence_required=True,
                S0_prior_reference=left['prior_reference'], C0_prior_reference=right['prior_reference'],
                raw_logits_warning='Raw F32 logits precede EOS masking and all samplers. Raw argmax is not necessarily the emitted ID; API logprobs are not raw logits.',
                attribution_gates_passed=True, timing_permitted=False, promotion_permitted=False)


OPERAND_SHAPES = {
    'blk.1.ffn_gate_inp.weight': (4096, 240),
    'blk.1.ffn_latent_down.weight': (4096, 1024),
}
OPERAND_STRINGS = {'phase', 'weight', 'role', 'backend_ctx', 'queue', 'type', 'data', 'base', 'file'}
OPERAND_FIELDS = OPERAND_STRINGS | set('device n reorder ne0 ne1 ne2 ne3 nb0 nb1 nb2 nb3 buffer_bytes offset bytes'.split())


def operand_record(line):
    assert line.count('BOLT_MOE_OPERAND ') == 1, 'operand framing'
    assert 'BOLT_ROUTE' not in line and 'BOLT_VERIFY' not in line, 'mixed operand record'
    pairs = line.split('BOLT_MOE_OPERAND ', 1)[1].split()
    assert pairs and all(re.fullmatch(r'[a-z_][a-z_0-9]*=[^\s=]+', p) for p in pairs)
    fields = [p.split('=', 1) for p in pairs]
    assert len(fields) == len({k for k, _ in fields}) and {k for k, _ in fields} == OPERAND_FIELDS
    row = {}
    for key, value in fields:
        if key in OPERAND_STRINGS:
            if key in ('backend_ctx', 'queue', 'data', 'base'):
                assert re.fullmatch(r'0x[0-9a-fA-F]+', value) and 0 < int(value, 16) < 2**64
            row[key] = value
        else:
            assert re.fullmatch(r'0|[1-9]\d*', value)
            row[key] = int(value)
    assert row['weight'] in OPERAND_SHAPES and row['role'] in ('weight', 'src1', 'dst')
    assert row['n'] in (1, 2) and row['device'] == 0
    weight, role, n = row['weight'], row['role'], row['n']
    k, m = OPERAND_SHAPES[weight]
    rows, cols = (k, m) if role == 'weight' else ((k, n) if role == 'src1' else (m, n))
    if role == 'weight':
        expected_type = 'f32' if weight == 'blk.1.ffn_gate_inp.weight' else 'q8_0'
        block = 4 if expected_type == 'f32' else 34
        stride = rows * 4 if expected_type == 'f32' else rows // 32 * 34
    else:
        expected_type, block, stride = 'f32', 4, rows * 4
    # Every role records the projection WEIGHT's reorder state, not its own dtype.
    expected_reorder = int(weight == 'blk.1.ffn_latent_down.weight')
    size = stride * cols
    assert row['type'] == expected_type and row['reorder'] == expected_reorder
    assert row['phase'] == ('after' if role == 'dst' else 'before')
    assert tuple(row[f'ne{i}'] for i in range(4)) == (rows, cols, 1, 1)
    assert tuple(row[f'nb{i}'] for i in range(4)) == (block, stride, size, size)
    assert row['bytes'] == size
    data, base = int(row['data'], 16), int(row['base'], 16)
    assert data % (16 if expected_type == 'q8_0' else 4) == 0
    assert data >= base and row['offset'] == data - base <= row['buffer_bytes'] < 2**64
    assert size <= row['buffer_bytes'] - row['offset'] and base + row['buffer_bytes'] <= 2**64
    stem = 'router' if weight == 'blk.1.ffn_gate_inp.weight' else 'latent_down'
    assert row['file'] == f'{stem}.{role}.bin'
    return row


def operand_bytes(directory, row):
    # Fixed basename, no traversal/symlinks, exact size, and no unbounded read.
    path = directory / row['file']
    assert not directory.is_symlink() and directory.is_dir()
    assert path.name == row['file'] and '/' not in row['file']
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        st = os.fstat(stream.fileno())
        assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size == row['bytes']
        assert 0 < row['bytes'] <= 4456448  # complete Q8_0 latent-down allocation slice
        data = stream.read(row['bytes'] + 1)
        assert len(data) == row['bytes']
    if row['type'] == 'f32':
        assert sys.byteorder == 'little'
        values = array.array('f')
        values.frombytes(data)
        assert all(math.isfinite(x) for x in values), 'nonfinite operand'
    return data


def first_operand_column(data, row):
    assert row['type'] == 'f32' and row['ne2'] == row['ne3'] == 1
    assert row['nb0'] >= 4 and row['ne0'] > 0
    assert (row['ne0'] - 1) * row['nb0'] + 4 <= len(data)
    return b''.join(data[i * row['nb0']:i * row['nb0'] + 4] for i in range(row['ne0']))


def analyze_operands(cell, label, alignment):
    log = (cell / 'diagnostic-window.log').read_text()
    assert 'BOLT_MOE_OPERAND' not in (cell / 'pre-diagnostic.log').read_text(), 'unarmed capture'
    route_groups = route_frames(log)
    frames, active = [], None
    for line_number, line in enumerate(log.splitlines()):
        if 'BOLT_VERIFY' in line:
            event = records(line)[0]
            if event['event'] == 'batch':
                assert active is None
                active = dict(batch=event, operands=[])
            elif event['event'] == 'decoded':
                assert active is not None and event['ctx'] == active['batch']['ctx'] and event['ret'] == 0
                frames.append(active)
                active = None
        if 'BOLT_MOE_OPERAND' in line:
            row = operand_record(line)
            assert active is not None, 'operand outside target decode'
            active['operands'].append((line_number, row))
    assert active is None
    sample = alignment['samples'][1]  # Actual first post-prompt raw frame: ordinal 2.
    batch = alignment['batches'][sample['batch_index']]
    n = 1 if label == 'S0' else 2
    assert sample['ordinal'] == 2 and sample['position'] == 55 and sample['batch_row'] == 0
    assert batch['n'] == n and batch['pos'] == list(range(55, 55 + n))
    candidates = [frame for frame in frames if frame['batch']['ctx'] == batch['ctx'] and
                  _route_batch_metadata(frame['batch']) == _route_batch_metadata(batch)]
    assert len(candidates) == 1, 'ambiguous actual first target frame'
    selected = candidates[0]
    route_frame = route_groups[frames.index(selected)]
    assert route_frame['batch'] == selected['batch']
    assert [r['weight'] for r in route_frame['routes']] == _expected_target_weights(n, label, output=True)
    assert sum(len(frame['operands']) for frame in frames) == len(selected['operands']) == 6
    expected = [(weight, role) for weight in OPERAND_SHAPES for role in ('weight', 'src1', 'dst')]
    assert [(r['weight'], r['role']) for _, r in selected['operands']] == expected
    # Existing route gates anchor the target backend, not a guessed pointer/first route.
    route_batch = alignment['route_analysis']['selected_batches'][0]
    assert route_batch['batch'] == _route_batch_metadata(batch)
    backend = route_frame['target_backend_ctx']
    assert backend == route_batch['target_backend_ctx']
    assert route_frame['routes'] == route_batch['routes']
    assert {r['backend_ctx'] for _, r in selected['operands']} == {backend}
    assert len({r['queue'] for _, r in selected['operands']}) == 1
    directory = cell / 'moe-operands'
    assert {p.name for p in directory.iterdir()} == {r['file'] for _, r in selected['operands']}
    result = {}
    for weight in OPERAND_SHAPES:
        rows = [(line, r) for line, r in selected['operands'] if r['weight'] == weight]
        assert len(rows) == 3, (weight, rows)
        routes = [r for r in route_frame['routes'] if r['weight'] == weight]
        router = weight == 'blk.1.ffn_gate_inp.weight'
        split = router and label == 'C0' and n == 2
        if router:
            assert_router_operations(routes, n, split=split)
            consumers = routes[1::2]  # actual MKL calls, not duplicate outer records
        else:
            assert len(routes) == 1, 'duplicate/nonselected latent-down route'
            route = routes[0]
            assert route['route'] == 'DMMV_ESIMD' and route['reorder'] == 1 and route['result'] == -1
            consumers = routes
        assert all(r['op'] == 'MUL_MAT' and r['expert'] == -1 and r['device'] == 0 for r in routes)
        # Common logger alone orders VERIFY and operand phases. Direct-stderr
        # route placement is independent; route_frames has already paired FIFOs.
        assert rows[0][0] < rows[1][0] < rows[2][0], 'copy phase ordering'
        result[weight] = {}
        for _, row in rows:
            assert row['n'] == n
            prefix = {'weight': '', 'src1': 'src1_', 'dst': 'dst_'}[row['role']]
            shape = tuple(row[f'ne{i}'] for i in range(4))
            if split and row['role'] != 'weight':
                # Capture brackets the FULL n2 compute-forward call. Its two
                # real singleton consumers change ne1 ONLY, retaining n2 nb2/nb3.
                assert shape[1] == 2
                shape = (shape[0], 1, shape[2], shape[3])
            strides = tuple(row[f'nb{i}'] for i in range(4))
            type_field = {'weight': 'src0_type', 'src1': 'src1_type', 'dst': 'dst_type'}[row['role']]
            for route in consumers:
                assert_route_tensor(route, prefix, shape, strides)
                assert route[type_field] == row['type']
            operand_bytes(directory, row)
            result[weight][row['role']] = row
    return dict(batch=_route_batch_metadata(batch), target_ctx=batch['ctx'], backend_ctx=backend,
                directory=str(directory), projections=result, complete=True)


def compare_operands(arms):
    for label in ('S0', 'C0'):
        assert arms[label]['observer_parent_invariant_passed'], 'observer changed parent'
        assert_parent_observer_invariant(arms[label]['parent_comparison'])
        assert arms[label]['operand_analysis']['complete']
    compared = {}
    for weight in OPERAND_SHAPES:
        compared[weight] = {}
        for role in ('weight', 'src1', 'dst'):
            data = []
            for label in ('S0', 'C0'):
                metadata = arms[label]['operand_analysis']
                row = metadata['projections'][weight][role]
                raw = operand_bytes(Path(metadata['directory']), row)
                data.append(raw if role == 'weight' else first_operand_column(raw, row))
            left, right = data
            assert len(left) == len(right)
            first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None) if left != right else None
            compared[weight][role] = dict(bytes_equal=left == right, compared_bytes=len(left), first_differing_byte=first)
    router, latent_down = (compared[w] for w in OPERAND_SHAPES)
    equal = lambda projection, role: projection[role]['bytes_equal']
    if not equal(router, 'src1'):
        location = 'before_block1_router'
    elif not equal(router, 'weight'):
        location = 'router_weight_bytes_differ'
    elif not equal(router, 'dst'):
        location = 'block1_router_projection'
    elif not equal(latent_down, 'src1'):
        location = 'between_router_and_latent_down'
    elif not equal(latent_down, 'weight'):
        location = 'latent_down_weight_bytes_differ'
    elif not equal(latent_down, 'dst'):
        location = 'latent_down_projection'
    else:
        location = 'block1_router_and_latent_down_excluded_in_this_fixture'
    return dict(comparisons=compared, localization=location, observer_parent_invariant_passed=True,
                limitation='Only the observed position55/token0 projections; no sole-root-cause, fidelity or performance claim.')


def main():
    assert socket.gethostname() == 'inference-host'
    control = runpy.run_path(str(ROOT / 'run-trace.py'))
    control['check_protected_sources']()
    old = runpy.run_path(str(WIDTH1 / 'run-controls.py'))
    native = runpy.run_path(str(WIDTH1 / 'analyze-controls.py'))
    helper = runpy.run_path(str(old['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(old['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(old['OLD'] / 'probe-api.py'))
    out = ROOT / 'run-r1'
    result = dict(attribution_gates_passed=False, reference_equivalence_required=True)
    try:
        summary = load(out / 'summary.json')
        assert summary['status'] == 'completed' and summary['restored'] and summary['order'] == ['S0', 'C0']
        assert summary['timing_permitted'] is False and summary['promotion_permitted'] is False
        assert summary.get('reference_equivalence_required') is True
        arms = {}
        for label in control['ORDER']:
            cell = out / label
            log = (cell / 'server.log').read_text()
            control['check_startup'](log, label, old, native)
            assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
            assert (cell / 'diagnostic-window.log').read_text() == control['diagnostic_window'](
                (cell / 'pre-diagnostic.log').read_text(), (cell / 'through-diagnostic.log').read_text())
            quality = load(cell / 'quality/summary.json')
            assert quality['pass'] and len(quality['checks']) == 20 and all(r['pass'] for r in quality['checks'])
            arms[label] = analyze_arm(cell, label, old)
            assert arms[label]['route_analysis']['actual_route_presence']
            control['save'](cell / 'alignment.json', arms[label])
        command = helper['command']
        live = json.loads(command('docker', 'inspect', old['ORIGINAL']))[0]
        control['check_restored'](load(out / 'original-container.json'), live)
        assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {old['ORIGINAL']}
        for name in ('accepted', 'parent', 'candidate'):
            assert control['binaries'](control[name.upper()]) == load(out / (name + '-binaries.json')) == load(ROOT / (name + '-binaries.json'))
        assert runner['hardware']() == load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
        assert api['request'](old['URL'], '/health')['status'] == 'ok'
        smoke = load(out / 'restored-smoke.result.json')
        assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
        result = compare(arms['S0'], arms['C0'])
        result['route_analysis'] = {label: arms[label]['route_analysis'] for label in control['ORDER']}
        result['parent_comparison'] = {label: arms[label]['parent_comparison'] for label in control['ORDER']}
        result['observer_parent_invariant_passed'] = all(arms[label]['observer_parent_invariant_passed'] for label in control['ORDER'])
        assert result['observer_parent_invariant_passed']
        result['operand_comparison'] = compare_operands(arms)
        result['attribution_gates_passed'] = True
        result['restoration_verified'] = True
    except BaseException as exc:
        result.update(attribution_gates_passed=False, error=repr(exc))
        raise
    finally:
        control['save'](out / 'analysis.json', result)
    print(json.dumps({k: result[k] for k in ('first_output_difference_ordinal', 'compared_through_ordinal',
                                             'restoration_verified', 'attribution_gates_passed')}))
    print('BOLT_VERIFY_ANALYZED_NO_TIMING')


if __name__ == '__main__':
    main()
