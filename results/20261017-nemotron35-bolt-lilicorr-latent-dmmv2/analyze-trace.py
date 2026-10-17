#!/usr/bin/env python3
"""Fail-closed analysis of this development-only latent DMMV control; no timing or promotion."""
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

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-latent-dmmv2')
WIDTH1 = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1'
PARENT_RESULTS = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-moe-fallback2'
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


def route_frames(log):
    """Frame routes only inside BOLT_VERIFY batch..decoded target boundaries."""
    frames = []
    active = None
    route_count = 0
    for line in log.splitlines():
        assert not ('BOLT_VERIFY' in line and 'BOLT_ROUTE' in line), 'mixed trace records'
        if 'BOLT_VERIFY' in line:
            assert 'BOLT_VERIFY ' in line, 'malformed BOLT_VERIFY framing'
            events = records(line)
            assert len(events) == 1
            event = events[0]
            if event['event'] == 'batch':
                assert active is None, 'nested BOLT_VERIFY batch'
                active = dict(batch=event, routes=[])
            elif event['event'] == 'decoded':
                assert active is not None, 'decoded without batch'
                assert event['ctx'] == active['batch']['ctx'], 'decoded context mismatch'
                assert event['ret'] == 0, 'decoded failed'
                active['decoded'] = event
                frames.append(active)
                active = None
            continue
        if 'BOLT_ROUTE' in line:
            route_count += 1
            route = route_records(line)[0]
            if active is not None:
                active['routes'].append(route)
    assert active is None, 'unterminated BOLT_VERIFY batch'
    assert route_count, 'empty route trace'
    assert frames, 'no BOLT_VERIFY batch frames'
    return frames


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
        anchors = [route for route in frame['routes'] if route['weight'] in
                   ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight')]
        contexts = {route['backend_ctx'] for route in anchors}
        assert len(contexts) == 1, 'missing or ambiguous target SSM backend context'
        target_context = next(iter(contexts))
        target_routes = [route for route in frame['routes'] if route['backend_ctx'] == target_context]
        other_routes = [route for route in frame['routes'] if route['backend_ctx'] != target_context]
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


def assert_actual_route_gates(route_analysis, label):
    assert label in ('S0', 'C0')
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
            assert block_routes, weight
            assert not any(route['route'] in ('MOE_GROUPED_GEMM_ATTEMPT', 'MOE_GROUPED_GEMM_RESULT')
                           for route in block_routes), block_routes
            for route in block_routes:
                assert route['src0_type'] == 'iq2_xxs', route
                assert route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
                assert route['reorder'] == 0, route
            if label == 'S0':
                assert all(route['route'] == 'MOE_SINGLE_MMVQ_PLAIN_Q8_1' for route in block_routes), block_routes
            else:
                assert any(route['route'] == 'MOE_PER_EXPERT_FALLBACK' for route in block_routes), block_routes
                assert any(route['route'] == 'MMVQ_PLAIN' for route in block_routes), block_routes


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
        reference_equivalence_required=False,
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


def report_parent_comparison(candidate, label, ordinary, diagnostic):
    parent_cell = PARENT_RESULTS / 'run-r1' / label
    parent = load(parent_cell / 'alignment.json')
    parent_capture = parent_cell / 'capture'
    candidate_capture = Path(candidate['capture_path'])
    parent_ordinary = load(parent_cell / 'ordinary.result.json')
    parent_diagnostic = load(parent_cell / 'diagnostic.response.json')
    assert parent_capture.is_dir()
    assert len(candidate['samples']) == len(parent['samples']) == 128
    prompt_ids_match = (candidate['prompt_ids'] == parent['prompt_ids'])
    assert prompt_ids_match, f'{label} prompt IDs differ from fallback2 parent'
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
    return dict(reference_equivalence_required=False,
                parent_alignment=str(parent_cell / 'alignment.json'),
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
    """S0 must match the fallback2 parent on all 128 raw vectors and identity fields."""
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
    reference = WIDTH1 / 'run-r1' / ('A0' if label == 'S0' else 'C0')
    ordinary, diagnostic = load(cell / 'ordinary.result.json'), load(cell / 'diagnostic.response.json')
    old['check_stream_result'](ordinary)
    old['check_stream_result'](load(cell / 'warmup.result.json'))
    tokens = old['check_diagnostic_result'](diagnostic)
    assert load(cell / 'diagnostic.request.json') == load(reference / 'diagnostic.request.json')
    for stem in ('warmup', 'ordinary'):
        assert load(cell / (stem + '.request.json')) == load(WIDTH1 / 'run-r1/frozen-payload.json')
    within_arm_text_consistency(ordinary, diagnostic)
    prior_reference = prior_reference_flags(
        ordinary, diagnostic,
        load(reference / 'repeat-00.result.json'),
        load(reference / 'diagnostic.response.json'))
    result = align(records((cell / 'diagnostic-window.log').read_text()), tokens, cell / 'capture')
    result['capture_path'] = str(cell / 'capture')
    route_analysis = select_route_batches((cell / 'diagnostic-window.log').read_text(), label)
    assert_actual_route_gates(route_analysis, label)
    parent_comparison = report_parent_comparison(result, label, ordinary, diagnostic)
    if label == 'S0':
        assert_parent_observer_invariant(parent_comparison)
    s0_parent_invariant_passed = (label == 'S0')
    verifications = result['verifications']
    accepted, proposed = sum(v['accepted'] for v in verifications), sum(len(v['drafts']) for v in verifications)
    assert (accepted, proposed) == (diagnostic['timings'].get('draft_n_accepted', 0), diagnostic['timings'].get('draft_n', 0))
    assert (not verifications) if label == 'S0' else (proposed > 0)
    result.update(within_arm_text_passed=True, prior_reference=prior_reference,
                  accepted=accepted, proposed=proposed,
                  rejection_events=sum(v['rollback'] > 0 for v in verifications), capture_path=str(cell / 'capture'),
                  route_analysis=route_analysis, parent_comparison=parent_comparison,
                  s0_parent_invariant_passed=s0_parent_invariant_passed,
                  c0_parent_report_only=(label == 'C0'))
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
                reference_equivalence_required=False,
                S0_prior_reference=left['prior_reference'], C0_prior_reference=right['prior_reference'],
                raw_logits_warning='Raw F32 logits precede EOS masking and all samplers. Raw argmax is not necessarily the emitted ID; API logprobs are not raw logits.',
                attribution_gates_passed=True, timing_permitted=False, promotion_permitted=False)


def main():
    assert socket.gethostname() == 'inference-host'
    control = runpy.run_path(str(ROOT / 'run-trace.py'))
    old = runpy.run_path(str(WIDTH1 / 'run-controls.py'))
    native = runpy.run_path(str(WIDTH1 / 'analyze-controls.py'))
    helper = runpy.run_path(str(old['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(old['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(old['OLD'] / 'probe-api.py'))
    out = ROOT / 'run-r1'
    result = dict(attribution_gates_passed=False, reference_equivalence_required=False)
    try:
        summary = load(out / 'summary.json')
        assert summary['status'] == 'completed' and summary['restored'] and summary['order'] == ['S0', 'C0']
        assert summary['timing_permitted'] is False and summary['promotion_permitted'] is False
        assert summary.get('reference_equivalence_required') is False
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
        result['s0_parent_invariant_passed'] = arms['S0']['s0_parent_invariant_passed']
        result['c0_parent_report_only'] = arms['C0']['c0_parent_report_only']
        assert result['s0_parent_invariant_passed'] and result['c0_parent_report_only']
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
