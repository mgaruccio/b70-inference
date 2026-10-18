#!/usr/bin/env python3
"""Remote CPU regressions. Synthetic 8K histories are NOT new live E2E evidence."""
import array
import ast
import copy
import json
import os
import runpy
import shlex
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

assert socket.gethostname() == 'inference-host'
HERE = Path(__file__).resolve().parent
C = runpy.run_path(str(HERE / 'contracts.py'))
M = runpy.run_path(str(HERE / 'analyze-long.py'))
D = runpy.run_path(str(HERE / 'run-long.py'))
A = C['A']
SCRATCH = Path(os.environ['BOLT_LONG_CPU_SCRATCH']).resolve()
assert SCRATCH.is_relative_to(C['ROOT'] / 'worker-scratch')


def event_line(event):
    fields = []
    for key in A['FIELDS'][event['event']].split():
        value = event[key]
        if key in A['LISTS']:
            value = ','.join(map(str, value)) or '-'
        fields.append(f'{key}={value}')
    return 'BOLT_VERIFY ' + event['event'] + ' ' + ' '.join(fields)


def frame_log(frames, routes_first=False, foreign=None):
    verify, routes = [], []
    for f in frames:
        verify.extend([event_line(dict(f['batch'], event='batch')), event_line(dict(event='decoded', ctx=f['batch']['ctx'], ret=0))])
        routes.extend('BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in r.items()) for r in f['routes'])
    routes = (foreign or []) + routes
    return '\n'.join(routes + verify if routes_first else verify + routes)


def synthetic_long(events, widths):
    """Replace only prefill, shift actual saved decode/rollback history; synthetic IDs."""
    first = next(i for i, e in enumerate(events) if e['event'] == 'sample')
    old_batches = [e for e in events[:first] if e['event'] == 'batch']
    old_prompt = [t for b in old_batches for t in b['tokens']]
    total = sum(widths)
    prompt = [17 + i % 100 for i in range(total - len(old_prompt))] + old_prompt
    delta, pos, prefix = total - len(old_prompt), 0, []
    ctx = old_batches[0]['ctx']
    for n in widths:
        prefix.append(dict(event='remove', ctx=ctx, slot=0, kind='prompt', p0=pos, p1=-1, rollback=0))
        prefix.append(dict(event='batch', ctx=ctx, off=0, n=n, tokens=prompt[pos:pos+n],
                           pos=list(range(pos, pos+n)), outputs=[int(p == total-1) for p in range(pos, pos+n)],
                           seq=','.join(['0']*n)))
        prefix.append(dict(event='decoded', ctx=ctx, ret=0))
        pos += n
    tail = copy.deepcopy(events[first:])
    tail[0]['idx'] = widths[-1] - 1
    assert tail[1]['event'] == 'single'
    tail[1]['row'] = widths[-1] - 1
    for e in tail:
        if e['event'] == 'batch':
            e['pos'] = [p + delta for p in e['pos']]
        elif e['event'] in ('remove', 'checkpoint'):
            e['p0'] += delta
            if e['p1'] >= 0:
                e['p1'] += delta
    return prefix + tail, prompt


class LongTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = C['frozen_jobs']()
        cls.cells = {'A0': C['NODRAFT'] / 'run-r1/A0', 'C0': C['REFERENCE'] / 'run-r1/C0'}
        cls.logs = {label: (cell / 'diagnostic-window.log').read_text() for label, cell in cls.cells.items()}
        cls.aligned = {label: C['load'](cell / 'alignment.json') for label, cell in cls.cells.items()}
        cls.frames = A['route_frames'](cls.logs['C0'])
        cls.synth = {}
        print('SYNTHETIC 8K transformations of existing A0/C0 logs and read-only captures; no new E2E.', flush=True)
        for label in C['ARMS']:
            events, prompt = synthetic_long(A['records'](cls.logs[label]), [5] + [1024]*8 + [73])
            log = '\n'.join(map(event_line, events))
            cls.synth[label] = (events, prompt, log)

    def rejected(self, fn, *args):
        with self.assertRaises((AssertionError, ValueError, KeyError, OSError)):
            fn(*args)

    def synthetic_frames(self, widths):
        result, pos = [], 0
        total = sum(widths)
        for n in widths:
            source = next(f for f in self.frames if f['batch']['n'] == (2 if n == 2 else 1 if n == 1 else 46 if n > 8 else 5))
            original_n = source['batch']['n']
            rows = []
            for weight in M['ORDER']:
                templates = [r for r in source['routes'] if r['weight'] == weight]
                want = M['dispatch'](weight, n)
                self.assertEqual(len(templates), len(want))
                for r, (route, prec) in zip(templates, want):
                    r = dict(r, route=route, operand_prec=prec, backend_ctx='0xabc')
                    expert = weight in A['IQ2_PROJECTIONS']
                    for prefix in ('src1_', 'dst_', 'ids_') if expert else ('src1_', 'dst_'):
                        axis = 1 if prefix == 'ids_' or not expert else 2
                        if n != 2:
                            r[prefix+'ne'+str(axis)] = n
                        for j in range(axis+1, 4):
                            self.assertEqual(r[prefix+'nb'+str(j)] % original_n, 0)
                            r[prefix+'nb'+str(j)] = r[prefix+'nb'+str(j)] // original_n * n
                    rows.append(r)
            if pos+n == total:
                head = next(r for f in self.frames if f['batch']['n'] == 1 for r in f['routes'] if r['weight'] == 'output.weight')
                rows.append(dict(head, backend_ctx='0xabc'))
            b = dict(ctx='0xdef', off=0, n=n, tokens=list(range(pos, pos+n)), pos=list(range(pos, pos+n)),
                     outputs=[int(p == total-1) for p in range(pos, pos+n)], seq=','.join(['0']*n))
            result.append(dict(batch=b, routes=rows))
            pos += n
        return result

    def test_01_real_fixtures_exact_argv_and_native_startup(self):
        self.assertEqual(len(self.jobs), 24)
        self.assertEqual(self.jobs[21]['payload'], self.jobs[22]['payload'])
        self.assertEqual(self.jobs[23]['payload'], C['OLD']['diagnostic_payload'](self.jobs[22]['payload']))
        self.assertEqual([j['historical_prompt_tokens'] for j in self.jobs[20:]], [8238, 8270, 8270, 8270])
        for label in C['ARMS']:
            argv = C['make_create'](label, C['ROOT'] / 'run-r1' / label / 'capture')
            self.assertIn('30/30 tests passed', argv[-1])
            old = C['load'](self.cells[label] / 'create-argv.json')
            reverted = list(argv)
            reverted[reverted.index('--name')+1] = old[old.index('--name')+1]
            reverted[reverted.index(str(C['ROOT'] / 'run-r1' / label / 'capture') + ':/capture')] = str(self.cells[label] / 'capture') + ':/capture'
            self.assertEqual(reverted, old)
            startup = (self.cells[label] / 'native-and-startup.log').read_text()
            # Match the new controller's --timestamps capture, and the independent
            # analyzer's full timestamped server-log input. Prefix-only transform
            # is SYNTHETIC formatting coverage, not a new native/GPU run.
            timestamped = '\n'.join('2026-10-18T00:00:00.000000000Z ' + line for line in startup.splitlines())
            for log in (startup, timestamped, (self.cells[label] / 'server.log').read_text()):
                meta = C['check_startup'](log, label)
                self.assertEqual(meta['n_outputs_max'], 1 if label == 'A0' else 2)
        self.rejected(C['make_create'], 'S0', C['ROOT'] / 'run-r1/S0/capture')
        self.rejected(C['check_startup'], (self.cells['C0'] / 'native-and-startup.log').read_text(), 'A0')

    def test_02_full_sources_and_five_binary_pins(self):
        C['source_contracts']()
        report = C['N']['check_sources_and_binaries']()
        C['save'](SCRATCH / 'hash-checks.json', report)
        self.assertTrue(report['all_source_hashes_match'])
        self.assertEqual(len(report['candidate_binaries']), 5)

    def test_03_public_tokenizer_pinned_source_and_mock_boundary(self):
        source = C['CANDIDATE']
        ctx = (source / 'tools/server/server-context.cpp').read_text()
        common = (source / 'tools/server/server-common.cpp').read_text()
        self.assertIn('this->post_apply_template =', ctx)
        self.assertIn('json data = oaicompat_chat_params_parse(', ctx)
        self.assertIn('inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt)', ctx)
        self.assertIn('inputs.add_generation_prompt  = json_value(body, "add_generation_prompt", true);', common)
        self.assertIn('inputs.chat_template_kwargs[item.key()] = item.value().dump();', common)
        calls = []
        def request(base, endpoint, payload):
            calls.append((endpoint, payload))
            return {'prompt': '<synthetic-template>not raw message</synthetic-template>'} if endpoint == '/apply-template' else {'tokens': [17, 42, 91]}
        stem = SCRATCH / 'public-mock'
        ids = C['public_tokenize'](self.jobs[22]['payload'], stem, request)
        self.assertEqual(ids, [17, 42, 91])
        self.assertEqual([e for e, p in calls], ['/apply-template', '/tokenize'])
        self.assertTrue(calls[0][1]['add_generation_prompt'])
        self.assertEqual(calls[0][1]['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(calls[1][1], {'content': '<synthetic-template>not raw message</synthetic-template>', 'add_special': True, 'parse_special': True, 'with_pieces': False})
        self.assertEqual(C['check_public'](self.jobs[22]['payload'], stem), ids)
        self.rejected(C['token_ids'], {'tokens': [{'id': 1}]})
        self.rejected(C['token_ids'], {'tokens': [True]})
        self.rejected(C['apply_request'], dict(self.jobs[22]['payload'], chat_template_kwargs={'enable_thinking': 'false'}))

    def test_04_budget_source_bound_and_refusal(self):
        sampling = (C['CANDIDATE'] / 'common/sampling.cpp').read_text()
        single = sampling.split('llama_token common_sampler_sample(', 1)[1].split('std::vector<llama_token> common_sampler_sample_and_accept_n', 1)[0]
        self.assertEqual(single.count('bolt_verify_write('), 1)
        self.assertNotIn('common_sampler_sample(', single)
        draft = (C['CANDIDATE'] / 'common/speculative.cpp').read_text().split('if (is_dflash2 || is_lilicorr)', 1)[1].split('if (is_dspark)', 1)[0]
        self.assertIn('continue;', draft)
        self.assertNotIn('common_sampler_sample', draft)
        server = (C['CANDIDATE'] / 'tools/server/server-context.cpp').read_text()
        self.assertIn('n_draft_max = std::min(n_draft_max, n_remaining() - 1)', server)
        b = C['budget'](self.jobs)
        self.assertEqual(b['remaining_output_caps'], 5632)
        self.assertEqual(b['raw_row_bound'], 5680)
        self.assertEqual(b['reserve'], 4 * 1024**3)
        self.assertTrue(C['capacity'](C['ROOT'], self.jobs, 11562463232)['fits'])
        self.assertTrue(C['capacity'](C['ROOT'], self.jobs, b['required_free'])['fits'])
        self.assertFalse(C['capacity'](C['ROOT'], self.jobs, b['required_free']-1)['fits'])
        self.assertLess(C['budget'](self.jobs[20:])['required_free'], b['required_free'])
        with patch.object(D['C']['shutil'], 'disk_usage', return_value=type('Disk', (), {'free': b['required_free']-1})()):
            self.rejected(D['check_capacity'], SCRATCH, 'C0', 99, self.jobs)
        C['save'](SCRATCH / 'disk-budget.json', C['capacity'](C['ROOT'], self.jobs))

    def test_05_source_bound_dispatch_thresholds(self):
        cpp = (C['CANDIDATE'] / 'ggml/src/ggml-sycl/ggml-sycl.cpp').read_text()
        fused = (C['CANDIDATE'] / 'ggml/src/ggml-sycl/fused-gemm.hpp').read_text()
        self.assertIn('GGML_SYCL_FG_MAX_N = 64', fused)
        self.assertIn('total_rows <= n_active * GGML_SYCL_FG_MAX_N', fused)
        self.assertIn('GGML_UNUSED(type);\n    return false;', cpp)
        for n in (65, 73, 78, 256, 1024):
            self.assertEqual(M['dispatch']('blk.1.ffn_up_shexp.weight', n)[1], ('LIBRARY_DNNL_F16', 20))
            self.assertEqual(M['dispatch']('blk.0.ssm_in.weight', n)[1], ('LIBRARY_DNNL_F32', 10))
            self.assertEqual(M['dispatch']('blk.7.attn_q.weight', n)[1], ('LIBRARY_DNNL_F16', 20))
        self.assertEqual(M['dispatch']('blk.1.ffn_up_shexp.weight', 64)[1][0], 'LIBRARY_FUSED_DEQUANT_GEMM')
        self.assertEqual(M['dispatch']('blk.1.ffn_gate_inp.weight', 16)[1][0], 'LIBRARY_MKL_F32')
        self.assertEqual(M['dispatch']('blk.48.ffn_gate_inp.weight', 16)[1][0], 'LIBRARY_DNNL_F32')
        self.rejected(M['dispatch'], 'blk.0.ssm_in.weight', 1025)

    def test_06_larger_prefill_fifo_and_all_frame_routes(self):
        for widths in ([5] + [1024]*8 + [73], [1024]*8 + [78], [1, 2, 3, 4, 8, 9, 16, 64, 65, 1024]):
            frames = self.synthetic_frames(widths)
            for first in (False, True):
                got = M['route_frames'](frame_log(frames, first), 'A0')
                self.assertEqual(len(got['frames']), len(widths))
            foreign = ['BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in dict(frames[-1]['routes'][-1], backend_ctx='0xfee').items())]
            self.assertEqual(len(M['route_frames'](frame_log(frames, True, foreign), 'C0')['other_backend_routes']), 1)
            self.rejected(M['route_frames'], frame_log(frames, True, foreign), 'A0')
        # Also check real C0 head width=2 layouts, not just synthetic prefills.
        self.assertEqual(len(M['route_frames'](self.logs['C0'], 'C0')['frames']), len(self.frames))

    def test_07_route_cardinality_metadata_fallback_rejected(self):
        good = self.synthetic_frames([1024, 73])
        for mutation in ('missing', 'extra', 'leading', 'order', 'ctx', 'stride', 'dtype', 'accum', 'operand', 'dynamic', 'result', 'fallback'):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(good)
                rows = bad[0]['routes']
                if mutation == 'missing': rows.pop(6)
                if mutation == 'extra': rows.append(dict(rows[-1]))
                if mutation == 'leading': rows.insert(0, dict(rows[-1]))
                if mutation == 'order': rows[4], rows[6] = rows[6], rows[4]
                if mutation == 'ctx': rows[4]['backend_ctx'] = '0xbad'
                if mutation == 'stride': rows[4]['src1_nb2'] += 4
                if mutation == 'dtype': rows[4]['src1_type'] = 'f16'
                if mutation == 'accum': rows[0]['accum_prec'] = 0
                if mutation == 'operand': rows[1]['operand_prec'] = 15
                if mutation == 'dynamic': rows[4]['dynamic_prec'] = 1
                if mutation == 'result': next(r for r in rows if r['route'] == 'MOE_GROUPED_GEMM_RESULT')['result'] = 0
                if mutation == 'fallback': next(r for r in rows if r['route'] == 'LIBRARY_DNNL_F16')['route'] = 'LIBRARY_FUSED_DEQUANT_GEMM'
                self.rejected(M['route_frames'], frame_log(bad), 'A0')
        lines = frame_log(good).splitlines()
        self.rejected(M['route_frames'], '\n'.join(lines[1:]), 'A0')
        self.rejected(M['route_frames'], '\n'.join(lines + lines[:2]), 'A0')

    def test_08_synthetic_long_alignment_single_and_real_rollback_history(self):
        for label in C['ARMS']:
            events, prompt, log = self.synth[label]
            result = M['align'](log, self.aligned[label]['emitted_ids'], self.cells[label] / 'capture', prompt, label)
            self.assertEqual(result['batch_metadata']['prompt_coverage'], 8270)
            self.assertEqual(result['batch_metadata']['prefill_widths'], [5] + [1024]*8 + [73])
            self.assertEqual(result['samples'][0]['position'], 8269)
            self.assertEqual(result['samples'][-1]['position'], 8396)
            self.assertEqual(result['proposed'], sum(len(v['drafts']) for v in self.aligned[label]['verifications']))
            if label == 'C0':
                self.assertGreater(result['rejection_events'], 0)
                self.assertGreater(result['accepted'], 0)
        # Same actual histories with a different legal contiguous partition.
        e, prompt = synthetic_long(A['records'](self.logs['A0']), [1024]*8 + [78])
        M['check_batches']([x for x in e if x['event'] == 'batch'], prompt, self.aligned['A0']['emitted_ids'], 'A0')

    def test_09_prompt_rollback_and_sample_row_fail_closed(self):
        events, prompt, _ = self.synth['C0']
        for mutation in ('token', 'gap', 'flag', 'rollback', 'sample', 'checkpoint', 'replay'):
            bad = copy.deepcopy(events)
            if mutation in ('token', 'gap', 'flag'):
                b = next(x for x in bad if x['event'] == 'batch')
                b[{'token': 'tokens', 'gap': 'pos', 'flag': 'outputs'}[mutation]][0] += 1
            if mutation == 'rollback': next(x for x in bad if x['event'] == 'remove' and x['kind'] == 'verify')['p0'] += 1
            if mutation == 'sample': next(x for x in bad if x['event'] == 'sample')['idx'] -= 1
            if mutation == 'checkpoint': bad.append(dict(event='checkpoint', ctx=events[0]['ctx'], slot=0, p0=0, p1=1, rollback=1))
            if mutation == 'replay': next(x for x in bad if x['event'] == 'verify')['replay'] = 1
            self.rejected(M['align'], '\n'.join(map(event_line, bad)), self.aligned['C0']['emitted_ids'], self.cells['C0'] / 'capture', prompt, 'C0')
        self.rejected(M['align'], self.synth['C0'][2], self.aligned['C0']['emitted_ids'], self.cells['C0'] / 'capture', prompt, 'A0')

    def test_10_full_raw_bounds_symlinks_bytes_and_finite(self):
        sample = self.aligned['A0']['samples'][0]
        data = A['raw_bytes'](self.cells['A0'] / 'capture', sample)
        for name in ('../row-ABCDEF', '/tmp/row-ABCDEF', 'row-wrong'):
            self.rejected(A['raw_bytes'], self.cells['A0'] / 'capture', dict(sample, file=name))
        self.rejected(A['raw_bytes'], self.cells['A0'] / 'capture', dict(sample, status='backend'))
        directory = SCRATCH / 'raw-guards'; directory.mkdir()
        (directory / 'row-SHORTX').write_bytes(data[:-4])
        self.rejected(A['raw_bytes'], directory, dict(sample, file='row-SHORTX'))
        (directory / 'row-SYMLNK').symlink_to(self.cells['A0'] / 'capture' / sample['file'])
        self.rejected(A['raw_bytes'], directory, dict(sample, file='row-SYMLNK'))
        # Test the actual finite guard through align, with in-memory mocked reads.
        bad = array.array('f'); bad.frombytes(data); bad[0] = float('inf')
        with patch.dict(M['A']['align'].__globals__, {'raw_bytes': lambda d, s: bad.tobytes()}), patch.dict(M['A'], {'raw_bytes': lambda d, s: bad.tobytes()}):
            self.rejected(M['align'], self.synth['A0'][2], self.aligned['A0']['emitted_ids'], self.cells['A0'] / 'capture', self.synth['A0'][1], 'A0')

    def test_11_full_vectors_inclusive_first_difference_no_parent_invariance(self):
        left, right = [copy.deepcopy(self.aligned[label]) for label in C['ARMS']]
        for label, result in zip(C['ARMS'], (left, right)):
            # The saved legacy C0 alignment has no label. The production align()
            # sets it explicitly; label this synthetic transformation honestly.
            result['label'] = label
            result['prompt_ids'] = self.synth['A0'][1]
            for s in result['samples']: s['position'] += 8215
        self.rejected(M['compare'], dict(left, label='C0'), right)
        self.rejected(M['compare'], left, dict(right, label='A0'))
        result = M['compare'](left, right)
        self.assertEqual(result['compared_through_ordinal'], 128)
        self.assertTrue(result['all_compared_raw_vectors_equal'])
        self.assertFalse(result['parent_invariance_required'])
        for ordinal in (1, 116, 128):
            bad = copy.deepcopy(right); bad['emitted_ids'][ordinal-1] += 1
            if ordinal < 128: bad['samples'][ordinal]['file'] = '../not-visited'
            self.assertEqual(M['compare'](left, bad)['compared_through_ordinal'], ordinal)
        bad = copy.deepcopy(right); bad['emitted_ids'][0] += 1
        directory = SCRATCH / 'raw-difference'; directory.mkdir()
        s = bad['samples'][0]
        raw = array.array('f'); raw.frombytes(A['raw_bytes'](Path(right['capture_path']), s)); raw[0] += 1
        (directory / s['file']).write_bytes(raw.tobytes())
        bad['capture_path'] = str(directory)
        result = M['compare'](left, bad)
        self.assertEqual(result['first_score_difference']['ordinal'], 1)
        self.assertEqual(result['first_score_difference']['first_differing_float_id'], 0)

    def test_12_exact_quality_and_anchor_criteria_not_tightened(self):
        summary = C['load'](self.cells['C0'] / 'quality/summary.json')
        for check in summary['checks']:
            self.assertEqual(C['functional_quality'](check['name'], check['result']), check['pass'])
        anchor = C['load'](C['FIXTURES'] / 'anchor.result.json')
        self.assertTrue(C['functional_anchor'](anchor))
        self.assertTrue(C['functional_anchor'](dict(anchor, text=' \nBOLT-7319-CEDAR\n')))
        self.assertFalse(C['functional_anchor'](dict(anchor, text='BOLT-7319-CEDAR extra')))
        self.assertFalse(C['functional_anchor'](dict(anchor, finish_reason='length')))
        self.assertFalse(C['functional_anchor'](dict(anchor, usage={'prompt_tokens': 8191})))
        # Compare copied quality criteria with actual legacy main, CPU-only mocks.
        api = runpy.run_path(str(C['OLD']['OLD'] / 'probe-api.py'))
        saved = {row['name']: row['result'] for row in summary['checks']}
        target = SCRATCH / 'legacy-quality-mock'
        def chat(base, prompt, out, **kwargs):
            self.assertEqual(kwargs, {'count': 256})
            self.assertEqual(prompt, C['load'](self.cells['C0'] / 'quality' / (out.name + '.request.json'))['messages'][0]['content'])
            return saved[out.name]
        with patch.object(sys, 'argv', ['probe-api.py', '--mode', 'quality', '--out', str(target)]), patch.dict(api['main'].__globals__, {'chat': chat, 'request': lambda *args: {'data': [{'id': 'nemotron-bolt'}]}}):
            api['main']()
        self.assertTrue(C['load'](target / 'summary.json')['pass'])

    def test_13_exact_restoration_and_failure_attempts_are_independent(self):
        original = C['load'](C['REFERENCE'] / 'run-r1/original-container.json')
        live = C['load'](C['REFERENCE'] / 'run-r1/restored-container.json')
        C['P']['check_restored'](original, live)
        for key in ('Config', 'HostConfig', 'Mounts', 'Id'):
            bad = copy.deepcopy(live); bad[key] = 'bad' if key == 'Id' else {}
            self.rejected(C['P']['check_restored'], original, bad)
        target = SCRATCH / 'restore-mock'; target.mkdir(); (target / 'A0').mkdir()
        calls = []
        def command(*args):
            calls.append(args)
            if args[:2] == ('docker', 'logs'): raise RuntimeError('synthetic capture-log failure')
            if args[:2] == ('docker', 'inspect'): return json.dumps([live])
            if args[:2] == ('docker', 'ps'): return C['OLD']['ORIGINAL'] + '\n'
            return ''
        api = {'request': lambda *a: {'status': 'ok'}, 'chat': lambda *a, **kw: {'text': 'READY', 'finish_reason': 'stop'}}
        with patch.dict(D['C']['N'], {'check_sources_and_binaries': lambda: {'synthetic': True}}):
            errors = D['restore'](target, original, {'synthetic': True}, ['A0'], True, command, lambda *a: {}, api, lambda: {'synthetic': True})
        self.assertEqual([e['action'] for e in errors], ['logs-A0'])
        self.assertIn(('docker', 'start', C['OLD']['ORIGINAL']), calls)
        self.assertTrue((target / 'hardware-after.json').exists())
        self.assertTrue((target / 'hash-checks-after.json').exists())

    def test_14_controller_no_analysis_and_exclusive_pair(self):
        text = (HERE / 'run-long.py').read_text()
        tree = ast.parse(text)
        self.assertIn('out.mkdir(exist_ok=False)', text)
        self.assertIn("for label in C['ARMS']", text)
        self.assertIn('finally:', text)
        self.assertNotIn('analyze-long.py', text)
        self.assertNotIn('docker\', \'rm', text)
        self.assertNotIn('probe-api.py\'),', text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                self.assertNotIn(ast.unparse(node.func.slice), ("'analyze_arm'", "'align'", "'compare'", "'main'"))
        self.assertEqual(C['code_hashes'](), C['code_hashes']())

    def test_15_postprocess_keeps_later_results_after_bad_response(self):
        target = SCRATCH / 'responses'; target.mkdir(); (target / 'public').mkdir()
        jobs = [dict(stem='anchor', payload=self.jobs[20]['payload']), dict(stem='ordinary', payload=self.jobs[22]['payload'])]
        for i, job in enumerate(jobs):
            stem = target / job['stem']; C['save'](stem.with_suffix('.request.json'), job['payload'])
            C['save'](stem.with_suffix('.http.json'), {'status': 200})
            C['public_tokenize'](job['payload'], target / 'public' / f'{i:02}', lambda base, path, p: {'prompt': 'synthetic'} if path == '/apply-template' else {'tokens': [17, 18]})
        (target / 'anchor.http-body').write_text('not JSON')
        body = {'choices': [{'delta': {'content': 'synthetic text'}, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 2, 'completion_tokens': 128}, 'timings': {'cache_n': 0, 'predicted_n': 128}}
        (target / 'ordinary.http-body').write_text('data: ' + json.dumps(body) + '\n\ndata: [DONE]\n\n')
        results, public, report = M['analyze_responses'](target, jobs)
        self.assertIn('ordinary', results)
        self.assertEqual(report['errors'][0]['stem'], 'anchor')
        self.assertFalse(report['pass_all'])
        self.assertTrue((target / 'ordinary.result.json').exists())
        self.assertEqual((target / 'anchor.http-body').read_text(), 'not JSON')


if __name__ == '__main__':
    unittest.main(verbosity=2)
