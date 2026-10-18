#!/usr/bin/env python3
"""Remote CPU-only tests. SYNTHETIC A0 fixtures are NOT live no-draft evidence."""
import array
import ast
import copy
import json
import shutil
import re
import runpy
import shlex
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

assert socket.gethostname() == 'inference-host'
C = runpy.run_path(str(Path(__file__).with_name('run-a0.py')))
M = runpy.run_path(str(Path(__file__).with_name('analyze-a0.py')))
A, R = M['A'], C['REFERENCE']
S0 = R / 'run-r1/S0'
SCRATCH = C['ROOT'] / 'worker-scratch'


def route_line(r):
    return 'BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in r.items())


def frame_log(frames, routes_first=False):
    verify, routes = [], []
    for frame in frames:
        b = frame['batch']
        fields = {k: b[k] for k in ('ctx', 'off', 'n', 'tokens', 'pos', 'outputs', 'seq')}
        for k in ('tokens', 'pos', 'outputs'):
            fields[k] = ','.join(map(str, fields[k]))
        verify.extend(['BOLT_VERIFY batch ' + ' '.join(f'{k}={v}' for k, v in fields.items()),
                       'BOLT_VERIFY decoded ctx=' + b['ctx'] + ' ret=0'])
        routes.extend(route_line(r) for r in frame['routes'])
    return '\n'.join(routes + verify if routes_first else verify + routes)


class A0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = (S0 / 'diagnostic-window.log').read_text()
        cls.alignment = C['load'](S0 / 'alignment.json')
        cls.frames = A['route_frames'](cls.log)
        cls.argv = C['load'](S0 / 'create-argv.json')
        cls.a0argv = C['make_create'](cls.argv, C['ROOT'] / 'run-r1/A0/capture')
        cls.temp = tempfile.TemporaryDirectory(prefix='synthetic-a0-', dir=SCRATCH)
        cls.scratch = Path(cls.temp.name)
        print('FIXTURES: transformed S0/C0 CPU fixtures only; real A0 remains PENDING', flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def rejected(self, fn, *args):
        with self.assertRaises((AssertionError, ValueError, KeyError, FileNotFoundError, OSError)):
            fn(*args)

    def test_argv_exact_four_changes_and_ro_native30(self):
        a = list(self.a0argv)
        a[a.index('--name')+1] = C['P']['PREFIX'] + 's0'
        a[a.index(str(C['ROOT'] / 'run-r1/A0/capture') + ':/capture')] = str(S0 / 'capture') + ':/capture'
        # Removed draft mount occupied the two slots immediately after capture.
        i = a.index(str(S0 / 'capture') + ':/capture') + 1
        a[i:i] = ['-v', str(C['OLD']['DRAFT'] / 'draft') + ':/draft:ro']
        a[-1] += ' ' + shlex.join(C['P']['spec_for_arm']('S0'))
        self.assertEqual(a, self.argv)
        self.assertIn('30/30 tests passed', self.a0argv[-1])
        self.assertEqual(len(C['P']['ROUTER_NATIVE_CASES']), 30)
        for flag in ('--spec-type none', '--spec-draft-n-max 0', '--cache-type-r f32'):
            bad = list(self.a0argv); bad[-1] += ' ' + flag
            self.rejected(C['assert_no_draft_argv'], bad)
        bad = list(self.a0argv); bad.insert(3, '/any:/draft:ro')
        self.rejected(C['assert_no_draft_argv'], bad)
        self.rejected(C['make_create'], self.argv, C['ROOT'] / 'run-r1/S0/capture')

    def model_fixture(self):
        # Prior REAL no-draft startup confirms K1, but not this candidate's routes.
        old = (C['P']['WIDTH1'] / 'run-r1/A0/native-and-startup.log').read_text()
        model = old.split(C['OLD']['SERVER_MARKER'], 1)[1]
        return model

    def test_real_prior_no_draft_startup(self):
        C['check_model_startup'](self.model_fixture())
        for field in ('n_rs_seq', 'n_outputs_max', 'n_outputs_max_per_seq'):
            bad = re.sub(r'(' + field + r'\s*=\s*)\d+', r'\g<1>2', self.model_fixture())
            self.rejected(C['check_model_startup'], bad)
        for marker in ("adding speculative implementation 'draft-dflash'", 'loading draft model',
                       'common_speculative_impl_draft_dflash', 'model=/draft/evil.gguf'):
            self.rejected(C['check_model_startup'], self.model_fixture() + '\n' + marker)
        C['check_model_startup'](self.model_fixture() + '\nsource=/src/lilicorr/runtime')

    def test_reused_actual_native_gates_then_synthetic_a0_startup(self):
        real = (S0 / 'native-and-startup.log').read_text()
        native = real.split(C['P']['MARKER'], 1)[0]
        fixture = native + C['P']['MARKER'] + self.model_fixture()
        self.assertEqual(C['check_startup'](fixture)['label'], 'A0')
        self.rejected(C['check_startup'], fixture.replace('30/30 tests passed', '29/30 tests passed'))
        self.rejected(C['check_startup'], real)  # S0 is not A0

    def test_runtime_sources_and_five_hashes_unchanged(self):
        result = C['check_sources_and_binaries']()
        self.assertTrue(result['all_source_hashes_match'])
        C['save'](Path('hash-checks.json'), result)

    def test_frozen_requests(self):
        payload, diag = C['frozen_requests']()
        self.assertFalse(diag['stream'])
        self.assertFalse(diag['backend_sampling'])
        self.assertEqual(payload['max_tokens'], 128)

    def test_restore_exact_configuration(self):
        original = C['load'](R / 'run-r1/original-container.json')
        live = C['load'](R / 'run-r1/restored-container.json')
        C['P']['check_restored'](original, live)
        for key in ('Config', 'HostConfig', 'Mounts', 'Id'):
            bad = copy.deepcopy(live); bad[key] = {} if key != 'Id' else 'other'
            self.rejected(C['P']['check_restored'], original, bad)

    def test_no_imported_main_and_analysis_after_restore_boundary(self):
        tree = ast.parse(Path('run-a0.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                self.assertNotIn(ast.unparse(node.func.slice), ("'main'", "'analyze_arm'"))
        self.assertNotIn('analyze-a0.py', Path('run-a0.py').read_text())
        self.assertIn("out.mkdir(exist_ok=False)", Path('run-a0.py').read_text())

    def synthetic_frames(self, widths):
        """SYNTHETIC width scaling of observed target layouts, source dispatch rules."""
        result, pos = [], 0
        for n in widths:
            # The existing C0 two-column model routes independently cover n=2.
            if n == 2:
                c0 = A['route_frames']((R / 'run-r1/C0/diagnostic-window.log').read_text())
                source = next(f for f in c0 if f['batch']['n'] == 2)
            else:
                source = self.frames[3 if n == 1 else 1 if n > 8 else 0]
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
                        if not (n == 2):
                            r[prefix+'ne'+str(axis)] = n
                        for j in range(axis+1, 4):
                            self.assertEqual(r[prefix+'nb'+str(j)] % original_n, 0)
                            r[prefix+'nb'+str(j)] = r[prefix+'nb'+str(j)] // original_n * n
                    rows.append(r)
            if pos+n == 55:
                rows.append(dict(self.frames[3]['routes'][-1], backend_ctx='0xabc'))
            b = dict(ctx='0xdef', off=0, n=n, tokens=list(range(pos, pos+n)),
                     pos=list(range(pos, pos+n)), outputs=[int(p == 54) for p in range(pos, pos+n)],
                     seq=','.join(['0']*n))
            result.append(dict(batch=b, routes=rows))
            pos += n
        return result

    def test_all_actual_s0_target_layouts_synthetic_no_draft_fifo(self):
        # Deliberately discard the S0 draft backend only in this SYNTHETIC fixture.
        frames = [dict(batch=f['batch'], routes=f['routes']) for f in self.frames]
        for first in (False, True):
            checked = M['route_frames'](frame_log(frames, first))
            self.assertEqual(len(checked), 130)

    def test_source_derived_dispatch_width_boundaries(self):
        for widths in ((55,), (5, 46, 4), (1, 2, 3, 4, 5, 8, 9, 23), (16, 17, 22)):
            with self.subTest(widths=widths):
                self.assertEqual(sum(widths), 55)
                frames = self.synthetic_frames(widths)
                self.assertEqual(len(M['route_frames'](frame_log(frames, True))), len(widths))
        # Individual F32 widths have DIFFERENT MKL/DNNL thresholds.
        self.assertEqual(M['dispatch']('blk.1.ffn_gate_inp.weight', 16)[1][0], 'LIBRARY_MKL_F32')
        self.assertEqual(M['dispatch']('blk.48.ffn_gate_inp.weight', 16)[1][0], 'LIBRARY_DNNL_F32')

    def test_route_fail_closed_missing_extra_foreign_order_metadata(self):
        good = self.synthetic_frames((55,))
        for mutation in ('missing', 'extra', 'leading', 'foreign', 'order', 'width', 'stride', 'unknown', 'field', 'prec', 'result'):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(good); rows = bad[0]['routes']
                if mutation == 'missing': rows.pop(5)
                if mutation == 'extra': rows.append(dict(rows[-1]))
                if mutation == 'leading': rows.insert(0, dict(rows[-1]))
                if mutation == 'foreign': rows[4]['backend_ctx'] = '0xbad'
                if mutation == 'order': rows[4], rows[6] = rows[6], rows[4]
                if mutation == 'width': bad[0]['batch']['n'] = 54
                if mutation == 'stride': rows[4]['src1_nb2'] += 4
                if mutation == 'unknown': rows[4]['route'] = 'MAGIC_GEMM'
                if mutation == 'field': rows[4].pop('ids_nb2')
                if mutation == 'prec': rows[4]['dynamic_prec'] = 1
                if mutation == 'result': rows[4]['result'] = 0
                self.rejected(M['route_frames'], frame_log(bad))
        self.rejected(M['route_frames'], self.log)  # REAL S0 has a draft backend
        lines = frame_log(good).splitlines()
        self.rejected(M['route_frames'], '\n'.join(lines[1:]))
        self.rejected(M['route_frames'], '\n'.join(lines + lines[:2]))

    def test_chunk_coverage_does_not_impose_parent_batch_shapes(self):
        prompt, ids = self.alignment['prompt_ids'], self.alignment['emitted_ids']
        decode = copy.deepcopy(self.alignment['batches'][3:])
        for widths in ((55,), (5, 46, 4), (1, 54), (17, 19, 19)):
            batches, pos = [], 0
            for n in widths:
                batches.append(dict(off=0, n=n, seq=','.join(['0']*n), tokens=prompt[pos:pos+n],
                                    pos=list(range(pos, pos+n)), outputs=[int(p == 54) for p in range(pos, pos+n)]))
                pos += n
            result = M['check_batches'](batches + decode, prompt, ids)
            self.assertEqual(result['prefill_widths'], list(widths))
        for field, value in (('pos', [1, 2, 3, 4, 5]), ('tokens', [0]*5), ('outputs', [1]*5), ('seq', '1,1,1,1,1')):
            bad = copy.deepcopy(self.alignment['batches']); bad[0][field] = value
            self.rejected(M['check_batches'], bad, prompt, ids)
        self.rejected(M['check_batches'], self.alignment['batches'][1:], prompt, ids)
        self.rejected(M['check_batches'], self.alignment['batches'] + self.alignment['batches'][-1:], prompt, ids)

    def test_single_only_alignment128_on_saved_common_capture_mechanics(self):
        result = M['align_a0'](self.log, self.alignment['emitted_ids'], S0 / 'capture', self.alignment['prompt_ids'])
        self.assertEqual(len(result['samples']), 128)
        self.assertEqual(result['proposed'], 0)
        self.assertEqual(result['batch_metadata']['decode_widths'], [1])
        # This tests COMMON raw alignment only, not live A0 dispatch.
        for extra in ('BOLT_VERIFY checkpoint ctx=0xabc slot=0 p0=0 p1=1 rollback=0',
                      'BOLT_VERIFY verify ctx=0xabc smpl=0xdef slot=0 rows=0,1 drafts=1 emitted=2 accepted=0 rollback=1 replay=0 rejection=0 synth=0',
                      'BOLT_VERIFY remove ctx=0xabc slot=0 kind=verify p0=55 p1=-1 rollback=0'):
            self.rejected(M['align_a0'], self.log+'\n'+extra, self.alignment['emitted_ids'], S0/'capture', self.alignment['prompt_ids'])
        bad = re.sub(r'BOLT_VERIFY single[^\n]+\n', '', self.log, count=1)
        self.rejected(M['align_a0'], bad, self.alignment['emitted_ids'], S0/'capture', self.alignment['prompt_ids'])

    def test_raw_file_guards(self):
        sample = dict(self.alignment['samples'][0])
        path = self.scratch / 'row-ABCDEF'
        sample['file'] = path.name
        data = A['raw_bytes'](S0/'capture', self.alignment['samples'][0])
        path.write_bytes(data)
        self.assertEqual(A['raw_bytes'](self.scratch, sample), data)
        for name in ('../row-ABCDEF', '/tmp/row-ABCDEF', 'row-wrong'):
            self.rejected(A['raw_bytes'], self.scratch, dict(sample, file=name))
        self.rejected(A['raw_bytes'], self.scratch, dict(sample, status='backend'))
        path.write_bytes(data[:-4]); self.rejected(A['raw_bytes'], self.scratch, sample)
        path.unlink(); path.symlink_to(S0/'capture'/self.alignment['samples'][0]['file'])
        self.rejected(A['raw_bytes'], self.scratch, sample)
        path.unlink(); path.mkdir(); self.rejected(A['raw_bytes'], self.scratch, sample)

    def test_fidelity_cutoff_inclusive_and_labels_and_shape_differences(self):
        left = dict(copy.deepcopy(self.alignment), label='A0')  # explicitly synthetic arm
        right = copy.deepcopy(self.alignment)
        left['batches'][0]['n'] = 55  # comparison reports rather than enforces batch invariance
        for label in ('S0', 'C0'):
            result = M['compare'](left, right, label)
            self.assertEqual(result['compared_through_ordinal'], 128)
            self.assertTrue(result['all_compared_raw_vectors_equal'])
            self.assertFalse(result['actual_batch_inputs_match'])
            self.assertEqual(set(result['comparisons'][0]['samples']), {'A0', label})
        for ordinal in (1, 116, 128):
            r = copy.deepcopy(right); r['emitted_ids'][ordinal-1] += 1
            # Poison the next row: no comparison beyond divergence is allowed.
            if ordinal < 128: r['samples'][ordinal]['file'] = '../not-visited'
            result = M['compare'](left, r, 'C0')
            self.assertEqual(result['compared_through_ordinal'], ordinal)
            self.assertEqual(result['first_output_difference_ordinal'], ordinal)
        for field in ('position', 'vocab', 'input_id'):
            bad = copy.deepcopy(right); bad['samples'][0][field] += 1
            self.rejected(M['compare'], left, bad, 'S0')
        changed = copy.deepcopy(right)
        capture = self.scratch/'changed'; capture.mkdir()
        for sample in changed['samples']:
            shutil.copyfile(S0/'capture'/sample['file'], capture/sample['file'])
        path = capture/changed['samples'][0]['file']
        values = array.array('f'); values.frombytes(path.read_bytes())
        values[0] += 1
        path.write_bytes(values.tobytes())  # scratch copy only; references are read-only
        changed['capture_path'] = str(capture)
        diff = M['compare'](left, changed, 'S0')
        self.assertEqual(diff['first_score_difference']['ordinal'], 1)
        self.assertEqual(diff['first_score_difference']['first_differing_float_id'], 0)
        self.assertIsNone(diff['first_output_difference_ordinal'])

    def test_within_arm_output_consistency_mandatory(self):
        ordinary = C['load'](S0/'ordinary.result.json')
        diagnostic = C['load'](S0/'diagnostic.response.json')
        A['within_arm_text_consistency'](ordinary, diagnostic)
        ordinary['text'] += 'different'
        self.rejected(A['within_arm_text_consistency'], ordinary, diagnostic)

    def test_quality20_saved_common_comparison_no_raw_id_claim(self):
        result = M['quality_compare'](S0, R/'run-r1/C0', 'C0')
        self.assertEqual(result['cases'], 20)
        self.assertTrue(result['all_requests_equal'])
        self.assertTrue(result['all_outputs_finish_counts_equal'])
        self.assertFalse(result['raw_token_ids_claimed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
