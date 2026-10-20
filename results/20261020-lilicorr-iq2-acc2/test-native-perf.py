#!/usr/bin/env python3
"""CPU-only fixture, exact-mount and native-body checks; no runtime or GPU calls."""
import copy
import contextlib
import io
import json
import sys
import tempfile
from unittest.mock import patch
import re
import runpy
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = runpy.run_path(str(HERE / 'run-native-perf.py'))


def cases():
    result = []
    for line in (HERE / 'native-perf-cases.txt').read_text().splitlines():
        fields = line.split()
        integers = iter(map(int, fields[:-1]))
        op, dtype = next(integers), next(integers)
        shape = [next(integers) for _ in range(4)]
        params = [next(integers) for _ in range(next(integers))]
        sources = [[next(integers) for _ in range(9)] for _ in range(next(integers))]
        assert list(integers) == []
        result.append(dict(op=op, dtype=dtype, shape=shape, params=params, sources=sources, name=fields[-1]))
    return result


class NativePerfTests(unittest.TestCase):
    def test_fixed_native_shapes_and_scope(self):
        rows = cases()
        self.assertEqual(len(rows), 2)
        for row, (k, m, nrows) in zip(rows, ((1024, 2048, 1), (2048, 1024, 9))):
            self.assertEqual((row['op'], row['dtype']), (30, 0))
            self.assertEqual(row['shape'], [m, 9, 4, 1])
            self.assertEqual(row['params'], [10] + [0] * 14 + [0x4e535631])
            self.assertEqual(len(row['sources']), 3)
            weights, inputs, ids = row['sources']
            stride = k // 256 * 66
            self.assertEqual(weights, [16, k, m, 240, 1, 66, stride, stride * m, stride * m * 240])
            self.assertEqual(inputs, [0, k, nrows, 4, 1, 4, k * 4, k * nrows * 4, k * nrows * 16])
            self.assertEqual(ids, [26, 9, 4, 1, 1, 4, 960, 3840, 3840])
            self.assertLess(k * m * 240, 2**31)
        self.assertEqual([row['name'] for row in rows], ['lilicorr-iq2-up-n4', 'lilicorr-iq2-down-n4'])

    def test_native_shell_syntax(self):
        subprocess.run(['bash', '-n'], input=P['NATIVE_BODY'], text=True, check=True)

    def test_actual_sequence_matches_declared_abba(self):
        text = P['NATIVE_BODY'].split('index=0\n', 1)[1]
        sequence = re.search(r'for variant in (.*?); do', text).group(1).split()
        self.assertEqual(tuple(sequence), P['SEQUENCE'])
        self.assertEqual(sequence.count('base'), 6)
        self.assertEqual(sequence.count('acc2'), 6)
        self.assertIn('GGML_SYCL_DEBUG=0', text)
        self.assertIn('env -u BOLT_VERIFY_CAPTURE_DIR', text)
        self.assertNotIn('SYCL_UR_TRACE=', text)
        self.assertNotIn('LD_DEBUG=', text)
        self.assertIn('unset LLAMA_LILICORR_PROFILE SYCL_UR_TRACE SYCL_PI_TRACE LD_DEBUG', P['NATIVE_BODY'])
        self.assertIn("'expected two measured operations'", text)
        self.assertIn('math.isfinite(time_us)', text)
        self.assertNotIn('n4 kernel missing from UR trace', P['NATIVE_BODY'])

    def check_perf(self, text):
        source = P['NATIVE_BODY'].split("<<'PERF'\n", 1)[1].split('\nPERF\n', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'perf.log'
            path.write_text(text)
            output = io.StringIO()
            with patch.object(sys, 'argv', ['inline-check', str(path)]), contextlib.redirect_stdout(output):
                exec(compile(source, 'inline-native-perf-check', 'exec'), {})
            return json.loads(output.getvalue())

    def sample_perf(self):
        return ''.join('  MUL_MAT_ID(name=' + name + ',type=f32):    1024 runs - 123.45 us/run - bandwidth\n'
                       for name in ('lilicorr-iq2-up-n4', 'lilicorr-iq2-down-n4'))

    def test_perf_validator_accepts_exactly_two_positive_results(self):
        rows = self.check_perf(self.sample_perf())
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['operator_graph_wall_us'] for row in rows], [123.45, 123.45])

    def test_perf_validator_rejects_missing_duplicate_or_unsupported_rows(self):
        sample = self.sample_perf()
        for text in ('', sample.splitlines()[0], sample + sample, sample.replace('lilicorr-iq2-down-n4', 'lilicorr-iq2-up-n4'), sample + 'not supported\n', sample + 'skipping large tensors\n'):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                self.check_perf(text)

    def test_perf_validator_rejects_nonfinite_nonpositive_or_zero_runs(self):
        for value in ('nan', 'inf', '-1', '0'):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                self.check_perf(self.sample_perf().replace('123.45', value))
        with self.assertRaises(AssertionError):
            self.check_perf(self.sample_perf().replace('1024 runs', '0 runs'))

    def test_frozen_native_gates_and_actual_loader_diagnostic(self):
        text = P['NATIVE_BODY']
        self.assertIn('for kind in q8 iq2;', text)
        self.assertIn('python3 -B /iq2-batch4/native-batch4.py --kind "$kind"', text)
        self.assertIn('"$bin/test-backend-ops"', text)
        self.assertIn('LD_DEBUG=libs SYCL_UR_TRACE=2', text)
        self.assertIn('UR_QUEUE_FLAG_PROFILING_ENABLE', text)
        self.assertIn("'calling init:'", text)
        self.assertNotIn('enable_profiling', text)
        self.assertNotIn('SYCL_UR_USE_LEVEL_ZERO_V2', text)
        self.assertNotIn('-ffast-math', text)

    def test_exact_create_delta_and_image_position(self):
        original = ['docker', 'create', '-v', str(P['R']['S']['CANDIDATE']) + ':/src:ro', 'image@sha256:frozen', 'bash', '-lc', 'original native and server command']
        result = P['derive'](original)
        self.assertEqual(result[-4], original[-4])
        self.assertEqual(result[-6:-4], ['-v', P['BASE_BIND']])
        self.assertEqual(result[-1], P['NATIVE_BODY'])
        self.assertEqual(original[-1], 'original native and server command')
        with self.assertRaises(AssertionError):
            P['derive'](result)
        with self.assertRaises(AssertionError):
            P['derive'](['docker', 'create', 'image', 'bash', '-lc', 'wrong source'])

    def record(self):
        return dict(HostConfig=dict(Binds=['other:/keep:ro', P['BASE_BIND']], ShmSize=123),
                    Mounts=[dict(Destination='/keep'), copy.deepcopy(P['BASE_MOUNT'])],
                    Config=dict(Cmd=['bash', '-lc', P['NATIVE_BODY']]))

    def test_mount_projection_preserves_original_and_every_other_field(self):
        record = self.record()
        before = copy.deepcopy(record)
        result = P['without_baseline_mount'](record)
        self.assertEqual(record, before)
        self.assertEqual(result['HostConfig'], dict(Binds=['other:/keep:ro'], ShmSize=123))
        self.assertEqual(result['Mounts'], [dict(Destination='/keep')])
        self.assertEqual(result['Config'], record['Config'])

    def test_baseline_mount_drift_is_rejected_not_discarded(self):
        for key, value in (('RW', True), ('Source', '/wrong'), ('Type', 'volume'), ('Mode', 'rw'), ('Propagation', 'shared')):
            record = self.record()
            record['Mounts'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                P['without_baseline_mount'](record)
        for binds in ([], [P['BASE_BIND'], P['BASE_BIND']]):
            record = self.record()
            record['HostConfig']['Binds'] = binds
            with self.assertRaises(AssertionError):
                P['without_baseline_mount'](record)


    def test_stopped_cell_comparison_only_normalizes_exact_native_id_and_restores_guard(self):
        runtime, c, _capture, cls = P['bind']()
        original = c['same_container']
        expected = dict(Id='exact-native-id', Name='/' + c['name']('VSPF-W3a'),
                        Config=dict(Cmd=['native']), HostConfig=dict(OomKillDisable=False, ShmSize=123), Mounts=[])
        actual = copy.deepcopy(expected)
        actual['HostConfig']['OomKillDisable'] = None
        control = object.__new__(cls)
        control.out = Path('/unused-cpu-contract')

        def inherited_check(_self):
            self.assertTrue(c['same_container'](expected, actual))
            bad = copy.deepcopy(actual)
            bad['HostConfig']['ShmSize'] = 124
            self.assertFalse(c['same_container'](expected, bad))
            before_original, after_original = copy.deepcopy(expected), copy.deepcopy(actual)
            before_original['Id'] = after_original['Id'] = 'original-service'
            self.assertFalse(c['same_container'](before_original, after_original))
            return {'checked': True}

        with patch.dict(c, load=lambda _path: copy.deepcopy(expected)), \
                patch.object(runtime['Runtime'], 'stopped_cells', inherited_check):
            self.assertEqual(control.stopped_cells(), {'checked': True})
        self.assertIs(c['same_container'], original)
        with patch.dict(c, load=lambda _path: copy.deepcopy(expected)), \
                patch.object(runtime['Runtime'], 'stopped_cells', side_effect=RuntimeError('stop-check failed')):
            with self.assertRaises(RuntimeError):
                control.stopped_cells()
        self.assertIs(c['same_container'], original)


if __name__ == '__main__':
    unittest.main()
