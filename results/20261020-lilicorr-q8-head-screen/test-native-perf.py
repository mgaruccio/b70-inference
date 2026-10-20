#!/usr/bin/env python3
"""CPU-only contract tests; no Docker, SSH, SDK, GPU, or evidence mutation."""
import contextlib
import copy
import io
import json
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
P = runpy.run_path(str(HERE / 'run-native-perf.py'))


def cases():
    result = []
    for line in (HERE / 'native-perf-cases.txt').read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        integers = iter(map(int, fields[:-1]))
        op, dtype = next(integers), next(integers)
        shape = [next(integers) for _ in range(4)]
        params = [next(integers) for _ in range(next(integers))]
        sources = []
        for _ in range(next(integers)):
            sources.append([next(integers) for _ in range(9)])
        assert list(integers) == []
        result.append(dict(op=op, dtype=dtype, shape=shape, params=params,
                           sources=sources, name=fields[-1]))
    return result




class NativePerfTests(unittest.TestCase):
    def test_exact_pinned_abi_shapes_strides_and_neutral_params(self):
        rows = cases()
        self.assertEqual(len(rows), 3)
        for row, n in zip(rows, (1, 4, 8)):
            # Pinned ggml.h: GGML_OP_MUL_MAT=29, F32=0, Q8_0=8.
            self.assertEqual((row['op'], row['dtype']), (29, 0))
            self.assertEqual(row['shape'], [131072, n, 1, 1])
            self.assertEqual(row['params'], [0] * 16)
            self.assertEqual(len(row['sources']), 2)
            self.assertEqual(row['sources'][0],
                             [8, 4096, 131072, 1, 1, 34, 4352,
                              570425344, 570425344])
            self.assertEqual(row['sources'][1],
                             [0, 4096, n, 1, 1, 4, 16384, 16384 * n,
                              16384 * n])
        self.assertEqual([row['name'] for row in rows], list(P['CASE_NAMES']))
        # The pinned test-file parser assigns only the final field to the
        # operation/test name; it has no source-weight name field.
        self.assertNotIn('output.weight', cases()[0]['sources'])

    def test_native_body_shell_and_requested_public_commands(self):
        subprocess.run(['bash', '-n'], input=P['NATIVE_BODY'], text=True, check=True)
        body = P['NATIVE_BODY']
        self.assertEqual(P['CELLS'], ('VSPF-W3a',))
        self.assertIn('mkdir "$root/capture"', body)
        self.assertIn('test-backend-ops" test -b SYCL0 -o MUL_MAT --test-file /q8-head-screen/native-perf-cases.txt -j 1', body)
        self.assertIn('LD_DEBUG=libs SYCL_UR_TRACE=2', body)
        self.assertIn('GGML_SYCL_DEBUG=1', body)
        self.assertIn('for index in 0 1 2; do', body)
        self.assertIn('perf -b SYCL0 -o MUL_MAT --test-file /q8-head-screen/native-perf-cases.txt', body)
        self.assertIn('env -u LLAMA_LILICORR_PROFILE -u SYCL_UR_TRACE -u SYCL_PI_TRACE -u LD_DEBUG -u GGML_SYCL_DEBUG', body)
        self.assertNotIn('unitrace', body.lower())
        self.assertNotIn('enable_profiling', body.lower())
        self.assertNotIn('SYCL_UR_USE_LEVEL_ZERO_V2', body)
        self.assertNotIn('-ffast-math', body)
        self.assertNotIn('/baseline', body)

    def diagnostic_sample(self):
        return ('loader: calling init: /src/build-sycl/bin/libggml-sycl.so.0\n'
                'urEnqueueKernelLaunch hKernel=0xabc\n' +
                ''.join('  MUL_MAT(name=' + name + ',type=f32,ne=[131072,1,1,1]):  OK\n'
                        for name in P['CASE_NAMES']) +
                '  3/3 tests passed\nBackend SYCL0: OK\n')
    def perf_sample(self, value='123.45', runs='1024'):
        return ''.join(
            '  MUL_MAT(name=' + name + ',type=f32):    ' + runs +
            ' runs - ' + value + ' us/run - bandwidth\n'
            for name in P['CASE_NAMES'])

    def check_perf(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'perf.log'
            path.write_text(text)
            output = io.StringIO()
            with patch.object(sys, 'argv', ['inline-check', str(path)]), contextlib.redirect_stdout(output):
                exec(compile(P['PERF_CHECKER'], 'inline-native-perf-check', 'exec'), {})
            return json.loads(output.getvalue())

    def check_diagnostic(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diagnostic.log'
            path.write_text(text)
            output = io.StringIO()
            with patch.object(sys, 'argv', ['inline-check', str(path), '/src/build-sycl/bin']), contextlib.redirect_stdout(output):
                exec(compile(P['DIAGNOSTIC_CHECKER'], 'inline-diagnostic-check', 'exec'), {})
            return json.loads(output.getvalue())

    def test_diagnostic_checker_requires_three_cpu_comparisons_exact_dll_and_no_profile(self):
        result = self.check_diagnostic(self.diagnostic_sample())
        self.assertEqual(result['operation_names'], list(P['CASE_NAMES']))
        self.assertEqual(result['ur_launches'], 1)
        self.assertEqual(result['kernel_handles'], ['0xabc'])
        for bad in ('', self.diagnostic_sample().replace('3/3 tests passed', '2/3 tests passed'),
                    self.diagnostic_sample().replace('libggml-sycl.so.0', 'libggml-sycl-other.so'),
                    self.diagnostic_sample() + 'UR_QUEUE_FLAG_PROFILING_ENABLE\n',
                    self.diagnostic_sample().replace('q8-head-n4', 'q8-head-missing')):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                self.check_diagnostic(bad)

    def test_perf_checker_accepts_three_positive_rows_and_rejects_bad_rows(self):
        rows = self.check_perf(self.perf_sample())
        self.assertEqual(len(rows), 3)
        self.assertEqual([row['name'] for row in rows], list(P['CASE_NAMES']))
        self.assertEqual([row['operator_graph_wall_us'] for row in rows], [123.45] * 3)
        for bad in ('', self.perf_sample().splitlines()[0],
                    self.perf_sample() + self.perf_sample(),
                    self.perf_sample().replace('q8-head-n4', 'q8-head-n1'),
                    self.perf_sample() + 'not supported\n',
                    self.perf_sample() + 'skipping large tensors\n'):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                self.check_perf(bad)
        for value in ('nan', 'inf', '-1', '0'):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                self.check_perf(self.perf_sample(value=value))
        with self.assertRaises(AssertionError):
            self.check_perf(self.perf_sample(runs='0'))

    def test_argv_delta_adds_one_package_mount_and_no_second_library(self):
        original = ['docker', 'create', '-v', '/candidate:/src:ro',
                    'image@sha256:frozen', 'bash', '-lc', 'original command']
        result = P['derive'](original)
        self.assertEqual(result[-3:-1], ['bash', '-lc'])
        self.assertEqual(result[-1], P['NATIVE_BODY'])
        self.assertEqual(result.count(P['PACKAGE_BIND']), 1)
        self.assertNotIn('/baseline', '\n'.join(result))
        reverse = list(result)
        package_index = reverse.index(P['PACKAGE_BIND'])
        del reverse[package_index - 1:package_index + 1]
        reverse[-1] = original[-1]
        self.assertEqual(reverse, original)

    def package_record(self):
        return dict(
            HostConfig=dict(Binds=['other:/keep:ro', P['PACKAGE_BIND']], ShmSize=123),
            Mounts=[dict(Destination='/keep'), copy.deepcopy(P['PACKAGE_MOUNT'])],
            Config=dict(Cmd=['bash', '-lc', P['NATIVE_BODY']]))

    def test_package_projection_preserves_inherited_container_guard_inputs(self):
        record = self.package_record()
        before = copy.deepcopy(record)
        result = P['without_package_mount'](record)
        self.assertEqual(record, before)
        self.assertEqual(result['HostConfig'], dict(Binds=['other:/keep:ro'], ShmSize=123))
        self.assertEqual(result['Mounts'], [dict(Destination='/keep')])
        self.assertEqual(result['Config'], record['Config'])
        for key, value in (('Source', '/wrong'), ('RW', True), ('Mode', 'rw'),
                           ('Propagation', 'shared'), ('Type', 'volume')):
            bad = self.package_record()
            bad['Mounts'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                P['without_package_mount'](bad)
        for binds in ([], [P['PACKAGE_BIND'], P['PACKAGE_BIND']]):
            bad = self.package_record()
            bad['HostConfig']['Binds'] = binds
            with self.assertRaises(AssertionError):
                P['without_package_mount'](bad)

    def test_lifecycle_is_inherited_and_only_one_w3_cell_is_declared(self):
        self.assertEqual(P['QUALIFIED_CASE'], HERE.parent / '20261020-lilicorr-q8-target-batch4')
        self.assertEqual(P['BASE_QUALIFIER_BIND'], P['R']['QUALIFIER_BIND'])
        self.assertEqual(P['OWN_FILES'], ('run-native-perf.py', 'test-native-perf.py',
                                          'native-perf-cases.txt', 'NATIVE_PERF_PLAN.md'))
        text = (HERE / 'run-native-perf.py').read_text()
        for required in ("runtime, _analysis, c = R['bind']()",
                         "runtime['_BOOT_K']['capture']", 'class NativePerf(runtime[\'Runtime\'])',
                         'super().preflight()', 'finally:', '--lead-verify-restored',
                         "self._stop_original_for_case()", "self.ownership(None)"):
            self.assertIn(required, text)
        self.assertNotIn('BASE_BIND', text)
        self.assertNotIn('baseline mount', text)


if __name__ == '__main__':
    print('CPU_CONTRACT_TESTS_ONLY_NO_GPU_VALIDATION')
    unittest.main(verbosity=2)
