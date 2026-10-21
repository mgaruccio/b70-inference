#!/usr/bin/env python3
"""CPU-only binding and collection-window tests; no remote or GPU execution."""
import copy
import json
import runpy
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-profile.py'))


class ProfileTests(unittest.TestCase):
    def test_defined_order_and_profile_only_contract(self):
        self.assertEqual(D['CELLS'], ('VSPF-A1', 'VSPF-W3a', 'VSPF-W3b', 'VSPF-A2'))
        self.assertEqual(D['CLEAN_CELLS'], frozenset(('VSPF-A1', 'VSPF-W3a')))
        self.assertEqual(D['TRACED_CELLS'], frozenset(('VSPF-W3b', 'VSPF-A2')))
        self.assertEqual(D['PAIRS'], (('VSPF-A1', 'VSPF-A2'), ('VSPF-W3a', 'VSPF-W3b')))
        self.assertEqual(D['TRACE_FLAGS'], ('-d', '-h', '-s', '-v', '--chrome-kernel-logging'))
        source = (HERE / 'run-profile.py').read_text()
        self.assertIn("self.mode = 'profiling'", source)
        self.assertIn("promotion_permitted", source)
        self.assertNotIn('--teardown-on-signal', source)

    def test_runpy_namespace_is_mutated_not_shadowed(self):
        qbind = D['Q8']['bind']
        self.assertIs(qbind.__globals__, D['Q8']['main'].__globals__)
        D['configure_q8_namespace']()
        self.assertIs(qbind.__globals__, D['Q8']['bind'].__globals__)
        self.assertEqual(qbind.__globals__['ROOT'], D['ROOT'])
        self.assertEqual(qbind.__globals__['OUT'], D['OUT'])
        self.assertEqual(qbind.__globals__['PREFIX'], D['PREFIX'])

    def test_native_and_server_transforms_preserve_shell_checks(self):
        body = (
            "set -eo pipefail; "
            "/src/build-sycl/bin/test-backend-ops test -b SYCL0 2>&1 | tee /tmp/native-a.log; "
            "grep -Fx 'passed' /tmp/native-a.log; "
            "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops perf -b SYCL0 "
            "> /tmp/native-b.log 2>&1; "
            "grep -F 'passed' /tmp/native-b.log; "
            "exec /src/build-sycl/bin/llama-server --model /models/model.gguf "
        )
        argv = ['docker', 'create', 'image', 'bash', '-lc', body]
        clean = D['wrap_argv'](argv, 'VSPF-W3a')
        self.assertEqual(clean, argv)
        with tempfile.TemporaryDirectory() as directory:
            wrap_globals = D['wrap_argv'].__globals__
            old_out = wrap_globals['OUT']
            wrap_globals['OUT'] = Path(directory)
            try:
                traced = D['wrap_argv'](argv, 'VSPF-W3b')
            finally:
                wrap_globals['OUT'] = old_out
        traced_body = traced[-1]
        self.assertEqual(traced_body.count('/tools/unitrace'), 3)
        self.assertEqual(traced_body.count(D['TRACE_FLAG_TEXT']), 3)
        self.assertEqual(traced_body.count('/src/build-sycl/bin/test-backend-ops'), 2)
        self.assertEqual(traced_body.count('--start-paused'), 1)
        self.assertIn('--result-dir /profile/native/native-01', traced_body)
        self.assertIn('--result-dir /profile/native/native-02', traced_body)
        self.assertIn('--result-dir /profile/server', traced_body)
        self.assertIn("tee /tmp/native-a.log", traced_body)
        self.assertIn("grep -Fx 'passed' /tmp/native-a.log", traced_body)
        self.assertIn("/tmp/native-b.log 2>&1", traced_body)
        self.assertEqual(traced[-4:-1], ['image', 'bash', '-lc'])
        self.assertTrue(any(':/profile:rw' in item for item in traced))
        self.assertTrue(any(':/tools:ro' in item for item in traced))

    def test_strict_declared_mounts_and_negative_drift(self):
        reference = {
            'HostConfig': {'Binds': ['/base:/base:ro']},
            'Mounts': [dict(Type='bind', Source='/base', Destination='/base',
                            Mode='ro', RW=False, Propagation='rprivate')],
        }
        actual = D['expected_reference'](reference, 'VSPF-W3b')
        before = copy.deepcopy(actual)
        self.assertTrue(D['check_declared_extra_mounts'](actual, reference, 'VSPF-W3b'))
        self.assertEqual(actual, before)
        bad = copy.deepcopy(actual)
        next(m for m in bad['Mounts'] if m['Destination'] == '/profile')['RW'] = False
        with self.assertRaises(AssertionError):
            D['check_declared_extra_mounts'](bad, reference, 'VSPF-W3b')
        self.assertEqual(actual, before)

    def test_resume_request_pause_and_failure_finally(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def command(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        contract = {'name': lambda cell: 'container-' + cell.lower()}
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / 'short' / 'ordinary'
            runtime = FakeRuntime()
            self.assertEqual(D['collect_request'](runtime, contract, 'VSPF-A2', stem,
                                                   lambda: 'ok'), 'ok')
            self.assertEqual([call[0][4] for call in runtime.calls], ['--resume', '--pause'])
            sidecar = json.loads(Path(str(stem) + '.trace-sidecar.json').read_text())
            span = sidecar['requests'][0]
            for key in ('before_resume', 'after_resume', 'before_request', 'after_request',
                        'before_pause', 'after_pause'):
                self.assertIn('realtime_ns', span[key])
                self.assertIn('monotonic_ns', span[key])
            self.assertTrue(span['request_succeeded'])

        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / 'long' / 'ordinary'
            runtime = FakeRuntime()
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                D['collect_request'](runtime, contract, 'VSPF-W3b', stem,
                                      lambda: (_ for _ in ()).throw(RuntimeError('request failed')))
            self.assertEqual([call[0][4] for call in runtime.calls], ['--resume', '--pause'])
            sidecar = json.loads(Path(str(stem) + '.trace-sidecar.json').read_text())
            self.assertFalse(sidecar['requests'][0]['request_succeeded'])
            self.assertIn('request failed', sidecar['requests'][0]['request_error'])

    def test_complete_trace_output_validation_rejects_missing_or_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in D['TRACE_FILENAMES']:
                path = root / name
                path.write_text('{"traceEvents":[{}]}\n' if name == 'chrome_trace.json' else 'sample\n')
            result = D['validate_trace_outputs'](root, 'fixture')
            self.assertTrue(result['complete'])
            (root / 'host_timing.txt').write_text('')
            with self.assertRaises(AssertionError):
                D['validate_trace_outputs'](root, 'empty')

    def test_pinned_hashes_and_no_application_or_queue_controls(self):
        self.assertEqual(D['TOOL_HASHES'], {
            'unitrace': '99ffdad3f3ae86aeee8e52f4b687a9d41e5e67491993952de37c112544a0ff30',
            'libunitrace_tool.so': '4bb8d93a64d9091867babd6e14d3b8ed7647b1f4c649802cc1e8570ad0aa2d04',
        })
        source = (HERE / 'run-profile.py').read_text()
        for forbidden in ('--teardown-on-signal', '--filter', '--metric', 'zeCommandQueue',
                          'SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS',
                          'UR_L0_USE_COPY_OFFLOAD'):
            self.assertNotIn(forbidden, source)
        self.assertIn('old_hashes(h)', source)
        self.assertIn('validated_backend=old_hashes(h)', source)
        self.assertIn('runtime_map[\'capture\'](rt)', source)

    def test_lead_script_is_bounded_and_always_verifies(self):
        source = (HERE / 'lead-run.sh').read_text()
        self.assertIn("mike@[fd7a:115c:a1e0::c933:4f38]", source)
        self.assertIn('HostKeyAlias', source)
        self.assertIn('BatchMode=yes', source)
        self.assertIn('ControlMaster=no', source)
        self.assertIn('ForwardAgent=no', source)
        self.assertIn('--lead-capture', source)
        self.assertIn('--lead-verify-restored', source)
        self.assertIn('|| verify_rc=$?', source)
        self.assertNotIn('--delete', source)
        self.assertEqual(D['OWN_FILES'], ('README.md', 'run-profile.py', 'test-profile.py',
                                          'lead-run.sh', '.gitignore'))


if __name__ == '__main__':
    unittest.main()
