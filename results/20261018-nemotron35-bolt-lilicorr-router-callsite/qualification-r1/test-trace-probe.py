#!/usr/bin/env python3
"""Remote CPU fixtures for trace-probe.py; no service, process, network, or GPU work."""
import copy
import os
import runpy
import socket
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ['BOLT_ROUTER_CALLSITE_TRACE_CPU_SCRATCH']).resolve()
assert socket.gethostname() == 'inference-host'
assert SCRATCH.is_relative_to(HERE / 'worker-scratch')


def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise AssertionError('trace CPU fixtures prohibit process/native/API execution: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        assert not os.fsdecode(args[0]).startswith(('/proc/', '/sys/', '/dev/dri')), args[0]


sys.addaudithook(audit)
T = runpy.run_path(str(HERE / 'trace-probe.py'))
P, C, K = T['P'], T['C'], T['K']


class TraceProbeFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=SCRATCH))

    def test_create_argv_only_declared_trace_delta(self):
        for cell in T['CELLS']:
            role = T['role_for'](cell)
            base = T['base_create_argv'](role)
            untouched = copy.deepcopy(base)
            derived = T['make_create'](cell, self.tmp / 'trace')
            T['assert_trace_argv'](base, derived, cell, self.tmp / 'trace')
            self.assertEqual(untouched, T['base_create_argv'](role))
            self.assertEqual(T['trace_name'](cell), derived[derived.index('--name') + 1])
            self.assertEqual(1, derived.count('BOLT_VERIFY_CAPTURE_DIR=/capture'))
            self.assertEqual(1, derived.count('LLAMA_TRACE=1'))
            self.assertIn('export GGML_SYCL_DEBUG=0', derived[-1])
            self.assertIn('export BOLT_VERIFY_CAPTURE_DIR=/capture', derived[-1])
            self.assertIn('export LLAMA_TRACE=1', derived[-1])
            self.assertIn('BOLT_TRACE_TENSORS=0', derived)
            self.assertIn('ONEDNN_VERBOSE=0', derived)

    def test_fresh_container_identity_without_relaxing_launch_contract(self):
        for cell in T['CELLS']:
            record = copy.deepcopy(T['_base_container'](T['role_for'](cell)))
            capture = T['capture_path'](cell, self.tmp)
            record['Id'] = 'a' * 64
            record['Name'] = '/' + T['trace_name'](cell)
            record['Config']['Hostname'] = record['Id'][:12]
            record['Config']['Cmd'] = T['make_create'](cell, self.tmp)[-3:]
            record['Config']['Env'] += [key + '=' + value for key, value in T['TRACE_ENV'].items()]
            record['HostConfig']['Binds'].append(str(capture) + ':/capture')
            record['Mounts'].append(dict(Type='bind', Source=str(capture), Destination='/capture',
                                         Mode='', RW=True, Propagation='rprivate'))
            untouched = copy.deepcopy(record)
            self.assertTrue(T['check_trace_container'](record, cell, self.tmp))
            self.assertEqual(record, untouched)
            for value in (None, False):
                default_oom = copy.deepcopy(record)
                default_oom['HostConfig']['OomKillDisable'] = value
                unchanged = copy.deepcopy(default_oom)
                self.assertTrue(T['check_trace_container'](default_oom, cell, self.tmp))
                self.assertEqual(default_oom, unchanged)
            disabled_oom = copy.deepcopy(record)
            disabled_oom['HostConfig']['OomKillDisable'] = True
            with self.assertRaises(AssertionError):
                T['check_trace_container'](disabled_oom, cell, self.tmp)
            for key, value in (('Hostname', 'wrong-host'), ('Image', 'wrong-image')):
                bad = copy.deepcopy(record)
                bad['Config'][key] = value
                with self.assertRaises(AssertionError):
                    T['check_trace_container'](bad, cell, self.tmp)
            bad = copy.deepcopy(record)
            self.assertTrue(bad['HostConfig']['PortBindings'])
            bad['HostConfig']['PortBindings'] = {}
            with self.assertRaises(AssertionError):
                T['check_trace_container'](bad, cell, self.tmp)

    def test_trace_warmup_routes_preserve_native_and_model_startup_gates(self):
        h = C['helpers']()
        for cell in T['CELLS']:
            baseline_cell = T['BASE_CELLS'][T['role_for'](cell)]
            directory = T['ROOT'] / 'qualification-r1' / baseline_cell
            raw = ''.join((directory / ('native-and-startup.raw.' + stream + '.log')).read_text()
                          for stream in ('stdout', 'stderr'))
            route = next(line for line in raw.splitlines() if 'BOLT_ROUTE route=' in line)
            # Synthetic fixture only: append a retained native route as a warmup log.
            traced = raw + '\n' + route + '\n'
            with self.assertRaises(AssertionError):
                P['check_startup'](traced, baseline_cell, h)
            self.assertTrue(T['check_trace_startup'](traced, cell, h))
            for bad in (traced.replace(P['MARKER'], 'MISSING_NATIVE_GATE', 1),
                        traced + '\nBOLT_TENSOR_TRACE_ENABLED\n',
                        traced + '\nSYCL exception\n'):
                with self.assertRaises(AssertionError):
                    T['check_trace_startup'](bad, cell, h)

    def test_raw_pid_trace_expectations_are_strict_and_non_mutating(self):
        baseline = dict(
            pid=1,
            origin='/proc/1/environ',
            diagnostics=[['GGML_SYCL_DEBUG', '0'], ['BOLT_TRACE_TENSORS', '0']],
            argv=['/src/build-sycl/bin/llama-server', '--port', '18098'],
            exe='/src/build-sycl/bin/llama-server',
            scoped_tmp_rows=[],
        )
        trace = copy.deepcopy(baseline)
        trace['diagnostics'] += [['BOLT_VERIFY_CAPTURE_DIR', '/capture'], ['LLAMA_TRACE', '1']]
        untouched = copy.deepcopy(trace)
        self.assertTrue(T['check_trace_pid'](trace, baseline))
        self.assertEqual(untouched, trace)
        for mutation in (
            lambda row: row['argv'].__setitem__(1, '--wrong'),
            lambda row: row['exe'].__setitem__(0, 'x') if isinstance(row['exe'], list) else row.__setitem__('exe', 'x'),
            lambda row: row['diagnostics'].append(['SECRET', 'x']),
            lambda row: row['diagnostics'].__setitem__(2, ['LLAMA_TRACE', '0']),
        ):
            bad = copy.deepcopy(trace)
            mutation(bad)
            with self.assertRaises(AssertionError):
                T['check_trace_pid'](bad, baseline)

    def test_job_selection_preserves_exact_23_payloads_and_reproduction_contract(self):
        jobs = [dict(stem='quality/q%02d' % i, payload={'i': i}) for i in range(20)]
        jobs += [
            dict(stem='anchor', payload={'i': 20}),
            dict(stem='short/ordinary', payload={'i': 21}),
            dict(stem='short/diagnostic', payload={'i': 22}),
            dict(stem='long/ordinary', payload={'i': 23}),
            dict(stem='long/diagnostic', payload={'i': 24}),
        ]
        selected = T['select_trace_jobs'](jobs)
        self.assertEqual(23, len(selected))
        self.assertEqual([job['payload'] for job in selected], [job['payload'] for job in jobs[:23]])
        self.assertIs(selected[0], jobs[0])
        with self.assertRaises(AssertionError):
            T['select_trace_jobs'](jobs[:-1])

        h = C['helpers']()
        diagnostic = T['trace_jobs'](h)[-1]
        expected_dir = T['ROOT'] / 'qualification-r1' / 'RCA-A'
        expected = P['read_result'](h, expected_dir, diagnostic)
        result = dict(kind='diagnostic', actual=copy.deepcopy(expected['actual']),
                      speculation=copy.deepcopy(expected['speculation']))
        _, comparison = T['compare_same_role'](
            h, self.tmp, expected_dir, diagnostic, result)
        self.assertTrue(comparison['passed'])
        result['actual']['ids'][0] += 1
        _, mismatch = T['compare_same_role'](
            h, self.tmp, expected_dir, diagnostic, result)
        self.assertFalse(mismatch['passed'])

    def test_failed_restore_is_fail_closed_and_callback_cleanup_runs(self):
        events = []
        failed_restore = Mock(side_effect=lambda: events.append('restore') or
                              [dict(action='restore', error='fixture')])
        runtime = type('RuntimeFixture', (), {
            'out': self.tmp,
            'mode': T['MODE'],
            'preflight': lambda self: events.append('preflight'),
            'begin': Mock(side_effect=lambda cell: events.append('begin-' + cell) or
                          (_ for _ in ()).throw(RuntimeError('fixture begin'))),
            'restore': failed_restore,
        })()
        with self.assertRaises(RuntimeError):
            K['capture'](runtime, T['CELLS'], Mock())
        self.assertEqual(['preflight', 'begin-RCT-A', 'restore'], events)
        summary = C['load'](self.tmp / 'summary.json')
        self.assertEqual('failed', summary['status'])
        self.assertFalse(summary['restored'])
        self.assertEqual([], summary['completed'])

        events = []
        runtime = type('RuntimeFixture', (), {
            'out': self.tmp / 'callback-failure',
            'mode': T['MODE'],
            'preflight': lambda self: events.append('preflight'),
            'begin': lambda self, cell: events.append('begin-' + cell),
            'end': lambda self, cell: events.append('end-' + cell),
            'restore': lambda self: events.append('restore') or [],
        })()
        with self.assertRaises(RuntimeError):
            K['capture'](runtime, ('RCT-A',),
                         lambda rt, cell: (_ for _ in ()).throw(RuntimeError('fixture callback')))
        self.assertEqual(['preflight', 'begin-RCT-A', 'end-RCT-A', 'restore'], events)
        summary = C['load'](runtime.out / 'summary.json')
        self.assertEqual('failed', summary['status'])
        self.assertTrue(summary['restored'])

    def test_docker_port_conflict_uses_inherited_failure_cleanup(self):
        events = []
        command = ['docker', 'start', T['trace_name']('RCT-A')]
        def begin(cell):
            events.append('created-' + cell)
            raise subprocess.CalledProcessError(1, command, stderr=b'address already in use')
        runtime = type('PortConflictFixture', (), {
            'out': self.tmp, 'mode': T['MODE'],
            'preflight': lambda self: None,
            'begin': staticmethod(begin),
            'restore': lambda self: events.extend(['stop-created', 'restore-original']) or [],
        })()
        with self.assertRaises(subprocess.CalledProcessError):
            K['capture'](runtime, ('RCT-A',), Mock())
        self.assertEqual(['created-RCT-A', 'stop-created', 'restore-original'], events)
        summary = C['load'](self.tmp / 'summary.json')
        self.assertEqual('failed', summary['status'])
        self.assertTrue(summary['restored'])
        self.assertEqual(command, C['load'](self.tmp / 'failed-command.json')['argv'])

    def test_independent_restore_failure_blocks_raw_analysis(self):
        C['save'](self.tmp / 'summary.json', dict(
            status='captured', restored=True, mode=T['MODE'],
            order=list(T['CELLS']), completed=list(T['CELLS']),
            timing_permitted=False, measured_attempts=[]))
        evidence = Mock()
        with patch.dict(T['analyze'].__globals__, OUT=self.tmp), \
                patch.object(T['Runtime'], 'verify_restored',
                             side_effect=C['InfraFailure']('fixture restore')), \
                patch.dict(T['analyze'].__globals__, check_cell_evidence=evidence):
            with self.assertRaises(C['InfraFailure']):
                T['analyze'](C['helpers']())
        evidence.assert_not_called()
        self.assertFalse((self.tmp / 'analysis.json').exists())


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
