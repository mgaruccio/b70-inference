#!/usr/bin/env python3
"""CPU-only tests for the full-trace Q8 adapter.

No Docker, subprocess, socket, service, GPU, DRM, /proc, or /sys access is allowed.
The positive log/container fixtures are retained artifacts from the existing trace
qualification and clean Q8 baseline; mutations must fail closed.
"""
import copy
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_RESULTS = HERE.parent
Q8_ROOT = REPO_RESULTS / '20261019-lilicorr-q8-drafter'
QUALIFIED = REPO_RESULTS / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'
TRACE_FIXTURES = REPO_RESULTS / '20261019-nemotron35-bolt-lilicorr-verify-singletons/trace-short55-r1'
WIDTH1 = REPO_RESULTS / '20261017-nemotron35-bolt-lilicorr-width1'
D = runpy.run_path(str(HERE / 'run-performance.py'))


# This fixture audit is deliberately narrow: it makes accidental native/API use fail
# instead of relying on a reviewer noticing it in a CPU test run.
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise AssertionError('fulltrace CPU tests prohibit process/native/API execution: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        assert not path.startswith(('/proc/', '/sys/', '/dev/dri')), path


sys.addaudithook(audit)


def public_fixture():
    return {
        'OUT': QUALIFIED,
        'C': {},
        'MARKER': 'BOLT_VERIFY_ROUTER_VERIFY_PHASE_NATIVE_PASSED',
        'SINGLETON_MARKER': 'BOLT_VERIFY_SINGLETONS_NATIVE_PASSED',
        'A': {'check_router_native_output': assert_route},
        'make_create': lambda cell, _h: json.loads(
            (QUALIFIED / cell / 'create-argv.json').read_text()),
    }


def trace_helpers():
    markers = {
        'iq2': 'BOLT_VERIFY_IQ2_NATIVE_PASSED',
        'q8': 'BOLT_VERIFY_Q8_LATENT_NATIVE_PASSED',
        'attention': 'BOLT_VERIFY_Q8_ATTENTION_NATIVE_PASSED',
        'attention_view': 'BOLT_VERIFY_Q8_ATTENTION_VIEW_NATIVE_PASSED',
    }

    def gate(log, marker):
        assert log.count(marker) == 1, marker
        return True

    p = {
        'WIDTH1': WIDTH1,
        'MARKER': 'BOLT_VERIFY_SERVER_START',
        'SINGLETON_MARKER': 'BOLT_VERIFY_SINGLETONS_NATIVE_PASSED',
        'ATTENTION_VIEW_NATIVE_MARKER': 'BOLT_VERIFY_Q8_ATTENTION_VIEW_NATIVE_PASSED',
        'ROUTER_NATIVE_MARKER': 'BOLT_VERIFY_ROUTER_NATIVE_PASSED',
        'A': {
            'check_router_native_output': assert_route,
        },
    }
    for name, marker in markers.items():
        p['check_' + name + '_native_gate'] = lambda log, marker=marker: gate(log, marker)
    old = {
        'N_RS_SEQ_RE': __import__('re').compile(r'never-used-(\d+)'),
        'DFLASH_PARAMS_RE': __import__('re').compile(r'never-used'),
        'DFLASH_BLOCK_RE': __import__('re').compile(r'never-used'),
        'check_model_log': lambda _model, _label: True,
    }
    off = {
        'P': p,
        'OLD': old,
        'N': {'check_model_startup': lambda _model: {'width': None}},
        'check_off_log': lambda _log: True,
    }
    return {'off': off}


def assert_route(segment, _tag=None):
    assert 'BOLT_ROUTE route=' in segment
    return {'passed': True, 'fixture': 'router_single2'}


def raw_startup(cell):
    directory = TRACE_FIXTURES / ('VSTB-A' if D['role_for'](cell) == 'A' else 'VSTB-W3')
    return ''.join((directory / ('native-and-startup.raw.' + stream + '.log')).read_text()
                   for stream in ('stdout', 'stderr'))


def raw_server(cell):
    directory = TRACE_FIXTURES / ('VSTB-A' if D['role_for'](cell) == 'A' else 'VSTB-W3')
    return ''.join((directory / ('server.' + stream + '.log')).read_text(errors='replace')
                   for stream in ('stdout', 'stderr'))


class FullTraceFixtures(unittest.TestCase):
    def setUp(self):
        self.runtime, self.analysis, self.c = D['bind']()
        self.public = public_fixture()
        self.c['configure'](self.public)
        self.h = trace_helpers()
        self.tmp = Path(tempfile.mkdtemp(prefix='fulltrace-q8-'))

    def record(self, cell):
        reference = json.loads((D['_reference_cell'](cell) / 'container-before.json').read_text())
        record = copy.deepcopy(reference)
        record['Id'] = 'a' * 64
        record['Name'] = '/' + self.c['name'](cell)
        record['Config']['Hostname'] = record['Id'][:12]
        record['Config']['Cmd'] = D['make_create'](cell, None, self.tmp)[-3:]
        record['Config']['Env'] += [key + '=' + value for key, value in D['TRACE_ENV'].items()]
        capture = D['capture_path'](cell, self.tmp)
        record['HostConfig']['Binds'].append(str(capture) + ':/capture')
        record['Mounts'].append(dict(Type='bind', Source=str(capture), Destination='/capture',
                                     Mode='', RW=True, Propagation='rprivate'))
        return record

    def test_private_namespace_and_analysis_hook(self):
        self.assertIs(self.analysis['analyze'].__globals__['C'], self.c)
        self.assertIs(self.analysis['analyze'].__globals__['check_runtime_evidence'],
                      self.analysis['check_runtime_evidence'])
        untouched = runpy.run_path(str(Q8_ROOT / 'run-performance.py'))
        self.assertNotEqual(untouched['bind'].__globals__['PREFIX'], D['PREFIX'])
        self.assertEqual(self.c['PREFIX'], D['PREFIX'])
        self.assertEqual(self.c['OUT'], D['OUT'])
        self.assertEqual(self.c['ROOT'], D['ROOT'])

    def test_role_cell_mapping_and_q8_candidate_delta(self):
        self.assertEqual(
            {cell: D['role_for'](cell) for cell in D['CELLS']},
            {'VSPF-A1': 'A', 'VSPF-W3a': 'W3', 'VSPF-W3b': 'W3', 'VSPF-A2': 'A'})
        for cell in D['CELLS']:
            argv = D['make_create'](cell, None, self.tmp)
            self.assertEqual(self.c['name'](cell), argv[argv.index('--name') + 1])
            self.assertEqual(D['capture_path'](cell, self.tmp), self.tmp / cell / 'capture')
            if D['role_for'](cell) == 'A':
                self.assertFalse(any('/draft/' in value for value in argv))
            else:
                self.assertGreaterEqual(sum('/draft/' in value for value in argv), 1)
                self.assertTrue(any(value.endswith(':/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf:ro')
                                    for value in argv))

    def test_trace_argv_round_trip_and_mutations(self):
        cell = 'VSPF-W3a'
        base = D['base_create_argv'](cell)
        derived = D['make_create'](cell, None, self.tmp)
        self.assertTrue(D['assert_trace_argv'](base, derived, cell, self.tmp))
        self.assertEqual(self.c['server_argv'](base), self.c['server_argv'](derived))

        bad_env = copy.deepcopy(derived)
        index = bad_env.index('LLAMA_TRACE=1')
        bad_env[index] = 'LLAMA_TRACE=0'
        with self.assertRaises(AssertionError):
            D['assert_trace_argv'](base, bad_env, cell, self.tmp)

        bad_marker = copy.deepcopy(derived)
        bad_marker[-1] = bad_marker[-1].replace('BOLT_VERIFY_SERVER_START', 'WRONG_START', 1)
        with self.assertRaises(AssertionError):
            D['assert_trace_argv'](base, bad_marker, cell, self.tmp)

        bad_mount = copy.deepcopy(derived)
        capture = str(D['capture_path'](cell, self.tmp)) + ':/capture'
        bad_mount[bad_mount.index(capture)] = str(D['capture_path'](cell, self.tmp)) + ':/capture:ro'
        with self.assertRaises(AssertionError):
            D['assert_trace_argv'](base, bad_mount, cell, self.tmp)

    def test_trace_container_positive_and_capture_mutations(self):
        for cell in D['CELLS']:
            record = self.record(cell)
            untouched = copy.deepcopy(record)
            self.assertTrue(D['check_trace_container'](record, cell, self.tmp))
            self.assertEqual(untouched, record)

        bad_env = self.record('VSPF-A1')
        env_index = next(i for i, value in enumerate(bad_env['Config']['Env'])
                         if value == 'BOLT_VERIFY_CAPTURE_DIR=/capture')
        bad_env['Config']['Env'][env_index] = 'BOLT_VERIFY_CAPTURE_DIR=/tmp'
        with self.assertRaises(AssertionError):
            D['check_trace_container'](bad_env, 'VSPF-A1', self.tmp)

        bad_argv = self.record('VSPF-A1')
        bad_argv['Config']['Cmd'][-1] = 'wrong-server-command'
        with self.assertRaises(AssertionError):
            D['check_trace_container'](bad_argv, 'VSPF-A1', self.tmp)

        bad_mount = self.record('VSPF-W3a')
        bad_mount['Mounts'][-1]['RW'] = False
        with self.assertRaises(AssertionError):
            D['check_trace_container'](bad_mount, 'VSPF-W3a', self.tmp)

        bad_bind = self.record('VSPF-W3a')
        bad_bind['HostConfig']['Binds'][-1] = str(self.tmp / 'wrong') + ':/capture'
        with self.assertRaises(AssertionError):
            D['check_trace_container'](bad_bind, 'VSPF-W3a', self.tmp)

    def test_pid_positive_and_wrong_capture_env_or_argv_fail_closed(self):
        for cell in ('VSPF-A1', 'VSPF-W3a'):
            baseline = json.loads((D['_reference_cell'](cell) / 'pid-before.json').read_text())
            traced = copy.deepcopy(baseline)
            traced['diagnostics'] += list(D['TRACE_ENV'].items())
            untouched = copy.deepcopy(traced)
            self.assertTrue(D['check_trace_pid'](traced, baseline))
            self.assertEqual(untouched, traced)

            bad_env = copy.deepcopy(traced)
            bad_env['diagnostics'][-1] = ['LLAMA_TRACE', '0']
            with self.assertRaises(AssertionError):
                D['check_trace_pid'](bad_env, baseline)

            bad_argv = copy.deepcopy(traced)
            bad_argv['argv'][1] = '--wrong'
            with self.assertRaises(AssertionError):
                D['check_trace_pid'](bad_argv, baseline)

            bad_extra = copy.deepcopy(traced)
            bad_extra['diagnostics'].append(['UNDECLARED', '1'])
            with self.assertRaises(AssertionError):
                D['check_trace_pid'](bad_extra, baseline)

    def test_actual_trace_startup_and_native_marker_failures(self):
        for cell in ('VSPF-A1', 'VSPF-W3a'):
            raw = raw_startup(cell)
            result = D['check_trace_startup'](raw, cell, self.h)
            self.assertEqual('unaltered raw startup and actual trace PID1', result['evidence'])

            bad_native = raw.replace('BOLT_VERIFY_Q8_LATENT_NATIVE_PASSED', 'NATIVE_FAILURE', 1)
            with self.assertRaises(AssertionError):
                D['check_trace_startup'](bad_native, cell, self.h)

            bad_marker = raw.replace(self.h['off']['P']['MARKER'], 'MISSING_MARKER', 1)
            with self.assertRaises(AssertionError):
                D['check_trace_startup'](bad_marker, cell, self.h)

            bad_forbidden = raw + '\nSYCL exception\n'
            with self.assertRaises(AssertionError):
                D['check_trace_startup'](bad_forbidden, cell, self.h)

    def test_actual_trace_server_log_marker_and_native_failures(self):
        raw = raw_server('VSPF-W3a')
        self.assertTrue(D['check_trace_server_log'](raw))
        for mutation in (
            raw.replace('BOLT_VERIFY sample ', 'MISSING_SAMPLE '),
            raw.replace('BOLT_VERIFY_ROUTER_VERIFY_PHASE_NATIVE_PASSED', 'MISSING_MARKER', 1),
            raw + '\nBOLT_VERIFY error\n',
        ):
            with self.assertRaises(AssertionError):
                D['check_trace_server_log'](mutation)

    def test_capture_manifest_requires_exact_cpu_row_size(self):
        trace_capture = TRACE_FIXTURES / 'VSTB-W3/capture'
        manifest = D['_capture_manifest'](trace_capture, require=True)
        self.assertGreater(len(manifest), 0)
        self.assertEqual({D['CAPTURE_ROW_BYTES']}, {item['bytes'] for item in manifest})

        with tempfile.TemporaryDirectory(prefix='fulltrace-row-') as directory:
            path = Path(directory) / 'row-bad'
            path.write_bytes(b'\0' * 4)
            with self.assertRaises(AssertionError):
                D['_capture_manifest'](directory, require=True)


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
