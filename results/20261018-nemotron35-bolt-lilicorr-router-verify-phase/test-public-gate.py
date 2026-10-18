#!/usr/bin/env python3
"""REMOTE CPU fixtures only; all changed startup/container inputs are SYNTHETIC.

Never creates a container, executes a native program, accesses DRM, or sends HTTP.
Recorded legacy HTTP/PID inputs are read-only; writable fixtures stay in scratch.
"""
import ast
import copy
import os
import re
import runpy
import shutil
import signal
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ['BOLT_ROUTER_VERIFY_PHASE_CPU_SCRATCH']).resolve()
assert socket.gethostname() == 'inference-host'
assert SCRATCH.is_relative_to(HERE / 'worker-scratch')


def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise AssertionError('CPU fixtures prohibit process/native/API execution: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        assert not os.fsdecode(args[0]).startswith(('/proc/', '/sys/', '/dev/dri')), args[0]


sys.addaudithook(audit)
P = runpy.run_path(str(HERE / 'public-gate.py'))
C, K, N = P['C'], P['K'], P['N']
H = C['helpers']()
FIXTURE = runpy.run_path(str(HERE / 'test-trace.py'))['fixture']
JOBS = P['jobs'](H)


def recorded(role, job):
    return P['read_result'](H, C['QUALIFY_DIR'] / role, job)


def synthetic_container(cell):
    role = P['ROLES'][cell]
    result = copy.deepcopy(C['load'](C['QUALIFY_DIR'] / role / 'container-before.json'))
    result['Name'] = '/' + P['name'](cell)
    result['Config']['Cmd'] = P['make_create'](cell, H)[-3:]
    old = str(N['PARENT']) + ':/src:ro'
    binds = result['HostConfig']['Binds']
    binds[binds.index(old)] = str(N['CANDIDATE']) + ':/src:ro'
    binds.append(str(HERE) + ':/router-verify-phase:ro')
    for mount in result['Mounts']:
        if mount['Destination'] == '/src':
            mount['Source'] = str(N['CANDIDATE'])
    result['Mounts'].append(dict(Type='bind', Source=str(HERE), Destination='/router-verify-phase', Mode='ro', RW=False))


def synthetic_startup(cell):
    directory = C['QUALIFY_DIR'] / P['ROLES'][cell]
    log = ''.join((directory / ('native-and-startup.' + stream + '.log')).read_text()
                  for stream in ('stdout', 'stderr'))
    p = H['off']['P']
    left, rest = log.split(p['ATTENTION_VIEW_NATIVE_MARKER'], 1)
    _, right = rest.split(p['ROUTER_NATIVE_MARKER'], 1)
    # ONLY an in-memory CPU fixture. Never write this as runtime evidence.
    synthetic = left + p['ATTENTION_VIEW_NATIVE_MARKER'] + '\n' + FIXTURE('router_single2') + p['ROUTER_NATIVE_MARKER'] + right
    return synthetic.replace(p['MARKER'], P['MARKER'] + '\n' + p['MARKER'])


class PublicGateFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=SCRATCH))
        for job in JOBS:
            for suffix in P['HTTP_SUFFIXES']:
                source = C['artifact'](C['QUALIFY_DIR'] / 'A' / job['stem'], suffix)
                target = C['artifact'](self.tmp / 'RVP-A' / job['stem'], suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

    def runtime(self, role='W3'):
        def run_job(cell, job, expected_for_compare=True):
            self.assertFalse(expected_for_compare)
            return recorded(role, job)
        return SimpleNamespace(h=H, off=H['off'], out=self.tmp, run_job=Mock(side_effect=run_job))

    def test_pinned_identity_and_inherited_lifecycle(self):
        for filename in P['FILES']:
            if filename.endswith('.py'):
                ast.parse((HERE / filename).read_text(), filename=filename)
        for method in ('restore', '_stop_original_for_case', 'save_logs', 'ownership', 'telemetry_mapping',
                       'public', 'run_job', '_exchange', 'observe'):
            self.assertIs(getattr(P['Runtime'], method), getattr(K['Runtime'], method))
        self.assertIs(C['FidelityFailure'], P['Runtime'].run_job.__globals__['C']['FidelityFailure'])
        self.assertFalse(set(P['CELLS']) & set(C['LABELS']))
        self.assertEqual('router-verify-phase-qualification-r1', P['MODE'])

    def test_exact_launch_delta_old_validator_replaced_and_native_isolated(self):
        for cell, role in P['ROLES'].items():
            base, argv = C['make_create'](role, H), P['make_create'](cell, H)
            self.assertEqual(C['server_argv'](base), C['server_argv'](argv))
            self.assertEqual(P['name'](cell), argv[argv.index('--name') + 1])
            self.assertNotIn(str(N['PARENT']) + ':/src:ro', argv)
            self.assertEqual(1, argv.count(str(N['CANDIDATE']) + ':/src:ro'))
            self.assertEqual(['-v', str(HERE) + ':/router-verify-phase:ro'], argv[-6:-4])
            self.assertEqual(len(base) + 2, len(argv))
            diff = P['positional_diff'](base, argv)
            self.assertEqual(len(argv) - 1, diff[-1]['index'])
            self.assertIsNone(diff[-1]['base'])  # Includes inserted trailing positions, no truncated zip.
            command = argv[-1]
            self.assertNotIn("['check_router_native_output']", command)
            self.assertEqual(1, command.count('--tag router_single2 /tmp/router-single2-native.log'))
            self.assertIn("-p '^router_single2=1,' 2>&1 | tee /tmp/router-single2-native.log", command)
            self.assertIn("grep -Fqx '  30/30 tests passed'", command)
            self.assertIn("cp /tmp/router-single2-native.log /tmp/originalrouter-single2-native.log;", command)
            self.assertIn("-p '^router_columns8=1,(?!router_single2=1,weight=blk\\.48\\.ffn_gate_inp\\.weight,layout=0,type_a=f32,type_b=f32,m=496,n=9,)' > /tmp/router-verify-phase-native.log 2>&1", command)
            self.assertIn("grep -Fqx '  61/61 tests passed'", command)
            self.assertEqual(1, command.count('--tag router_columns8_r3 /tmp/router-verify-phase-native.log'))
            self.assertIn("-p '^router_prefill=1,' > /tmp/router-prefill-native.log 2>&1", command)
            self.assertIn("grep -Fqx '  2/2 tests passed'", command)
            self.assertEqual(1, command.count('--tag router_prefill /tmp/router-prefill-native.log'))
            self.assertEqual(1, command.count(P['NEW_NATIVE']))
            boundary = H['off']['OFF_BOUNDARY']
            self.assertIn(P['NEW_NATIVE'] + boundary, command)
            self.assertTrue(command.startswith(H['off']['NATIVE_ON']))
            C['diagnostics_off_server'](command)
            for gate in ('IQ2', 'Q8', 'ATTENTION', 'ATTENTION_VIEW'):
                self.assertIn(H['off']['P'][gate + '_NATIVE_COMMAND'], command)
            self.assertEqual(base, C['make_create'](role, H))  # No mutated helper globals/argv.

    def test_actual_pid_records_pass_without_rewriting_and_mounts_fail_closed(self):
        for cell, role in P['ROLES'].items():
            record = C['load'](C['QUALIFY_DIR'] / role / 'pid-before.json')
            untouched = copy.deepcopy(record)
            C['check_pid'](record, role, H)
            self.assertEqual(record['argv'], C['server_argv'](P['make_create'](cell, H)))
            self.assertEqual(untouched, record)
            bad = copy.deepcopy(record)
            bad['diagnostics'].append(['BOLT_VERIFY_CAPTURE_DIR', '0'])
            with self.assertRaises(AssertionError):
                C['check_pid'](bad, role, H)
            container = synthetic_container(cell)
            P['check_container'](container, cell, H)
            for field, value in (('Source', str(N['PARENT'])), ('RW', True), ('Mode', 'rw')):
                bad = copy.deepcopy(container)
                next(m for m in bad['Mounts'] if m['Destination'] == '/src')[field] = value
                with self.assertRaises(AssertionError):
                    P['check_container'](bad, cell, H)
            bad = copy.deepcopy(container)
            bad['Mounts'].append(copy.deepcopy(bad['Mounts'][-1]))
            with self.assertRaises(AssertionError):
                P['check_container'](bad, cell, H)

    def test_native_startup_gates_and_prefill_negative_controls(self):
        for cell in P['CELLS']:
            log = synthetic_startup(cell)
            result = P['check_startup'](log, cell, H)
            expected_old = P['A']['check_router_native_output'](FIXTURE('router_single2'), 'router_single2')
            self.assertEqual(expected_old['route_records'], result['router_single2']['route_records'])
            for bad in (log.replace(P['MARKER'], ''), log + P['MARKER'],
                        log + 'router_columns8=1,', log + 'router_prefill=1,',
                        log + 'BOLT_ROUTE route=MODEL_LEAK',
                        log.replace('GGML_SYCL_DEBUG: 0', 'GGML_SYCL_DEBUG: 1')):
                with self.assertRaises(AssertionError):
                    P['check_startup'](bad, cell, H)
            p = H['off']['P']
            too_late = log.replace(P['MARKER'], '').replace(p['MARKER'], p['MARKER'] + '\n' + P['MARKER'])
            with self.assertRaises(AssertionError):
                P['check_startup'](too_late, cell, H)
        a = P['A']
        r3 = a['check_router_native_output'](FIXTURE('router_columns8_r3'), 'router_columns8_r3')
        self.assertEqual((61, 402, 35), (r3['cases'], r3['route_records'], r3['split_cases']))
        old = a['check_router_native_output'](FIXTURE('router_single2'), 'router_single2')
        self.assertEqual(old['route_records'], a['check_router_native_output'](FIXTURE('router_single2'), 'router_single2')['route_records'])
        prefill = a['check_router_native_output'](FIXTURE('router_prefill'), 'router_prefill')
        self.assertEqual((2, 4, 0), (prefill['cases'], prefill['route_records'], prefill['split_cases']))
        key = ('blk.1.ffn_gate_inp.weight', 240, 0, 3)
        unsplit_n3 = FIXTURE('router_single2', lambda k, rows:
                             runpy.run_path(str(HERE / 'test-trace.py'))['rows_for'](k, 'router_single2', False) if k == key else rows)
        a['check_router_native_output'](unsplit_n3, 'router_single2')
        split_n3 = FIXTURE('router_single2', lambda k, rows:
                           runpy.run_path(str(HERE / 'test-trace.py'))['rows_for'](k, 'router_single2', True) if k == key else rows)
        with self.assertRaises(AssertionError):
            a['check_router_native_output'](split_n3, 'router_single2')
        with self.assertRaises(AssertionError):
            a['check_router_native_output'](FIXTURE('router_columns8_r3').replace('61/61', '60/61'),
                                            'router_columns8_r3')
        bad_prefill = FIXTURE('router_prefill', lambda k, rows:
                              runpy.run_path(str(HERE / 'test-trace.py'))['rows_for'](k, 'router_prefill', True))
        with self.assertRaises(AssertionError):
            a['check_router_native_output'](bad_prefill, 'router_prefill')

    def test_names_only_new_cells_and_all_25_unchanged_requests(self):
        old = [C['ORIGINAL']] + [C['PREFIX'] + c.lower() for c in C['QUALIFY_ORDER'] + C['TIMING_ORDER']]
        self.assertEqual([P['name'](c) for c in P['CELLS']], P['check_names'](old)['planned_names'])
        for cell in P['CELLS']:
            with self.assertRaises(AssertionError):
                P['check_names'](old + [P['name'](cell)])
        self.assertEqual(JOBS, C['qualification_jobs'](H))
        self.assertEqual(25, len(JOBS))
        for job in JOBS[-4:]:
            self.assertEqual(128, job['payload']['max_tokens'])
            self.assertTrue(job['payload']['ignore_eos'])
            self.assertFalse(job['payload']['cache_prompt'])
            self.assertEqual(42, job['payload']['seed'])
            self.assertEqual(55 if job['stem'].startswith('short') else 8270,
                             len(H['off']['reference_prompt']('A0', job)))

    def test_build_success_full_candidate_and_legacy_proofs_frozen(self):
        frozen = P['freeze'](H)
        self.assertEqual(C['source_binary_hashes'](H), frozen['source_binaries'])
        self.assertEqual(N['source_hashes'](N['CANDIDATE']), frozen['candidate']['source'])
        self.assertEqual(N['binaries'](N['CANDIDATE']), frozen['candidate']['binaries'])
        for filename in P['FILES'] + P['BUILD_FILES']:
            self.assertIn(str(HERE / filename), frozen['hashes'])
        for role in P['ROLES'].values():
            self.assertIn(str(C['QUALIFY_DIR'] / role / 'pid-before.json'), frozen['references'])
            for job in JOBS:
                self.assertIn(str(C['artifact'](C['QUALIFY_DIR'] / role / job['stem'], '.http-body')), frozen['references'])
        with patch.object(Path, 'read_text', return_value='1'), self.assertRaises(AssertionError):
            P['candidate_proof']()  # Failed build cannot qualify, before touching sources/binaries.

    def test_valid_mismatches_keep_25_and_diagnostic_rejection_behavior(self):
        for cell in ('RVP-W3', 'RVP-W7'):
            rt = self.runtime(P['ROLES'][cell])
            self.assertEqual([], P['collect'](rt, cell))
            report = C['load'](self.tmp / cell / 'qualification.json')
            self.assertEqual(25, rt.run_job.call_count)
            self.assertFalse(report['passed'])
            self.assertEqual([], report['errors'])
            self.assertEqual(21, report['behavior']['natural_stop_cases'])
            self.assertEqual(2, report['behavior']['fixed_length_cases'])
            self.assertEqual(2, report['behavior']['diagnostics_exercised'])
            self.assertGreater(report['behavior']['rejected_draft_observed'], 0)
            self.assertEqual([], report['behavior']['not_exercised'])
            for row in report['jobs']:
                if row['kind'] == 'diagnostic':
                    self.assertEqual(116 if row['stem'].startswith('short') else 14,
                        row['comparisons']['frozen_a']['id_comparison']['first_difference_ordinal'])

    def test_baseline_failure_and_transport_abort_preserve_partial_report(self):
        rt = self.runtime('A')
        original = rt.run_job.side_effect
        def mismatch(*args, **kwargs):
            result = original(*args, **kwargs)
            result['actual']['text'] += ' CPU fixture mismatch'
            return result
        rt.run_job.side_effect = mismatch
        with self.assertRaises(C['InfraFailure']):
            P['collect'](rt, 'RVP-A')
        report = C['load'](self.tmp / 'RVP-A/qualification.json')
        self.assertEqual(1, rt.run_job.call_count)
        self.assertEqual(1, len(report['jobs']))
        self.assertTrue(report['fidelity_failures'])
        rt = self.runtime()
        rt.run_job.side_effect = ConnectionError('CPU transport fixture')
        with self.assertRaises(C['InfraFailure']):
            P['collect'](rt, 'RVP-W3')
        self.assertEqual(1, rt.run_job.call_count)
        self.assertTrue(C['load'](self.tmp / 'RVP-W3/qualification.json')['errors'])

    def test_fully_passing_report_requires_real_rejected_draft(self):
        for cell, rejects, passed in (('RVP-A', False, True), ('RVP-W3', False, False), ('RVP-W7', True, True)):
            report = P['new_report'](cell)
            for job in JOBS:
                result = recorded('A', job)
                expected = copy.deepcopy(result)
                if result['kind'] == 'diagnostic':
                    result['actual']['proposal_acceptance_counts'] = dict(draft_n=8 if rejects else 0, draft_n_accepted=0)
                    result['speculation'] = C['check_speculation_counts'](result['actual'])
                P['add_result'](report, job, result, expected, expected)
            self.assertEqual(passed, P['finish_report'](report)['passed'])

    def test_inherited_capture_restores_on_begin_failure_no_timings(self):
        events = []
        rt = SimpleNamespace(out=self.tmp, mode=P['MODE'], preflight=lambda: events.append('preflight'),
                             begin=Mock(side_effect=RuntimeError('CPU failed native fixture')),
                             end=Mock(), restore=lambda: events.append('restore') or [])
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            with self.assertRaises(RuntimeError):
                K['capture'](rt, P['CELLS'], Mock())
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
        self.assertEqual(['preflight', 'restore'], events)
        summary = C['load'](self.tmp / 'summary.json')
        self.assertTrue(summary['restored'])
        self.assertFalse(summary['timing_permitted'])
        self.assertEqual([], summary['measured_attempts'])
        self.assertEqual([], summary['completed'])

    def test_failed_start_preserves_native_via_stopped_container_without_blocking_restore(self):
        rt = P['Runtime'](H, self.tmp, P['MODE'])
        cell = 'RVP-W3'
        (self.tmp / cell).mkdir()
        rt.created = [cell]
        rt._stop_if_running = Mock()
        rt.logs = Mock(return_value=dict(stdout=b'CPU raw stdout', stderr=b'CPU raw stderr'))
        rt.command = Mock(side_effect=RuntimeError('CPU missing native file fixture'))
        rt.inspect = Mock(return_value={'CPU': 'preserved'})
        rt.verify_restored = Mock()
        rt.capacity = Mock()
        self.assertEqual([], rt.restore())
        rt._stop_if_running.assert_called_once_with(P['name'](cell))
        rt.verify_restored.assert_called_once()
        self.assertEqual(len(P['NATIVE_LOGS']), rt.command.call_count)
        for call in rt.command.call_args_list:
            self.assertEqual(('docker', 'cp'), call.args[:2])
            self.assertTrue(call.args[2].startswith(P['name'](cell) + ':/tmp/'))
        self.assertEqual(len(P['NATIVE_LOGS']), len(C['load'](self.tmp / cell / 'restored-native-copy.json')['errors']))
        self.assertEqual(b'CPU raw stdout', (self.tmp / cell / 'restored-final.stdout.log').read_bytes())

    def test_independent_restoration_failure_prevents_raw_analysis(self):
        C['save'](self.tmp / 'summary.json', dict(status='captured', restored=True, mode=P['MODE'],
            order=list(P['CELLS']), completed=list(P['CELLS']), timing_permitted=False, measured_attempts=[]))
        # Patch ONLY this adapter's test boundary, never inherited helper globals.
        with patch.dict(P['analyze'].__globals__, OUT=self.tmp), \
             patch.object(P['Runtime'], 'verify_restored', side_effect=C['InfraFailure']('CPU restore fixture')) as restore, \
             patch.dict(P['analyze'].__globals__, check_cell_evidence=Mock()) as globals_:
            with self.assertRaises(C['InfraFailure']):
                P['analyze'](H)
            restore.assert_called_once()
            globals_['check_cell_evidence'].assert_not_called()


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
