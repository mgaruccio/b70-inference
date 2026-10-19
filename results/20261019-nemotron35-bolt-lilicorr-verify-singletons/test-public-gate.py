#!/usr/bin/env python3
"""REMOTE CPU fixtures only; all changed startup/container inputs are SYNTHETIC.

Never creates a container, executes a native program, accesses DRM, or sends HTTP.
Recorded legacy HTTP/PID inputs are read-only; writable fixtures stay in scratch.
"""
import ast
import copy
import os
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ['BOLT_VERIFY_SINGLETONS_CPU_SCRATCH']).resolve()
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
F = runpy.run_path(str(P['PHASE'] / 'test-trace.py'))
FIXTURE = F['fixture']
JOBS = P['jobs'](H)


def recorded(role, job):
    return P['read_result'](H, C['QUALIFY_DIR'] / role, job)


def synthetic_container(cell):
    role = P['ROLES'][cell]
    result = copy.deepcopy(C['load'](C['QUALIFY_DIR'] / role / 'container-before.json'))
    result['Id'] = 'a' * 64
    result['Name'] = '/' + P['name'](cell)
    result['Config']['Hostname'] = result['Id'][:12]
    result['Config']['Cmd'] = P['make_create'](cell, H)[-3:]
    old = str(N['PARENT']) + ':/src:ro'
    binds = result['HostConfig']['Binds']
    binds[binds.index(old)] = str(P['CANDIDATE']) + ':/src:ro'
    binds.append(str(P['PHASE']) + ':/router-verify-phase:ro')
    binds.append(str(P['ROOT']) + ':/verify-singletons:ro')
    for mount in result['Mounts']:
        if mount['Destination'] == '/src':
            mount['Source'] = str(P['CANDIDATE'])
    result['Mounts'].append(dict(Type='bind', Source=str(P['PHASE']), Destination='/router-verify-phase',
                                 Mode='ro', RW=False, Propagation='rprivate'))
    result['Mounts'].append(dict(Type='bind', Source=str(P['ROOT']), Destination='/verify-singletons',
                                 Mode='ro', RW=False, Propagation='rprivate'))
    return result


def synthetic_startup(cell):
    directory = C['QUALIFY_DIR'] / P['ROLES'][cell]
    log = ''.join((directory / ('native-and-startup.' + stream + '.log')).read_text()
                  for stream in ('stdout', 'stderr'))
    p = H['off']['P']
    left, rest = log.split(p['ATTENTION_VIEW_NATIVE_MARKER'], 1)
    _, right = rest.split(p['ROUTER_NATIVE_MARKER'], 1)
    # ONLY an in-memory CPU fixture. Never write this as runtime evidence.
    synthetic = left + p['ATTENTION_VIEW_NATIVE_MARKER'] + '\n' + FIXTURE('router_single2') + p['ROUTER_NATIVE_MARKER'] + right
    return synthetic.replace(p['MARKER'], P['MARKER'] + '\n' + P['SINGLETON_MARKER'] + '\n' + p['MARKER'])


class PublicGateFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=SCRATCH))
        for job in JOBS:
            for suffix in P['HTTP_SUFFIXES']:
                source = C['artifact'](C['QUALIFY_DIR'] / 'A' / job['stem'], suffix)
                target = C['artifact'](self.tmp / 'VSB-A' / job['stem'], suffix)
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
        self.assertIs(C['InfraFailure'], P['Runtime'].restore.__globals__['C']['InfraFailure'])
        self.assertFalse(set(P['CELLS']) & set(C['LABELS']))
        self.assertEqual('verify-singletons-qualification-r2', P['MODE'])
        self.assertNotEqual(P['PARENT'], N['CANDIDATE'])  # Helper namespace remains the inherited phase parser.
        self.assertEqual(P['PHASE'], N['ROOT'])

    def test_exact_launch_delta_old_validator_replaced_and_native_isolated(self):
        for cell, role in P['ROLES'].items():
            base, argv = C['make_create'](role, H), P['make_create'](cell, H)
            self.assertEqual(C['server_argv'](base), C['server_argv'](argv))
            self.assertEqual(P['name'](cell), argv[argv.index('--name') + 1])
            self.assertNotIn(str(N['PARENT']) + ':/src:ro', argv)
            self.assertEqual(1, argv.count(str(P['CANDIDATE']) + ':/src:ro'))
            self.assertEqual(['-v', str(P['PHASE']) + ':/router-verify-phase:ro',
                              '-v', str(P['ROOT']) + ':/verify-singletons:ro'], argv[-8:-4])
            self.assertEqual(len(base) + 4, len(argv))
            diff = P['positional_diff'](base, argv)
            self.assertEqual(len(argv) - 1, diff[-1]['index'])
            self.assertIsNone(diff[-1]['base'])
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
            self.assertIn(P['NEW_NATIVE'] + P['singleton_native_command']() + boundary, command)
            self.assertTrue(command.startswith(H['off']['NATIVE_ON']))
            C['diagnostics_off_server'](command)
            for gate in ('IQ2', 'Q8', 'ATTENTION', 'ATTENTION_VIEW'):
                self.assertIn(H['off']['P'][gate + '_NATIVE_COMMAND'], command)
            self.assertEqual(base, C['make_create'](role, H))

    def test_actual_pid_and_full_container_contract_without_rewriting(self):
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
            for value in (None, False):
                container['HostConfig']['OomKillDisable'] = value
                untouched = copy.deepcopy(container)
                P['check_container'](container, cell, H)
                self.assertEqual(container, untouched)
            reordered = copy.deepcopy(container)
            reordered['Config']['Env'].reverse()
            reordered['HostConfig']['Binds'].reverse()
            untouched = copy.deepcopy(reordered)
            P['check_container'](reordered, cell, H)
            self.assertEqual(reordered, untouched)
            original_env = container['Config']['Env']
            for values in (original_env + [original_env[0]], original_env[1:],
                           original_env + ['UNEXPECTED_ENV=1'],
                           [original_env[0].split('=', 1)[0] + '=wrong'] + original_env[1:]):
                bad = copy.deepcopy(container)
                bad['Config']['Env'] = values
                with self.assertRaises(AssertionError):
                    P['check_container'](bad, cell, H)
            original_binds = container['HostConfig']['Binds']
            for values in (original_binds + [original_binds[0]], original_binds[1:],
                           original_binds + ['/unexpected:/unexpected:ro']):
                bad = copy.deepcopy(container)
                bad['HostConfig']['Binds'] = values
                with self.assertRaises(AssertionError):
                    P['check_container'](bad, cell, H)
            for field, value in (('Source', str(P['PARENT'])), ('RW', True), ('Mode', 'rw')):
                bad = copy.deepcopy(container)
                next(m for m in bad['Mounts'] if m['Destination'] == '/src')[field] = value
                with self.assertRaises(AssertionError):
                    P['check_container'](bad, cell, H)
            for section, field, value in (('HostConfig', 'OomKillDisable', True),
                                          ('HostConfig', 'PortBindings', {}),
                                          ('Config', 'Hostname', 'wrong'), ('Config', 'Image', 'wrong')):
                bad = copy.deepcopy(container)
                bad[section][field] = value
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
                        log.replace(P['SINGLETON_MARKER'], ''), log + P['SINGLETON_MARKER'],
                        log + 'verify_singletons=1,',
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
        self.assertEqual((30, 70, 5), (old['cases'], old['route_records'], old['split_cases']))
        self.assertTrue(all(row['accum_prec'] == 0 for row in a['route_records'](FIXTURE('router_single2'))))
        prefill = a['check_router_native_output'](FIXTURE('router_prefill'), 'router_prefill')
        self.assertEqual((2, 4, 0), (prefill['cases'], prefill['route_records'], prefill['split_cases']))
        key = ('blk.1.ffn_gate_inp.weight', 240, 0, 3)
        unsplit_n3 = FIXTURE('router_single2', lambda k, rows:
                             F['rows_for'](k, 'router_single2', False) if k == key else rows)
        a['check_router_native_output'](unsplit_n3, 'router_single2')
        split_n3 = FIXTURE('router_single2', lambda k, rows:
                           F['rows_for'](k, 'router_single2', True) if k == key else rows)
        with self.assertRaises(AssertionError):
            a['check_router_native_output'](split_n3, 'router_single2')
        with self.assertRaises(AssertionError):
            a['check_router_native_output'](FIXTURE('router_columns8_r3').replace('61/61', '60/61'),
                                            'router_columns8_r3')
        bad_prefill = FIXTURE('router_prefill', lambda k, rows: F['rows_for'](k, 'router_prefill', True))
        with self.assertRaises(AssertionError):
            a['check_router_native_output'](bad_prefill, 'router_prefill')

    def test_names_only_new_cells_and_all_25_unchanged_requests(self):
        old = [C['ORIGINAL']] + [C['PREFIX'] + c.lower() for c in C['QUALIFY_ORDER'] + C['TIMING_ORDER']]
        old += [C['PREFIX'] + c for c in ('rvp-a', 'rvp-w3', 'rvp-w7')]
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

    def test_native_contract_and_raw_parsers(self):
        proof = P['candidate_proof']()
        self.assertEqual(proof['source'], N['source_hashes'](P['CANDIDATE']))
        self.assertEqual(proof['binaries'], N['binaries'](P['CANDIDATE']))
        fixture = runpy.run_path(str(HERE / 'test-native.py'))['synthetic']
        for kind, count in (('q8', 82), ('iq2', 34)):
            raw = fixture(kind)
            self.assertEqual(count, P['check_native_log'](kind, raw)['cases'])
            with self.assertRaises(AssertionError):
                P['check_native_log'](kind, raw + raw)
        command = P['singleton_native_command']()
        self.assertNotIn('/capture/', command)
        self.assertIn('> "/tmp/verify-singletons-$kind.log" 2>&1', command)
        self.assertEqual(1, command.count(P['SINGLETON_MARKER']))

    def test_valid_mismatches_keep_25_and_diagnostic_rejection_behavior(self):
        for cell in ('VSB-W3', 'VSB-W7'):
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
            P['collect'](rt, 'VSB-A')
        report = C['load'](self.tmp / 'VSB-A/qualification.json')
        self.assertEqual(1, rt.run_job.call_count)
        self.assertEqual(1, len(report['jobs']))
        self.assertTrue(report['fidelity_failures'])
        rt = self.runtime()
        rt.run_job.side_effect = ConnectionError('CPU transport fixture')
        with self.assertRaises(C['InfraFailure']):
            P['collect'](rt, 'VSB-W3')
        self.assertEqual(1, rt.run_job.call_count)
        self.assertTrue(C['load'](self.tmp / 'VSB-W3/qualification.json')['errors'])

    def test_fully_passing_report_requires_real_rejected_draft(self):
        for cell, rejects, passed in (('VSB-A', False, True), ('VSB-W3', False, False), ('VSB-W7', True, True)):
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

    def test_docker_atomic_port_failure_uses_real_begin_and_inherited_cleanup(self):
        rt = P['Runtime'](H, self.tmp, P['MODE'])
        cell = 'VSB-W3'
        rt.preflight = Mock()
        rt.capacity = Mock()
        rt.ownership = Mock()
        rt.prefixes = {'ORIGINAL': {}}
        rt.inspect = Mock(side_effect=lambda name: {'State': {'Running': False}}
                          if name == rt.original_name else synthetic_container(cell))
        def command(*argv, **kwargs):
            if argv[:2] == ('docker', 'start'):
                raise subprocess.CalledProcessError(1, list(argv), stderr=b'address already in use')
            return ''
        rt.command = Mock(side_effect=command)
        rt._stop_if_running = Mock()
        rt.logs = Mock(return_value=dict(stdout=b'', stderr=b''))
        rt.verify_restored = Mock()
        with self.assertRaises(subprocess.CalledProcessError):
            K['capture'](rt, (cell,), Mock())
        self.assertEqual([cell], rt.created)
        rt._stop_if_running.assert_called_once_with(P['name'](cell))
        rt.verify_restored.assert_called_once()
        summary = C['load'](self.tmp / 'summary.json')
        self.assertTrue(summary['restored'])
        self.assertEqual('failed', summary['status'])

    def test_failed_start_preserves_native_via_stopped_container_without_blocking_restore(self):
        rt = P['Runtime'](H, self.tmp, P['MODE'])
        cell = 'VSB-W3'
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
