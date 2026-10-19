#!/usr/bin/env python3
"""CPU-only fixtures for the bounded W3 timing adapter.

No Docker, native process, socket, API, /proc, /sys, or DRM access is allowed.
Remote qualification artifacts are exercised only by the lead-side run.
"""
import copy
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get('BOLT_SINGLETONS_W3_CPU_SCRATCH', HERE / 'worker-scratch')).resolve()
SCRATCH.mkdir(parents=True, exist_ok=True)
C = runpy.run_path(str(HERE / 'contracts.py'))
R = runpy.run_path(str(HERE / 'controller.py'))
A = runpy.run_path(str(HERE / 'analyze-timing.py'))


def _result(value=1.0, counters=True):
    timings = dict(predicted_ms=2000, predicted_per_second=64,
                   prompt_ms=500, prompt_per_second=110,
                   cache_n=0, predicted_n=128)
    if counters:
        timings.update(draft_n=69, draft_n_accepted=57)
    return dict(text='fixture output', finish_reason='length',
                usage=dict(prompt_tokens=55, completion_tokens=128, total_tokens=183),
                timings=timings, elapsed_s=value + 4, ttfc_s=value)


def _rows():
    rows = []
    ordinal = 0
    for cell in C['CELLS']:
        for point in C['POINTS']:
            for repeat in range(3):
                result = _result(value=1.0 + ordinal / 10)
                rows.append(dict(cell=cell, point=point, phase='measured', ordinal=repeat,
                                 passed=True, actual=C['result_view'](result),
                                 timings=result['timings'], metrics=C['timing_metrics'](result),
                                 equals_no_draft=True, equals_fresh_a=(cell.startswith('VSPF-A'))))
                ordinal += 1
    return rows


def _fake_public():
    def make_create(cell, _h):
        return ['docker', 'create', '--name', 'qualified-' + cell.lower(),
                'bash', '-lc', 'set -e; exec /src/build-sycl/bin/llama-server --host 0.0.0.0']

    def check_output(result, _job, expected=None, _prompt=None):
        if expected is not None:
            assert result['text'] == expected['text']
        return True

    return dict(make_create=make_create, C={'check_output': check_output})


class TimingContracts(unittest.TestCase):
    def test_exact_name_only_argv_delta_and_pid_shape(self):
        C['configure'](_fake_public())
        base = _fake_public()['make_create']('VSB-A', None)
        derived = C['make_create']('VSPF-A1', None)
        self.assertEqual([dict(index=3, base='qualified-vsb-a',
                               derived=C['name']('VSPF-A1'))], C['argv_diff'](base, derived))
        self.assertEqual(C['server_argv'](base), C['server_argv'](derived))
        self.assertEqual(C['name']('VSPF-A1'), C['PREFIX'] + 'vspf-a1')
        self.assertEqual('VSB-W3', C['base_cell']('VSPF-W3a'))

    def test_w7_is_rejected_and_abba_schedule_has_24_rows(self):
        with self.assertRaises(AssertionError):
            C['eligible_order']({7})
        self.assertEqual(C['CELLS'], C['eligible_order']({3}))
        schedule = C['schedule']()
        self.assertEqual(32, len(schedule))
        self.assertEqual(24, len(C['expected_measurements']()))
        self.assertEqual(['VSPF-A1', 'VSPF-W3a', 'VSPF-W3b', 'VSPF-A2'],
                         [row['cell'] for row in schedule[::8]])
        for point in C['POINTS']:
            for arm in ('A', 'W3'):
                rows = [row for row in C['expected_measurements']()
                        if row['point'] == point and C['ROLES'][row['cell']] == arm]
                self.assertEqual(6, len(rows))

    def test_missing_counters_are_unavailable_not_zero(self):
        result = _result(counters=False)
        metrics = C['timing_metrics'](result)
        self.assertIsNone(metrics['draft_n'])
        self.assertIsNone(metrics['draft_n_accepted'])
        result['timings']['draft_n'] = 0
        result['timings']['draft_n_accepted'] = 0
        metrics = C['timing_metrics'](result)
        self.assertEqual(0, metrics['draft_n'])
        self.assertEqual(0, metrics['draft_n_accepted'])
        result['timings']['draft_n_accepted'] = 1
        with self.assertRaises(AssertionError):
            C['timing_metrics'](result)

    def test_type7_and_six_per_arm_point_summaries(self):
        self.assertEqual(dict(n=6, median=3.5, q1=2.25, q3=4.75, iqr=2.5),
                         C['describe']([1, 2, 3, 4, 5, 6]))
        report = A['summarize'](_rows())
        self.assertEqual(24, report['measured_rows'])
        self.assertEqual(6, report['samples_per_arm_point'])
        self.assertEqual(6, report['per_arm']['short']['A']['elapsed_s']['summary']['n'])
        self.assertTrue(report['first_content_decode_proxy_is_not_decode_speed'])
        self.assertEqual(6, report['per_arm']['short']['W3']['elapsed_s']['summary']['n'])

    def test_source_and_reference_guards_are_case_local(self):
        self.assertIn('contracts.py', C['FILES'])
        hashes = C['adapter_code_hashes']()
        self.assertEqual(C['sha'](HERE / 'contracts.py'), hashes[str(HERE / 'contracts.py')])
        self.assertNotIn('VSPF-A1', C['ROLES'].keys() - set(C['CELLS']))
        self.assertNotIn('VSB-W7', C['BASE_CELLS'].values())
        self.assertEqual(C['planned_names'](), list(map(C['name'], C['CELLS'])))
        with self.assertRaises(AssertionError):
            C['check_name_collisions']([C['name']('VSPF-A1')])


    def test_inherited_functions_bind_the_adapter_contract_namespace(self):
        adapter_contracts = R['C']
        self.assertIs(R['_BOOT_K']['C'], adapter_contracts)
        functions = []
        for owner in R['Runtime'].__mro__[1:]:
            for name in ('end', 'restore', 'run_job', 'timed_request', 'save_logs'):
                function = owner.__dict__.get(name)
                if function is not None:
                    functions.append(function)
        for name in ('stream_chat', 'sampled_request', 'capture', '_save_exception'):
            function = R['_BOOT_K'].get(name)
            if function is not None:
                functions.append(function)
        self.assertTrue(functions)
        for function in functions:
            self.assertIs(function.__globals__['C'], adapter_contracts, function.__qualname__)

    def test_actual_inherited_restore_and_end_use_synthetic_adapter_names(self):
        adapter_contracts = R['C']
        out = Path(tempfile.mkdtemp(dir=SCRATCH))
        cell = 'VSPF-A1'
        directory = out / cell
        directory.mkdir()
        (directory / 'native-and-startup.stdout.log').write_bytes(
            b'raw stdout' + bytes((0,)))
        (directory / 'native-and-startup.stderr.log').write_bytes(
            b'raw stderr' + bytes((0xff,)))
        commands = []
        runtime = object.__new__(R['Runtime'])
        runtime.out = out
        runtime.h = {}
        runtime.hashes = {'source': 'same'}
        runtime.active = cell
        runtime.pid = Mock(return_value={'argv': ['synthetic']})
        runtime.observe = Mock()
        runtime.save_logs = Mock(return_value='server log')
        runtime.off = {'check_off_log': Mock()}
        runtime.ownership = Mock()
        runtime.command = lambda *argv, **kwargs: commands.append((argv, kwargs)) or ''
        runtime.inspect = Mock(side_effect=[
            {'State': {'Running': True}}, {'State': {'Running': False}}])
        with patch.dict(adapter_contracts, {
                'source_binary_hashes': lambda _h: {'source': 'same'},
                'log_duration_stats': lambda _text: {'available': True}}, clear=False):
            R['Runtime'].end(runtime, cell)
        expected_name = adapter_contracts['PREFIX'] + cell.lower()
        self.assertIn((('docker', 'stop', '-t', '90', expected_name), {}), commands)
        runtime.pid.assert_called_once_with(cell, 'after')
        runtime.observe.assert_called_once_with(cell + '-after', expected_name)

        commands.clear()
        created = 'VSPF-W3a'
        runtime = object.__new__(R['Runtime'])
        runtime.out = out
        runtime.h = {}
        runtime.created = [created]
        runtime.prefixes = {}
        runtime.original_was_running = False
        runtime.command = lambda *argv, **kwargs: commands.append((argv, kwargs)) or ''
        runtime.inspect = Mock(return_value={'State': {'Running': True}})
        runtime._preserve_restore_log = Mock()
        runtime.verify_restored = Mock(return_value={'passed': True})
        runtime.capacity = Mock(return_value={'fits': True})
        self.assertEqual([], R['Runtime'].restore(runtime))
        expected_name = adapter_contracts['PREFIX'] + created.lower()
        self.assertIn((('docker', 'stop', '-t', '90', expected_name), {}), commands)
        runtime._preserve_restore_log.assert_called_once_with(created, expected_name)
        restoration = adapter_contracts['load'](out / 'restoration.json')
        self.assertTrue(restoration['passed'])

    def test_raw_startup_bytes_are_saved_without_decode_loss(self):
        out = Path(tempfile.mkdtemp(dir=SCRATCH))
        stdout = bytes((0x80,)) + b'stdout' + bytes((10,))
        stderr = bytes((0xff,)) + b'stderr' + bytes((10,))
        raw = Mock(stdout=stdout, stderr=stderr)
        R['_save_raw_startup'](out, raw)
        self.assertEqual(stdout, (out / 'native-and-startup.raw.stdout.log').read_bytes())
        self.assertEqual(stderr, (out / 'native-and-startup.raw.stderr.log').read_bytes())

    def test_adapter_hashes_reject_missing_files_and_drift(self):
        hashes = C['adapter_code_hashes']()
        drifted = dict(hashes)
        first = next(iter(drifted))
        drifted[first] = '0' * 64
        with self.assertRaises(AssertionError):
            C['check_adapter_code'](drifted)
        missing_root = Path(tempfile.mkdtemp(dir=SCRATCH))
        with patch.dict(C['adapter_code_hashes'].__globals__,
                        {'PACKAGE_ROOT': missing_root}, clear=False):
            with self.assertRaises(AssertionError):
                C['adapter_code_hashes']()

    def test_raw_analysis_round_trip_rejects_mutated_events_and_result(self):
        out = Path(tempfile.mkdtemp(dir=SCRATCH))
        stem = out / 'VSPF-A1' / 'short' / 'measured-00'
        stem.parent.mkdir(parents=True)
        bodies = [
            {'choices': [{'delta': {'content': 'hello'}}]},
            {'choices': [{'delta': {'content': ' world'}}]},
        ]
        line_ending = bytes((10,))
        raw = b''.join(b'data: ' + json.dumps(body).encode() + line_ending
                       for body in bodies)
        raw += b'data: [DONE]' + line_ending
        events = [dict(elapsed_s=0.5, body=bodies[0]),
                  dict(elapsed_s=1.0, body=bodies[1])]
        transport = dict(status=200, headers=[], complete=True, bytes=len(raw),
                         elapsed_s=1.5, ttfc_s=0.5)
        A['C']['save'](A['_artifact'](stem, '.events.json'), events)
        A['C']['save'](A['_artifact'](stem, '.http.json'), transport)
        normalize = lambda _raw, _payload: dict(
            text='hello world', finish_reason='stop', usage={}, timings={})
        protocol = normalize(raw, {})
        self.assertNotIn('elapsed_s', protocol)
        self.assertNotIn('ttfc_s', protocol)
        result = A['_augment_timing'](stem, raw, protocol, transport)
        A['C']['save'](A['_artifact'](stem, '.result.json'), result)
        self.assertEqual(result, A['_compare_stored_result'](stem, result))

        events[1]['elapsed_s'] = 2.0
        A['C']['save'](A['_artifact'](stem, '.events.json'), events)
        with self.assertRaises(AssertionError):
            A['_event_timing'](stem, raw, transport)
        bad_result = dict(result, text='mutated')
        A['C']['save'](A['_artifact'](stem, '.result.json'), bad_result)
        with self.assertRaises(AssertionError):
            A['_compare_stored_result'](stem, result)

    def test_timing_rows_require_six_distinct_rows_per_cell(self):
        rows = [row for row in _rows() if row['cell'] == 'VSPF-A1']
        A['_validate_timing_rows'](rows, 'VSPF-A1')
        duplicate = copy.deepcopy(rows)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaises(AssertionError):
            A['_validate_timing_rows'](duplicate, 'VSPF-A1')
    def test_raw_timing_and_fidelity_rejection(self):
        result = _result()
        result['ttfc_s'] = result['elapsed_s']
        with self.assertRaises(AssertionError):
            C['timing_metrics'](result)
        result = _result()
        expected = copy.deepcopy(result)
        expected['text'] = 'different'
        C['configure'](_fake_public())
        job = dict(stem='short/ordinary', payload={})
        with self.assertRaises(AssertionError):
            C['check_output'](result, job, expected, list(range(55)))

    def test_inherited_capture_restores_on_error(self):
        # This uses K.capture with the adapter's C namespace but never creates
        # a Runtime, container, process, socket, or API connection.
        C['configure'](_fake_public())
        K = R['_BOOT_K']
        self.assertIsNotNone(K)
        out = Path(tempfile.mkdtemp(dir=SCRATCH))
        events = []

        class Fake:
            mode = 'timing'
            def __init__(self):
                self.out = out
            def preflight(self):
                events.append('preflight')
            def begin(self, _cell):
                events.append('begin')
                raise RuntimeError('fixture begin failure')
            def restore(self):
                events.append('restore')
                return []

        with self.assertRaises(RuntimeError):
            K['capture'](Fake(), C['CELLS'], lambda *_: [])
        self.assertEqual(['preflight', 'begin', 'restore'], events)
        summary = C['load'](out / 'summary.json')
        self.assertEqual('failed', summary['status'])
        self.assertTrue(summary['restored'])
        self.assertTrue(summary['timing_permitted'])
        self.assertEqual([], summary['measured_attempts'])


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
