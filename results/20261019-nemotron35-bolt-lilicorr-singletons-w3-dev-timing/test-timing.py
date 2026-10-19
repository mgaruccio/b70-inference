#!/usr/bin/env python3
"""CPU-only fixtures for the bounded W3 timing adapter.

No Docker, native process, socket, API, /proc, /sys, or DRM access is allowed.
Remote qualification artifacts are exercised only by the lead-side run.
"""
import copy
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

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
        K['C'] = C
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
