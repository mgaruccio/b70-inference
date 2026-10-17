#!/usr/bin/env python3
"""Boundary/cycle accounting checks; run on inference-host, not the interactive host."""
from pathlib import Path
import runpy
import unittest

A = runpy.run_path(str(Path(__file__).with_name('analyze-decode-profile.py')))


def fixture():
    gpu = []
    for i in range(128):
        t = i * 100
        gpu.extend([
            {'name': 'ggml_sycl_ssm_scan_prefill(test)' if i == 0 else 'decode_kernel', 'ts': t, 'dur': 10},
            {'name': A['OUTPUT'] + '(test)[SIMD_ANY {65536; 1; 1} {8; 1; 1}]', 'ts': t + 20, 'dur': 20},
            {'name': A['COPY'], 'ts': t + 40, 'dur': 5},
        ])
    return gpu


class AnalysisTests(unittest.TestCase):
    def test_decode_windows(self):
        gpu = fixture()
        indices = A['boundaries'](gpu)
        for first, count in ((1, 127), (16, 112)):
            result = A['window'](gpu, [], indices, first)
            self.assertEqual(result['steps'], count)
            self.assertEqual(result['device_events'], count * 3)
            self.assertAlmostEqual(result['window_ms'], count * .1)
            self.assertAlmostEqual(result['summed_device_ms'], count * .035)
            self.assertAlmostEqual(result['uncovered_ms'], count * .065)
            self.assertTrue(all(x['calls_per_step_min'] == x['calls_per_step_max'] == 1 for x in result['top_implementations']))

    def test_union_does_not_double_count_overlap(self):
        events = [{'ts': 0, 'dur': 10}, {'ts': 2, 'dur': 4}, {'ts': 8, 'dur': 10}, {'ts': 20, 'dur': 2}]
        self.assertEqual(A['union_us'](events), 20)

    def test_missing_copy_rejected(self):
        with self.assertRaises(AssertionError):
            A['boundaries'](fixture()[:-1])

    def test_wrong_projection_rejected(self):
        gpu = fixture()
        gpu[4]['name'] = 'unrelated_kernel'
        with self.assertRaises(AssertionError):
            A['boundaries'](gpu)

    def test_prefill_leakage_rejected(self):
        gpu = fixture()
        gpu[30]['name'] = 'ggml_sycl_ssm_scan_prefill(test)'
        with self.assertRaises(AssertionError):
            A['boundaries'](gpu)

    def test_cross_boundary_event_rejected(self):
        gpu = fixture()
        gpu[3]['ts'] = 42
        with self.assertRaises(AssertionError):
            A['boundaries'](gpu)

    def test_host_duration_is_clipped(self):
        gpu = fixture()
        cpu = [{'name': 'wait', 'ts': 0, 'dur': 20000}]
        result = A['window'](gpu, cpu, A['boundaries'](gpu), 16)
        self.assertAlmostEqual(result['host_api_sums_ms_not_cpu_work'][0]['sum_ms'], result['window_ms'])

    def test_other_q8_projections_are_not_logits(self):
        gpu = fixture()
        for i in range(1, 128):
            gpu[i * 3]['name'] = A['OUTPUT'] + '(test)[SIMD_ANY {512; 1; 1} {8; 1; 1}]'
        self.assertEqual(len(A['boundaries'](gpu)), 128)

    def test_epoch_sized_union_preserves_short_durations(self):
        events = [{'ts': 1792224780000000.0 + i * 10, 'dur': 1.666} for i in range(100)]
        self.assertAlmostEqual(A['union_us'](events), 166.6)

    def test_epoch_shift_does_not_change_window_accounting(self):
        gpu = fixture()
        expected = A['window'](gpu, [], A['boundaries'](gpu), 16)
        shifted = [dict(e, ts=e['ts'] + 1792224780000000.0) for e in gpu]
        actual = A['window'](shifted, [], A['boundaries'](shifted), 16)
        for key in ('window_ms', 'summed_device_ms', 'device_busy_union_ms', 'uncovered_ms'):
            self.assertAlmostEqual(actual[key], expected[key])


if __name__ == '__main__':
    unittest.main()
