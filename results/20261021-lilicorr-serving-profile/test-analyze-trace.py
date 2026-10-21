#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('analyze_trace', Path(__file__).with_name('analyze-trace.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class TraceTests(unittest.TestCase):
    def test_pinned_unitrace_clock_domain_is_not_realtime(self):
        stamp = dict(realtime_ns=1_790_000_000_000_000_000,
                     boottime_ns=700_016_000_000_000,
                     monotonic_raw_ns=700_000_000_000_000,
                     clock_read_span_ns=250)
        self.assertEqual(M.trace_us(stamp), (stamp['realtime_ns'] - 16_000_000_000) / 1000)
        stamp['clock_read_span_ns'] = -1
        with self.assertRaises(AssertionError):
            M.trace_us(stamp)

    def test_union(self):
        self.assertEqual(M.union([(5, 8), (1, 4), (2, 3), (4, 5), (9, 9)]), [[1, 8]])
        self.assertEqual(M.duration([(1, 3), (2, 4)]), 3)

    def test_partition_overlap_is_not_double_counted(self):
        result = M.partition({'gpu': [(1, 4), (3, 5)], 'host_sync_api': [(2, 7)]}, 1, 9)
        self.assertEqual(result, {'gpu': .001, 'gpu+host_sync_api': .003, 'host_sync_api': .002, 'untraced': .002})
        self.assertAlmostEqual(sum(result.values()), .008)

    def test_partition_clips(self):
        self.assertEqual(M.partition({'gpu': [(-1, 3), (8, 20)]}, 1, 9), {'gpu': .003, 'untraced': .005})

    def test_invalid_intervals(self):
        for rows in [[(2, 1)], [(0, float('nan'))], [(0, float('inf'))]]:
            with self.assertRaises(AssertionError):
                M.union(rows)

    def trace(self, root, events):
        path = Path(root) / 'trace.json'
        path.write_text('{ "traceEvents":[\n' + ',\n'.join(json.dumps(e) for e in events) + '\n]\n}\n')
        return path

    def window(self, **extra):
        return dict(cell='VSPF-W3b', point='short', stem='measured-00', start_us=100, end_us=200,
                    fidelity_passed=True, traced=True, **extra)

    def event(self, cat, ts, dur, name='unknown'):
        return dict(ph='X', cat=cat, ts=ts, dur=dur, name=name)

    def test_unsorted_events_clipping_and_nonadditive_sums(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.trace(root, [self.event('cpu_op', 95, 15, 'zeCommandListAppendLaunchKernel'),
                                     self.event('gpu_op', 120, 60, 'iq2_kernel'),
                                     self.event('gpu_op', 110, 40, 'iq2_kernel'),
                                     self.event('cpu_op', 130, 80, 'zeEventHostSynchronize')])
            rule = dict(substring='iq2', category='target_iq2', evidence='fixture source')
            row = M.analyze(path, [self.window()], [rule])['requests'][0]
            self.assertEqual(row['gpu_busy_union_ms'], .07)
            self.assertEqual(row['kernel_event_sum_ms'], .1)
            self.assertEqual(row['gpu_category_union_ms'], {'target_iq2': .07})
            self.assertAlmostEqual(sum(row['wall_partition_ms'].values()), .1)
            self.assertEqual(row['gpu_events'], 2)

    def test_multiple_windows_and_straddling_event(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.trace(root, [self.event('gpu_op', 50, 300), self.event('cpu_op', 50, 300)])
            second = dict(self.window(), stem='measured-01', start_us=250, end_us=300)
            rows = M.analyze(path, [second, self.window()])['requests']
            self.assertEqual([r['gpu_busy_union_ms'] for r in rows], [.1, .05])

    def test_qualification_and_window_guards(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.trace(root, [self.event('gpu_op', 110, 10), self.event('cpu_op', 110, 10)])
            for key, value in [('fidelity_passed', False), ('traced', False), ('end_us', 50)]:
                window = self.window()
                window[key] = value
                with self.assertRaises(AssertionError):
                    M.analyze(path, [window])
            with self.assertRaises(AssertionError):
                M.analyze(path, [self.window(), self.window()])
            with self.assertRaises(AssertionError):
                M.analyze(path, [dict(self.window(), start_us=500, end_us=600)])

    def test_ambiguous_rules_rejected(self):
        with self.assertRaises(AssertionError):
            M.kernel_kind('abc', [dict(substring='a', category='x'), dict(substring='b', category='y')])

    def test_trace_truncation_and_separators_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            good = self.trace(root, [self.event('gpu_op', 100, 10)])
            original = good.read_text()
            self.assertEqual(len(list(M.events(good))), 1)
            for invalid in [original[:-4], original.replace('\n]\n}', ',\n]\n}'), original + '{}',
                            original.replace('\n]\n}', '\n{}\n]\n}')]:
                good.write_text(invalid)
                with self.assertRaises((AssertionError, ValueError)):
                    list(M.events(good))


if __name__ == '__main__':
    unittest.main()
