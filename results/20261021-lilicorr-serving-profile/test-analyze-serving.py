#!/usr/bin/env python3
import copy
import json
import runpy
import tempfile
import unittest
from pathlib import Path

D = runpy.run_path(str(Path(__file__).with_name('analyze-serving.py')))


class ServingAnalysisTests(unittest.TestCase):
    def sidecar(self):
        def clock(position):
            return dict(realtime_ns=1_790_000_000_000_000_000 + position,
                        boottime_ns=700_016_000_000_000 + position,
                        monotonic_raw_ns=700_000_000_000_000 + position,
                        monotonic_ns=700_016_000_000_000 + position,
                        clock_read_span_ns=200)
        span = dict(kind='timed-measured', resumed=True, request_succeeded=True, pause_succeeded=True)
        for key, position in [('before_resume', 0), ('after_resume', 1_000_000),
                              ('before_request', 2_000_000), ('after_request', 1_002_000_000),
                              ('before_pause', 1_003_000_000), ('after_pause', 1_004_000_000)]:
            span[key] = clock(position)
        return dict(cell='VSPF-W3b', requests=[span])

    def test_clock_window_uses_raw_domain(self):
        row = D['clock_window'](self.sidecar(), {'elapsed_s': 1}, 'VSPF-W3b', 'short', 0)
        self.assertEqual(row['end_us'] - row['start_us'], 1_000_000)
        self.assertEqual(row['clock']['boottime_minus_raw_begin_ns'], 16_000_000_000)
        self.assertEqual(row['clock']['epoch_minus_boot_spread_ns'], 0)
        self.assertEqual(row['clock']['monotonic_window_ms'], 1000)

    def test_clock_steps_or_excess_read_uncertainty_rejected(self):
        for mutate in [lambda s: s['requests'][0]['after_request'].__setitem__('realtime_ns', 2_000_000_000_000_000_000),
                       lambda s: s['requests'][0]['before_request'].__setitem__('clock_read_span_ns', 2_000_000)]:
            sidecar = self.sidecar()
            mutate(sidecar)
            with self.assertRaises(AssertionError):
                D['clock_window'](sidecar, {'elapsed_s': 1}, 'VSPF-W3b', 'short', 0)

    def test_inactive_failed_or_multiple_requests_rejected(self):
        for key in ['resumed', 'request_succeeded', 'pause_succeeded']:
            sidecar = self.sidecar()
            sidecar['requests'][0][key] = False
            with self.assertRaises(AssertionError):
                D['clock_window'](sidecar, {'elapsed_s': 1}, 'VSPF-W3b', 'short', 0)
        sidecar = self.sidecar()
        sidecar['requests'] *= 2
        with self.assertRaises(AssertionError):
            D['clock_window'](sidecar, {'elapsed_s': 1}, 'VSPF-W3b', 'short', 0)
        with self.assertRaises(AssertionError):
            D['clock_window'](self.sidecar(), {'elapsed_s': 3}, 'VSPF-W3b', 'short', 0)

    def fixture(self, root):
        cell = Path(root) / 'VSPF-W3a'
        cell.mkdir()
        records = []
        proposed = accepted = 0
        for index in range(33):
            generated_delta = accepted_delta = 1
            if index in (26, 27, 28, 30, 31, 32):
                point = 'short' if index < 29 else 'long'
                ordinal = index - (26 if point == 'short' else 30)
                generated_delta, accepted_delta = (129, 84) if point == 'short' else (133, 82)
                path = cell / point / f'measured-{ordinal:02}.result.json'
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({'timings': dict(draft_n=generated_delta,
                    draft_n_accepted=accepted_delta, predicted_ms=100)}))
            proposed += generated_delta
            accepted += accepted_delta
            records.append(dict(cumulative=True, calls=dict(b=index+1, g=index+1, a=index+1),
                                generated=dict(tokens=proposed), accepted=dict(tokens=accepted),
                                duration_ms=dict(b=(index+1)*.001, g=(index+1)*10, a=(index+1)*.001),
                                text='fixture'))
        path = cell / 'server-duration-stats.json'
        path.write_text(json.dumps(dict(records=records)))
        return path, records

    def test_callback_deltas_are_not_cumulative_totals(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root)
            result = D['callback_deltas'](Path(root), 'VSPF-W3a')
            self.assertEqual(len(result['rows']), 6)
            for row in result['rows']:
                self.assertEqual(row['callback_host_ms']['g'], 10)
                self.assertEqual(row['draft_callback_fraction_of_server_decode'], .1)
                self.assertEqual(row['calls']['g'], 1)

    def test_callback_extra_request_or_counter_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, records = self.fixture(root)
            changed = copy.deepcopy(records)
            changed[27]['generated']['tokens'] += 1
            path.write_text(json.dumps(dict(records=changed)))
            with self.assertRaises(AssertionError):
                D['callback_deltas'](Path(root), 'VSPF-W3a')
            path.write_text(json.dumps(dict(records=records + records[-1:])))
            with self.assertRaises(AssertionError):
                D['callback_deltas'](Path(root), 'VSPF-W3a')

    def test_analysis_never_overwrites_prior_report(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'report.json'
            D['save'](path, {'first': True})
            with self.assertRaises(AssertionError):
                D['save'](path, {'first': False})
            self.assertEqual(json.loads(path.read_text()), {'first': True})


if __name__ == '__main__':
    unittest.main()
