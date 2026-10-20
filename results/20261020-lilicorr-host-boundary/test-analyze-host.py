#!/usr/bin/env python3
"""Synthetic host-mode parser fixtures only; no actual GPU evidence is fabricated."""
import json
import re
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE / 'analyze-host.py'))
F = runpy.run_path(str(HERE.parent / '20261020-lilicorr-device-timing/test-analyze-device.py'))


def synthetic_host_request():
    lines = []
    for line in F['valid_request']().splitlines():
        if 'stage=event ' in line:
            line = line.replace('status=ok ', 'status=unsupported_profiling ')
            line = re.sub(r'(device_(?:start_ns|end_ns|elapsed_ns))=\d+', r'\1=unavailable', line)
        lines.append(line)
    return '\n'.join(lines)


class HostAnalysisTests(unittest.TestCase):
    def test_positive_components_without_any_gpu_duration(self):
        value = A['check_request'](synthetic_host_request())
        self.assertEqual(value['status'], 'ok')
        self.assertFalse(value['gpu_time_claim'])
        self.assertEqual(value['gpu_timestamps'], 'unavailable')
        self.assertEqual(value['host']['totals_us']['target_getter'], 72)
        self.assertNotIn('device_elapsed_ns', json.dumps(value))
        self.assertEqual({r['width'] for r in value['launch_metadata']}, {4, 8, 46})

    def test_profiled_or_failed_or_incomplete_mode_is_not_host_success(self):
        for bad in (F['valid_request'](),
                    synthetic_host_request().replace('dropped=0', 'dropped=1'),
                    synthetic_host_request().replace('status=unsupported_profiling', 'status=profiling_query_error', 1),
                    synthetic_host_request().replace('backend_ctx=0xabc', 'backend_ctx=0xdef', 1),
                    synthetic_host_request().replace('device_elapsed_ns=unavailable', 'device_elapsed_ns=0', 1)):
            with self.assertRaises((AssertionError, KeyError, ValueError)):
                A['check_request'](bad)

    def test_missing_and_duplicate_component_ownership_rejected(self):
        text = synthetic_host_request()
        for bad in ('\n'.join(s for s in text.splitlines() if 'PROCESS_COMPONENTS' not in s),
                    text + '\n' + next(s for s in text.splitlines() if 'PROCESS_COMPONENTS' in s and 'stage=chunk ' in s),
                    text.replace('process_rc=0', 'process_rc=1')):
            with self.assertRaises((AssertionError, KeyError, ValueError)):
                A['check_request'](bad)

    def test_only_new_output_is_written_and_previous_failure_is_preserved(self):
        (HERE / '.scratch').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE / '.scratch') as temporary:
            out = Path(temporary)
            (out / 'analysis-r1').mkdir()
            originals = {'analysis.json': json.dumps({'rows': [{'passed': True}] * 24}),
                         'profile.json': 'unchanged baseline\n'}
            for name, text in originals.items():
                (out / 'analysis-r1' / name).write_text(text)
            context = dict(status='analyzed', errors=[], requests=[{}] * 16)
            validator = mock.Mock(return_value=context)
            c = dict(load=lambda path: json.loads(Path(path).read_text()))
            meta = dict(text=synthetic_host_request(), cell='VSPF-W3a', point='short', phase='measured', ordinal=0)
            with mock.patch.dict(A['D']['CONTEXT_ANALYZER'], analyze=validator), \
                    mock.patch.dict(A['D'], _selected_requests=lambda *_a: ([meta], {}),
                                    _context_graph_record=lambda *_a: {'graphs': [{'graph': 10, 'graph_nodes': 59}]}):
                result = A['analyze'](out, c)
            self.assertEqual(result['status'], 'analyzed')
            self.assertEqual(result['context_measured_outputs'], 24)
            self.assertEqual(len(context['requests']), 16)
            validator.assert_called_once()
            validator.call_args.args[1]['save'](out / 'analysis-r1/profile.json', {'overwrite': True})
            self.assertNotIn('save', c)
            for name, text in originals.items():
                self.assertEqual((out / 'analysis-r1' / name).read_text(), text)
            before = (out / 'analysis-r1/host-analysis.json').read_bytes()
            with self.assertRaises(AssertionError): A['analyze'](out, c)
            self.assertEqual((out / 'analysis-r1/host-analysis.json').read_bytes(), before)

    def test_failed_context_or_measured_output_gates_still_fail_closed(self):
        (HERE / '.scratch').mkdir(exist_ok=True)
        valid = dict(status='analyzed', errors=[], requests=[{}] * 16)
        for context, rows in ((dict(valid, status='failed'), [{'passed': True}] * 24),
                              (dict(valid, errors=['failed gate']), [{'passed': True}] * 24),
                              (valid, [{'passed': True}] * 23),
                              (valid, [{'passed': True}] * 23 + [{'passed': False}])):
            with self.subTest(context=context['status'], rows=len(rows)), \
                    tempfile.TemporaryDirectory(dir=HERE / '.scratch') as temporary:
                c = dict(load=lambda _path: {'rows': rows})
                with mock.patch.dict(A['D']['CONTEXT_ANALYZER'], analyze=lambda *_a: context):
                    result = A['analyze'](Path(temporary), c)
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['errors'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
