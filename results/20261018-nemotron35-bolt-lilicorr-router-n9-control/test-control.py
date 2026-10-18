#!/usr/bin/env python3
"""Remote CPU checks; synthetic logs are not native evidence."""
import runpy
import socket
import unittest
from pathlib import Path

assert __debug__ and socket.gethostname() == 'inference-host'
ROOT = Path(__file__).resolve().parent
M = runpy.run_path(str(ROOT / 'run-control.py'))
F = runpy.run_path(str(M['SOURCE'] / 'test-trace.py'))
N9 = M['N9']

def fixture(passed=False):
    rows = F['rows_for'](('blk.48.ffn_gate_inp.weight', 496, 0, 9), 'router_columns8')
    lines = ['Backend 1/2: SYCL0']
    lines += ['BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in sorted(row.items())) for row in rows]
    lines += [N9['MARKER'] + '00' * N9['EXPECTED_BYTES']]
    if not passed:
        lines += ['[MUL_MAT] ERR = 0.000000139 > 0.000000100']
    lines += [f'  {int(passed)}/1 tests passed', 'Backend 2/2: CPU', '  Skipping',
              f'{2 if passed else 1}/2 backends passed']
    return '\n'.join(lines) + '\n'


class ControlTests(unittest.TestCase):
    def test_failure_remains_failure(self):
        result = M['analyze_native'](fixture(), 1)
        self.assertFalse(result['numerical_pass'])
        self.assertFalse(result['qualification_waiver'])
        self.assertFalse(result['original_random_operands_replayed'])
        self.assertFalse(result['raw_output_byte_equality_claimed'])
        self.assertEqual(result['reported_error'], '0.000000139')
        self.assertEqual(result['tolerance'], 1e-7)

    def test_success_does_not_invent_error_measurement(self):
        result = M['analyze_native'](fixture(True), 0)
        self.assertTrue(result['numerical_pass'])
        self.assertIsNone(result['reported_error'])
        self.assertFalse(result['qualification_waiver'])

    def test_tolerance_cannot_be_relaxed(self):
        with self.assertRaises(AssertionError):
            M['analyze_native'](fixture().replace('> 0.000000100', '> 0.000001000'), 1)

    def test_wrong_exit_or_extra_cases_rejected(self):
        for raw, code in ((fixture(), 0), (fixture(True), 1),
                          (fixture().replace('0/1 tests', '0/2 tests'), 1)):
            with self.subTest(raw=raw, code=code), self.assertRaises(AssertionError):
                M['analyze_native'](raw, code)

    def test_missing_or_extra_error_rejected(self):
        for raw, code in ((fixture().replace('[MUL_MAT] ERR = 0.000000139 > 0.000000100', ''), 1),
                          (fixture(True) + '[MUL_MAT] ERR = 0.000000139 > 0.000000100\n', 0)):
            with self.subTest(raw=raw, code=code), self.assertRaises(AssertionError):
                M['analyze_native'](raw, code)

    def test_output_marker_fails_closed_and_pair_is_literal(self):
        raw = fixture()
        marker = next(line for line in raw.splitlines() if line.startswith(N9['MARKER']))
        wrong_length = N9['MARKER'] + '00' * (N9['EXPECTED_BYTES'] - 1)
        for changed in (raw.replace(marker + '\n', '', 1), raw + marker + '\n',
                        raw.replace(marker, wrong_length, 1)):
            with self.subTest(changed=changed[:len(N9['MARKER'])]), self.assertRaises(AssertionError):
                M['analyze_native'](changed, 1)
        one_byte_mismatch = N9['MARKER'] + '01' + '00' * (2 * N9['EXPECTED_BYTES'] - 2)
        with self.assertRaises(AssertionError):
            N9['check_pair'](raw, raw.replace(marker, one_byte_mismatch, 1))
    def test_wrong_route_or_duplicate_rejected(self):
        raw = fixture()
        extra = next(line for line in raw.splitlines() if line.startswith('BOLT_ROUTE')) + '\n'
        for changed in (raw + extra, raw.replace('router_columns8-m496-l0-n9', 'router_columns8-m496-l0-n8'),
                        raw.replace('LIBRARY_DNNL_F32', 'LIBRARY_MKL_F32')):
            with self.subTest(raw=changed), self.assertRaises(AssertionError):
                M['analyze_native'](changed, 1)

    def test_pair_changes_only_name_and_backend(self):
        h = M['C']['helpers']()
        cells = list(M['BACKENDS'])
        argv = [M['argv_for'](cell, h) for cell in cells]
        self.assertEqual(len(M['C']['argv_diff'](*argv)), 2)
        for cell, command in zip(cells, argv):
            self.assertEqual(command[-1], M['COMMAND'])
            self.assertEqual(command.count(str(M['TEST_BINARY']) + ':/src/build-sycl/bin/test-backend-ops:ro'), 1)
            self.assertEqual(command.count(str(M['BACKENDS'][cell]) + ':/src:ro'), 1)
        self.assertIn('GGML_SYCL_ENABLE_DNN=1', M['COMMAND'])
        self.assertIn('GGML_SYCL_ENABLE_GRAPH=0', M['COMMAND'])

    def test_restore_implementation_is_inherited(self):
        self.assertIs(M['Runtime'].restore, M['K']['Runtime'].restore)


if __name__ == '__main__':
    unittest.main(verbosity=2)
