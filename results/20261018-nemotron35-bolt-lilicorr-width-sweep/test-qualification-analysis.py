#!/usr/bin/env python3
"""CPU-only qualification-selection regressions; no live work."""
import copy
from pathlib import Path
import runpy
import unittest

HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE / 'analyze-qualify.py'))


def reports():
    return {
        cell: dict(cell=cell, width=width, passed=True,
                   jobs=[dict(stem=f'fixture-{i}', passed=True) for i in range(25)],
                   errors=[], fidelity_failures=[],
                   behavior=dict(natural_stop_cases=21, fixed_length_cases=2,
                                 diagnostics_exercised=2, rejected_draft_observed=1 if width else 0))
        for cell, width in (('A', None), ('W3', 3), ('W7', 7))
    }


def reject(value, cell):
    value[cell]['passed'] = False
    value[cell]['jobs'][0]['passed'] = False
    value[cell]['behavior']['natural_stop_cases'] = 20
    value[cell]['fidelity_failures'] = [dict(stem='fixture-0', error='text differs')]


class QualificationSelectionTests(unittest.TestCase):
    def test_valid_and_fidelity_rejected_combinations(self):
        for rejected, expected in (((), [3, 7]), (('W3',), [7]), (('W7',), [3]),
                                   (('W3', 'W7'), [])):
            value = reports()
            for cell in rejected:
                reject(value, cell)
            before = copy.deepcopy(value)
            with self.subTest(rejected=rejected):
                self.assertEqual(A['select_qualified'](value), expected)
                self.assertEqual(value, before)

    def test_missing_behavior_reject_does_not_block_sibling(self):
        value = reports()
        value['W3'].update(passed=False, fidelity_failures=[dict(scope='behavior', error='no rejection observed')])
        value['W3']['behavior']['rejected_draft_observed'] = 0
        self.assertEqual(A['select_qualified'](value), [7])

    def test_baseline_infrastructure_incomplete_and_false_pass_fail(self):
        for defect in ('baseline', 'infra', 'missing-job', 'duplicate-job', 'false-pass',
                       'missing-diagnostic', 'no-rejection', 'unexplained-reject'):
            value = reports()
            if defect == 'baseline':
                reject(value, 'A')
            elif defect == 'infra':
                value['W3']['errors'] = ['transport failed']
            elif defect == 'missing-job':
                value['W3']['jobs'].pop()
            elif defect == 'duplicate-job':
                value['W3']['jobs'][1] = dict(value['W3']['jobs'][0])
            elif defect == 'false-pass':
                value['W3']['jobs'][0]['passed'] = False
            elif defect == 'missing-diagnostic':
                value['W3']['behavior']['diagnostics_exercised'] = 1
            elif defect == 'no-rejection':
                value['W3']['behavior']['rejected_draft_observed'] = 0
            else:
                value['W3']['passed'] = False
            with self.subTest(defect=defect), self.assertRaises(AssertionError):
                A['select_qualified'](value)


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
