#!/usr/bin/env python3
"""Remote CPU-only regression for mount-order correction; no new measurements."""
import copy
import json
from pathlib import Path
import runpy
import socket
import sys
import unittest

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing')
assert socket.gethostname() == 'inference-host'
assert Path(__file__).resolve().parent == ROOT

def deny_live(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise AssertionError('CPU-only regression forbids ' + event)

sys.addaudithook(deny_live)
OLD = runpy.run_path(str(ROOT / 'analyze-timing.py'))
NEW = runpy.run_path(str(ROOT / 'analyze-timing-r2.py'))
H = NEW['C']['helpers']()
OUT = ROOT / 'run-r1'
ORIGINAL = NEW['load'](OUT / 'original-container.json')

class MountOrderTests(unittest.TestCase):
    def test_frozen_capture_code_and_references_unchanged(self):
        NEW['C']['check_frozen'](H)
        H['off']['N']['check_sources_and_binaries']()

    def test_original_failure_reproduces(self):
        for cell in ('O1', 'O2'):
            with self.subTest(cell=cell), self.assertRaisesRegex(AssertionError, 'Mounts'):
                OLD['check_cell'](H, OUT / cell, ORIGINAL)

    def test_reordered_complete_mount_objects_pass(self):
        reordered = copy.deepcopy(ORIGINAL)
        reordered['Mounts'].reverse()
        for cell in ('O1', 'O2'):
            self.assertEqual(len(NEW['check_cell'](H, OUT / cell, reordered)), 21)

    def test_every_mount_field_and_multiplicity_remain_protected(self):
        for index, mount in enumerate(ORIGINAL['Mounts']):
            for field, value in mount.items():
                changed = copy.deepcopy(ORIGINAL)
                changed['Mounts'][index][field] = not value if isinstance(value, bool) else '__CHANGED__'
                with self.subTest(index=index, field=field), self.assertRaisesRegex(AssertionError, 'Mounts'):
                    NEW['check_cell'](H, OUT / 'O1', changed)
        for mode in ('missing', 'duplicate', 'extra-field'):
            changed = copy.deepcopy(ORIGINAL)
            if mode == 'missing':
                changed['Mounts'].pop()
            elif mode == 'duplicate':
                changed['Mounts'].append(copy.deepcopy(changed['Mounts'][0]))
            else:
                changed['Mounts'][0]['unexpected'] = 'not allowed'
            with self.subTest(mode=mode), self.assertRaisesRegex(AssertionError, 'Mounts'):
                NEW['check_cell'](H, OUT / 'O1', changed)

    def test_other_original_fields_still_exact(self):
        for field in ('Id', 'Config', 'HostConfig'):
            changed = copy.deepcopy(ORIGINAL)
            if field == 'Id':
                changed[field] = '__CHANGED__'
            else:
                changed[field]['r2_test_unexpected'] = True
            with self.subTest(field=field), self.assertRaises(AssertionError):
                NEW['check_cell'](H, OUT / 'O1', changed)

    def test_all_saved_cells_pass_without_changing_any_measurement_row(self):
        failed = NEW['load'](OUT / 'analysis-r1/analysis.json')
        self.assertEqual(failed['status'], 'failed')
        result = NEW['analyze'](OUT, H)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['errors'], [])
        self.assertEqual(len(result['rows']), 36)
        self.assertEqual(result['rows'], failed['rows'])
        self.assertEqual(NEW['load'](OUT / 'analysis-r1/analysis.json'), failed)
        self.assertFalse((OUT / 'analysis-r2').exists())

if __name__ == '__main__':
    unittest.main(verbosity=2)
