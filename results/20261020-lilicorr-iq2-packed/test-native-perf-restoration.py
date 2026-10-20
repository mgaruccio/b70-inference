#!/usr/bin/env python3
"""CPU-only checks of the exact native-container false/null normalization."""
import copy
import json
import runpy
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = runpy.run_path(str(HERE / 'verify-native-perf-restored.py'))


class RestorationChecks(unittest.TestCase):
    def setUp(self):
        _runtime, c, _capture, _cls = V['R']['bind']()
        root = HERE / 'native-perf-r1/VSPF-W3a'
        self.before, self.after = [json.loads((root / name).read_text()) for name in
                                  ('container-before.json', 'container-after.json')]
        self.compare = V['native_oom_comparator'](c['same_container'], self.before['Id'])

    def test_only_known_native_drift_passes(self):
        before, after = copy.deepcopy(self.before), copy.deepcopy(self.after)
        self.assertTrue(self.compare(self.before, self.after))
        self.assertEqual(self.before, before)
        self.assertEqual(self.after, after)

    def test_other_configuration_or_mount_drift_fails(self):
        for part, key, value in (('Config', 'Cmd', ['bad']), ('HostConfig', 'ShmSize', 1),
                                 ('HostConfig', 'Binds', [])):
            record = copy.deepcopy(self.after)
            record[part][key] = value
            with self.subTest(key=key):
                self.assertFalse(self.compare(self.before, record))
        record = copy.deepcopy(self.after)
        record['Mounts'][0]['RW'] = not record['Mounts'][0]['RW']
        self.assertFalse(self.compare(self.before, record))
        record = copy.deepcopy(self.after)
        record['Id'] = 'different'
        self.assertFalse(self.compare(self.before, record))

    def test_true_numeric_and_string_oom_values_fail(self):
        for value in (True, 0, 'false'):
            record = copy.deepcopy(self.after)
            record['HostConfig']['OomKillDisable'] = value
            with self.subTest(value=value), self.assertRaises(AssertionError):
                self.compare(self.before, record)

    def test_original_container_is_not_normalized(self):
        before, after = copy.deepcopy(self.before), copy.deepcopy(self.after)
        before['Id'] = after['Id'] = 'original-service'
        self.assertFalse(self.compare(before, after))


if __name__ == '__main__':
    unittest.main()
