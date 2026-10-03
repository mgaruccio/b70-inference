"""Resume portability ignores physical UUID only, never runtime compatibility."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('mtp_resume_env', Path(__file__).parents[1] / 'scripts/experiments/glimmer_recursive_mtp.py')
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class ResumeEnvironmentTests(unittest.TestCase):
    def environment(self):
        return {'model': 'meta-models/Muse-Glimmer-30B', 'revision': 'pinned',
                'torch': '2.14.1+cu130', 'transformers': '5.15.1', 'cuda': '13.0',
                'gpu': 'NVIDIA A100-SXM4-80GB', 'dtype': 'bfloat16', 'attention': 'sdpa',
                'python': '3.10.13', 'platform': 'Linux-x86_64',
                'nvidia_smi': 'name, uuid, driver_version, memory.total [MiB], power.limit [W]\n'
                              'NVIDIA A100-SXM4-80GB, GPU-old, 580.126.09, 81920 MiB, 500.00 W\n'}

    def test_replacement_uuid_accepted_without_mutating_recorded_environment(self):
        saved = self.environment()
        original = copy.deepcopy(saved)
        current = copy.deepcopy(saved)
        current['nvidia_smi'] = current['nvidia_smi'].replace('GPU-old', 'GPU-new')
        self.assertTrue(pilot.resume_environment_matches(saved, current))
        self.assertEqual(saved, original)
        self.assertIn('GPU-new', current['nvidia_smi'])

    def test_all_other_environment_fields_stay_strict(self):
        saved = self.environment()
        for key in saved:
            if key == 'nvidia_smi':
                continue
            with self.subTest(key=key):
                current = copy.deepcopy(saved)
                current[key] += '-changed'
                self.assertFalse(pilot.resume_environment_matches(saved, current))
        current = copy.deepcopy(saved)
        current['extra_runtime_field'] = 'unexpected'
        self.assertFalse(pilot.resume_environment_matches(saved, current))

    def test_driver_memory_power_and_device_count_stay_strict(self):
        saved = self.environment()
        for old, new in [('580.126.09', '581.0'), ('81920 MiB', '40960 MiB'),
                         ('500.00 W', '300.00 W'), ('NVIDIA A100', 'NVIDIA H100')]:
            with self.subTest(change=old):
                current = copy.deepcopy(saved)
                current['nvidia_smi'] = current['nvidia_smi'].replace(old, new)
                self.assertFalse(pilot.resume_environment_matches(saved, current))
        current = copy.deepcopy(saved)
        current['nvidia_smi'] += current['nvidia_smi'].splitlines()[1] + '\n'
        self.assertFalse(pilot.resume_environment_matches(saved, current))

    def test_unexpected_schema_rejected_and_cpu_records_remain_exact(self):
        saved = self.environment()
        current = copy.deepcopy(saved)
        current['nvidia_smi'] = 'unexpected output'
        with self.assertRaisesRegex(ValueError, 'nvidia-smi schema'):
            pilot.resume_environment_matches(saved, current)
        self.assertTrue(pilot.resume_environment_matches({'runtime': 'cpu'}, {'runtime': 'cpu'}))
        self.assertFalse(pilot.resume_environment_matches({'runtime': 'cpu'}, {'runtime': 'different'}))


if __name__ == '__main__':
    unittest.main()
