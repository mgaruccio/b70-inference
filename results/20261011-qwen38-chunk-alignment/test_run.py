"""Supplementary lifecycle/gating checks; not real B70 evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('chunk_campaign', Path(__file__).with_name('run.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CampaignGateTests(unittest.TestCase):
    def exercise(self, after, stop_error=False):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = module.Campaign.__new__(module.Campaign)
            campaign.root = Path(temporary)
            campaign.summary = {'cells': []}
            campaign.process = None
            campaign.log = Mock()
            def start(*args, **kwargs):
                campaign.log = Mock()
                return campaign.root, {}
            campaign.start = Mock(side_effect=start)
            campaign.raw = Mock()
            campaign.decode = Mock()
            campaign.canaries = Mock()
            campaign.stop = Mock()
            if stop_error:
                campaign.stop.side_effect = [None, RuntimeError('owned client wait failed')]
            profiles = {'baseline-before': 10.0, 'baseline-after': after,
                        'candidate-8320': 9.5, 'candidate-6656': 11.0}
            campaign.screen = Mock(side_effect=lambda label, budget: {
                'budget': budget, 'median_ttft_s': {'16384': profiles[label], '32768': profiles[label]}})
            with patch.object(module, 'power', return_value={'cap': '275000000'}), patch.object(module, 'command', side_effect=['', 'qwen38\n']):
                if stop_error:
                    with self.assertRaisesRegex(RuntimeError, 'owned client wait failed'):
                        campaign.run()
                else:
                    campaign.run()
            labels = [c.args[0] for c in campaign.start.call_args_list]
            boundary_calls = [c for c in campaign.raw.call_args_list if len(c.args) == 4 and c.args[3] == 'boundary']
            prefixes = [module.ids(c.args[1], c.args[2])[0] for c in boundary_calls]
            self.assertEqual(len(prefixes), 132)
            self.assertEqual(len(set(prefixes)), 132)
            self.assertTrue(campaign.summary['cleanup']['production_running'])
            self.assertEqual(labels[-1], 'production-restored')
            self.assertTrue(campaign.start.call_args_list[-1].kwargs['production'])
            return labels, campaign.summary

    def test_first_baseline_win_then_second_loss_never_extends(self):
        labels, summary = self.exercise(9.0)
        self.assertEqual(labels, ['guard-prefix-on', 'production-restored'])
        self.assertEqual(summary['primary_pass_budgets'], [])

    def test_both_baseline_wins_extend_control_and_only_qualifying_candidate(self):
        labels, summary = self.exercise(10.0)
        self.assertEqual(labels, ['guard-prefix-on', 'extended-8192', 'extended-8320', 'production-restored'])
        self.assertEqual(summary['primary_pass_budgets'], [8320])

    def test_stop_failure_does_not_skip_safe_production_restore(self):
        labels, summary = self.exercise(9.0, stop_error=True)
        self.assertEqual(labels[-1], 'production-restored')
        self.assertIn('stop_error', summary['cleanup'])
        self.assertTrue(summary['cleanup']['production_running'])


if __name__ == '__main__':
    unittest.main()
