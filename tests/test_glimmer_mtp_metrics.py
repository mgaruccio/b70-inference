"""Stdlib-only checks for truthful monitoring; no model runtime required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('mtp_metrics', Path(__file__).parents[1] / 'scripts/experiments/glimmer_mtp_metrics.py')
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class MetricsTests(unittest.TestCase):
    def data(self, **overrides):
        return {'observed_at': 123, 'active': False, 'commands': [], 'complete': False,
                'training': [], 'captures': [], 'gpu': None, **overrides}

    def test_unreachable_does_not_replay_old_training(self):
        self.assertEqual(metrics.render(self.data(), False), 'mtp_source_reachable 0\n')

    def test_capture_does_not_claim_training_loss(self):
        text = metrics.render(self.data(active=True, commands=[['python', 'capture-generated']]))
        self.assertIn('mtp_phase 1\n', text)
        self.assertNotIn('mtp_train_loss', text)

    def test_idle_is_not_completed_or_failed(self):
        self.assertIn('mtp_phase 0\n', metrics.render(self.data()))
        self.assertIn('mtp_phase 5\n', metrics.render(self.data(latest_log_error=True)))
        self.assertIn('mtp_phase 4\n', metrics.render(self.data(complete=True)))

    def test_diagnostic_is_not_validation_acceptance(self):
        data = self.data(training=[{'stage': 'stage0/overfit', 'mtime': 100,
             'latest': {'variant': 'shared-ce', 'update': 500, 'loss': 0.1,
                        'diagnostic': {'per_depth': [{'depth': 1, 'teacher_argmax_agreement': .95}]},
                        'probe': {'draft_acceptance_rate': .4}}}])
        text = metrics.render(data)
        self.assertIn('mtp_diagnostic_teacher_argmax_agreement', text)
        self.assertNotIn('mtp_validation_teacher_argmax_agreement', text)
        self.assertIn('mtp_validation_draft_acceptance{stage="stage0/overfit",variant="shared-ce"} 0.4', text)

    def test_partial_jsonl_keeps_last_complete_record(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            log = root / 'shared-ce-training.jsonl'
            log.write_text(json.dumps({'update': 1, 'variant': 'shared-ce', 'loss': .2}) + '\n{"update":2')
            with patch.object(metrics, 'active_commands', return_value=[]), patch.object(metrics.subprocess, 'run', side_effect=OSError):
                data = metrics.snapshot(root)
            self.assertEqual(data['training'][0]['latest']['update'], 1)
            self.assertFalse(data['complete'])

    def test_selected_head_acceptance_is_actual_validation_report(self):
        data = self.data(validation_reports=[{'stage': 'stage0/selected-head',
              'probe': {'draft_acceptance_rate': .25},
              'offline': {'per_depth': [{'depth': 1, 'teacher_kl': 2.0}]}}])
        text = metrics.render(data)
        self.assertIn('mtp_validation_draft_acceptance{stage="stage0/selected-head",variant="selected-head"} 0.25', text)
        self.assertNotIn('mtp_train_loss', text)

    def test_live_capture_counts_are_observed_not_completed_totals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'stage0').mkdir()
            (root / 'stage0/stage0-complete.json').write_text('{}')
            event = {'event': 'capture_progress', 'capture_dir': str(root / 'capture'),
                     'split': 'train', 'sequence_token_count': 100, 'root_count': 50, 'budget': 3000000}
            (root / 'capture.log').write_text(json.dumps(event) + '\n')
            with patch.object(metrics, 'active_commands', return_value=[]), patch.object(metrics.subprocess, 'run', side_effect=OSError):
                data = metrics.snapshot(root)
            text = metrics.render(data)
            self.assertIn('mtp_capture_split_tokens{split="train",stage="capture"} 100', text)
            self.assertNotIn('mtp_capture_sequence_tokens', text)
            self.assertFalse(data['complete'])
    def test_nonfinite_and_missing_values_are_omitted(self):
        data = self.data(training=[{'stage': 'x', 'mtime': 1,
            'latest': {'variant': 'shared-ce', 'update': 1, 'loss': float('nan')}}])
        text = metrics.render(data)
        self.assertNotIn('mtp_train_loss', text)
        self.assertNotIn('nan', text)


if __name__ == '__main__':
    unittest.main()
