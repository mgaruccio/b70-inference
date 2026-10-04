"""CPU regression checks for the exact Python gates embedded in the run recipe.

These temporary fixtures test orchestration, not model quality or real training.
The real GPU CLI remains the end-to-end training/decoding acceptance boundary.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'scripts/experiments/glimmer_mtp_state_supervision.sh'
TRAINER = ROOT / 'scripts/experiments/glimmer_recursive_mtp.py'
BLOCKS = re.findall(r"<<'PY'\n(.*?)\nPY", RECIPE.read_text(), re.S)


def execute(block, *args):
    with patch.object(sys, 'argv', ['recipe-gate', *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(BLOCKS[block], str(RECIPE), 'exec'), {'__name__': '__main__'})


@unittest.skipIf(torch is None, 'CPU PyTorch is required')
class StateRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'stage1').mkdir()
        (self.root / 'state-smoke').mkdir()
        self.initial = {
            'model': 'meta-models/Muse-Glimmer-30B',
            'revision': 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68',
            'state_transition': 'postnorm-gated-residual-v1',
            'variant': 'shared-ce', 'max_depth': 1, 'rank': 128,
            'head': {'weight': torch.zeros(2, 2)},
        }
        self.control = {
            **self.initial, 'update': 1, 'root_exposures': 64, 'state_weight': 0.,
            'head': {'weight': torch.ones(2, 2)},
            'optimizer': {'state': {0: {'step': torch.tensor(1.)}}},
            'sampler': {'schema': 'glimmer-mtp-sampler-v1', 'root_cursor': 64,
                        'root_order': [1, 2, 0],
                        'generator_state': torch.tensor([1, 2, 3], dtype=torch.uint8),
                        'root_generator_state': torch.tensor([4, 5, 6], dtype=torch.uint8)},
            'init_head_identity': {'path': 'checkpoint-best.pt', 'bytes': 100, 'mtime_ns': 123},
        }
        self.state = copy.deepcopy(self.control)
        self.state.update(variant='shared-state', state_weight=.2, head={'weight': torch.full((2, 2), 2.)})

    def run_gate(self):
        torch.save(self.initial, self.root / 'stage1/checkpoint-best.pt')
        torch.save(self.control, self.root / 'state-smoke/shared-ce-last.pt')
        torch.save(self.state, self.root / 'state-smoke/shared-state-last.pt')
        execute(1, self.root, TRAINER)

    def test_equal_tensor_sampler_states_pass_actual_recipe_gate(self):
        self.run_gate()
        self.assertTrue(json.loads((self.root / 'state-smoke.json').read_text())['passed'])

    def test_mismatched_sampler_states_are_rejected(self):
        original = copy.deepcopy(self.state)
        for key in ('generator_state', 'root_generator_state', 'root_cursor'):
            with self.subTest(key=key):
                self.state = copy.deepcopy(original)
                if torch.is_tensor(self.state['sampler'][key]):
                    self.state['sampler'][key][0] += 1
                else:
                    self.state['sampler'][key] += 1
                with self.assertRaisesRegex(AssertionError, key):
                    self.run_gate()

    def test_different_initial_checkpoint_is_rejected(self):
        self.state['init_head_identity']['bytes'] += 1
        with self.assertRaises(AssertionError):
            self.run_gate()

    def test_failed_preflight_is_rejected(self):
        path = self.root / 'preflight.json'
        path.write_text(json.dumps({'passed': False}))
        with self.assertRaisesRegex(AssertionError, 'preflight'):
            execute(0, path)

    def test_report_excludes_divergent_pairs_from_speedup(self):
        for arm, weight, acceptance in (('shared-ce', 0., .4), ('shared-state', .2, .5)):
            (self.root / arm).mkdir()
            (self.root / (arm + '.json')).write_text(json.dumps({
                'updates_completed': 10000, 'root_exposures': 640000,
                'best_update': 9000, 'state_weight': weight}))
            pairs = [{'exact_token_identity': exact,
                      'baseline': {'decode_tokens_per_s': 100.},
                      'candidate': {'decode_tokens_per_s': speed}}
                     for exact, speed in ((True, 120.), (True, 130.), (False, 100000.))]
            (self.root / arm / 'validation.json').write_text(json.dumps({
                'offline': {'per_depth': [{'depth': 1, 'teacher_kl': 1.}]},
                'probe': {'pairs': pairs, 'draft_acceptance_rate': acceptance,
                          'accepted_drafts': int(acceptance * 100), 'proposed_drafts': 100,
                          'categories': [], 'fidelity': {'exact_token_identity': False}}}))
        execute(2, self.root)
        report = json.loads((self.root / 'comparison.json').read_text())
        self.assertAlmostEqual(report['acceptance_delta'], .1)
        self.assertEqual(report['arms']['shared-state']['exact_pairs'], 2)
        self.assertAlmostEqual(report['arms']['shared-state']['median_decode_speedup_exact_pairs_only'], 1.25)
        self.assertFalse(report['arms']['shared-state']['fidelity']['exact_token_identity'])


if __name__ == '__main__':
    unittest.main()
