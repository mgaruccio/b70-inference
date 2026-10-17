#!/usr/bin/env python3
"""Remote CPU-only contracts, using preserved response fixtures; no new inference."""
import ast
import copy
import json
from pathlib import Path
import runpy
import socket
import unittest

assert socket.gethostname() == 'inference-host'
M = runpy.run_path('run-replay.py')
P = runpy.run_path(str(M['PREVIOUS'] / 'run-controls.py'))
BASE = json.loads((P['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
FIXTURE = json.loads((M['PREVIOUS'] / 'run-r1/A0/diagnostic.response.json').read_text())['__verbose']


class Replay(unittest.TestCase):
    def test_syntax(self):
        ast.parse(Path('run-replay.py').read_text())

    def test_fixed_gray_order(self):
        self.assertEqual(M['CELLS'], (('F1E1', 1, 1), ('F0E1', 0, 1), ('F0E0', 0, 0), ('F1E0', 1, 0)))

    def test_only_declared_argv_changes(self):
        original = copy.deepcopy(BASE)
        allowed = {BASE.index('--name') + 1, BASE.index('GGML_SYCL_ENABLE_FUSION=1'),
                   BASE.index('GGML_SYCL_ENABLE_ESIMD=1'), len(BASE) - 1}
        for label, fusion, esimd in M['CELLS']:
            argv = M['make_create'](BASE, label, fusion, esimd)
            self.assertEqual(len(argv), len(BASE))
            self.assertTrue({i for i, (a,b) in enumerate(zip(argv, BASE)) if a != b} <= allowed)
            self.assertIn(f'GGML_SYCL_ENABLE_FUSION={fusion}', argv)
            self.assertIn(f'GGML_SYCL_ENABLE_ESIMD={esimd}', argv)
            self.assertEqual(argv[-1].rsplit('exec ', 1)[1], BASE[-1].rsplit('exec ', 1)[1])
            self.assertIn(M['MARKER'], argv[-1])
            self.assertNotIn('--spec-type', argv[-1])
        self.assertEqual(BASE, original)

    def test_payload_preserves_integer_prefix(self):
        ids = [12, 42, 100, 1046]
        payload = M['native_payload'](ids, 1)
        ids.append(999)
        self.assertEqual(payload['prompt'], [12, 42, 100, 1046])
        self.assertEqual(payload['n_predict'], 1)
        self.assertFalse(payload['cache_prompt'])
        self.assertFalse(payload['backend_sampling'])
        self.assertTrue(payload['ignore_eos'])
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['seed'], 42)
        self.assertNotIn('logit_bias', payload)

    def test_saved_native_fixture(self):
        M['check_native'](FIXTURE, 55, 128)

    def test_reject_cached_fixture(self):
        body = copy.deepcopy(FIXTURE)
        body['timings']['cache_n'] = 1
        with self.assertRaises(AssertionError): M['check_native'](body, 55, 128)

    def test_reject_wrong_count(self):
        with self.assertRaises(AssertionError): M['check_native'](FIXTURE, 94, 1)

    def test_reject_speculation(self):
        body = copy.deepcopy(FIXTURE)
        body['generation_settings']['speculative.types'] = 'draft-dflash'
        with self.assertRaises(AssertionError): M['check_native'](body, 55, 128)

    def test_reject_boolean_token(self):
        body = copy.deepcopy(FIXTURE)
        body['tokens'][0] = True
        with self.assertRaises(AssertionError): M['check_native'](body, 55, 128)

    def test_frozen_prefix_all_six(self):
        for label in P['ORDER']:
            body = json.loads((M['PREVIOUS'] / f'run-r1/{label}/diagnostic.response.json').read_text())
            self.assertEqual(body['__verbose']['tokens'][:39], FIXTURE['tokens'][:39])
            self.assertEqual(body['__verbose']['prompt'], FIXTURE['prompt'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
