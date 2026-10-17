#!/usr/bin/env python3
"""Remote CPU-only controller contract tests; no Docker or inference."""
import ast
import copy
import json
from pathlib import Path
import runpy
import socket
import unittest

assert socket.gethostname() == 'inference-host'
M = runpy.run_path('run-controls.py')
BASE = json.loads((M['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
IMAGE = 'intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04'
PAYLOAD = json.loads((M['PROFILE'] / 'run-r1/baseline/bench/measured-00.request.json').read_text())


class Controls(unittest.TestCase):
    def test_syntax(self):
        ast.parse(Path('run-controls.py').read_text())

    def test_order(self):
        self.assertEqual(M['ORDER'], ('A0', 'B0', 'C0', 'C1', 'B1', 'A1'))
        self.assertEqual([x[0] for x in M['ORDER']], list('ABCCBA'))

    def test_all_names_unique(self):
        results = [M['make_create'](BASE, label, IMAGE) for label in M['ORDER']]
        self.assertEqual(len({a[a.index('--name') + 1] for a in results}), 6)
        self.assertEqual(BASE[BASE.index('--name') + 1], 'nemotron-bolt-probe')

    def test_no_base_mutation(self):
        before = copy.deepcopy(BASE)
        for label in M['ORDER']:
            M['make_create'](BASE, label, IMAGE)
        self.assertEqual(BASE, before)

    def test_accepted_preserved(self):
        argv = M['make_create'](BASE, 'A0', IMAGE)
        self.assertIn(str(M['ACCEPTED']) + ':/src:ro', argv)
        self.assertNotIn('/native-outprod.sh:ro', ' '.join(argv))
        self.assertNotIn('--spec-type', argv[-1])
        self.assertEqual(argv[-1].rsplit('exec ', 1)[1], BASE[-1].rsplit('exec ', 1)[1])

    def test_patched_off(self):
        argv = M['make_create'](BASE, 'B0', IMAGE)
        self.assertIn(str(M['CANDIDATE']) + ':/src:ro', argv)
        self.assertNotIn(str(M['ACCEPTED']) + ':/src:ro', argv)
        self.assertIn('bash /native-outprod.sh;', argv[-1])
        self.assertNotIn('--spec-type', argv[-1])
        self.assertNotIn(':/draft:ro', ' '.join(argv))

    def test_patched_on(self):
        argv = M['make_create'](BASE, 'C0', IMAGE)
        self.assertIn(str(M['CANDIDATE']) + ':/src:ro', argv)
        self.assertIn(str(M['DRAFT'] / 'draft') + ':/draft:ro', argv)
        self.assertIn('--spec-type draft-dflash', argv[-1])
        self.assertIn('--spec-draft-n-max 4', argv[-1])
        self.assertIn('--no-spec-draft-backend-sampling', argv[-1])
        self.assertNotIn('spec-synth', argv[-1])

    def test_matched_logging(self):
        for label in M['ORDER']:
            argv = M['make_create'](BASE, label, IMAGE)
            self.assertIn('unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; exec ', argv[-1])
            self.assertIn('BOLT_SSM_PREFILL_NUMERICAL_PASSED', argv[-1])
            self.assertNotIn('GGML_SYCL_DEBUG=1', argv[:-1])

    def test_diagnostic_only_declared_differences(self):
        before = copy.deepcopy(PAYLOAD)
        diag = M['diagnostic_payload'](PAYLOAD)
        self.assertEqual(PAYLOAD, before)
        changed = {k for k in PAYLOAD.keys() | diag.keys() if PAYLOAD.get(k) != diag.get(k)}
        self.assertEqual(changed, {'stream', 'stream_options', 'verbose', 'return_tokens', 'logprobs', 'top_logprobs', 'backend_sampling'})
        self.assertFalse(diag['stream'])
        self.assertFalse(diag['backend_sampling'])
        self.assertEqual(diag['temperature'], 0)
        self.assertEqual(diag['seed'], 42)
        self.assertFalse(diag['cache_prompt'])
        self.assertEqual(diag['max_tokens'], 128)


if __name__ == '__main__':
    unittest.main(verbosity=2)
