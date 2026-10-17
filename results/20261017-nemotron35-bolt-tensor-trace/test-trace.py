#!/usr/bin/env python3
"""Remote CPU-only launch/payload contracts; real callback validation is API-owned."""
import ast
import copy
import json
from pathlib import Path
import runpy
import socket
import unittest

assert socket.gethostname() == 'inference-host'
M=runpy.run_path('run-trace.py')
P=runpy.run_path(str(M['PREVIOUS']/'run-controls.py'))
R=runpy.run_path(str(M['PREFIX']/'run-replay.py'))
BASE=json.loads((P['OLD']/'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())


class Trace(unittest.TestCase):
    def test_syntax(self):
        ast.parse(Path('run-trace.py').read_text())

    def test_debug_before_fresh_non_debug(self):
        self.assertEqual(M['CELLS'],(('on',1),('off',0)))

    def test_only_declared_argv_changes(self):
        before=copy.deepcopy(BASE)
        for label,enabled in M['CELLS']:
            argv=M['make_create'](BASE,label,enabled,P['ACCEPTED'])
            i=argv.index(f'BOLT_TRACE_TENSORS={enabled}')-1
            self.assertEqual(argv[i:i+2],['-e',f'BOLT_TRACE_TENSORS={enabled}'])
            del argv[i:i+2]
            argv[argv.index('--name')+1]=BASE[BASE.index('--name')+1]
            argv[argv.index(str(M['CANDIDATE'])+':/src:ro')]=str(P['ACCEPTED'])+':/src:ro'
            self.assertEqual(argv[:-1],BASE[:-1])
            self.assertEqual(argv[-1].rsplit('exec ',1)[1],BASE[-1].rsplit('exec ',1)[1])
            self.assertIn('unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1;',argv[-1])
            self.assertIn('BOLT_SSM_PREFILL_NUMERICAL_PASSED',argv[-1])
        self.assertEqual(BASE,before)

    def test_exact_prior_replay_payload(self):
        prior=M['PREFIX']/'run-r1/F1E1'
        ids=json.loads((prior/'summary.json').read_text())['prompt_ids']
        prefix=json.loads((M['PREFIX']/'run-r1/frozen-prefix.json').read_text())['generated_prefix_ids']
        self.assertEqual(len(ids+prefix),94)
        self.assertEqual(R['native_payload'](ids+prefix,1),json.loads((prior/'replay-00.request.json').read_text()))

    def test_saved_replay_response(self):
        body=json.loads((M['PREFIX']/'run-r1/F1E1/replay-00.response.json').read_text())
        R['check_native'](body,94,1)
        body['timings']['cache_n']=1
        with self.assertRaises(AssertionError):R['check_native'](body,94,1)


if __name__=='__main__':
    unittest.main(verbosity=2)
