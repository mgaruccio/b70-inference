#!/usr/bin/env python3
"""Remote CPU-only tests for the two additional controls; no new inference."""
import ast
import copy
import json
from pathlib import Path
import runpy
import socket
import unittest

assert socket.gethostname() == 'inference-host'
M = runpy.run_path('run-controls.py')
OLD_PATH = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefix-replay/run-replay.py')
OLD = runpy.run_path(str(OLD_PATH))
P = runpy.run_path(str(M['PREVIOUS'] / 'run-controls.py'))
BASE = json.loads((P['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())


def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


class Controls(unittest.TestCase):
    def test_syntax_and_no_stale_variables(self):
        tree = ast.parse(Path('run-controls.py').read_text())
        self.assertFalse({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & {'fusion','esimd'})

    def test_order_and_namespace(self):
        self.assertEqual(M['CELLS'], (('O0V1',0,1),('O1V0',1,0)))
        self.assertEqual(M['ROOT'].name, '20261017-nemotron35-bolt-state-controls')

    def test_only_declared_argv_changes(self):
        before = copy.deepcopy(BASE)
        for label, opt, vmm in M['CELLS']:
            argv = M['make_create'](BASE, label, opt, vmm)
            i = argv.index(f'GGML_SYCL_ENABLE_OPT={opt}') - 1
            self.assertEqual(argv[i:i+4], ['-e',f'GGML_SYCL_ENABLE_OPT={opt}','-e',f'GGML_SYCL_ENABLE_VMM={vmm}'])
            del argv[i:i+4]
            argv[argv.index('--name') + 1] = BASE[BASE.index('--name') + 1]
            self.assertEqual(argv[:-1], BASE[:-1])
            self.assertEqual(argv[-1].rsplit('exec ',1)[1], BASE[-1].rsplit('exec ',1)[1])
            self.assertIn('GGML_SYCL_ENABLE_FUSION=1',argv)
            self.assertIn('GGML_SYCL_ENABLE_ESIMD=1',argv)
            self.assertIn('BOLT_SSM_PREFILL_NUMERICAL_PASSED',argv[-1])
        self.assertEqual(BASE,before)

    def test_payload_and_response_checks_unchanged(self):
        new,old=functions(Path('run-controls.py')),functions(OLD_PATH)
        for name in ['native_payload','check_native','save']:
            self.assertEqual(ast.dump(new[name]),ast.dump(old[name]))

    def test_restoration_unchanged(self):
        new,old=functions(Path('run-controls.py')),functions(OLD_PATH)
        a=next(n for n in new['main'].body if isinstance(n,ast.Try))
        b=next(n for n in old['main'].body if isinstance(n,ast.Try))
        self.assertEqual([ast.dump(n) for n in a.finalbody],[ast.dump(n) for n in b.finalbody])

    def test_saved_fixture(self):
        body=json.loads((M['PREVIOUS']/'run-r1/A0/diagnostic.response.json').read_text())['__verbose']
        M['check_native'](body,55,128)
        bad=copy.deepcopy(body);bad['timings']['cache_n']=1
        with self.assertRaises(AssertionError):M['check_native'](bad,55,128)


if __name__ == '__main__':
    unittest.main(verbosity=2)
