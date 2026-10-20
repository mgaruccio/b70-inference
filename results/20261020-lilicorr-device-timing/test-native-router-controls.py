#!/usr/bin/env python3
"""Host-only checks for the fixed native comparison; synthetic log cases are not GPU evidence."""
import copy
import json
import runpy
import subprocess
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
N = runpy.run_path(str(HERE / 'native-router-controls.py'))
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'


def synthetic(fusion):
    return ('BOLT_ROUTER floats: mismatches=0 BOLT_ROUTER indices: mismatches=0 \n' * 82
            + 'BOLT_ROUTER allocator: logits=probs exact (2 graphs)\n' * 5
            + 'BOLT_ROUTER consumer bytes: mismatches=0 \n' * 2
            + 'call ggml_sycl_op_router_fused:\n' * (67 if fusion else 0)
            + '  82/82 tests passed\n')


class NativeControlTests(unittest.TestCase):
    def test_only_native_body_and_process_profile_environment_change(self):
        _runtime, _analysis, c = N['D']['bind']()
        c['configure'](dict(OUT=QUALIFIED, C={}, CANDIDATE=N['D']['S']['PARENT'],
            make_create=lambda cell, _h: json.loads((QUALIFIED / cell / 'create-argv.json').read_text())))
        for cell in N['CELLS']:
            base = c['make_create'](cell, None)
            frozen = copy.deepcopy(base)
            result = N['derive'](base)
            self.assertEqual(base, frozen)
            self.assertEqual(result[-4:-1], base[-4:-1])
            self.assertEqual(result[-1], N['NATIVE_BODY'])
            expected = list(base[:-1])
            for i in reversed(range(len(expected) - 1)):
                if expected[i:i+2] == ['-e', 'LLAMA_LILICORR_PROFILE=1']:
                    del expected[i:i+2]
            self.assertEqual(result[:-1], expected)
        subprocess.run(['bash', '-n'], input=N['NATIVE_BODY'], text=True, check=True)
        self.assertNotIn('llama-server', N['NATIVE_BODY'])

    def test_existing_lifecycle_and_signal_capture_keep_private_namespace(self):
        runtime, c, capture, cls = N['bind']()
        self.assertIs(capture.__globals__['C'], c)
        self.assertEqual(c['PREFIX'], N['PREFIX'])
        self.assertEqual(c['OUT'], N['OUT'])
        obj = object.__new__(cls)
        obj.mode = 'native-router-correctness'
        with mock.patch.object(runtime['Runtime'], 'preflight', autospec=True,
                               side_effect=lambda self: self.mode) as parent:
            self.assertEqual(obj.preflight(), 'timing')
            parent.assert_called_once_with(obj)
        self.assertEqual(obj.mode, 'native-router-correctness')
        with mock.patch.object(runtime['Runtime'], 'preflight', side_effect=RuntimeError('guard')):
            with self.assertRaises(RuntimeError): obj.preflight()
        self.assertEqual(obj.mode, 'native-router-correctness')
        self.assertTrue(issubclass(cls, runtime['Runtime']))

    def test_native_container_identity_uses_existing_exact_checker(self):
        _runtime, c, _capture, _cls = N['bind']()
        c['configure'](dict(OUT=QUALIFIED, C={}, CANDIDATE=N['D']['S']['PARENT'],
            make_create=lambda cell, _h: json.loads((QUALIFIED / cell / 'create-argv.json').read_text())))
        old = runpy.run_path(str(HERE / 'test-runtime.py'))['RuntimeTests']()
        for cell in N['CELLS']:
            record = old.record(c, cell)
            record['Config']['Cmd'] = N['derive'](c['make_create'](cell, None))[-3:]
            record['Config']['Env'] = [v for v in record['Config']['Env'] if not v.startswith('LLAMA_LILICORR_PROFILE=')]
            before = copy.deepcopy(record)
            self.assertTrue(c['check_container'](record, cell, None))
            self.assertEqual(record, before)
            bad = copy.deepcopy(record)
            bad['Config']['Env'].append('LLAMA_LILICORR_PROFILE=1')
            with self.assertRaises(AssertionError): c['check_container'](bad, cell, None)
            bad = copy.deepcopy(record)
            next(m for m in bad['Mounts'] if m['Destination'] == '/src')['RW'] = True
            with self.assertRaises(AssertionError): c['check_container'](bad, cell, None)

    def test_strict_82_numerical_and_fusion_marker_requirements(self):
        for fusion in (0, 1):
            text = synthetic(fusion)
            self.assertTrue(N['check_log'](text, 0, fusion)['passed'])
            for bad in (text.replace('82/82', '77/82'),
                        text.replace('floats: mismatches=0', 'floats: mismatches=192', 1),
                        text.replace('indices: mismatches=0', 'indices: mismatches=1136', 1),
                        text + 'not supported\n'):
                self.assertFalse(N['check_log'](bad, 0, fusion)['passed'])
            self.assertFalse(N['check_log'](text, 1, fusion)['passed'])
            self.assertFalse(N['check_log'](text, 0, 1-fusion)['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
