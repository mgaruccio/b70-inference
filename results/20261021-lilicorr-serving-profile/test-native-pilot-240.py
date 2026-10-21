#!/usr/bin/env python3
import copy
import json
import runpy
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = runpy.run_path(str(HERE / 'native-pilot-240.py'))


class PilotTests(unittest.TestCase):
    def test_only_declared_mounts_and_body_change(self):
        argv = ['docker', 'create', '--name', 'fixture', 'image', 'bash', '-lc', 'original body']
        actual = P['derive'](argv)
        self.assertEqual(actual[-4:-1], ['image', 'bash', '-lc'])
        reverse = actual[:]
        del reverse[-8:-4]
        reverse[-1] = argv[-1]
        self.assertEqual(reverse, argv)
        args = '/src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -p nvfp4_reorder=1'
        self.assertEqual(P['BODY'].count(args), 2)
        self.assertEqual(P['BODY'].count("56/56 tests passed"), 2)
        self.assertIn('-d -h -s -v --chrome-kernel-logging', P['BODY'])

    def test_expected_mounts_do_not_mutate_record_and_keep_other_fields(self):
        reference = dict(HostConfig=dict(Binds=['/src:/src:ro'], Other='unchanged'),
                         Mounts=[dict(Destination='/src', Source='/src', RW=False)])
        saved = copy.deepcopy(reference)
        result = P['expected'](reference)
        self.assertEqual(reference, saved)
        self.assertEqual(result['HostConfig']['Other'], 'unchanged')
        self.assertEqual(len(result['Mounts']), 3)
        self.assertEqual(len(result['HostConfig']['Binds']), 3)
        with self.assertRaises(AssertionError):
            P['expected'](result)

    def test_binding_preserves_backend_and_finally_identity_scope(self):
        runtime, c, cls = P['bind']()
        self.assertEqual(c['OUT'], P['OUT'])
        self.assertEqual(c['PREFIX'], P['PREFIX'])
        self.assertEqual(P['R']['Q8']['bind'].__globals__['HERE'], P['R']['Q8_PATH'].parent)
        source = (HERE / 'native-pilot-240.py').read_text()
        self.assertIn('timeout=180', source)
        self.assertIn("OOM_COMPARATOR(original, before['Id'])", source)
        self.assertIn("c['same_container'] = original", source)
        self.assertIn("runtime['_BOOT_K']['capture']", source)
        for token in ['--teardown-on-signal', 'PROFILING_ENABLE', 'UR_L0_', 'SYCL_PI_LEVEL_ZERO_']:
            self.assertNotIn(token, P['BODY'])


if __name__ == '__main__':
    unittest.main()
