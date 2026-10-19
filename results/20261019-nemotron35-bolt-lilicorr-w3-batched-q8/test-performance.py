#!/usr/bin/env python3
"""CPU-only namespace and expected-binding tests, using saved Docker fixtures."""
import copy
import json
import runpy
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-performance.py'))
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.analysis, self.c = D['bind']()
        self.public = dict(OUT=QUALIFIED, C={}, make_create=lambda cell, h:
            json.loads((QUALIFIED / cell / 'create-argv.json').read_text()))
        self.c['configure'](self.public)

    def fixture(self, cell):
        # This is an explicitly synthetic future-container fixture derived from
        # saved input, not an altered observation or evidence for a live run.
        base = self.c['qualified_dir'](self.c['role_for'](cell))
        actual = json.loads((base / 'container-before.json').read_text())
        actual.update(Name='/' + self.c['name'](cell), Id='b' * 64)
        actual['Config']['Hostname'] = 'b' * 12
        actual['Config']['Cmd'] = self.c['make_create'](cell, None)[-3:]
        old = str(D['PARENT']) + ':/src:ro'
        new = str(D['CANDIDATE']) + ':/src:ro'
        actual['HostConfig']['Binds'] = [new if b == old else b
                                        for b in actual['HostConfig']['Binds']]
        actual['HostConfig']['Binds'].reverse()
        actual['HostConfig']['OomKillDisable'] = False
        for mount in actual['Mounts']:
            if mount['Destination'] == '/src':
                mount['Source'] = str(D['CANDIDATE'])
        return actual

    def test_all_actual_function_globals_share_private_contracts(self):
        c = self.c
        for fn in (self.runtime['Runtime'].__init__, self.runtime['Runtime'].begin,
                   self.runtime['Runtime'].end, self.runtime['Runtime'].restore,
                   self.runtime['_BOOT_K']['capture'], self.runtime['_BOOT_K']['stream_chat'],
                   self.analysis['analyze'], self.analysis['reread_measurement']):
            self.assertIs(fn.__globals__['C'], c)
        self.assertIs(c['configure'].__globals__, c)
        self.assertEqual(c['PREFIX'], D['PREFIX'])
        self.assertEqual(c['OUT'], D['OUT'])
        untouched = runpy.run_path(str(D['BASE'] / 'controller.py'))
        self.assertNotEqual(untouched['C']['PREFIX'], c['PREFIX'])
        self.assertEqual(untouched['C']['OUT'].name, 'run-r3')

    def test_only_name_and_explicit_source_bind_change(self):
        for cell in self.c['CELLS']:
            base = self.public['make_create'](self.c['base_cell'](cell), None)
            actual = self.c['make_create'](cell, None)
            differences = [(i, a, b) for i, (a, b) in enumerate(zip(base, actual)) if a != b]
            self.assertEqual(len(base), len(actual))
            self.assertEqual(2, len(differences))
            self.assertEqual({base.index('--name') + 1,
                              base.index(str(D['PARENT']) + ':/src:ro')},
                             {i for i, _, _ in differences})
            self.assertEqual(base[-3:], actual[-3:])

    def test_container_validation_does_not_rewrite_observations(self):
        for cell in self.c['CELLS']:
            actual = self.fixture(cell)
            before = copy.deepcopy(actual)
            self.assertTrue(self.c['check_container'](actual, cell, None))
            self.assertEqual(before, actual)

    def test_reject_wrong_source_permissions_image_and_identity(self):
        cell = 'VSPF-W3a'
        pristine = self.fixture(cell)
        def wrong_bind(x):
            x['HostConfig']['Binds'].append(str(D['PARENT']) + ':/src:ro')
        def wrong_mount(x):
            next(m for m in x['Mounts'] if m['Destination'] == '/src')['Source'] = str(D['PARENT'])
        def writable(x):
            next(m for m in x['Mounts'] if m['Destination'] == '/src')['RW'] = True
        mutations = (wrong_bind, wrong_mount, writable,
                     lambda x: x.update(Image='sha256:wrong'),
                     lambda x: x.update(Name='/wrong'),
                     lambda x: x['Config'].update(Hostname='wrong'),
                     lambda x: x['Config'].update(Cmd=['bash', '-lc', 'wrong']),
                     lambda x: x['HostConfig'].update(OomKillDisable=0))
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__):
                actual = copy.deepcopy(pristine)
                mutation(actual)
                with self.assertRaises(AssertionError):
                    self.c['check_container'](actual, cell, None)

    def test_source_replacement_rejects_missing_duplicate_and_nonbind(self):
        old = str(D['PARENT']) + ':/src:ro'
        for argv in (['docker', 'create'], ['-v', old, '-v', old], ['--label', old]):
            with self.assertRaises(AssertionError):
                D['replace_source_bind'](argv)

    def test_qualification_namespace_is_not_rebound(self):
        prerequisite = D['ROOT'].parent / '20261018-nemotron35-bolt-lilicorr-width-sweep/contracts.py'
        if not prerequisite.is_file():
            self.skipTest('actual frozen prerequisite paths are present only on inference-host')
        p = runpy.run_path(str(QUALIFIED.parent / 'public-gate.py'))
        before = (p['ROOT'], p['CANDIDATE'], dict(p['ROLES']))
        self.c['configure'](p)
        self.assertEqual(before, (p['ROOT'], p['CANDIDATE'], p['ROLES']))
        self.assertEqual(p['candidate_proof'].__globals__['CANDIDATE'], D['PARENT'])
        self.assertEqual(p['name'].__globals__['ROLES'], before[2])


if __name__ == '__main__':
    unittest.main()
