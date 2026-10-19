#!/usr/bin/env python3
"""CPU-only namespace, bind, and candidate-file contract tests.

No Docker, native process, socket, API, /proc, /sys, GPU, or DRM access is allowed.
The inherited singleton controller owns the lead-only lifecycle.
"""
import copy
import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-performance.py'))
BASE = D['BASE']
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.analysis, self.c = D['bind']()
        self.public = dict(
            OUT=QUALIFIED,
            C={},
            make_create=lambda cell, _h: json.loads(
                (QUALIFIED / cell / 'create-argv.json').read_text()),
        )
        self.c['configure'](self.public)

    def fixture(self, cell):
        """Build a synthetic inspect result from saved qualified input only."""
        base = self.c['qualified_dir'](self.c['role_for'](cell))
        actual = json.loads((base / 'container-before.json').read_text())
        actual.update(Name='/' + self.c['name'](cell), Id='b' * 64)
        actual['Config']['Hostname'] = 'b' * 12
        actual['Config']['Cmd'] = self.c['make_create'](cell, None)[-3:]
        if cell.startswith('VSPF-W3'):
            old = D['old_draft_bind']()
            new = D['candidate_draft_bind']()
            actual['HostConfig']['Binds'] = [new if item == old else item
                                             for item in actual['HostConfig']['Binds']]
            for mount in actual['Mounts']:
                if mount['Destination'] == '/draft':
                    mount['Source'] = str(D['CANDIDATE'])
                    mount['Destination'] = D['DRAFT_DEST']
        actual['HostConfig']['Binds'].reverse()
        actual['HostConfig']['OomKillDisable'] = False
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
        untouched = runpy.run_path(str(BASE / 'controller.py'))
        self.assertNotEqual(untouched['C']['PREFIX'], c['PREFIX'])
        self.assertEqual(untouched['C']['OUT'].name, 'run-r3')

    def test_A_is_unchanged_and_W3_changes_only_name_and_draft_bind(self):
        for cell in self.c['CELLS']:
            base = self.public['make_create'](self.c['base_cell'](cell), None)
            actual = self.c['make_create'](cell, None)
            differences = [(i, a, b) for i, (a, b) in enumerate(zip(base, actual)) if a != b]
            expected_count = 1 if self.c['role_for'](cell) == 'A' else 2
            self.assertEqual(expected_count, len(differences), cell)
            self.assertEqual(base.index('--name') + 1, differences[0][0])
            self.assertEqual(base[-3:], actual[-3:])
            self.assertEqual(self.c['server_argv'](base), self.c['server_argv'](actual))
            if self.c['role_for'](cell) == 'A':
                self.assertNotIn(D['old_draft_bind'](), actual)
                self.assertNotIn(D['candidate_draft_bind'](), actual)
            else:
                self.assertEqual(1, actual.count(D['candidate_draft_bind']()))
                self.assertNotIn(D['old_draft_bind'](), actual)
                bind_index = actual.index(D['candidate_draft_bind']())
                self.assertEqual(base.index(D['old_draft_bind']()), bind_index)
                self.assertEqual(D['DRAFT_DEST'], actual[bind_index].split(':', 2)[1])

    def test_candidate_file_pin_checks_size_hash_and_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.gguf'
            payload = b'fp16 fixture\x00' * 17
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(
                dict(path=str(path), size_bytes=len(payload), sha256=digest),
                D['file_pin'](path, len(payload), digest))
            with self.assertRaises(AssertionError):
                D['file_pin'](path, len(payload) + 1, digest)
            with self.assertRaises(AssertionError):
                D['file_pin'](path, len(payload), '0' * 64)
            link = Path(directory) / 'candidate-link.gguf'
            link.symlink_to(path)
            with self.assertRaises(AssertionError):
                D['file_pin'](link, len(payload), digest)

    def test_container_validation_inspects_candidate_file_bind_and_does_not_mutate(self):
        for cell in self.c['CELLS']:
            actual = self.fixture(cell)
            before = copy.deepcopy(actual)
            self.assertTrue(self.c['check_container'](actual, cell, None))
            self.assertEqual(before, actual)

        pristine = self.fixture('VSPF-W3a')
        mutations = (
            lambda x: x['HostConfig']['Binds'].append(D['old_draft_bind']()),
            lambda x: next(m for m in x['Mounts']
                           if m['Destination'] == D['DRAFT_DEST']).__setitem__('RW', True),
            lambda x: next(m for m in x['Mounts']
                           if m['Destination'] == D['DRAFT_DEST']).__setitem__('Source', str(D['DRAFT_DIR'])),
            lambda x: x.update(Image='sha256:wrong'),
            lambda x: x.update(Name='/wrong'),
            lambda x: x['Config'].update(Hostname='wrong'),
            lambda x: x['Config'].update(Cmd=['bash', '-lc', 'wrong']),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(pristine)
                mutation(changed)
                with self.assertRaises(AssertionError):
                    self.c['check_container'](changed, 'VSPF-W3a', None)

    def test_draft_replacement_rejects_missing_duplicate_and_nonbind(self):
        old = D['old_draft_bind']()
        for argv in (['docker', 'create'], ['-v', old, '-v', old], ['--label', old]):
            with self.assertRaises(AssertionError):
                D['replace_draft_bind'](argv)

    def test_candidate_proof_is_saved_in_private_adapter_hash_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.gguf'
            payload = b'private-pin'
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(path, Path(D['candidate_file_proof'](
                path, len(payload), digest)['path']))

    def test_qualification_namespace_is_not_rebound(self):
        prerequisite = D['ROOT'].parent / '20261018-nemotron35-bolt-lilicorr-width-sweep/contracts.py'
        if not prerequisite.is_file():
            self.skipTest('actual frozen prerequisite paths are present only on inference-host')
        p = runpy.run_path(str(QUALIFIED.parent / 'public-gate.py'))
        before = (p['ROOT'], p['CANDIDATE'], dict(p['ROLES']))
        self.c['configure'](p)
        self.assertEqual(before, (p['ROOT'], p['CANDIDATE'], p['ROLES']))
        self.assertEqual(p['CANDIDATE'], D['QUALIFIED_SOURCE'])
        self.assertEqual(p['candidate_proof'].__globals__['CANDIDATE'], D['QUALIFIED_SOURCE'])
        self.assertEqual(p['name'].__globals__['ROLES'], before[2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
