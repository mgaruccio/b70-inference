#!/usr/bin/env python3
"""CPU-only expected-profile-binding checks; fixtures are not live evidence."""
import copy
import json
import runpy
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-profile.py'))
Q = runpy.run_path(str(D['Q8'] / 'test-performance.py'))


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.prior, self.runtime, self.c = D['bind']()
        qualified = Q['QUALIFIED']
        self.public = dict(OUT=qualified, C={}, make_create=lambda cell, h:
            json.loads((qualified / cell / 'create-argv.json').read_text()))
        self.c['configure'](self.public)
        self.template = Q['BindingTests']('test_container_validation_does_not_rewrite_observations')
        self.template.c = self.c

    def fixture(self, cell):
        return D['expected_reference'](self.template.fixture(cell), cell)

    def test_namespace_and_output_scope(self):
        self.assertEqual(self.c['CELLS'], D['CELLS'])
        self.assertEqual(len(self.c['planned_names']()), 2)
        for fn in (self.runtime['Runtime'].__init__, self.runtime['Runtime'].begin,
                   self.runtime['Runtime'].end, self.runtime['Runtime'].timed_request,
                   self.runtime['Runtime'].restore, self.runtime['_BOOT_K']['capture']):
            self.assertIs(fn.__globals__['C'], self.c)
        for cell in D['CELLS']:
            output = D['extra_mounts'](cell)[1][0]
            self.assertFalse(output.is_relative_to(D['OUT'] / cell))

    def test_only_declared_wrap_mounts_and_prior_deltas(self):
        for cell in D['CELLS']:
            old = self.public['make_create'](self.c['base_cell'](cell), None)
            actual = self.c['make_create'](cell, None)
            reverted = list(actual)
            self.assertEqual(reverted[-8:-4], [item for src, dst, mode in D['extra_mounts'](cell)
                                              for item in ('-v', f'{src}:{dst}:{mode}')])
            del reverted[-8:-4]
            reverted[-1] = reverted[-1].replace(D['WRAPPED'].format(session=D['session_name'](cell)), D['SERVER'])
            reverted[reverted.index('--name') + 1] = old[old.index('--name') + 1]
            candidate_bind = str(self.prior['CANDIDATE']) + ':/src:ro'
            reverted[reverted.index(candidate_bind)] = str(self.prior['PARENT']) + ':/src:ro'
            self.assertEqual(reverted, old)

    def test_strict_expected_mounts_without_rewriting_records(self):
        for cell in D['CELLS']:
            actual = self.fixture(cell)
            before = copy.deepcopy(actual)
            self.assertTrue(self.c['check_container'](actual, cell, None))
            self.assertEqual(actual, before)
            for destination in ('/src', '/tools', '/profile'):
                bad = copy.deepcopy(actual)
                next(m for m in bad['Mounts'] if m['Destination'] == destination)['RW'] ^= True
                with self.assertRaises(AssertionError):
                    self.c['check_container'](bad, cell, None)
            bad = copy.deepcopy(actual)
            bad['Config']['Cmd'][-1] += ' --changed-model-argument'
            with self.assertRaises(AssertionError):
                self.c['check_container'](bad, cell, None)

    def test_collection_is_scoped_and_paused_even_on_request_failure(self):
        for fail in (False, True):
            calls = []
            class Runtime:
                def command(self, *args, **kwargs):
                    calls.append(args)
                    self_timeout = kwargs['timeout']
                    assert self_timeout == 30
                def timed_request(self, *args):
                    calls.append(args)
                    if fail:
                        raise RuntimeError('request failed')
                    return 'actual result'
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'request failed'):
                    D['collect_request'](Runtime(), self.c, 'VSPF-A1')
            else:
                self.assertEqual(D['collect_request'](Runtime(), self.c, 'VSPF-A1'), 'actual result')
            session = D['session_name']('VSPF-A1')
            name = self.c['name']('VSPF-A1')
            self.assertEqual(calls, [
                ('docker', 'exec', name, '/tools/unitrace', '--resume', session),
                ('VSPF-A1', 'short', 'profile', 0),
                ('docker', 'exec', name, '/tools/unitrace', '--pause', session)])
            self.assertTrue(session.isalnum())
            self.assertNotEqual(session, D['session_name']('VSPF-W3a'))

    def test_duplicate_or_missing_server_is_rejected(self):
        for body in ('echo no-server', D['SERVER'] + D['SERVER']):
            with self.assertRaises(AssertionError):
                D['wrap_argv'](['docker', 'create', 'image', 'bash', '-lc', body], 'VSPF-A1')


if __name__ == '__main__':
    unittest.main()
