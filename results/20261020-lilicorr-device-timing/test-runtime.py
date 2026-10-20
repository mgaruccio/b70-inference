#!/usr/bin/env python3
"""CPU-only lifecycle binding checks. No network, Docker or GPU operations."""
import copy
import inspect
import json
import runpy
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-profile.py'))
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'


def load(path):
    return json.loads(Path(path).read_text())


class RuntimeTests(unittest.TestCase):
    def bound(self):
        runtime, analysis, c = D['bind']()
        p = dict(OUT=QUALIFIED, C={}, CANDIDATE=D['S']['PARENT'],
                 make_create=lambda cell, _h: load(QUALIFIED / cell / 'create-argv.json'))
        c['configure'](p)
        return runtime, analysis, c, p

    def record(self, c, cell):
        record = load(c['qualified_dir'](c['role_for'](cell)) / 'container-before.json')
        record.update(Name='/' + c['name'](cell), Id='b' * 64)
        record['Config']['Hostname'] = 'b' * 12
        record['Config']['Cmd'] = c['make_create'](cell, None)[-3:]
        if c['role_for'](cell) == 'W3':
            record['Config']['Env'].append(D['PROFILE_ENV'] + '=1')
            changes = {D['Q']['old_draft_bind'](): D['Q']['candidate_draft_bind'](),
                       str(D['S']['PARENT']) + ':/src:ro': str(D['S']['CANDIDATE']) + ':/src:ro'}
            record['HostConfig']['Binds'] = [changes.get(x, x) for x in record['HostConfig']['Binds']]
            record['HostConfig']['Binds'].append(D['QUALIFIER_BIND'])
            record['Mounts'] = D['Q']['_expected_mounts'](record, cell)
            next(m for m in record['Mounts'] if m['Destination'] == '/src')['Source'] = str(D['S']['CANDIDATE'])
            record['Mounts'].append(dict(Type='bind', Source=str(D['ROOT'].parent / '20261020-lilicorr-iq2-batch4'),
                                         Destination='/iq2-batch4', Mode='ro', RW=False, Propagation='rprivate'))
        return record

    def test_private_namespaces_original_ancestry_and_role_isolation(self):
        runtime, analysis, c, p = self.bound()
        original = copy.deepcopy(p)
        self.assertIs(c['configure'].__globals__, c)
        self.assertIs(analysis['analyze'].__globals__['C'], c)
        self.assertIs(runtime['capture'].__globals__['C'], c)
        for method in ('pid', 'verify_restored', '__init__', 'begin', 'run_job', 'preserve_native'):
            function = getattr(runtime['Runtime'], method)
            if 'C' in function.__globals__:
                self.assertIs(function.__globals__['C'], c)
            else:
                self.assertIs(inspect.getclosurevars(function).nonlocals['contract' if method == 'run_job' else 'c'], c)
        for cell in c['CELLS']:
            argv = c['make_create'](cell, None)
            base = p['make_create'](c['base_cell'](cell), None)
            active = c['role_for'](cell) == 'W3'
            self.assertEqual(c['server_argv'](argv), c['server_argv'](base))
            self.assertIn(str(D['S']['CANDIDATE'] if active else D['S']['PARENT']) + ':/src:ro', argv)
            self.assertEqual(active, D['QUALIFIER_BIND'] in argv)
            self.assertEqual(active, D['NEW_QUALIFIER'] in argv[-1])
            env = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == '-e']
            self.assertEqual(active, D['PROFILE_ENV'] + '=1' in env)
            self.assertFalse(any(v.startswith(('LLAMA_TRACE=', 'BOLT_VERIFY_CAPTURE_DIR=')) for v in env))
            if active:
                i = argv.index(D['QUALIFIER_BIND'])
                self.assertEqual(argv[i + 2:i + 4], ['bash', '-lc'])
            self.assertNotIn('W7', cell)
        self.assertEqual(p, original)
        _r, _a, other, _p = self.bound()
        self.assertIsNot(c, other)

    def test_exact_observation_environment_and_mounts_without_rewriting(self):
        _runtime, _analysis, c, _p = self.bound()
        for cell in c['CELLS']:
            record = self.record(c, cell)
            original = copy.deepcopy(record)
            self.assertTrue(c['check_container'](record, cell, None))
            self.assertEqual(record, original)
            for field in ('env', 'source', 'write', 'id', 'command', 'host', 'qualifier'):
                bad = copy.deepcopy(record)
                if field == 'env': bad['Config']['Env'].append('LLAMA_TRACE=1')
                if field == 'source': next(m for m in bad['Mounts'] if m['Destination'] == '/src')['Source'] = '/wrong'
                if field == 'write': next(m for m in bad['Mounts'] if m['Destination'] == '/src')['RW'] = True
                if field == 'id': bad['Config']['Hostname'] = 'wrong'
                if field == 'command': bad['Config']['Cmd'][-1] += '; true'
                if field == 'host': bad['HostConfig']['Privileged'] = True
                if field == 'qualifier': bad['HostConfig']['Binds'].append('/wrong:/iq2-batch4:rw')
                with self.subTest(cell=cell, field=field), self.assertRaises(AssertionError):
                    c['check_container'](bad, cell, None)

    def test_qualifier_transform_is_minimal_reversible_and_fail_closed(self):
        _runtime, _analysis, c, _p = self.bound()
        argv = c['make_create']('VSPF-W3a', None)
        with self.assertRaises(AssertionError):
            D['qualifier_argv'](argv)
        bad = [*argv[:-1], argv[-1].replace(D['NEW_QUALIFIER'], 'wrong')]
        with self.assertRaises(AssertionError):
            D['qualifier_argv'](bad)

    def test_first_public_guard_uses_current_request_and_preserves_old_scope(self):
        globals_ = D['check_first_job_scope'].__globals__
        seen = []
        with mock.patch.dict(globals_, ORIGINAL_SCOPE_CHECK=lambda _s: dict(passed=True, request_start_line=2)):
            with mock.patch.object(globals_['runpy'], 'run_path', return_value={
                    'check_request': lambda text: seen.append(text) or dict(status='ok')}):
                proof = D['check_first_job_scope']('old startup\ncurrent request\n')
        self.assertEqual(seen, ['current request'])
        self.assertTrue(proof['passed'] and proof['bounded_device']['status'] == 'ok')
        with mock.patch.dict(globals_, ORIGINAL_SCOPE_CHECK=lambda _s: dict(passed=True, request_start_line=2)):
            with mock.patch.object(globals_['runpy'], 'run_path', return_value={
                    'check_request': lambda _text: dict(status='failed')}):
                with self.assertRaises(AssertionError):
                    D['check_first_job_scope']('old startup\ncurrent request\n')

    def test_independent_reread_uses_private_native_checker_only_for_w3(self):
        runtime, analysis, c = D['Q']['bind']()
        c.update(ROOT=D['ROOT'], OUT=D['OUT'], PREFIX=D['PREFIX'])
        seen = []
        def original(public, _h, _out, cell):
            seen.append((cell, public))
            return dict(passed=True)
        analysis['check_runtime_evidence'] = original
        namespace = dict(result=(runtime, analysis, c))
        exec('def bind():\n    return result\n', namespace)
        with mock.patch.dict(D['CTX'], bind=namespace['bind']):
            _runtime, bound, _c = D['bind']()
        public = dict(NATIVE_LOGS={'iq2': 'iq2'}, check_native_log=lambda *_args: None)
        before = copy.copy(public)
        for cell in c['CELLS']:
            self.assertTrue(bound['check_runtime_evidence'](public, None, Path('.'), cell)['passed'])
        for cell, actual in seen:
            if c['role_for'](cell) == 'A':
                self.assertIs(actual, public)
            else:
                self.assertIsNot(actual, public)
                self.assertIsNot(actual['check_native_log'], public['check_native_log'])
                self.assertIs(actual['NATIVE_LOGS'], public['NATIVE_LOGS'])
        self.assertEqual(public, before)
        self.assertIs(bound['analyze'].__globals__['check_runtime_evidence'], bound['check_runtime_evidence'])

    def test_profile_pid_and_acceptance_stay_exact(self):
        checker = D['CTX']['_FIRST_RUN']['check_profile_pid']
        for cell in ('VSPF-A1', 'VSPF-W3a'):
            record = dict(pid=1, origin='/proc/1/environ', profile_environment_origin='/proc/1/environ',
                          profile_environment=[[D['PROFILE_ENV'], '1']] if 'W3' in cell else [])
            self.assertTrue(checker(record, cell))
            record['profile_environment'] = [] if 'W3' in cell else [[D['PROFILE_ENV'], '1']]
            with self.assertRaises(AssertionError): checker(record, cell)
        for point, counts in (('short', (129, 84)), ('long', (133, 82))):
            row = dict(cell='VSPF-W3a', point=point, timings=dict(zip(('draft_n', 'draft_n_accepted'), counts)))
            self.assertIs(D['BATCH']['check_acceptance'](row), row)
            row['timings']['draft_n_accepted'] -= 1
            with self.assertRaises(AssertionError): D['BATCH']['check_acceptance'](row)


if __name__ == '__main__':
    unittest.main(verbosity=2)
