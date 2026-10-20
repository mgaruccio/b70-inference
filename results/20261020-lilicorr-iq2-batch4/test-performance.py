#!/usr/bin/env python3
"""CPU adapter preparation only; no Docker, GPU, network or service execution."""
import copy
import json
import runpy
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-performance.py'))
S = D['S']
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'
NATIVE_TEST = runpy.run_path(str(HERE / 'test-native-batch4.py'))


def load(path):
    return json.loads(Path(path).read_text())


class RunnerTests(unittest.TestCase):
    def bound(self):
        runtime, analysis, c = D['bind']()
        p = dict(OUT=QUALIFIED, C={}, CANDIDATE=S['PARENT'],
                 make_create=lambda cell, _h: load(QUALIFIED / cell / 'create-argv.json'))
        c['configure'](p)
        return runtime, analysis, c, p

    def test_private_globals_and_role_isolation(self):
        runtime, analysis, c, p = self.bound()
        original = copy.deepcopy(p)
        self.assertIs(c['configure'].__globals__, c)
        self.assertIs(analysis['analyze'].__globals__['C'], c)
        self.assertIs(runtime['capture'].__globals__['C'], c)
        for method in ('pid', 'verify_restored', '__init__', 'begin'):
            self.assertIs(getattr(runtime['Runtime'], method).__globals__['C'], c)
        for cell in c['CELLS']:
            argv = c['make_create'](cell, None)
            base = p['make_create'](c['base_cell'](cell), None)
            active = c['role_for'](cell) == 'W3'
            self.assertEqual(c['server_argv'](argv), c['server_argv'](base))
            self.assertIn(D['source_bind'](S['CANDIDATE'] if active else S['PARENT']), argv)
            self.assertNotIn(D['source_bind'](S['PARENT'] if active else S['CANDIDATE']), argv)
            self.assertEqual(active, D['QUALIFIER_BIND'] in argv)
            self.assertEqual(active, D['NEW_QUALIFIER'] in argv[-1])
            self.assertEqual(not active, D['OLD_QUALIFIER'] in argv[-1])
            # Native qualification intentionally enables capture before the frozen OFF boundary.
            # Serving argv is equal above; no diagnostic opt-in may enter Docker's environment.
            env = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == '-e']
            for flag in ('LLAMA_TRACE=', 'BOLT_VERIFY_CAPTURE_DIR=', 'LLAMA_LILICORR_PROFILE='):
                self.assertFalse(any(value.startswith(flag) for value in env))
            if active:
                i = argv.index(D['QUALIFIER_BIND'])
                self.assertEqual(argv[i + 2:i + 4], ['bash', '-lc'])
            self.assertNotIn('W7', cell)
        self.assertEqual(original, p)
        for name in D['OWN_FILES']:
            self.assertTrue((HERE / name).is_file(), name)
        _r2, _a2, c2, _p2 = self.bound()
        self.assertIsNot(c, c2)

    def test_real_argv_transform_is_reversible_and_fail_closed(self):
        root = HERE.parent / '20261019-lilicorr-q8-drafter/timing-20261020-baseline-r1'
        for cell in ('VSPF-W3a', 'VSPF-W3b'):
            argv = load(root / cell / 'create-argv.json')
            original = copy.deepcopy(argv)
            new = D['derive_candidate_argv'](argv)
            self.assertEqual(original, argv)
            self.assertEqual(len(new), len(argv) + 2)
            self.assertEqual(new[-4], argv[-4])  # IMAGE unchanged.
            for bad in (new, [*argv[:-1], argv[-1].replace(D['OLD_QUALIFIER'], 'wrong')],
                        [x for x in argv if x != D['source_bind'](S['PARENT'])]):
                with self.assertRaises(AssertionError): D['derive_candidate_argv'](bad)

    def record(self, c, cell):
        record = load(c['qualified_dir'](c['role_for'](cell)) / 'container-before.json')
        record.update(Name='/' + c['name'](cell), Id='b' * 64)
        record['Config']['Hostname'] = 'b' * 12
        record['Config']['Cmd'] = c['make_create'](cell, None)[-3:]
        if c['role_for'](cell) == 'W3':
            changes = {D['Q']['old_draft_bind'](): D['Q']['candidate_draft_bind'](),
                       D['source_bind'](S['PARENT']): D['source_bind'](S['CANDIDATE'])}
            record['HostConfig']['Binds'] = [changes.get(x, x) for x in record['HostConfig']['Binds']]
            record['HostConfig']['Binds'].append(D['QUALIFIER_BIND'])
            record['Mounts'] = D['Q']['_expected_mounts'](record, cell)
            next(m for m in record['Mounts'] if m['Destination'] == '/src')['Source'] = str(S['CANDIDATE'])
            record['Mounts'].append(dict(Type='bind', Source=str(D['ROOT']), Destination='/iq2-batch4',
                                         Mode='ro', RW=False, Propagation='rprivate'))
        return record

    def test_exact_container_identity_environment_mounts_and_no_rewriting(self):
        _rt, _a, c, _p = self.bound()
        for cell in c['CELLS']:
            record = self.record(c, cell); original = copy.deepcopy(record)
            self.assertTrue(c['check_container'](record, cell, None))
            self.assertEqual(original, record)
            for field in ('env', 'source', 'write', 'id', 'command', 'host'):
                bad = copy.deepcopy(record)
                if field == 'env': bad['Config']['Env'].append('LLAMA_TRACE=1')
                if field == 'source': next(m for m in bad['Mounts'] if m['Destination'] == '/src')['Source'] = '/wrong'
                if field == 'write': next(m for m in bad['Mounts'] if m['Destination'] == '/src')['RW'] = True
                if field == 'id': bad['Config']['Hostname'] = 'wrong'
                if field == 'command': bad['Config']['Cmd'][-1] += '; true'
                if field == 'host': bad['HostConfig']['Privileged'] = True
                with self.subTest(cell=cell, field=field), self.assertRaises(AssertionError):
                    c['check_container'](bad, cell, None)

    def test_native_dispatch_keeps_frozen_gates_and_requires_actual_batch(self):
        calls = []
        public = dict(NATIVE_LOGS=dict(iq2='iq2', q8='q8', router='router'),
                      check_native_log=lambda tag, log: calls.append((tag, log)))
        for tag in ('q8', 'router'):
            D['check_native'](public, tag, 'unaltered\n')
        self.assertEqual(calls, [('q8', 'unaltered\n'), ('router', 'unaltered\n')])
        D['check_native'](public, 'iq2', NATIVE_TEST['synthetic_batch_fixture']())
        with self.assertRaises(AssertionError):
            D['check_native'](public, 'iq2', NATIVE_TEST['FROZEN_TEST']['synthetic']('iq2'))
        with self.assertRaises(AssertionError): D['check_native'](public, 'unknown', '')
        self.assertEqual(len(calls), 2)

    def test_build_evidence_couples_current_source_and_binaries(self):
        value = dict(source={'changed.cpp': 'compiled'}, binaries={'sycl': 'new'}, parent_binaries={'sycl': 'old'})
        source = dict(candidate=value['source'], parent={'parent.cpp': 'qualified'})
        records = {'candidate-source-before.json': source, 'candidate-source-after.json': copy.deepcopy(source),
                   'candidate-binaries.json': value['binaries'], 'parent-binaries.json': value['parent_binaries']}
        def read(path): return copy.deepcopy(records[Path(path).name])
        with mock.patch.dict(S, proof=lambda: copy.deepcopy(value), source_map=lambda _p: source['parent']):
            self.assertEqual(D['candidate_proof'](read), value)
            for field in ('candidate-source-after.json', 'candidate-binaries.json', 'parent-binaries.json'):
                original = records[field]; records[field] = {'wrong': 'bytes'}
                with self.subTest(field=field), self.assertRaises(AssertionError): D['candidate_proof'](read)
                records[field] = original

    def test_independent_reread_uses_private_native_dispatch_only_for_w3(self):
        runtime, analysis, c = D['Q']['bind']()
        seen = []
        def original(p, _h, _out, cell):
            seen.append((cell, p))
            return {'passed': True}
        analysis['check_runtime_evidence'] = original
        with mock.patch.dict(D['Q'], bind=lambda: (runtime, analysis, c)):
            _r, bound, _c = D['bind']()
        public = dict(NATIVE_LOGS={'iq2': 'iq2'}, check_native_log=lambda *_a: None)
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

    def test_acceptance_is_exact(self):
        for cell in ('VSPF-W3a', 'VSPF-W3b'):
            for point, counts in (('short', (129, 84)), ('long', (133, 82))):
                row = dict(cell=cell, point=point, timings=dict(zip(('draft_n', 'draft_n_accepted'), counts)))
                self.assertIs(D['check_acceptance'](row), row)
                row['timings']['draft_n_accepted'] -= 1
                with self.assertRaises(AssertionError): D['check_acceptance'](row)
        self.assertIsNotNone(D['check_acceptance'](dict(cell='VSPF-A1')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
