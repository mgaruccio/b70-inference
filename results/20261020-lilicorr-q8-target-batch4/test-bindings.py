#!/usr/bin/env python3
"""Supplementary CPU-only composition/route/parser/lifecycle checks; no GPU or SDK."""
import contextlib
import copy
import gzip
import io
import json
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / 'run-performance.py'))
BUILD = runpy.run_path(str(HERE / 'build-candidate.py'))
OLD_TEST = runpy.run_path(str(R['BASE_CASE'] / 'test-performance.py'))
OLD_TEST['RunnerTests'].bound.__globals__.update(D=dict(R['B'], **R), S=R['S'], HERE=HERE)


class RuntimeTests(OLD_TEST['RunnerTests']):
    def setUp(self):
        R['configure']()

    def test_native_dispatch_keeps_frozen_gates_and_requires_actual_batch(self):
        calls = []
        public = dict(NATIVE_LOGS=dict(iq2='iq2', q8='q8', router='router'),
                      check_native_log=lambda tag, log: calls.append(('original', tag, log)))
        with mock.patch.dict(R['N'], check=lambda log, kind: calls.append(('private', kind, log))):
            for tag in ('q8', 'iq2', 'router'):
                R['check_native'](public, tag, 'unaltered\n')
            with self.assertRaises(AssertionError):
                R['check_native'](public, 'unknown', '')
        self.assertEqual(calls, [('private', 'q8', 'unaltered\n'),
                                 ('private', 'iq2', 'unaltered\n'),
                                 ('original', 'router', 'unaltered\n')])


class SourceBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.parent, self.seed, self.candidate = (root / name for name in ('parent', 'seed', 'candidate'))
        for path in (self.parent, self.seed, self.candidate):
            path.mkdir()
            (path / 'UPSTREAM_COMMIT').write_text('b6334e37e6ef286fc5091a41b6561b2113d915fb\n')
        self.g = g = R['S']['proof'].__globals__
        pins = g['load_pins']()
        original = dict(pins['original'], other='frozen')
        seed, candidate = dict(original, **pins['seed']), dict(original, **pins['candidate'])
        self.maps = {self.parent: original, self.seed: seed, self.candidate: candidate}
        parent_bins = {name: 'original-' + name for name in g['ARTIFACTS']}
        seed_bins = dict(parent_bins, **{'libggml-sycl.so.0': 'passed-batch4'})
        candidate_bins = dict(parent_bins, **{'libggml-sycl.so.0': 'target-q8'})
        self.bins = {self.parent: parent_bins, self.seed: seed_bins, self.candidate: candidate_bins}
        self.root = root / 'experiment'
        self.attempt = self.root / 'build-attempt-r1'
        self.attempt.mkdir(parents=True)
        base_case = root / 'baseline'
        base_attempt = base_case / 'build-attempt-r1'
        base_attempt.mkdir(parents=True)
        self.save(base_attempt / 'candidate-source-after.json', dict(parent=original, candidate=seed))
        self.save(base_attempt / 'candidate-binaries.json', seed_bins)
        fake_base = dict(proof=lambda: None, patch_stats=lambda: dict(sha256=pins['seed_patch_sha256']),
                         check_sources=lambda parent: dict(parent=self.maps[Path(parent)]))
        patch = mock.patch.dict(g, ROOT=self.root, PARENT=self.parent, SEED=self.seed, CANDIDATE=self.candidate,
                                BASE_CASE=base_case, BASE=fake_base,
                                source_map=lambda path: self.maps[Path(path)],
                                binaries=lambda path: self.bins[Path(path)])
        patch.start()
        self.addCleanup(patch.stop)
        source = g['check_sources'](self.parent, self.candidate)
        for name, value in (('candidate-source-before.json', source), ('candidate-source-after.json', source),
                            ('seed-source-before.json', seed), ('seed-source-after.json', seed),
                            ('parent-binaries.json', parent_bins), ('candidate-binaries.json', candidate_bins),
                            ('seed-built-before.json', seed_bins), ('seed-built-after.json', seed_bins),
                            ('graph-before.json', dict(frozen='graph')), ('graph-after.json', dict(frozen='graph'))):
            self.save(self.attempt / name, value)
        (self.attempt / 'exit-code.txt').write_text('0\n')
        (self.attempt / 'post-build-verification.log').write_text(g['BUILD_MARKER'] + '\n')
        self.fresh = mock.Mock(return_value=copy.deepcopy(dict(source=original, binaries=parent_bins)))
        self.qualified = dict(CANDIDATE=self.parent, candidate_proof=self.fresh)

    @staticmethod
    def save(path, value):
        Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

    def test_three_seed_files_five_original_files_one_library(self):
        result = self.g['proof'](self.qualified)
        self.assertEqual(result['source'], self.maps[self.candidate])
        self.assertEqual(result['changed_binaries'], ['libggml-sycl.so.0'])
        self.assertEqual(len(self.g['SOURCE_SCOPE']), 5)
        self.assertEqual(self.g['check_seed'](self.seed, self.candidate)['changed_relative_to_seed'], sorted(self.g['SEED_SOURCE_FILES']))
        self.fresh.assert_called_once()

    def test_whole_source_drift_and_protected_library_fail_closed(self):
        for path in (self.parent, self.seed, self.candidate):
            self.maps[path]['unexpected'] = 'drift'
            with self.subTest(path=path), self.assertRaises(AssertionError):
                self.g['proof'](self.qualified)
            del self.maps[path]['unexpected']
        self.bins[self.candidate]['llama-server'] = 'wrong'
        with self.assertRaises(AssertionError):
            self.g['proof'](self.qualified)

    def test_source_build_graph_and_exit_records_are_mandatory(self):
        for name, bad in (('candidate-source-after.json', {}), ('seed-source-before.json', {}),
                          ('graph-after.json', dict(changed=True)), ('seed-built-after.json', {})):
            path = self.attempt / name
            previous = path.read_bytes()
            self.save(path, bad)
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.g['proof'](self.qualified)
            path.write_bytes(previous)
        (self.attempt / 'exit-code.txt').write_text('1\n')
        with self.assertRaises(AssertionError):
            self.g['proof'](self.qualified)

    def test_seed_cli_is_unwrapped_actual_map(self):
        with mock.patch('sys.argv', ['source-proof.py', '--seed-sources']), contextlib.redirect_stdout(io.StringIO()) as output:
            self.g['main']()
        self.assertEqual(json.loads(output.getvalue()), self.maps[self.seed])

    def test_frozen_renderer_and_actual_postcheck(self):
        text = BUILD['render']()
        subprocess.run(['bash', '-n'], input=text, text=True, check=True)
        self.assertIn('assert len(objects) == 105', text)
        self.assertNotIn('--device', text)
        self.assertNotIn('iq2-packed', text)
        marker = "<<'PY' | tee \"$ATTEMPT/post-build-verification.log\""
        post = text.rsplit(marker, 1)[1].split('\n', 1)[1].split('\nPY', 1)[0]
        with mock.patch.object(runpy, 'run_path', return_value=self.g), \
                mock.patch('sys.argv', ['postcheck', str(self.root), str(self.attempt)]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            exec(compile(post, 'frozen-postcheck', 'exec'), {})
        self.assertIn(self.g['BUILD_MARKER'], output.getvalue())


class NativeMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = R['N']
        raw = HERE.parent / '20261020-lilicorr-iq2-packed/native-perf-r1/VSPF-W3a/raw/base'
        def read_fixture(name):
            path = raw / (name + '.gz')
            if path.exists():
                with gzip.open(path, 'rt') as file:
                    return file.read()
            return (raw / name).read_text()
        cls.original = read_fixture('verify-singletons-q8.log')
        cls.iq2 = read_fixture('verify-singletons-iq2.log')
        # Synthetic parser-only metadata appended to unchanged real passed logs.
        # This fixture is never used as GPU qualification or a candidate result.
        lines = []
        routes = cls.n['OLD_FROZEN']['route_records'](cls.original)
        for ident, key in sorted(cls.n['expected_q8_cases']().items()):
            k, m, _ = cls.n['OLD_FROZEN']['dimensions'](key)
            parent = [row for row in routes if row['dst_name'] == ident + '-wide']
            assert len(parent) == 1
            blocks = k // 32
            wg = 8 if blocks >= 64 else 4 if blocks >= 32 else 2 if blocks >= 16 else 1
            lines.append(f"BOLT_Q8_VERIFY_BATCH4 weight={parent[0]['weight']} dst_name={ident}-wide n=4 row_low=0 row_high={m} rowpairs={(m+1)//2} wg={wg} physical_launches=1")
        cls.lines = lines
        cls.fixture = cls.original + '\n' + '\n'.join(lines) + '\n'

    def test_existing_numeric_routes_and_bytes_plus_seven_physical_records(self):
        result = self.n['check'](self.fixture, 'q8')
        self.assertEqual((result['cases'], result['byte_exact_cases'], result['physical_n4_launches']), (82, 54, 7))
        iq2 = self.n['check'](self.iq2, 'iq2')
        self.assertEqual((iq2['cases'], iq2['byte_exact_cases']), (34, 16))

    def test_unmodified_old_logs_cannot_qualify_as_new_q8_candidate(self):
        with self.assertRaises(AssertionError):
            self.n['check'](self.original, 'q8')

    def test_physical_metadata_missing_duplicate_negative_and_forged_fields_fail(self):
        variants = [self.lines[:-1], self.lines + [self.lines[0]]]
        for old, new in (('physical_launches=1', 'physical_launches=4'), ('n=4', 'n=3'),
                         ('row_low=0', 'row_low=1'), ('wg=8', 'wg=4'), ('-wide', '-s0'),
                         ('-n4-', '-n3-'), ('weight=blk.1.', 'weight=wrong.')):
            variants.append([self.lines[0].replace(old, new)] + self.lines[1:])
        variants.append([self.lines[0] + ' extra=1'] + self.lines[1:])
        for lines in variants:
            fixture = self.original + '\n' + '\n'.join(lines) + '\n'
            with self.subTest(first=lines[0]), self.assertRaises(AssertionError):
                self.n['check'](fixture, 'q8')

    def test_new_metadata_cannot_waive_byte_mismatch_or_escape_iq2_scope(self):
        bad = self.fixture.replace('required=1 mismatches=0', 'required=1 mismatches=1', 1)
        self.assertNotEqual(bad, self.fixture)
        with self.assertRaises(AssertionError):
            self.n['check'](bad, 'q8')
        with self.assertRaises(AssertionError):
            self.n['check'](self.iq2 + '\n' + self.lines[0] + '\n', 'iq2')


if __name__ == '__main__':
    unittest.main(verbosity=2)
