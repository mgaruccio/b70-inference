#!/usr/bin/env python3
"""CPU-only composition/build/lifecycle bindings; no SDK, remote or GPU use."""
import contextlib
import copy
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
D = dict(R['B'], **R)
OLD_TEST['RunnerTests'].bound.__globals__.update(D=D, S=R['S'], HERE=HERE)


class RuntimeTests(OLD_TEST['RunnerTests']):
    def setUp(self):
        R['configure']()


class SourceBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.parent, self.seed, self.candidate = (root / x for x in ('parent', 'seed', 'candidate'))
        for p in (self.parent, self.seed, self.candidate):
            p.mkdir()
            (p / 'UPSTREAM_COMMIT').write_text('b6334e37e6ef286fc5091a41b6561b2113d915fb\n')
        g = self.g = R['S']['proof'].__globals__
        pins = g['load_pins']()
        original = {'ggml/src/ggml-sycl/' + k: v for k, v in pins['qualified_sycl_sha256'].items()}
        original.update({'UPSTREAM_COMMIT': 'marker', 'ggml/src/ggml-common.h': pins['codebook_sha256'], 'other': 'frozen'})
        seed = dict(original)
        for name in g['SOURCE_SCOPE']: seed[name] = 'seed-' + name
        seed[g['SEED_SOURCE_FILE']] = pins['batch4_mmvq_sha256']
        candidate = dict(seed, **{g['SEED_SOURCE_FILE']: pins['packed_mmvq_sha256']})
        self.maps = {self.parent: original, self.seed: seed, self.candidate: candidate}
        parent_bins = {k: 'original-' + k for k in g['ARTIFACTS']}
        seed_bins = dict(parent_bins, **{'libggml-sycl.so.0': 'batch4-sycl'})
        candidate_bins = dict(parent_bins, **{'libggml-sycl.so.0': 'packed-sycl'})
        self.bins = {self.parent: parent_bins, self.seed: seed_bins, self.candidate: candidate_bins}
        self.root = root / 'experiment'; self.attempt = self.root / 'build-attempt-r1'
        self.attempt.mkdir(parents=True)
        baseline_case = root / 'baseline'; baseline_attempt = baseline_case / 'build-attempt-r1'
        baseline_attempt.mkdir(parents=True)
        self.save(baseline_attempt / 'candidate-source-after.json', {'parent': original, 'candidate': seed})
        self.save(baseline_attempt / 'candidate-binaries.json', seed_bins)
        fake_base = dict(proof=lambda: None, patch_stats=lambda: {'sha256': pins['batch4_patch_sha256']},
                         check_sources=lambda parent: {'parent': self.maps[Path(parent)]})
        patch = mock.patch.dict(g, ROOT=self.root, PARENT=self.parent, SEED=self.seed, CANDIDATE=self.candidate,
                                BASE_CASE=baseline_case, BASE=fake_base,
                                source_map=lambda path: self.maps[Path(path)],
                                binaries=lambda path: self.bins[Path(path)])
        patch.start(); self.addCleanup(patch.stop)
        sources = g['check_sources'](self.parent, self.candidate)
        for name, value in (('candidate-source-before.json', sources), ('candidate-source-after.json', sources),
                            ('seed-source-before.json', seed), ('seed-source-after.json', seed),
                            ('parent-binaries.json', parent_bins), ('candidate-binaries.json', candidate_bins),
                            ('seed-built-before.json', seed_bins), ('seed-built-after.json', seed_bins),
                            ('graph-before.json', {'frozen': 'graph'}), ('graph-after.json', {'frozen': 'graph'})):
            self.save(self.attempt / name, value)
        (self.attempt / 'exit-code.txt').write_text('0\n')
        (self.attempt / 'post-build-verification.log').write_text(g['BUILD_MARKER'] + '\n')
        self.fresh = mock.Mock(return_value=copy.deepcopy(dict(source=original, binaries=parent_bins)))
        self.qualified = dict(CANDIDATE=self.parent, candidate_proof=self.fresh)

    @staticmethod
    def save(path, value):
        Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')

    def test_fresh_frozen_parent_seed_single_source_and_single_library(self):
        result = self.g['proof'](self.qualified)
        self.assertEqual(result['source'], self.maps[self.candidate])
        self.assertEqual(result['changed_binaries'], ['libggml-sycl.so.0'])
        self.fresh.assert_called_once()
        self.assertEqual(self.g['check_seed'](self.seed, self.candidate)['changed_relative_to_seed'], [self.g['SEED_SOURCE_FILE']])

    def test_seed_parent_candidate_drift_and_protected_library_fail_closed(self):
        for path in (self.parent, self.seed, self.candidate):
            with self.subTest(path=path):
                self.maps[path]['unexpected'] = 'drift'
                with self.assertRaises(AssertionError): self.g['proof'](self.qualified)
                del self.maps[path]['unexpected']
        self.bins[self.candidate]['llama-server'] = 'wrong'
        with self.assertRaises(AssertionError): self.g['proof'](self.qualified)

    def test_source_graph_and_exit_records_remain_mandatory(self):
        for name, bad in (('candidate-source-after.json', {}), ('seed-source-before.json', {}),
                          ('graph-after.json', {'changed': True}), ('seed-built-after.json', {})):
            p = self.attempt / name; previous = p.read_bytes(); self.save(p, bad)
            with self.subTest(name=name), self.assertRaises(AssertionError): self.g['proof'](self.qualified)
            p.write_bytes(previous)
        (self.attempt / 'exit-code.txt').write_text('1\n')
        with self.assertRaises(AssertionError): self.g['proof'](self.qualified)

    def test_actual_seed_cli_emits_unwrapped_map(self):
        with mock.patch('sys.argv', ['source-proof.py', '--seed-sources']), contextlib.redirect_stdout(io.StringIO()) as out:
            self.g['main']()
        self.assertEqual(json.loads(out.getvalue()), self.maps[self.seed])

    def test_frozen_template_render_and_actual_postcheck(self):
        text = BUILD['render']()
        subprocess.run(['bash', '-n'], input=text, text=True, check=True)
        self.assertNotIn('source-proof.py" --built > "$ATTEMPT/host-proof.json"', text)
        self.assertIn('assert len(objects) == 105', text)
        self.assertNotIn('--device', text)
        marker = "<<'PY' | tee \"$ATTEMPT/post-build-verification.log\""
        post = text.rsplit(marker, 1)[1].split('\n', 1)[1].split('\nPY', 1)[0]
        with mock.patch.object(runpy, 'run_path', return_value=self.g), \
                mock.patch('sys.argv', ['postcheck', str(self.root), str(self.attempt)]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            exec(compile(post, 'build-candidate.generated.sh', 'exec'), {})
        self.assertIn(self.g['BUILD_MARKER'], output.getvalue())
        self.assertEqual(json.loads((self.attempt / 'candidate-binaries.json').read_text()), self.bins[self.candidate])


if __name__ == '__main__':
    unittest.main(verbosity=2)
