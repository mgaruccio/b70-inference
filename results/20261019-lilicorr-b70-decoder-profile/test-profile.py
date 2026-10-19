#!/usr/bin/env python3
"""CPU-only continuation tests; no Docker, GPU, network, or remote mutation."""
import gzip
import hashlib
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYCL = HERE.parent / '20261019-lilicorr-b70-sycl-profile'
D = runpy.run_path(str(HERE / 'run-profile.py'))
S = D['S']
A = runpy.run_path(str(HERE / 'analyze-profile.py'))


def fixture_bytes(name):
    return gzip.decompress((HERE / 'fixtures' / (name + '.gz')).read_bytes())


def materialize_sources(root):
    originals = {}
    for name in S['SOURCE_SCOPE']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        originals[name] = fixture_bytes(name)
        path.write_bytes(originals[name])
    return originals


def apply_patch(root, patch, *flags):
    result = subprocess.run(
        ['patch', '--fuzz=0', '--batch', '-p1', *flags, '-i', str(patch)],
        cwd=root, check=True, capture_output=True, text=True)
    assert not re.search(r'fuzz|offset|failed', (result.stdout + result.stderr).lower())
    return result


def event_line(marker):
    return ('LLAMA_LILICORR_SYCL_PROFILE v=1 stage=gate graph=7 requested=1 '
            'enabled=1 marker=1 graph_nodes=1 marker_name=' + marker)


class PatchAndContractTests(unittest.TestCase):
    def test_build_markers_match_the_inherited_built_proof(self):
        script = (HERE / 'build-candidate.sh').read_text()
        proof = (HERE.parent / '20261019-lilicorr-b70-profile/source-proof.py').read_text()
        for marker in ('LILICORR_PROFILE_BUILD_PASSED', 'LILICORR_PROFILE_DEPENDENCIES_PASSED'):
            self.assertIn(marker, proof)
            self.assertIn(marker, script)
        subprocess.run(['bash', '-n', str(HERE / 'build-candidate.sh')], check=True)

    def test_combined_and_incremental_patches_apply_reverse_and_hash_exactly(self):
        pins = S['load'](HERE / 'source-hashes.json')
        S['check_fixtures']()
        self.assertEqual(hashlib.sha256((HERE / 'lilicorr-profile.patch').read_bytes()).hexdigest(),
                         pins['patch_sha256'])
        self.assertEqual(hashlib.sha256((HERE / 'seed-incremental.patch').read_bytes()).hexdigest(),
                         pins['seed_patch_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_tree, seed_tree = root / 'full', root / 'seed'
            originals = materialize_sources(full_tree)
            for name, data in originals.items():
                path = seed_tree / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            first_patch = SYCL / 'lilicorr-profile.patch'
            apply_patch(seed_tree, first_patch, '--dry-run', '--forward')
            apply_patch(seed_tree, first_patch, '--forward')
            apply_patch(full_tree, HERE / 'lilicorr-profile.patch', '--dry-run', '--forward')
            apply_patch(full_tree, HERE / 'lilicorr-profile.patch', '--forward')
            self.assertEqual(
                {name: hashlib.sha256((full_tree / name).read_bytes()).hexdigest()
                 for name in S['SOURCE_SCOPE']}, pins['candidate'])
            old_line = ('    return tensor && tensor->name[0] && '
                        '(std::strcmp(tensor->name, "lilicorr_lattice") == 0 || '
                        'std::strncmp(tensor->name, "lilicorr.", 9) == 0);')
            new_line = ('    return tensor && tensor->name[0] && '
                        '(std::strcmp(tensor->name, "lilicorr_lattice") == 0 || '
                        'std::strcmp(tensor->name, "inp_noise_embd") == 0 || '
                        'std::strncmp(tensor->name, "lilicorr.", 9) == 0);')
            seed_text = (seed_tree / 'ggml/src/ggml-sycl/ggml-sycl.cpp').read_text()
            full_text = (full_tree / 'ggml/src/ggml-sycl/ggml-sycl.cpp').read_text()
            self.assertEqual(full_text, seed_text.replace(old_line, new_line, 1))
            self.assertEqual(1, full_text.count('std::strcmp(tensor->name, "inp_noise_embd") == 0'))
            self.assertLess(full_text.index('bool ok = ggml_sycl_compute_forward'),
                            full_text.index('stage=op'))
            apply_patch(seed_tree, HERE / 'seed-incremental.patch', '--dry-run', '--forward')
            apply_patch(seed_tree, HERE / 'seed-incremental.patch', '--forward')
            for name in S['SOURCE_SCOPE']:
                self.assertEqual((full_tree / name).read_bytes(), (seed_tree / name).read_bytes())
            apply_patch(seed_tree, HERE / 'seed-incremental.patch', '--dry-run', '--reverse')
            apply_patch(seed_tree, HERE / 'seed-incremental.patch', '--reverse')
            apply_patch(full_tree, HERE / 'lilicorr-profile.patch', '--dry-run', '--reverse')
            apply_patch(full_tree, HERE / 'lilicorr-profile.patch', '--reverse')
            apply_patch(seed_tree, first_patch, '--dry-run', '--reverse')
            apply_patch(seed_tree, first_patch, '--reverse')
            for name, data in originals.items():
                self.assertEqual(data, (full_tree / name).read_bytes())
                self.assertEqual(data, (seed_tree / name).read_bytes())

    def test_exact_new_marker_preserves_old_markers_and_rejects_target_names(self):
        for marker in ('inp_noise_embd', 'lilicorr_lattice', 'lilicorr.out_head'):
            self.assertTrue(A['_valid_marker_name'](marker))
            self.assertEqual(marker, A['event'](event_line(marker))['marker_name'])
        for marker in ('inp_noise_embd_target', 'targetNemotron', 'target.inp_noise_embd'):
            self.assertFalse(A['_valid_marker_name'](marker))
            with self.assertRaises(AssertionError):
                A['event'](event_line(marker))
        self.assertIs(A['request_sycl'].__globals__, A['CURRENT'])
        self.assertIs(A['CURRENT']['_valid_marker_name'], A['_valid_marker_name'])

    def test_actual_paths_source_bind_and_q8_model_constant_are_private_rebound(self):
        runtime, analysis, contract = D['bind']()
        self.assertIsNotNone(runtime)
        self.assertIsNotNone(analysis)
        self.assertEqual(Path('/home/mike/b70-evals/20261019-lilicorr-b70-decoder-profile'), D['ROOT'])
        self.assertEqual(D['ROOT'] / 'profile-r1', D['OUT'])
        self.assertEqual('bolt-lilicorr-b70-decoder-profile-r1-', D['PREFIX'])
        self.assertEqual(D['ROOT'], contract['ROOT'])
        self.assertEqual(D['OUT'], contract['OUT'])
        self.assertEqual(D['PREFIX'], contract['PREFIX'])
        self.assertIs(D['_FIRST_GLOBALS']['S'], D['S'])
        self.assertIs(D['_FIRST_GLOBALS']['_FIRST_GLOBALS']['S'], D['S'])
        self.assertEqual(D['S']['PARENT'], D['_FIRST_RUN']['S']['PARENT'])
        self.assertEqual(D['Q']['CANDIDATE'], D['_FIRST_RUN']['Q']['CANDIDATE'])
        self.assertEqual(D['Q']['CANDIDATE_SIZE'], D['_FIRST_RUN']['Q']['CANDIDATE_SIZE'])
        self.assertEqual(D['Q']['CANDIDATE_SHA256'], D['_FIRST_RUN']['Q']['CANDIDATE_SHA256'])
        source_bind = D['_FIRST_RUN']['_FIRST_RUN']['source_bind']
        self.assertEqual(source_bind(D['S']['PARENT']),
                         str(D['S']['PARENT']) + ':/src:ro')
        self.assertEqual(source_bind(D['S']['CANDIDATE']),
                         str(D['S']['CANDIDATE']) + ':/src:ro')
        for imported in (
            '../20261019-lilicorr-b70-sycl-profile/run-profile.py',
            '../20261019-lilicorr-b70-sycl-profile/analyze-profile.py',
            '../20261019-lilicorr-b70-sycl-profile/source-proof.py',
            '../20261019-lilicorr-b70-profile/run-profile.py',
            '../20261019-lilicorr-b70-profile/analyze-profile.py',
            '../20261019-lilicorr-b70-profile/source-proof.py',
        ):
            self.assertIn(imported, D['OWN_FILES'])

    def test_source_binary_and_model_repins_reject_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, candidate = root / 'parent', root / 'candidate'
            originals = materialize_sources(parent)
            for name, data in originals.items():
                path = candidate / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (parent / 'UPSTREAM_COMMIT').write_text(S['load'](HERE / 'source-hashes.json')['upstream_commit'] + '\n')
            (candidate / 'UPSTREAM_COMMIT').write_text((parent / 'UPSTREAM_COMMIT').read_text())
            apply_patch(candidate, HERE / 'lilicorr-profile.patch', '--forward')
            source_result = S['check_sources'](parent, candidate)
            self.assertEqual(S['SOURCE_SCOPE'], source_result['changed_relative_to_parent'])
            original = candidate / 'common/speculative.cpp'
            changed = original.read_bytes()
            original.write_bytes(bytes([changed[0] ^ 1]) + changed[1:])
            with self.assertRaises(AssertionError):
                S['check_sources'](parent, candidate)
            original.write_bytes(changed)
            for name in S['ARTIFACTS']:
                for tree in (parent, candidate):
                    path = tree / 'build-sycl/bin' / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((name + ('-changed' if tree == candidate and
                                               name in S['CHANGED_BINARIES'] else '')).encode())
            parent_bins, candidate_bins = S['binaries'](parent), S['binaries'](candidate)
            self.assertEqual(S['CHANGED_BINARIES'],
                             sorted(name for name in S['ARTIFACTS']
                                    if parent_bins[name] != candidate_bins[name]))
            rogue = candidate / 'build-sycl/bin/llama-server'
            rogue.write_bytes(b'rogue')
            candidate_bins = S['binaries'](candidate)
            self.assertNotEqual(S['CHANGED_BINARIES'],
                                sorted(name for name in S['ARTIFACTS']
                                       if parent_bins[name] != candidate_bins[name]))
            model = root / 'q8.gguf'
            model.write_bytes(b'q8-fixture')
            expected = hashlib.sha256(model.read_bytes()).hexdigest()
            self.assertEqual(expected, D['Q']['candidate_file_proof'](
                model, expected_size=model.stat().st_size, expected_sha256=expected)['sha256'])
            model.write_bytes(b'q8-drift!!')
            with self.assertRaises(AssertionError):
                D['Q']['candidate_file_proof'](
                    model, expected_size=model.stat().st_size, expected_sha256=expected)

    def test_inherited_completed_sycl_eighteen_test_suite_is_unchanged(self):
        result = subprocess.run([sys.executable, '-B', str(SYCL / 'test-profile.py')],
                                cwd=SYCL, check=True, capture_output=True, text=True)
        self.assertRegex(result.stdout + result.stderr, r'Ran 18 tests')


if __name__ == '__main__':
    unittest.main(verbosity=2)
