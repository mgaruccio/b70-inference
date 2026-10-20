#!/usr/bin/env python3
"""CPU-only guards for the host-boundary source proof and 105-object build slice."""
import gzip
import contextlib
import io
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEVICE_CASE = HERE.parent / '20261020-lilicorr-device-timing'
IQ2_CASE = HERE.parent / '20261020-lilicorr-iq2-batch4'
SHELL = HERE / 'build-candidate.sh'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Keep the existing host composition helpers as read-only fixtures.
FIXTURE = load_module('host_composition_fixture', HERE / 'test-host-source.py')
ACTUAL = load_module('host_source_proof', HERE / 'source-proof.py')
MODULE = load_module('iq2_build_module', IQ2_CASE / 'build-module.py')
REFERENCE = MODULE.expected()


class BuildProofTests(unittest.TestCase):
    def setUp(self):
        (HERE / '.scratch').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=HERE / '.scratch')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / 'parent'
        self.composed = self.root / 'composed'
        self.seed = self.root / 'seed'
        self.candidate = self.root / 'candidate'
        FIXTURE.parent(self.parent)
        shutil.copytree(self.parent, self.composed)
        FIXTURE.apply(self.composed, FIXTURE.CONTEXT / 'lilicorr-profile.patch')
        FIXTURE.apply(self.composed, FIXTURE.IQ2 / 'iq2-batch4.patch')
        # The SDK seed is the completed device composition.
        shutil.copytree(self.composed, self.seed)
        FIXTURE.apply(self.seed, DEVICE_CASE / 'device-timing.patch')
        shutil.copytree(self.seed, self.candidate)
        FIXTURE.apply(self.candidate, HERE / 'disable-device-profiling.patch')
        self._make_artifacts()
        self._bind_proof_globals()

    def _make_artifacts(self):
        for source in (self.parent, self.seed, self.candidate):
            (source / 'build-sycl/bin').mkdir(parents=True)
        changed = set(ACTUAL.CHANGED_BINARIES)
        for name in ACTUAL.ARTIFACTS:
            (self.parent / 'build-sycl/bin' / name).write_text('original ' + name + '\n')
            value = ('seed ' if name in changed else 'original ') + name + '\n'
            (self.seed / 'build-sycl/bin' / name).write_text(value)
            (self.candidate / 'build-sycl/bin' / name).write_text(value)
        path = self.candidate / 'build-sycl/bin/libggml-sycl.so.0'
        path.write_text('host rebuilt libggml-sycl.so.0\n')

    def _bind_proof_globals(self):
        self.case = self.root / 'case'
        self.attempt = self.case / 'build-attempt-r1'
        self.attempt.mkdir(parents=True)
        ACTUAL.ROOT = self.case
        ACTUAL.PARENT = self.parent
        ACTUAL.SEED = self.seed
        ACTUAL.CANDIDATE = self.candidate
        self.addCleanup(self._restore_proof_globals)

    def _restore_proof_globals(self):
        ACTUAL.ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-host-boundary')
        ACTUAL.PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
        ACTUAL.SEED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-timing-r1')
        ACTUAL.CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-host-boundary-r1')

    def _write_attempt(self):
        sources = ACTUAL.check_sources(self.parent, self.candidate)
        seed_sources = ACTUAL.check_seed(self.seed, self.candidate)
        (self.attempt / 'candidate-source-before.json').write_text(json.dumps(sources, sort_keys=True, indent=2) + '\n')
        (self.attempt / 'candidate-source-after.json').write_text(json.dumps(sources, sort_keys=True, indent=2) + '\n')
        (self.attempt / 'seed-source-before.json').write_text(json.dumps(seed_sources['seed'], sort_keys=True, indent=2) + '\n')
        (self.attempt / 'seed-source-after.json').write_text(json.dumps(seed_sources['seed'], sort_keys=True, indent=2) + '\n')
        parent_bins = ACTUAL.binaries(self.parent)
        seed_bins = ACTUAL.binaries(self.seed)
        candidate_bins = ACTUAL.binaries(self.candidate)
        for name, value in (
            ('parent-binaries.json', parent_bins),
            ('seed-built-before.json', seed_bins),
            ('seed-built-after.json', seed_bins),
            ('candidate-binaries.json', candidate_bins),
        ):
            (self.attempt / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
        graph = {'build.ninja': 'graph', 'CMakeFiles/rules.ninja': 'rules', 'CMakeCache.txt': 'cache'}
        (self.attempt / 'graph-before.json').write_text(json.dumps(graph, sort_keys=True, indent=2) + '\n')
        (self.attempt / 'graph-after.json').write_text(json.dumps(graph, sort_keys=True, indent=2) + '\n')
        (self.attempt / 'exit-code.txt').write_text('0\n')
        (self.attempt / 'post-build-verification.log').write_text(ACTUAL.BUILD_MARKER + '\n')
        return sources

    def _qualified(self, calls=None):
        original = dict(source=ACTUAL.source_map(self.parent), binaries=ACTUAL.binaries(self.parent))
        if calls is None:
            calls = []

        def candidate_proof():
            calls.append('proof')
            return original

        return dict(CANDIDATE=self.parent, candidate_proof=candidate_proof)

    def test_source_composition_full_map_and_helper_only_seed_delta(self):
        sources = ACTUAL.check_sources(self.parent, self.candidate)
        self.assertIn('common/speculative.cpp', sources['changed_relative_to_parent'])
        self.assertIn('ggml/src/ggml-sycl/lilicorr-device-timing.hpp', sources['changed_relative_to_parent'])
        for name, expected in FIXTURE.PINS['diagnostic'].items():
            self.assertEqual(sources['candidate'][name], expected, name)
        seed = ACTUAL.check_seed(self.seed, self.candidate)
        self.assertEqual(seed['changed_relative_to_seed'], [ACTUAL.HELPER])
        self.assertEqual(ACTUAL.source_map(self.seed)[ACTUAL.HELPER], FIXTURE.PINS['device_diagnostic'][ACTUAL.HELPER])
        self.assertEqual(ACTUAL.source_map(self.candidate)[ACTUAL.HELPER], FIXTURE.PINS['diagnostic'][ACTUAL.HELPER])

    def test_seed_sources_cli_emits_the_unwrapped_map_consumed_by_proof(self):
        saved = sys.argv
        output = io.StringIO()
        try:
            sys.argv = ['source-proof.py', '--seed-sources']
            with contextlib.redirect_stdout(output):
                ACTUAL.main()
        finally:
            sys.argv = saved
        emitted = json.loads(output.getvalue())
        consumed = ACTUAL.check_seed(self.seed, self.candidate)['seed']
        self.assertEqual(emitted, consumed)
        self.assertNotIn('seed', emitted)
        self.assertTrue(all(isinstance(name, str) and isinstance(value, str)
                            for name, value in emitted.items()))

    def test_full_map_drift_extra_missing_and_source_escape_rejected(self):
        extra_name = 'unlisted-source.cpp'
        (self.parent / extra_name).write_text('// baseline outside the pin subset\n')
        shutil.copy2(self.parent / extra_name, self.candidate / extra_name)
        ACTUAL.check_sources(self.parent, self.candidate)
        baseline = (self.candidate / extra_name).read_bytes()
        (self.candidate / extra_name).write_text('unexpected drift\n')
        with self.assertRaises(AssertionError):
            ACTUAL.check_sources(self.parent, self.candidate)
        (self.candidate / extra_name).write_bytes(baseline)
        extra = self.candidate / 'unexpected-source.cpp'
        extra.write_text('unrequested source\n')
        with self.assertRaises(AssertionError):
            ACTUAL.check_sources(self.parent, self.candidate)
        extra.unlink()
        (self.candidate / extra_name).unlink()
        with self.assertRaises(AssertionError):
            ACTUAL.check_sources(self.parent, self.candidate)
        (self.candidate / extra_name).write_bytes(baseline)
        outside = self.root / 'outside.cpp'
        outside.write_text('outside\n')
        escaping = self.candidate / 'escaping.cpp'
        escaping.symlink_to(outside)
        with self.assertRaises(AssertionError):
            ACTUAL.check_sources(self.parent, self.candidate)

    def test_fresh_public_proof_rehashes_original_and_protected_artifacts(self):
        sources = self._write_attempt()
        calls = []
        qualified = self._qualified(calls)
        first = ACTUAL.proof(qualified)
        second = ACTUAL.proof(qualified)
        self.assertEqual(first['source'], sources['candidate'])
        self.assertEqual(first['binaries'], ACTUAL.binaries(self.candidate))
        self.assertEqual(first, second)
        self.assertEqual(2, len(calls), 'public candidate_proof must be called on every proof')
        protected = next(name for name in ACTUAL.ARTIFACTS if name != 'libggml-sycl.so.0')
        protected_path = self.candidate / 'build-sycl/bin' / protected
        protected_bytes = protected_path.read_bytes()
        protected_path.write_bytes(b'unexpected protected artifact change\n')
        with self.assertRaises(AssertionError):
            ACTUAL.proof(qualified)
        protected_path.write_bytes(protected_bytes)
        (self.candidate / ACTUAL.HELPER).write_text('post-build source drift\n')
        with self.assertRaises(AssertionError):
            ACTUAL.proof(qualified)

    def test_seed_and_final_artifact_deltas_are_exact(self):
        seed = ACTUAL.binaries(self.seed)
        candidate = ACTUAL.binaries(self.candidate)
        parent = ACTUAL.binaries(self.parent)
        self.assertEqual(['libggml-sycl.so.0'], sorted(name for name in ACTUAL.ARTIFACTS if seed[name] != candidate[name]))
        self.assertEqual(ACTUAL.CHANGED_BINARIES, sorted(name for name in ACTUAL.ARTIFACTS if parent[name] != candidate[name]))
        self.assertEqual([name for name in ACTUAL.ARTIFACTS if name != 'libggml-sycl.so.0'],
                         [name for name in ACTUAL.ARTIFACTS if seed[name] == candidate[name]])

    def test_old_iq2_closure_flags_graph_and_template_counts(self):
        r = REFERENCE
        argv, objects, commands = MODULE.validate_closure(
            r['link'], r['targets'], r['queries'], '\n'.join(r['commands']), r)
        self.assertEqual(105, len(objects))
        self.assertEqual(105, len(commands))
        self.assertEqual(59, sum('/template-instances/' not in x for x in objects))
        self.assertEqual(46, sum('/template-instances/' in x for x in objects))
        self.assertEqual(MODULE.COMPILER, argv[0])
        self.assertEqual('-Xs', argv[4])
        self.assertEqual('-ze-intel-greater-than-4GB-buffer-required', argv[5])
        self.assertNotIn('DPCT_PROFILING_ENABLED', ' '.join(r['commands']))
        self.assertTrue(all('-DGGML_SYCL_WARP_SIZE=16' in command for command in r['commands']))
        self.assertNotIn('GGML_SYCL_MMV_Y', '\n'.join(r['commands']))
        MODULE.check_plan(r['plan'], objects)
        graph_root = self.root / 'graph'
        for name in r['graph_sha256']:
            path = graph_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + '\n')
        before = MODULE.file_map(graph_root, r['graph_sha256'])
        after = MODULE.file_map(graph_root, r['graph_sha256'])
        self.assertEqual(before, after)
        (graph_root / 'build.ninja').write_text('graph drift\n')
        self.assertNotEqual(before, MODULE.file_map(graph_root, r['graph_sha256']))

    def test_object_cleanup_is_guarded_and_preserves_everything_else(self):
        script = SHELL.read_text()
        start_marker = "python3 -B - /package/build-inputs.json.gz /src/build-sycl <<'PY'\n"
        end_marker = "\nPY\n# LILICORR_OBJECT_CLEAN_END"
        start = script.index(start_marker) + len(start_marker)
        end = script.index(end_marker, start)
        cleanup = script[start:end]
        compile(cleanup, str(SHELL) + ':object-cleanup', 'exec')
        for forbidden in ('build.ninja', 'CMakeCache.txt', '.ninja_log', 'bin/', 'source'):
            self.assertNotIn(forbidden, cleanup)
        self.assertIn('reference["objects"]', cleanup)
        self.assertIn('len(objects) == 105', cleanup)
        self.assertIn('path.unlink()', cleanup)

        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            inputs = root / 'build-inputs.json.gz'
            build = root / 'build-sycl'
            build.mkdir()
            json_data = dict(objects=REFERENCE['objects'])
            inputs.write_bytes(gzip.compress(json.dumps(json_data).encode()))
            for name in REFERENCE['objects']:
                path = build / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'object')
            for name in ('build.ninja', 'CMakeCache.txt', '.ninja_log'):
                (build / name).write_text(name)
            output = build / 'bin/libggml-sycl.so.0'
            output.parent.mkdir()
            output.write_text('library')
            subprocess.run([sys.executable, '-B', '-c', cleanup, str(inputs), str(build)], check=True)
            self.assertTrue(all(not (build / name).exists() for name in REFERENCE['objects']))
            self.assertTrue((build / 'build.ninja').exists())
            self.assertTrue((build / 'CMakeCache.txt').exists())
            self.assertTrue((build / '.ninja_log').exists())
            self.assertTrue(output.exists())

            unsafe = list(REFERENCE['objects'])
            unsafe[0] = '../outside.o'
            inputs.write_bytes(gzip.compress(json.dumps(dict(objects=unsafe)).encode()))
            outside = root / 'outside.o'
            outside.write_bytes(b'protected')
            result = subprocess.run([sys.executable, '-B', '-c', cleanup, str(inputs), str(build)], capture_output=True)
            self.assertNotEqual(0, result.returncode)
            self.assertEqual(b'protected', outside.read_bytes())

    def test_shell_wrapper_is_frozen_and_cpu_safe_to_parse(self):
        subprocess.run(['bash', '-n', str(SHELL)], check=True)
        script = SHELL.read_text()
        self.assertIn('python3 -B "$DEVICE_CASE/source-proof.py" --built', script)
        self.assertLess(script.index('device-proof-before-copy.json'), script.index('cp -a --reflink=auto "$SDKSEED" "$CANDIDATE"'))
        self.assertIn('disable-device-profiling.patch', script)
        self.assertIn('-v "$CANDIDATE:/src" -v "$SDKSEED:/qualified:ro" -v "$IQ2_PACKAGE:/package:ro"', script)
        self.assertIn('python3 -B /package/build-module.py', script)
        self.assertIn('aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0', script)
        self.assertIn('apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config', script)
        self.assertEqual(2, script.count('source /opt/intel/oneapi/setvars.sh --force\nset -u'))
        self.assertIn('cmp /build-evidence/graph-before.json /build-evidence/graph-after.json', script)
        for forbidden in ('--device', '/dev/dri', '--privileged', 'cmake --build', 'cmake -S', 'docker push', 'ssh '):
            self.assertNotIn(forbidden, script)


if __name__ == '__main__':
    print('CPU-only source/build-boundary checks; no SDK, container, compiler, GPU, or remote access.', flush=True)
    unittest.main(verbosity=2)
