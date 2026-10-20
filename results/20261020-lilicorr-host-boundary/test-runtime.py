#!/usr/bin/env python3
"""Reuse the reviewed lifecycle checks for the explicit host-only composition."""
import contextlib
import io
import json
import tempfile
import runpy
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-profile.py'))
OLD = runpy.run_path(str(HERE.parent / '20261020-lilicorr-device-timing/test-runtime.py'))
BASE = OLD['RuntimeTests']
# Tests, like production adapters, must change their functions' actual namespace.
BASE.bound.__globals__['D'] = D


class HostRuntimeTests(BASE):
    def test_first_public_guard_uses_current_request_and_preserves_old_scope(self):
        globals_ = D['check_first_job_scope'].__globals__
        seen = []
        positive = dict(status='ok', gpu_time_claim=False, gpu_timestamps='unavailable')
        with mock.patch.dict(globals_, ORIGINAL_SCOPE_CHECK=lambda _s: dict(passed=True, request_start_line=2)):
            with mock.patch.object(globals_['runpy'], 'run_path', return_value={
                    'check_request': lambda text: seen.append(text) or positive}):
                proof = D['check_first_job_scope']('old startup\ncurrent request\n')
        self.assertEqual(seen, ['current request'])
        self.assertEqual(proof['host_boundary'], positive)
        for bad in (dict(positive, status='failed'), dict(positive, gpu_time_claim=True),
                    dict(positive, gpu_timestamps='measured')):
            with mock.patch.dict(globals_, ORIGINAL_SCOPE_CHECK=lambda _s: dict(passed=True, request_start_line=2)):
                with mock.patch.object(globals_['runpy'], 'run_path', return_value={
                        'check_request': lambda _text, value=bad: value}):
                    with self.assertRaises(AssertionError):
                        D['check_first_job_scope']('old startup\ncurrent request\n')


    def test_actual_build_postcheck_uses_changed_pins_and_leaves_exit_to_trap(self):
        script = (HERE / 'build-candidate.sh').read_text()
        marker = "<<'PY' | tee \"$ATTEMPT/post-build-verification.log\""
        post = script.rsplit(marker, 1)[1].split('\n', 1)[1].split('\nPY', 1)[0]
        self.assertNotIn('python3 -B "$ROOT/source-proof.py" --built', script)
        pins = json.loads((HERE / 'source-pins.json').read_text())
        expected = sorted(k for k in pins['diagnostic'] if pins['parent'].get(k) != pins['diagnostic'][k])
        self.assertEqual(len(expected), 8)
        original = {name: 'original-' + name for name in D['S']['ARTIFACTS']}
        seed = dict(original)
        for name in D['S']['CHANGED_BINARIES']: seed[name] = 'seed-' + name
        candidate = dict(seed, **{'libggml-sycl.so.0': 'host-sycl'})
        (HERE / '.scratch').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE / '.scratch') as temporary:
            root = Path(temporary)
            attempt = root / 'attempt'
            attempt.mkdir()
            parent_path, seed_path, candidate_path = (Path(s) for s in ('original', 'seed', 'candidate'))
            fake = dict(PARENT=parent_path, SEED=seed_path, CANDIDATE=candidate_path,
                        ARTIFACTS=D['S']['ARTIFACTS'], CHANGED_BINARIES=D['S']['CHANGED_BINARIES'],
                        SEED_CHANGED_BINARIES=['libggml-sycl.so.0'], HELPER=D['S']['HELPER'],
                        BUILD_MARKER=D['S']['BUILD_MARKER'], load_pins=lambda: pins,
                        check_sources=lambda *_a: {'changed_relative_to_parent': expected},
                        check_seed=lambda *_a: {'changed_relative_to_seed': [D['S']['HELPER']]},
                        binaries=lambda path: {parent_path: original, seed_path: seed, candidate_path: candidate}[path])
            with mock.patch.object(runpy, 'run_path', return_value=fake), \
                    mock.patch('sys.argv', ['postcheck', str(root), str(attempt)]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                exec(compile(post, str(HERE / 'build-candidate.sh'), 'exec'), {})
            self.assertIn(D['S']['BUILD_MARKER'], output.getvalue())
            self.assertEqual(json.loads((attempt / 'candidate-binaries.json').read_text()), candidate)


del BASE  # Avoid collecting the imported base TestCase a second time.

if __name__ == '__main__':
    unittest.main(verbosity=2)
