#!/usr/bin/env python3
"""CPU-only reuse checks for the private full-source/six-artifact boundary."""
import json
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = runpy.run_path(str(HERE / 'source-proof.py'))
FIXTURE = runpy.run_path(str(HERE / 'test-device-timing.py'))


class SourceProofTests(unittest.TestCase):
    def setUp(self):
        (HERE / '.scratch').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=HERE / '.scratch')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent, self.candidate = self.root / 'parent', self.root / 'candidate'
        FIXTURE['parent'](self.parent)
        (self.parent / 'unlisted-source.cpp').write_text('// baseline outside the pin subset\n')
        shutil.copytree(self.parent, self.candidate)
        for patch in (FIXTURE['CONTEXT'] / 'lilicorr-profile.patch',
                      FIXTURE['IQ2'] / 'iq2-batch4.patch', HERE / 'device-timing.patch'):
            FIXTURE['apply'](self.candidate, patch)

    def check(self):
        return SOURCE['check_sources'](self.parent, self.candidate)

    def test_real_pinned_composition_and_entire_unlisted_source(self):
        proof = self.check()
        self.assertIn('unlisted-source.cpp', proof['parent'])
        self.assertEqual(proof['parent']['unlisted-source.cpp'], proof['candidate']['unlisted-source.cpp'])
        self.assertIn('common/speculative.cpp', proof['changed_relative_to_parent'])
        self.assertIn('ggml/src/ggml-sycl/lilicorr-device-timing.hpp', proof['changed_relative_to_parent'])

    def test_unlisted_drift_extra_and_missing_source_rejected(self):
        baseline = (self.candidate / 'unlisted-source.cpp').read_bytes()
        (self.candidate / 'unlisted-source.cpp').write_text('unexpected drift\n')
        with self.assertRaises(AssertionError):
            self.check()
        (self.candidate / 'unlisted-source.cpp').write_bytes(baseline)
        extra = self.candidate / 'unexpected-source.cpp'
        extra.write_text('unrequested source\n')
        with self.assertRaises(AssertionError):
            self.check()
        extra.unlink()
        (self.candidate / 'unlisted-source.cpp').unlink()
        with self.assertRaises(AssertionError):
            self.check()

    def test_parent_drift_and_symlink_escape_rejected(self):
        target = self.parent / 'ggml/src/ggml-sycl/presets.hpp'
        baseline = target.read_bytes()
        target.write_text('wrong parent\n')
        with self.assertRaises(AssertionError):
            self.check()
        target.write_bytes(baseline)
        outside = self.root / 'outside.cpp'
        outside.write_text('outside\n')
        (self.candidate / 'escaping.cpp').symlink_to(outside)
        with self.assertRaises(AssertionError):
            self.check()

    def test_fresh_source_six_artifacts_and_three_library_delta(self):
        globals_ = SOURCE['proof'].__globals__
        previous = {name: globals_[name] for name in ('ROOT', 'PARENT', 'CANDIDATE')}
        self.addCleanup(globals_.update, previous)
        globals_.update(ROOT=self.root / 'case', PARENT=self.parent, CANDIDATE=self.candidate)
        attempt = globals_['ROOT'] / 'build-attempt-r1'
        attempt.mkdir(parents=True)
        sources = self.check()
        for name in ('candidate-source-before.json', 'candidate-source-after.json'):
            (attempt / name).write_text(json.dumps(sources))
        (attempt / 'exit-code.txt').write_text('0\n')
        (attempt / 'post-build-verification.log').write_text(SOURCE['BUILD_MARKER'] + '\n')
        for source in (self.parent, self.candidate):
            (source / 'build-sycl/bin').mkdir(parents=True)
            for name in SOURCE['ARTIFACTS']:
                value = 'parent artifact ' + name
                if source == self.candidate and name in SOURCE['CHANGED_BINARIES']:
                    value = 'changed library ' + name
                (source / 'build-sycl/bin' / name).write_text(value)
        for source, name in ((self.parent, 'parent-binaries.json'), (self.candidate, 'candidate-binaries.json')):
            (attempt / name).write_text(json.dumps(SOURCE['binaries'](source)))
        original = dict(source=sources['parent'], binaries=SOURCE['binaries'](self.parent))
        qualified = dict(CANDIDATE=self.parent, candidate_proof=lambda: original)
        public = self.root / '20261019-nemotron35-bolt-lilicorr-verify-singletons/public-gate.py'
        public.parent.mkdir()
        public.write_text('import json\nfrom pathlib import Path\nCANDIDATE = Path(' + repr(str(self.parent)) + ')\ndef candidate_proof():\n    return json.loads(' + repr(json.dumps(original)) + ')\n')
        self.assertEqual(SOURCE['proof']()['source'], sources['candidate'])
        self.assertEqual(SOURCE['proof'](qualified)['source'], sources['candidate'])
        with self.assertRaises(AssertionError):
            SOURCE['proof'](dict(qualified, CANDIDATE=self.candidate))
        wrong_original = dict(original, source=dict(original['source'], unexpected='drift'))
        with self.assertRaises(AssertionError):
            SOURCE['proof'](dict(qualified, candidate_proof=lambda: wrong_original))
        server = self.candidate / 'build-sycl/bin/llama-server'
        server.write_text('unexpected fourth artifact change\n')
        (attempt / 'candidate-binaries.json').write_text(json.dumps(SOURCE['binaries'](self.candidate)))
        with self.assertRaises(AssertionError):
            SOURCE['proof']()
        server.write_bytes((self.parent / 'build-sycl/bin/llama-server').read_bytes())
        (attempt / 'candidate-binaries.json').write_text(json.dumps(SOURCE['binaries'](self.candidate)))
        (self.candidate / 'unlisted-source.cpp').write_text('drift after build\n')
        with self.assertRaises(AssertionError):
            SOURCE['proof']()


if __name__ == '__main__':
    unittest.main(verbosity=2)
