#!/usr/bin/env python3
"""Small case-local checks for the seeded test executable build."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-n9-control')
PROOF = ROOT / 'r2'
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-columns8-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-n9-control-r2')
ACCEPTED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so')
SOURCE_SCOPE = ('tests/test-backend-ops.cpp',)
BASE = runpy.run_path(str(ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-columns8/run-trace.py'))
N9 = runpy.run_path(str(ROOT / 'n9-equivalence.py'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binaries(source):
    return {name: sha256(source / 'build-sycl/bin' / name) for name in ARTIFACTS}


def source_hashes(root):
    return {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
            if p.is_file() and p.relative_to(root).parts[0] != 'build-sycl'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def proof(name):
    return PROOF / name


def check_consumed(source, label):
    BASE['check_consumed_sources'](PARENT, 'candidate')
    original = (PARENT / SOURCE_SCOPE[0]).read_bytes()
    assert label in ('parent', 'candidate')
    expected = original if label == 'parent' else N9['expected_test_source'](original)
    assert (source / SOURCE_SCOPE[0]).read_bytes() == expected, label


def check_protected(candidate=True):
    BASE['check_protected_sources']()
    BASE['check_protected_binaries']()
    assert binaries(PARENT) == json.loads((BASE['ROOT'] / 'candidate-binaries.json').read_text())
    expected = json.loads(proof('protected-source-hashes.json').read_text())
    assert source_hashes(ACCEPTED) == expected['accepted']
    assert source_hashes(PARENT) == expected['parent']
    check_consumed(PARENT, 'parent')
    if candidate:
        actual = source_hashes(CANDIDATE)
        changed = sorted(k for k in actual.keys() | expected['parent'].keys()
                         if actual.get(k) != expected['parent'].get(k))
        assert changed == ['tests/test-backend-ops.cpp'], changed
        check_consumed(CANDIDATE, 'candidate')


def check_protected_binaries():
    for label, source in (('accepted', ACCEPTED), ('parent', PARENT)):
        manifest = json.loads(proof(label + '-binaries.json').read_text())
        assert set(manifest) == set(ARTIFACTS), label
        assert binaries(source) == manifest, label
