#!/usr/bin/env python3
"""Small case-local checks for the seeded test executable build."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-n9-control')
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-columns8-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-n9-control-r1')
ACCEPTED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so')
SOURCE_SCOPE = ('tests/test-backend-ops.cpp',)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binaries(source):
    return {name: sha256(source / 'build-sycl/bin' / name) for name in ARTIFACTS}


def source_hashes(root):
    return {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
            if p.is_file() and p.relative_to(root).parts[0] != 'build-sycl'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def check_consumed(source, label):
    expected = json.loads((ROOT / 'source-hashes.json').read_text())[label]
    actual = {name: sha256(source / name) for name in SOURCE_SCOPE}
    assert actual == expected, (label, actual, expected)


def check_protected(candidate=False):
    expected = json.loads((ROOT / 'protected-source-hashes.json').read_text())
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
        manifest = json.loads((ROOT / (label + '-binaries.json')).read_text())
        assert tuple(manifest) == ARTIFACTS, label
        assert binaries(source) == manifest, label
