#!/usr/bin/env python3
"""Case-local read-only source/build proof. No proof cache: every call rehashes."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261019-lilicorr-b70-profile')
QUALIFIED = ROOT.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons'
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-profile-r1')
SOURCE_SCOPE = ['common/speculative.cpp', 'src/llama-context.cpp']
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so', 'libllama.so.0.7.0')
CHANGED_BINARIES = ['libllama-common.so', 'libllama.so.0.7.0']


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_map(source):
    # Exactly the qualified ancestor convention; include ALL non-build files.
    return {str(p.relative_to(source)): sha(p) for p in source.rglob('*')
            if p.is_file() and p.relative_to(source).parts[0] != 'build-sycl'}


def binaries(source):
    return {name: sha(source / 'build-sycl/bin' / name) for name in ARTIFACTS}


def check_sources(parent=PARENT, candidate=None):
    pins = load(HERE / 'source-hashes.json')
    assert sha(HERE / 'lilicorr-profile.patch') == pins['patch_sha256']
    assert sorted(pins['parent']) == sorted(pins['candidate']) == SOURCE_SCOPE
    for label, source in [('parent', parent)] + ([('candidate', candidate)] if candidate else []):
        assert not (source / '.git').exists(), 'expected qualified exported tree, not a checkout'
        assert (source / 'UPSTREAM_COMMIT').read_text().strip() == pins['upstream_commit']
        assert {name: sha(source / name) for name in SOURCE_SCOPE} == pins[label], label
    if candidate is None:
        return dict(parent=source_map(parent))
    before, after = source_map(parent), source_map(candidate)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert changed == SOURCE_SCOPE, changed
    return dict(parent=before, candidate=after, changed_relative_to_parent=changed)


def proof(p=None):
    """Retain P's entire immutable ancestry, then extend it to this actual build."""
    assert __debug__, 'assertions required'
    if p is None:
        p = runpy.run_path(str(QUALIFIED / 'public-gate.py'))
    assert p['CANDIDATE'] == PARENT
    qualified = p['candidate_proof']()  # Never rebind or memoize the qualified gate.
    sources = check_sources(PARENT, CANDIDATE)
    assert sources['parent'] == qualified['source']
    assert (HERE / 'build-attempt-r1/exit-code.txt').read_text().strip() == '0'
    assert 'LILICORR_PROFILE_BUILD_PASSED' in (HERE / 'post-build-verification.log').read_text()
    assert 'LILICORR_PROFILE_DEPENDENCIES_PASSED' in (HERE / 'runtime-precheck.log').read_text()
    assert load(HERE / 'source-diff.json') == dict(changed_relative_to_parent=SOURCE_SCOPE)
    parent_bins, candidate_bins = binaries(PARENT), binaries(CANDIDATE)
    assert parent_bins == qualified['binaries'] == load(HERE / 'parent-binaries.json')
    assert candidate_bins == load(HERE / 'candidate-binaries.json')
    assert sorted(k for k in ARTIFACTS if parent_bins[k] != candidate_bins[k]) == CHANGED_BINARIES
    return dict(path=str(CANDIDATE), source=sources['candidate'], binaries=candidate_bins,
                parent_binaries=parent_bins, upstream_commit=load(HERE / 'source-hashes.json')['upstream_commit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--built', action='store_true')
    args = parser.parse_args()
    assert __debug__
    result = proof() if args.built else check_sources(PARENT, CANDIDATE if args.candidate else None)
    print(json.dumps(result, indent=2))
