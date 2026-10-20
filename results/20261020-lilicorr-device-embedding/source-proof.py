#!/usr/bin/env python3
"""Placement-only extension of the qualified whole-tree/six-artifact proof."""
import argparse
import gzip
import hashlib
import json
import runpy
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-device-embedding')
QUALIFIED = ROOT.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons'
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
# Retain the qualified original source and six artifacts; R2 rebuilds only its
# complete llama object closure because whole-server Ninja log lineage is missing.
SEED = PARENT
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-embedding-r2')
SOURCE_SCOPE = ['src/llama-context.cpp', 'src/llama-context.h', 'src/llama-cparams.h', 'src/models/dflash.cpp']
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so', 'libllama.so.0.7.0')
CHANGED_BINARIES = ['libllama.so.0.7.0']
FIRST = HERE.parent / '20261019-lilicorr-b70-profile'
_BASE = runpy.run_path(str(FIRST / 'source-proof.py'))
load, sha = _BASE['load'], _BASE['sha']
source_map, binaries = _BASE['source_map'], _BASE['binaries']


def check_fixtures(here=HERE):
    pins = load(here / 'source-hashes.json')
    assert sha(here / 'device-embedding.patch') == pins['patch_sha256']
    assert sorted(pins['parent']) == sorted(pins['candidate']) == SOURCE_SCOPE
    for name in SOURCE_SCOPE:
        archive = here / 'fixtures' / (name + '.gz')
        assert sha(archive) == pins['fixtures'][name]['compressed_sha256'], name
        assert hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest() == pins['parent'][name]
        assert pins['fixtures'][name]['decompressed_sha256'] == pins['parent'][name]
    archive = here / 'fixtures/support.tar.gz'
    assert sha(archive) == pins['support_archive_sha256']
    with tarfile.open(archive) as stream:
        actual = {}
        for member in stream:
            assert member.isfile() and member.name not in actual
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            actual[member.name] = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
        assert actual == pins['support']


def check_sources(parent=PARENT, candidate=None):
    check_fixtures()
    pins = load(HERE / 'source-hashes.json')
    for label, source in [('parent', parent)] + ([('candidate', candidate)] if candidate else []):
        assert not (source / '.git').exists(), 'expected qualified exported source'
        assert (source / 'UPSTREAM_COMMIT').read_text().strip() == pins['upstream_commit']
        assert {name: sha(source / name) for name in SOURCE_SCOPE} == pins[label], label
    # These are exact original headers and backend bytes used by CPU syntax/layout tests.
    assert {name: sha(parent / name) for name in pins['support']} == pins['support']
    before = source_map(parent)
    if candidate is None:
        return dict(parent=before)
    after = source_map(candidate)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert changed == SOURCE_SCOPE, changed
    return dict(parent=before, candidate=after, changed_relative_to_parent=changed)


def proof(p=None):
    """Same qualified whole-tree/six-binary gate, with R2-local build evidence."""
    assert __debug__, 'assertions required'
    if p is None:
        p = runpy.run_path(str(QUALIFIED / 'public-gate.py'))
    assert p['CANDIDATE'] == PARENT
    qualified = p['candidate_proof']()  # Never rebind or memoize qualified ancestry.
    sources = check_sources(PARENT, CANDIDATE)
    assert sources['parent'] == qualified['source']
    attempt = HERE / 'build-attempt-r2'
    assert (attempt / 'exit-code.txt').read_text().strip() == '0'
    assert 'LILICORR_PROFILE_BUILD_PASSED' in (attempt / 'post-build-verification.log').read_text()
    assert 'LILICORR_PROFILE_DEPENDENCIES_PASSED' in (attempt / 'runtime-precheck.log').read_text()
    assert 'LILICORR_DEVICE_EMBEDDING_MODULE_R2_PASSED' in (attempt / 'build.log').read_text()
    assert load(attempt / 'source-diff.json') == dict(changed_relative_to_parent=SOURCE_SCOPE)
    parent_bins, candidate_bins = binaries(PARENT), binaries(CANDIDATE)
    assert parent_bins == qualified['binaries'] == load(attempt / 'parent-binaries.json')
    assert candidate_bins == load(attempt / 'candidate-binaries.json')
    assert sorted(k for k in ARTIFACTS if parent_bins[k] != candidate_bins[k]) == CHANGED_BINARIES
    return dict(path=str(CANDIDATE), source=sources['candidate'], binaries=candidate_bins,
                parent_binaries=parent_bins, upstream_commit=load(HERE / 'source-hashes.json')['upstream_commit'])


_G = proof.__globals__  # The CPU tests bind only this private runpy namespace.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--built', action='store_true')
    args = parser.parse_args()
    assert __debug__
    result = proof() if args.built else check_sources(PARENT, CANDIDATE if args.candidate else None)
    print(json.dumps(result, indent=2))
