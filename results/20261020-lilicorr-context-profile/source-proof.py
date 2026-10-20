#!/usr/bin/env python3
"""Fresh complete qualified-source/six-binary proof for the context diagnostic."""
import argparse
import gzip
import hashlib
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-context-profile')
QUALIFIED = ROOT.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons'
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
SEED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-decoder-profile-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-context-profile-r1')
SOURCE_SCOPE = ['common/speculative.cpp', 'ggml/src/ggml-sycl/ggml-sycl.cpp', 'src/llama-context.cpp']
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so', 'libllama.so.0.7.0')
CHANGED_BINARIES = ['libggml-sycl.so.0', 'libllama-common.so', 'libllama.so.0.7.0']
FIRST = HERE.parent / '20261019-lilicorr-b70-profile'

# One direct import, not a recursive continuation chain. Qualified P/C are untouched.
_BASE = runpy.run_path(str(FIRST / 'source-proof.py'))
_BASE_GLOBALS = _BASE['proof'].__globals__
_BASE_GLOBALS.update(HERE=HERE, ROOT=ROOT, QUALIFIED=QUALIFIED, PARENT=PARENT,
                     CANDIDATE=CANDIDATE, SOURCE_SCOPE=SOURCE_SCOPE,
                     ARTIFACTS=ARTIFACTS, CHANGED_BINARIES=CHANGED_BINARIES)
_BASE_CHECK_SOURCES = _BASE['check_sources']
load, sha = _BASE['load'], _BASE['sha']
source_map, binaries = _BASE['source_map'], _BASE['binaries']


def check_fixtures():
    pins = load(HERE / 'source-hashes.json')
    assert sha(HERE / 'lilicorr-profile.patch') == pins['patch_sha256']
    assert sha(HERE / 'seed-incremental.patch') == pins['seed_patch_sha256']
    assert sorted(pins['seed']) == SOURCE_SCOPE
    for name in SOURCE_SCOPE:
        archive = HERE / 'fixtures' / (name + '.gz')
        assert sha(archive) == pins['fixtures'][name]['compressed_sha256'], name
        data = gzip.decompress(archive.read_bytes())
        assert hashlib.sha256(data).hexdigest() == pins['parent'][name], name
        assert pins['fixtures'][name]['decompressed_sha256'] == pins['parent'][name]
    assert sha(HERE / 'fixtures/combined-startup-measured.json.gz') == pins['combined_fixture_sha256']


def check_sources(parent=PARENT, candidate=None):
    check_fixtures()
    return _BASE_CHECK_SOURCES(parent, candidate)


_BASE_GLOBALS['check_sources'] = check_sources
proof = _BASE['proof']  # Calls the qualified proof and rehashes all sources/binaries every time.

if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--built', action='store_true')
    args = parser.parse_args()
    assert __debug__
    result = proof() if args.built else check_sources(PARENT, CANDIDATE if args.candidate else None)
    print(json.dumps(result, indent=2))
