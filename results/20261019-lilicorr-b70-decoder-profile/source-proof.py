#!/usr/bin/env python3
"""Case-local source/build proof for the decoder-marker continuation."""
import argparse
import gzip
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261019-lilicorr-b70-decoder-profile')
QUALIFIED = ROOT.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons'
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
SEED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-sycl-profile-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-decoder-profile-r1')
SOURCE_SCOPE = ['common/speculative.cpp', 'ggml/src/ggml-sycl/ggml-sycl.cpp',
                'src/llama-context.cpp']
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so', 'libllama.so.0.7.0')
CHANGED_BINARIES = ['libggml-sycl.so.0', 'libllama-common.so', 'libllama.so.0.7.0']
SYCL = HERE.parent / '20261019-lilicorr-b70-sycl-profile'

# Load the completed SYCL proof fresh.  Its own source proof imports the first
# common profile proof, so redirect both actual globals dictionaries rather than
# memoizing any result or rebinding the qualified ancestry.
_BASE = runpy.run_path(str(SYCL / 'source-proof.py'))
_BASE_GLOBALS = _BASE['check_sources'].__globals__
_INNER_GLOBALS = _BASE_GLOBALS['_BASE_GLOBALS']
for _globals in (_BASE_GLOBALS, _INNER_GLOBALS):
    _globals.update(HERE=HERE, ROOT=ROOT, QUALIFIED=QUALIFIED, PARENT=PARENT,
                    CANDIDATE=CANDIDATE, SOURCE_SCOPE=SOURCE_SCOPE,
                    ARTIFACTS=ARTIFACTS, CHANGED_BINARIES=CHANGED_BINARIES)
_BASE_CHECK_SOURCES = _BASE['check_sources']


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_path(name):
    return HERE / 'fixtures' / (name + '.gz')


def check_fixtures():
    pins = load(HERE / 'source-hashes.json')
    assert sha(HERE / 'lilicorr-profile.patch') == pins['patch_sha256']
    assert sha(HERE / 'seed-incremental.patch') == pins['seed_patch_sha256']
    for name in SOURCE_SCOPE:
        archive = _fixture_path(name)
        assert archive.is_file(), name
        assert sha(archive) == pins['fixtures'][name]['compressed_sha256'], name
        with gzip.open(archive, 'rb') as stream:
            data = stream.read()
        assert hashlib.sha256(data).hexdigest() == pins['parent'][name], name


def check_sources(parent=PARENT, candidate=None):
    check_fixtures()
    return _BASE_CHECK_SOURCES(parent, candidate)


# The imported proof resolves all of these names through its actual namespace.
# Replace only the case-local checker; qualified P/C proof and six-binary logic
# stay inherited and uncached.
_BASE_GLOBALS['check_sources'] = check_sources
_BASE['check_sources'] = check_sources


def proof(p=None):
    return _BASE['proof'](p)


def source_map(source):
    return _BASE['source_map'](source)


def binaries(source):
    return _BASE['binaries'](source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--built', action='store_true')
    args = parser.parse_args()
    assert __debug__
    result = proof() if args.built else check_sources(PARENT, CANDIDATE if args.candidate else None)
    print(json.dumps(result, indent=2))
