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
# The qualified original has a complete /src Ninja cache and all six pinned artifacts.
# Use it directly: no diagnostic seed, diagnostic reversal, or SYCL kernel rebuild.
SEED = PARENT
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-embedding-r1')
SOURCE_SCOPE = ['src/llama-context.cpp', 'src/llama-context.h', 'src/llama-cparams.h', 'src/models/dflash.cpp']
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so', 'libllama.so.0.7.0')
CHANGED_BINARIES = ['libllama.so.0.7.0']
FIRST = HERE.parent / '20261019-lilicorr-b70-profile'
_BASE = runpy.run_path(str(FIRST / 'source-proof.py'))
_G = _BASE['proof'].__globals__
_G.update(HERE=HERE, ROOT=ROOT, QUALIFIED=QUALIFIED, PARENT=PARENT,
          CANDIDATE=CANDIDATE, SOURCE_SCOPE=SOURCE_SCOPE,
          ARTIFACTS=ARTIFACTS, CHANGED_BINARIES=CHANGED_BINARIES)
load, sha = _BASE['load'], _BASE['sha']
source_map, binaries = _BASE['source_map'], _BASE['binaries']


def check_fixtures():
    pins = load(HERE / 'source-hashes.json')
    assert sha(HERE / 'device-embedding.patch') == pins['patch_sha256']
    assert sorted(pins['parent']) == sorted(pins['candidate']) == SOURCE_SCOPE
    for name in SOURCE_SCOPE:
        archive = HERE / 'fixtures' / (name + '.gz')
        assert sha(archive) == pins['fixtures'][name]['compressed_sha256'], name
        assert hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest() == pins['parent'][name]
        assert pins['fixtures'][name]['decompressed_sha256'] == pins['parent'][name]
    archive = HERE / 'fixtures/support.tar.gz'
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


_G['check_sources'] = check_sources
# Keeps original qualified ancestry, fresh whole-tree hashing, all SIX binaries,
# and the actual LILICORR_PROFILE_{BUILD,DEPENDENCIES}_PASSED build markers.
proof = _BASE['proof']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--built', action='store_true')
    args = parser.parse_args()
    assert __debug__
    result = proof() if args.built else check_sources(PARENT, CANDIDATE if args.candidate else None)
    print(json.dumps(result, indent=2))
