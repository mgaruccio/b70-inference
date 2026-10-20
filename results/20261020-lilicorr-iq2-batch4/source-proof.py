#!/usr/bin/env python3
"""Case-local source and six-artifact proof for the IQ2 batch-4 candidate.

This proof is deliberately local: it rehashes the qualified exported source
and the candidate on every invocation.  It does not treat a cached receipt,
filename glob, or an inferred object name as source identity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-iq2-batch4')
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-batch4-r1')
SOURCE_SCOPE = [
    'ggml/src/ggml-sycl/ggml-sycl.cpp',
    'ggml/src/ggml-sycl/mmvq.cpp',
    'ggml/src/ggml-sycl/mmvq.hpp',
]
ARTIFACTS = (
    'llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
    'test-backend-ops', 'libllama-common.so', 'libllama.so.0.7.0',
)
CHANGED_BINARIES = ['libggml-sycl.so.0']
PATCH_FILES = tuple(SOURCE_SCOPE)
PATCH_SHA256 = 'ab6d9f014eb82303489d0750d48e9b6614fb79f3e65d076c980006e5c3d27cb1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_pins(path=None):
    path = HERE / 'source-pins.sha256' if path is None else Path(path)
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        assert name not in result, ('duplicate source pin', name)
        result[name] = digest
    return result


def patch_stats():
    patch = HERE / 'iq2-batch4.patch'
    assert sha(patch) == PATCH_SHA256, 'candidate patch changed'
    files = []
    additions = deletions = 0
    for line in patch.read_text().splitlines():
        if line.startswith('--- '):
            old = line[4:].split('\t', 1)[0]
            assert old.startswith('a/')
            old = old[2:]
        elif line.startswith('+++ '):
            new = line[4:].split('\t', 1)[0]
            assert new.startswith('b/') and old == new[2:]
            files.append(new[2:])
        elif line.startswith('+'):
            additions += 1
        elif line.startswith('-'):
            deletions += 1
    assert tuple(files) == PATCH_FILES, ('patch scope drift', files)
    assert additions == 74 and deletions == 0, (additions, deletions)
    return dict(files=files, additions=additions, deletions=deletions, sha256=PATCH_SHA256)


def _safe_path(root, path):
    root = Path(root).resolve()
    path = Path(path)
    resolved = path.resolve(strict=False)
    assert resolved.is_relative_to(root), ('source path escaped root', path, resolved)
    return resolved


def source_map(source):
    """Hash every non-build source file, including symlinked files."""
    source = Path(source)
    root = source.resolve()
    result = {}
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if not relative.parts or relative.parts[0] in {'build-sycl', '.git'}:
            continue
        if not (path.is_file() or path.is_symlink()):
            continue
        _safe_path(root, path)
        result[str(relative)] = sha(path)
    return result


def binaries(source):
    source = Path(source)
    result = {}
    root = source.resolve()
    for name in ARTIFACTS:
        path = source / 'build-sycl' / 'bin' / name
        _safe_path(root, path)
        assert path.is_file() or path.is_symlink(), ('missing artifact', name)
        result[name] = sha(path)
    return result


def _check_pinned_files(source, pins, names):
    for name in names:
        path = source / name
        _safe_path(source, path)
        assert sha(path) == pins[name], ('source pin drift', name)


def check_sources(parent=PARENT, candidate=None):
    """Check the qualified parent and, when supplied, exactly three patch files."""
    assert __debug__, 'assertions required'
    patch = patch_stats()
    pins = load_pins()
    assert set(pins) == {'UPSTREAM_COMMIT', *SOURCE_SCOPE,
                         'ggml/src/ggml-sycl/quantize.hpp',
                         'ggml/src/ggml-sycl/vecdotq.hpp',
                         'ggml/src/ggml-sycl/quants.hpp',
                         'ggml/src/ggml-sycl/common.hpp',
                         'ggml/src/ggml-sycl/presets.hpp',
                         'ggml/src/ggml-sycl/CMakeLists.txt'}
    unchanged = [name for name in pins if name not in SOURCE_SCOPE and name != 'UPSTREAM_COMMIT']
    for label, source in [('parent', parent)] + ([('candidate', candidate)] if candidate else []):
        source = Path(source)
        assert source.exists() and source.is_dir(), ('missing source root', label, source)
        assert not (source / '.git').exists(), 'expected exported source, not a checkout'
        commit = source / 'UPSTREAM_COMMIT'
        _safe_path(source, commit)
        assert sha(commit) == pins['UPSTREAM_COMMIT'], ('upstream marker drift', label)
        if label == 'parent':
            _check_pinned_files(source, pins, SOURCE_SCOPE + unchanged)
        else:
            _check_pinned_files(source, pins, unchanged)

    before = source_map(parent)
    if candidate is None:
        return dict(parent=before, patch=patch)
    after = source_map(candidate)
    changed = sorted(name for name in before.keys() | after.keys()
                     if before.get(name) != after.get(name))
    assert changed == sorted(SOURCE_SCOPE), ('unexpected source delta', changed)
    return dict(parent=before, candidate=after,
                changed_relative_to_parent=changed, patch=patch)


def proof():
    """Validate a completed lead build without widening the artifact claim."""
    sources = check_sources(PARENT, CANDIDATE)
    attempt = ROOT / 'build-attempt-r1'
    assert (attempt / 'exit-code.txt').read_text().strip() == '0'
    log = (attempt / 'post-build-verification.log').read_text()
    assert 'LILICORR_IQ2_BATCH4_GGML_SYCL_BUILD_PASSED' in log
    parent_bins, candidate_bins = binaries(PARENT), binaries(CANDIDATE)
    changed = sorted(name for name in ARTIFACTS if parent_bins[name] != candidate_bins[name])
    assert changed == CHANGED_BINARIES, ('unexpected artifact delta', changed)
    return dict(path=str(CANDIDATE), source=sources['candidate'],
                binaries=candidate_bins, parent_binaries=parent_bins,
                changed_binaries=changed,
                upstream_commit=(PARENT / 'UPSTREAM_COMMIT').read_text().strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--built', action='store_true')
    args = parser.parse_args()
    result = proof() if args.built else check_sources(PARENT, CANDIDATE if args.candidate else None)
    print(json.dumps(result, indent=2, sort_keys=True))
