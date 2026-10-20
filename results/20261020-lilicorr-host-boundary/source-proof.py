#!/usr/bin/env python3
"""Private host-boundary source and six-artifact proof; no runtime work.

The qualified R2 tree remains the public-gate baseline.  The SDK seed is the
completed device-r1 composition; the host candidate is required to differ from
that seed only in the restored SYCL queue helper, while the final candidate is
still checked against the complete qualified source map and six-artifact
boundary.
"""
import argparse
import hashlib
import importlib.util
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-host-boundary')
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
SEED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-timing-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-host-boundary-r1')
HOST_PATCH = HERE / 'host-boundary.patch'
ROLLBACK_PATCH = HERE / 'disable-device-profiling.patch'
DEVICE_CASE = HERE.parent / '20261020-lilicorr-device-timing'
IQ2_PROOF = HERE.parent / '20261020-lilicorr-iq2-batch4/source-proof.py'
HELPER = 'ggml/src/ggml-sycl/dpct/helper.hpp'
BUILD_MARKER = 'LILICORR_HOST_BOUNDARY_GGML_SYCL_MODULE_PASSED'
CHANGED_BINARIES = ['libggml-sycl.so.0', 'libllama-common.so', 'libllama.so.0.7.0']
SEED_CHANGED_BINARIES = ['libggml-sycl.so.0']


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module(IQ2_PROOF, '_lilicorr_iq2_source_proof')
sha = BASE.sha
source_map = BASE.source_map
binaries = BASE.binaries
ARTIFACTS = BASE.ARTIFACTS


def load_pins():
    pins = json.loads((HERE / 'source-pins.json').read_text())
    assert pins['upstream_commit'] == 'b6334e37e6ef286fc5091a41b6561b2113d915fb'
    assert sha(HOST_PATCH) == pins['patch_sha256']
    assert sha(ROLLBACK_PATCH) == pins['rollback_patch_sha256']
    assert sha(DEVICE_CASE / 'device-timing.patch') == pins['device_patch_sha256']
    for group in ('ancestry_files', 'reused_fixtures'):
        for name, expected in pins[group].items():
            relative = Path(name).relative_to('results')
            assert '..' not in relative.parts and not relative.is_absolute()
            assert sha(HERE.parent / relative) == expected, ('fixture drift', name)
    for stage in ('parent', 'context_seed', 'composed_seed', 'diagnostic', 'device_diagnostic'):
        assert 'UPSTREAM_COMMIT' in pins[stage]
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts
                   for name in pins[stage]), ('unsafe source pin', stage)
    assert pins['diagnostic'][HELPER] == pins['parent'][HELPER]
    assert pins['device_diagnostic'][HELPER] != pins['diagnostic'][HELPER]
    return pins


def _assert_source_root(root):
    root = Path(root)
    assert root.is_dir() and not (root / '.git').exists(), ('invalid exported source', root)
    return root


def _patch_record(pins):
    return dict(
        sha256=pins['patch_sha256'],
        ancestry=pins['ancestry_files'],
        device_sha256=pins['device_patch_sha256'],
        rollback_sha256=pins['rollback_patch_sha256'],
    )


def check_sources(parent=PARENT, candidate=None):
    """Check the entire exported source tree, not only patch-mentioned files."""
    assert __debug__, 'assertions required'
    pins = load_pins()
    parent = _assert_source_root(parent)
    assert (parent / 'UPSTREAM_COMMIT').read_text().strip() == pins['upstream_commit']
    before = source_map(parent)
    for name, expected in pins['parent'].items():
        assert before.get(name) == expected, ('qualified source drift', name)
    patch = _patch_record(pins)
    if candidate is None:
        return dict(parent=before, patch=patch)

    candidate = _assert_source_root(candidate)
    assert (candidate / 'UPSTREAM_COMMIT').read_text().strip() == pins['upstream_commit']
    expected = dict(before)
    expected.update(pins['diagnostic'])
    after = source_map(candidate)
    changed = sorted(name for name in before.keys() | after.keys()
                     if before.get(name) != after.get(name))
    assert after == expected, ('unexpected host source delta', sorted(
        name for name in expected.keys() | after.keys()
        if expected.get(name) != after.get(name)))
    return dict(parent=before, candidate=after,
                changed_relative_to_parent=changed, patch=patch)


def check_seed(seed=SEED, candidate=CANDIDATE):
    """Require the host candidate to be exactly the device seed plus one helper rollback."""
    pins = load_pins()
    seed = _assert_source_root(seed)
    candidate = _assert_source_root(candidate)
    assert (seed / 'UPSTREAM_COMMIT').read_text().strip() == pins['upstream_commit']
    assert (candidate / 'UPSTREAM_COMMIT').read_text().strip() == pins['upstream_commit']
    before = source_map(seed)
    after = source_map(candidate)
    changed = sorted(name for name in before.keys() | after.keys()
                     if before.get(name) != after.get(name))
    assert changed == [HELPER], ('seed-to-host source delta drift', changed)
    assert before[HELPER] == pins['device_diagnostic'][HELPER]
    assert after[HELPER] == pins['diagnostic'][HELPER]
    return dict(seed=before, candidate=after, changed_relative_to_seed=changed)


def _read_json(path):
    return json.loads(Path(path).read_text())


def proof(qualified=None):
    """Validate a completed single-module build without widening its artifact claim."""
    pins = load_pins()
    if qualified is None:
        qualified = runpy.run_path(str(ROOT.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/public-gate.py'))
    assert Path(qualified['CANDIDATE']).resolve() == PARENT.resolve(), 'qualified ancestry must not be rebound'

    # This call intentionally rehashes the frozen public baseline on every proof invocation.
    original = qualified['candidate_proof']()
    sources = check_sources(PARENT, CANDIDATE)
    seed_sources = check_seed(SEED, CANDIDATE)
    assert sources['parent'] == original['source'], 'qualified whole-source drift'

    attempt = ROOT / 'build-attempt-r1'
    assert (attempt / 'exit-code.txt').read_text().strip() == '0'
    assert BUILD_MARKER in (attempt / 'post-build-verification.log').read_text()

    before = _read_json(attempt / 'candidate-source-before.json')
    after = _read_json(attempt / 'candidate-source-after.json')
    assert before == after == sources, 'candidate source changed during or after build'
    seed_before = _read_json(attempt / 'seed-source-before.json')
    seed_after = _read_json(attempt / 'seed-source-after.json')
    assert seed_before == seed_after == seed_sources['seed'], 'SDK seed source changed'
    if (attempt / 'graph-before.json').exists() or (attempt / 'graph-after.json').exists():
        assert _read_json(attempt / 'graph-before.json') == _read_json(attempt / 'graph-after.json'), 'build graph changed'

    seed_bins = binaries(SEED)
    current = binaries(CANDIDATE)
    parent_bins = binaries(PARENT)
    assert seed_bins == _read_json(attempt / 'seed-built-before.json')
    assert seed_bins == _read_json(attempt / 'seed-built-after.json')
    assert current == _read_json(attempt / 'candidate-binaries.json')
    assert parent_bins == original['binaries'] == _read_json(attempt / 'parent-binaries.json')
    assert sorted(name for name in ARTIFACTS if seed_bins[name] != current[name]) == SEED_CHANGED_BINARIES
    assert sorted(name for name in ARTIFACTS if parent_bins[name] != current[name]) == CHANGED_BINARIES
    return dict(path=str(CANDIDATE), source=sources['candidate'], binaries=current,
                parent_binaries=parent_bins, upstream_commit=pins['upstream_commit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--parent', action='store_true')
    mode.add_argument('--candidate-sources', action='store_true')
    mode.add_argument('--seed-sources', action='store_true')
    mode.add_argument('--built', action='store_true')
    args = parser.parse_args()
    if args.built:
        result = proof()
    elif args.seed_sources:
        result = dict(seed=source_map(_assert_source_root(SEED)))
    else:
        result = check_sources(PARENT, CANDIDATE if args.candidate_sources else None)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
