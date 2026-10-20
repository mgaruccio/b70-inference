#!/usr/bin/env python3
"""Private diagnostic composition and existing six-artifact checks; no runtime work."""
import argparse
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-device-timing')
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-timing-r1')
BASE = runpy.run_path(str(HERE.parent / '20261020-lilicorr-iq2-batch4/source-proof.py'))
sha, source_map, binaries = (BASE[name] for name in ('sha', 'source_map', 'binaries'))
ARTIFACTS = BASE['ARTIFACTS']
CHANGED_BINARIES = sorted(('libggml-sycl.so.0', 'libllama.so.0.7.0', 'libllama-common.so'))
BUILD_MARKER = 'LILICORR_DEVICE_TIMING_THREE_MODULE_BUILD_PASSED'


def load_pins():
    pins = json.loads((HERE / 'source-pins.json').read_text())
    assert pins['upstream_commit'] == 'b6334e37e6ef286fc5091a41b6561b2113d915fb'
    assert sha(HERE / 'device-timing.patch') == pins['patch_sha256']
    for group in ('ancestry_files', 'reused_fixtures'):
        for name, expected in pins[group].items():
            relative = Path(name).relative_to('results')
            assert '..' not in relative.parts and not relative.is_absolute()
            assert sha(HERE.parent / relative) == expected, ('ancestry drift', name)
    for stage in ('parent', 'context_seed', 'composed_seed', 'diagnostic'):
        assert 'UPSTREAM_COMMIT' in pins[stage]
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts
                   for name in pins[stage]), ('unsafe source pin', stage)
    return pins


def check_sources(parent=PARENT, candidate=None):
    """Check the actual entire export, not just the files mentioned by a patch."""
    assert __debug__, 'assertions required'
    pins = load_pins()
    parent = Path(parent)
    assert parent.is_dir() and not (parent / '.git').exists()
    assert (parent / 'UPSTREAM_COMMIT').read_text().strip() == pins['upstream_commit']
    before = source_map(parent)
    for name, expected in pins['parent'].items():
        assert before.get(name) == expected, ('qualified source drift', name)
    patch = dict(sha256=pins['patch_sha256'], ancestry=pins['ancestry_files'])
    if candidate is None:
        return dict(parent=before, patch=patch)
    candidate = Path(candidate)
    assert candidate.is_dir() and not (candidate / '.git').exists()
    expected = dict(before)
    expected.update(pins['diagnostic'])
    after = source_map(candidate)
    changed = sorted(name for name in before.keys() | expected.keys()
                     if before.get(name) != expected.get(name))
    assert after == expected, ('unexpected diagnostic source delta', sorted(
        name for name in expected.keys() | after.keys() if expected.get(name) != after.get(name)))
    assert changed, 'diagnostic source is unchanged'
    return dict(parent=before, candidate=after, changed_relative_to_parent=changed, patch=patch)


def proof():
    sources = check_sources(PARENT, CANDIDATE)
    attempt = ROOT / 'build-attempt-r1'
    assert (attempt / 'exit-code.txt').read_text().strip() == '0'
    assert BUILD_MARKER in (attempt / 'post-build-verification.log').read_text()
    before = json.loads((attempt / 'candidate-source-before.json').read_text())
    after = json.loads((attempt / 'candidate-source-after.json').read_text())
    assert before == after == sources, 'source changed during or after build'
    current, parent = binaries(CANDIDATE), binaries(PARENT)
    assert current == json.loads((attempt / 'candidate-binaries.json').read_text())
    assert parent == json.loads((attempt / 'parent-binaries.json').read_text())
    assert sorted(name for name in ARTIFACTS if current[name] != parent[name]) == CHANGED_BINARIES
    return dict(**sources, binaries=current, parent_binaries=parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--parent', action='store_true')
    mode.add_argument('--candidate-sources', action='store_true')
    mode.add_argument('--built', action='store_true')
    args = parser.parse_args()
    result = proof() if args.built else check_sources(PARENT, CANDIDATE if args.candidate_sources else None)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
