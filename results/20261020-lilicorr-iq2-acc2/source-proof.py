#!/usr/bin/env python3
"""Private exact two-accumulator n4 composition; reuse the qualified source/artifact primitives."""
import argparse
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-iq2-acc2')
BASE_CASE = HERE.parent / '20261020-lilicorr-iq2-batch4'
BASE = runpy.run_path(str(BASE_CASE / 'source-proof.py'))
PARENT = BASE['PARENT']
SEED = BASE['CANDIDATE']
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-acc2-r1')
SEED_SOURCE_FILE = 'ggml/src/ggml-sycl/mmvq.cpp'
SOURCE_SCOPE = BASE['SOURCE_SCOPE']
ARTIFACTS = BASE['ARTIFACTS']
CHANGED_BINARIES = ['libggml-sycl.so.0']
SEED_CHANGED_BINARIES = CHANGED_BINARIES
BUILD_MARKER = 'LILICORR_IQ2_ACC2_GGML_SYCL_MODULE_PASSED'
PATCH_SHA256 = 'b2701fff63762d8bd397cbe24b42eb1c852986a1c5d691f69d847d5dc8ed501b'
sha, source_map, binaries = (BASE[k] for k in ('sha', 'source_map', 'binaries'))


def read_json(path):
    return json.loads(Path(path).read_text())


def load_pins():
    pins = read_json(HERE / 'source-pins.json')
    assert sha(HERE / 'iq2-acc2.patch') == PATCH_SHA256
    assert BASE['patch_stats']()['sha256'] == pins['batch4_patch_sha256']
    assert pins['batch4_mmvq_sha256'] != pins['acc2_mmvq_sha256']
    return pins


def seed_sources():
    """Tie the unchanged SDK seed to its actual completed build, not its name."""
    pins = load_pins()
    BASE['proof']()
    record = read_json(BASE_CASE / 'build-attempt-r1/candidate-source-after.json')
    current = source_map(SEED)
    assert current == record['candidate'], 'completed seed source drift'
    assert current[SEED_SOURCE_FILE] == pins['batch4_mmvq_sha256']
    assert binaries(SEED) == read_json(BASE_CASE / 'build-attempt-r1/candidate-binaries.json')
    return current, record['parent']


def check_sources(parent=PARENT, candidate=None):
    assert __debug__
    parent = Path(parent)
    assert parent.is_dir() and not (parent / '.git').exists()
    pins = load_pins()
    before = BASE['check_sources'](parent)['parent']
    assert (parent / 'UPSTREAM_COMMIT').read_text().strip() == 'b6334e37e6ef286fc5091a41b6561b2113d915fb'
    for name, expected in pins['qualified_sycl_sha256'].items():
        assert before['ggml/src/ggml-sycl/' + name] == expected, name
    assert before['ggml/src/ggml-common.h'] == pins['codebook_sha256']
    seed, qualified_before = seed_sources()
    assert before == qualified_before, 'whole qualified parent source drift'
    patch = dict(sha256=PATCH_SHA256, seed_patch_sha256=pins['batch4_patch_sha256'])
    if candidate is None:
        return dict(parent=before, patch=patch)
    candidate = Path(candidate)
    assert candidate.is_dir() and not (candidate / '.git').exists()
    expected = dict(seed)
    expected[SEED_SOURCE_FILE] = pins['acc2_mmvq_sha256']
    after = source_map(candidate)
    assert after == expected, 'unexpected packed source delta'
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert changed == sorted(SOURCE_SCOPE)
    return dict(parent=before, candidate=after, changed_relative_to_parent=changed, patch=patch)


def check_seed(seed=SEED, candidate=CANDIDATE):
    assert Path(seed) == SEED
    before, _parent = seed_sources()
    after = source_map(candidate)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert changed == [SEED_SOURCE_FILE]
    assert after[SEED_SOURCE_FILE] == load_pins()['acc2_mmvq_sha256']
    return dict(seed=before, candidate=after, changed_relative_to_seed=changed)


def proof(qualified=None):
    if qualified is None:
        qualified = runpy.run_path(str(ROOT.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/public-gate.py'))
    assert Path(qualified['CANDIDATE']) == PARENT, 'qualified ancestry must not be rebound'
    original = qualified['candidate_proof']()  # Fresh whole-source/six-artifact frozen baseline proof.
    sources = check_sources(PARENT, CANDIDATE)
    seed = check_seed(SEED, CANDIDATE)
    assert sources['parent'] == original['source']
    attempt = ROOT / 'build-attempt-r1'
    assert (attempt / 'exit-code.txt').read_text().strip() == '0'
    assert BUILD_MARKER in (attempt / 'post-build-verification.log').read_text()
    assert read_json(attempt / 'candidate-source-before.json') == read_json(attempt / 'candidate-source-after.json') == sources
    assert read_json(attempt / 'seed-source-before.json') == read_json(attempt / 'seed-source-after.json') == seed['seed']
    assert read_json(attempt / 'graph-before.json') == read_json(attempt / 'graph-after.json')
    parent_bins, seed_bins, current = (binaries(path) for path in (PARENT, SEED, CANDIDATE))
    assert parent_bins == original['binaries'] == read_json(attempt / 'parent-binaries.json')
    assert seed_bins == read_json(attempt / 'seed-built-before.json') == read_json(attempt / 'seed-built-after.json')
    assert current == read_json(attempt / 'candidate-binaries.json')
    assert sorted(k for k in ARTIFACTS if seed_bins[k] != current[k]) == SEED_CHANGED_BINARIES
    assert sorted(k for k in ARTIFACTS if parent_bins[k] != current[k]) == CHANGED_BINARIES
    return dict(path=str(CANDIDATE), source=sources['candidate'], binaries=current,
                parent_binaries=parent_bins, changed_binaries=CHANGED_BINARIES,
                upstream_commit='b6334e37e6ef286fc5091a41b6561b2113d915fb')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('parent', 'candidate-sources', 'seed-sources', 'built'):
        mode.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    result = (proof() if args.built else seed_sources()[0] if args.seed_sources else
              check_sources(PARENT, CANDIDATE if args.candidate_sources else None))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
