#!/usr/bin/env python3
"""Lead-only explicit context diagnostic; directly reuse the first common runner."""
import argparse
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-context-profile')
OUT = ROOT / 'profile-r1'
PREFIX = 'bolt-lilicorr-context-profile-r1-'
FIRST = HERE.parent / '20261019-lilicorr-b70-profile'
_FIRST_RUN = runpy.run_path(str(FIRST / 'run-profile.py'))
_FIRST_GLOBALS = _FIRST_RUN['bind'].__globals__
Q = _FIRST_RUN['Q']
S = runpy.run_path(str(HERE / 'source-proof.py'))
PROFILE_ENV = _FIRST_RUN['PROFILE_ENV']
OWN_FILES = (
    'README.md', '.gitignore', 'run-profile.py', 'source-proof.py',
    'build-candidate.sh', 'analyze-profile.py', 'test-profile.py',
    'source-hashes.json', 'lilicorr-profile.patch', 'seed-incremental.patch',
    'fixtures/common/speculative.cpp.gz', 'fixtures/src/llama-context.cpp.gz',
    'fixtures/ggml/src/ggml-sycl/ggml-sycl.cpp.gz',
    'fixtures/combined-startup-measured.json.gz', 'fixtures/scope-guard-driver.cpp',
    '../20261019-lilicorr-b70-profile/run-profile.py',
    '../20261019-lilicorr-b70-profile/source-proof.py',
    '../20261019-lilicorr-b70-profile/analyze-profile.py',
    '../20261019-lilicorr-q8-drafter/run-performance.py',
    '../20261019-lilicorr-b70-sycl-profile/analyze-profile.py',
    '../20261019-lilicorr-b70-sycl-profile/source-proof.py',
    '../20261019-lilicorr-b70-decoder-profile/source-proof.py',
    '../20261019-lilicorr-b70-decoder-profile/source-hashes.json',
    '../20261019-lilicorr-b70-decoder-profile/lilicorr-profile.patch',
)
BUILD_FILES = (
    'candidate-binaries.json', 'parent-binaries.json', 'source-diff.json',
    'post-build-verification.log', 'runtime-precheck.log',
    'build-attempt-r1/exit-code.txt', 'build-attempt-r1/image.json',
    'build-attempt-r1/seed-built-before.json', 'build-attempt-r1/seed-built-after.json',
    'build-attempt-r1/candidate-source-before.json',
    'build-attempt-r1/candidate-source-after.json',
)


def bind():
    _FIRST_GLOBALS.update(HERE=HERE, ROOT=ROOT, OUT=OUT, PREFIX=PREFIX, S=S,
                          OWN_FILES=OWN_FILES, BUILD_FILES=BUILD_FILES)
    runtime, analysis, contract = _FIRST_GLOBALS['bind']()
    assert contract['ROOT'] == ROOT and contract['OUT'] == OUT and contract['PREFIX'] == PREFIX
    assert _FIRST_GLOBALS['S']['CANDIDATE'] == S['CANDIDATE']
    assert S['PARENT'] == _FIRST_RUN['S']['PARENT']
    assert Q['CANDIDATE'] == _FIRST_RUN['Q']['CANDIDATE']
    assert Q['CANDIDATE_SIZE'] == _FIRST_RUN['Q']['CANDIDATE_SIZE']
    assert Q['CANDIDATE_SHA256'] == _FIRST_RUN['Q']['CANDIDATE_SHA256']
    return runtime, analysis, contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-profile', action='store_true')
    action.add_argument('--lead-after-profile-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == ROOT
    runtime, analysis, c = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == S['PARENT'], 'qualified ancestry must not be rebound'
    if args.lead_run_profile:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, OUT))
        assert result['status'] == 'captured' and result['restored']
        print('LILICORR_CONTEXT_PROFILE_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        profile = runpy.run_path(str(HERE / 'analyze-profile.py'))['analyze'](OUT, c)
        assert profile['status'] == 'analyzed', profile['errors']
        print('LILICORR_CONTEXT_PROFILE_ANALYZED_RESTORED_DEVELOPMENT_ONLY_NO_SPEED_CLAIM', flush=True)


if __name__ == '__main__':
    main()
