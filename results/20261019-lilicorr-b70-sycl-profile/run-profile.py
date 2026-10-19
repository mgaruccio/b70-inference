#!/usr/bin/env python3
"""Lead-only second timing diagnostic layered on the first profile adapter."""
import argparse
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261019-lilicorr-b70-sycl-profile')
OUT = ROOT / 'profile-r1'
PREFIX = 'bolt-lilicorr-b70-sycl-profile-r1-'
FIRST = HERE.parent / '20261019-lilicorr-b70-profile'

# Freshly load the first runner and reuse its Q8 adapter/controller lifecycle.  Only
# this case's private paths and proof map are rebound; the qualified P/C ancestry is not.
_FIRST_RUN = runpy.run_path(str(FIRST / 'run-profile.py'))
_FIRST_GLOBALS = _FIRST_RUN['bind'].__globals__
Q = _FIRST_RUN['Q']
S = runpy.run_path(str(HERE / 'source-proof.py'))
PROFILE_ENV = _FIRST_GLOBALS['PROFILE_ENV']
OWN_FILES = (
    'README.md', '.gitignore', 'run-profile.py', 'source-proof.py', 'build-candidate.sh',
    'analyze-profile.py', 'test-profile.py', 'source-hashes.json', 'lilicorr-profile.patch',
    'fixtures/common/speculative.cpp.gz', 'fixtures/src/llama-context.cpp.gz',
    'fixtures/ggml/src/ggml-sycl/ggml-sycl.cpp.gz',
    'fixtures/combined-startup-measured.json.gz',
    '../20261019-lilicorr-b70-profile/run-profile.py',
    '../20261019-lilicorr-b70-profile/analyze-profile.py',
    '../20261019-lilicorr-b70-profile/source-proof.py',
)
BUILD_FILES = (
    'candidate-binaries.json', 'parent-binaries.json', 'source-diff.json',
    'post-build-verification.log', 'runtime-precheck.log', 'build-attempt-r1/exit-code.txt',
    'build-attempt-r1/image.json',
)


def bind():
    # Functions in _FIRST_RUN close over their actual module globals.  Rebind only
    # the case-local evidence paths and proof; all runtime checks remain inherited.
    _FIRST_GLOBALS.update(HERE=HERE, ROOT=ROOT, OUT=OUT, PREFIX=PREFIX, S=S,
                          OWN_FILES=OWN_FILES, BUILD_FILES=BUILD_FILES)
    return _FIRST_GLOBALS['bind']()


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
        print('LILICORR_SYCL_PROFILE_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        profile = runpy.run_path(str(HERE / 'analyze-profile.py'))['analyze'](OUT, c)
        assert profile['status'] == 'analyzed', profile['errors']
        print('LILICORR_SYCL_PROFILE_ANALYZED_RESTORED_DEVELOPMENT_ONLY_NO_SPEED_CLAIM', flush=True)


if __name__ == '__main__':
    main()
