#!/usr/bin/env python3
"""Lead-only host-boundary diagnostic on the qualified original queue initializer."""
import argparse
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-host-boundary')
OUT = ROOT / 'profile-r1'
PREFIX = 'bolt-lilicorr-host-boundary-r1-'
DEVICE = HERE.parent / '20261020-lilicorr-device-timing'
LEGACY = runpy.run_path(str(DEVICE / 'run-profile.py'))
S = runpy.run_path(str(HERE / 'source-proof.py'))
ORIGINAL_SCOPE_CHECK = LEGACY['ORIGINAL_SCOPE_CHECK']
Q, CTX, BATCH = (LEGACY[name] for name in ('Q', 'CTX', 'BATCH'))
PROFILE_ENV = LEGACY['PROFILE_ENV']
QUALIFIER_BIND, NEW_QUALIFIER, OLD_QUALIFIER = (LEGACY[name] for name in ('QUALIFIER_BIND', 'NEW_QUALIFIER', 'OLD_QUALIFIER'))
qualifier_argv = LEGACY['qualifier_argv']
OWN_FILES = (
    'README.md', '.gitignore', 'host-boundary.patch', 'disable-device-profiling.patch',
    'source-pins.json', 'test-host-source.py', 'cpu-checks.txt',
    'source-proof.py', 'build-candidate.sh', 'test-build-proof.py', 'build-cpu-checks.txt',
    'run-profile.py', 'test-runtime.py', 'analyze-host.py', 'test-analyze-host.py',
    '../20261020-lilicorr-iq2-batch4/build-module.py',
    '../20261020-lilicorr-iq2-batch4/build-inputs.json.gz',
    str(DEVICE / 'run-profile.py'), str(DEVICE / 'analyze-device.py'),
) + tuple(str(DEVICE / name) for name in LEGACY['OWN_FILES'])


def check_first_job_scope(stderr):
    scope = ORIGINAL_SCOPE_CHECK(stderr)
    current = '\n'.join(stderr.splitlines()[scope['request_start_line'] - 1:])
    host = runpy.run_path(str(HERE / 'analyze-host.py'))['check_request'](current)
    assert host['status'] == 'ok' and not host['gpu_time_claim']
    assert host['gpu_timestamps'] == 'unavailable'
    return dict(**scope, host_boundary=host)


def bind():
    # Build proof is authoritative. The list below only hashes all retained build
    # evidence for before/after lifecycle immutability; it is not source identity.
    attempt = HERE / 'build-attempt-r1'
    build_files = tuple(str(p.relative_to(HERE)) for p in sorted(attempt.rglob('*')) if p.is_file())
    globals_ = LEGACY['bind'].__globals__
    globals_.update(HERE=HERE, ROOT=ROOT, OUT=OUT, PREFIX=PREFIX, S=S,
                    OWN_FILES=OWN_FILES, BUILD_FILES=build_files,
                    check_first_job_scope=check_first_job_scope)
    runtime, analysis, c = LEGACY['bind']()
    assert c['ROOT'] == ROOT and c['OUT'] == OUT and c['PREFIX'] == PREFIX
    return runtime, analysis, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--lead-run-profile', action='store_true')
    mode.add_argument('--lead-after-profile-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == ROOT
    runtime, analysis, c = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == S['PARENT'], 'qualified ancestry must not be rebound'
    if args.lead_run_profile:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, OUT))
        assert result['status'] == 'captured' and result['restored'], result
        print('HOST_BOUNDARY_CAPTURED_RESTORED_DIAGNOSTIC_ONLY', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        full = runpy.run_path(str(DEVICE.parent / '20261020-lilicorr-context-profile/analyze-profile.py'))['analyze'](OUT, c)
        assert full['status'] == 'analyzed', full['errors']
        host = runpy.run_path(str(HERE / 'analyze-host.py'))['analyze'](OUT, c)
        assert host['status'] == 'analyzed', host['errors']
        print('HOST_BOUNDARY_ANALYZED_SEPARATELY_RESTORED_NO_GPU_TIME_CLAIM', flush=True)


if __name__ == '__main__':
    main()
