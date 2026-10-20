#!/usr/bin/env python3
"""Use unchanged qualified clean ABBA with strict extra Q8 physical-launch checks."""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-q8-target-batch4')
OUT = ROOT / 'candidate-r1'
PREFIX = 'bolt-lilicorr-q8-target-batch4-candidate-r1-'
BASE_CASE = HERE.parent / '20261020-lilicorr-iq2-batch4'
B = runpy.run_path(str(BASE_CASE / 'run-performance.py'))
S = runpy.run_path(str(HERE / 'source-proof.py'))
N = runpy.run_path(str(HERE / 'native-batch4.py'))
QUALIFIER_BIND = str(ROOT) + ':/iq2-batch4:ro'
OWN_FILES = (
    'README.md', 'q8-target-batch4.patch', 'source-pins.json', 'test-candidate.py', 'cpu-checks.txt',
    'source-proof.py', 'build-candidate.py', 'run-performance.py', 'native-batch4.py', 'iq2-native-base.py',
    'test-bindings.py', 'bindings-cpu-checks.txt',
    str(HERE.parent / '20261020-lilicorr-host-boundary/build-candidate.sh'),
    str(BASE_CASE / 'run-performance.py'), str(BASE_CASE / 'source-proof.py'),
) + tuple(str(BASE_CASE / name) for name in B['OWN_FILES'] + B['BUILD_FILES'])


def check_native(public, tag, log):
    assert tag in public['NATIVE_LOGS']
    if tag in ('q8', 'iq2'):
        return N['check'](log, tag)
    return public['check_native_log'](tag, log)


def configure():
    globals_ = B['bind'].__globals__
    assert B['main'].__globals__ is globals_
    build_files = (('build-candidate.generated.sh',) if (HERE / 'build-candidate.generated.sh').is_file() else ()) + tuple(
        str(path.relative_to(HERE)) for path in sorted((HERE / 'build-attempt-r1').rglob('*')) if path.is_file())
    globals_.update(HERE=HERE, ROOT=ROOT, OUT=OUT, PREFIX=PREFIX, S=S, N=N,
                    QUALIFIER_BIND=QUALIFIER_BIND, OWN_FILES=OWN_FILES, BUILD_FILES=build_files,
                    check_native=check_native)


def bind():
    configure()
    return B['bind']()


def main():
    configure()
    B['main']()  # Original ancestry/NoDraft/config/workload/restoration are unchanged.


if __name__ == '__main__':
    main()
