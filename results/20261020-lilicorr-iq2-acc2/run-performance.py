#!/usr/bin/env python3
"""Bind the unchanged qualified IQ2 clean lifecycle to the exact two-accumulator n4 source."""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-iq2-acc2')
OUT = ROOT / 'candidate-r1'
PREFIX = 'bolt-lilicorr-iq2-acc2-candidate-r1-'
BASE_CASE = HERE.parent / '20261020-lilicorr-iq2-batch4'
B = runpy.run_path(str(BASE_CASE / 'run-performance.py'))
S = runpy.run_path(str(HERE / 'source-proof.py'))
N = runpy.run_path(str(HERE / 'native-batch4.py'))
QUALIFIER_BIND = str(ROOT) + ':/iq2-batch4:ro'
OWN_FILES = (
    'README.md', 'iq2-acc2.patch', 'source-pins.json',
    'test-candidate.py', 'cpu-checks.txt', 'source-proof.py',
    'build-candidate.py', 'run-performance.py',
    'test-bindings.py', 'bindings-cpu-checks.txt', 'native-batch4.py',
    str(BASE_CASE / 'run-performance.py'), str(BASE_CASE / 'source-proof.py'),
    str(HERE.parent / '20261020-lilicorr-host-boundary/build-candidate.sh'),
) + tuple(str(BASE_CASE / name) for name in B['OWN_FILES'] + B['BUILD_FILES'])


def configure():
    globals_ = B['bind'].__globals__
    assert B['main'].__globals__ is globals_
    build_files = (('build-candidate.generated.sh',) if (HERE / 'build-candidate.generated.sh').is_file() else ()) + tuple(str(p.relative_to(HERE)) for p in sorted((HERE / 'build-attempt-r1').rglob('*')) if p.is_file())
    globals_.update(HERE=HERE, ROOT=ROOT, OUT=OUT, PREFIX=PREFIX, S=S, N=N,
                    QUALIFIER_BIND=QUALIFIER_BIND, OWN_FILES=OWN_FILES, BUILD_FILES=build_files)


def bind():
    configure()
    return B['bind']()


def main():
    configure()
    B['main']()  # Same flags, original baseline, checks, separate restore and cleanup.


if __name__ == '__main__':
    main()
