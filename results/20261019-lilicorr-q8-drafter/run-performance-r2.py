#!/usr/bin/env python3
"""Fresh Q8 retry after approved lossless log cleanup; preserve failed timing-r1."""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent


def configure():
    adapter = runpy.run_path(str(HERE / 'run-performance.py'))
    globals_ = adapter['bind'].__globals__
    globals_['OUT'] = globals_['ROOT'] / 'timing-r2'
    globals_['PREFIX'] = 'bolt-lilicorr-q8-drafter-timing-r2-'
    globals_['OWN_FILES'] = (*globals_['OWN_FILES'], 'run-performance-r2.py')
    return globals_


if __name__ == '__main__':
    configure()['main']()
