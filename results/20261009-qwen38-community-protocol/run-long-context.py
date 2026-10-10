#!/usr/bin/env python3
"""Extend the matched 230 W comparison through the 212992-token total limit."""
from pathlib import Path
import runpy

CAMPAIGN = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-community-protocol')
SUITE = CAMPAIGN / 'power-long-230w-01'
CELLS = ((163840, 128), (196608, 128), (212864, 128))


def main():
    assert all(prompt + output <= 212992 for prompt, output in CELLS)
    run = runpy.run_path(str(CAMPAIGN / 'run-230w.py'))
    run['main'].__globals__.update(
        SUITE=SUITE, RUN_SUFFIX='long-230w-01',
        RUN_OPTIONS={'CELLS': CELLS, 'DECODE_PROMPTS': SUITE / 'prompts.json'})
    try:
        run['main']()
    finally:
        if SUITE.is_dir():
            (SUITE / 'run-long-context.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
