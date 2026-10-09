#!/usr/bin/env python3
"""Reuse the guarded operator runner with a new, isolated diagnostic input set."""
import argparse
import hashlib
from pathlib import Path
import runpy
import socket
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'inference-host':
        raise RuntimeError('GPU diagnostics must run on inference-host')
    root = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
    previous = root / '20261009-qwen38-native-shared-kv/run-cell.py'
    if hashlib.sha256(previous.read_bytes()).hexdigest() != '58fc22ce410d84c6559c07109973568078e6ac80004a2c8c0769a25dc277fd09':
        raise RuntimeError('Frozen operator runner changed')
    loaded = runpy.run_path(str(previous))
    loaded['verify_sources']()
    ns = loaded['main'].__globals__
    original_inputs = ns['BUILD'] / 'inputs'
    out = args.out.resolve()
    # Only private copies change; do not mutate any original campaign dependency.
    assets = out.with_name(out.name + '-assets')
    assets.mkdir(parents=True, exist_ok=False)
    inputs = assets / 'inputs'
    inputs.mkdir()
    for source, name in (
        (original_inputs / 'probe.py', 'prior-probe.py'),
        (original_inputs / 'grouped_verify.py', 'grouped_verify.py'),
        (original_inputs / 'check-grouped-split-k.py', 'check-grouped-split-k.py'),
        (Path(__file__).with_name('probe-auto.py'), 'probe.py'),
    ):
        (inputs / name).write_bytes(source.read_bytes())
    ns['BUILD'] = assets
    ns['HASHES'] = {**ns['HASHES'], **{p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()},
                    previous: hashlib.sha256(previous.read_bytes()).hexdigest(),
                    Path(__file__): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    sys.argv = [str(previous), '--arm', 'operator', '--out', str(out)]
    try:
        return int(loaded['main']())
    finally:
        if out.is_dir():
            (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    raise SystemExit(main())
