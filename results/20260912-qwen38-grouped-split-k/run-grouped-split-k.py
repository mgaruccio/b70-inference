#!/usr/bin/env python3
"""Development-only MTP4 A/B; reuse the existing real-API driver and gates."""
import argparse
import hashlib
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT.parent / '20260911-qwen38-step-profile-64k'
DIAGNOSTICS = ROOT.parent / '20260911-qwen38-dspark-layer-norm'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--candidate', action='store_true')
    options, remaining = parser.parse_known_args()
    driver = runpy.run_path(str(PROFILE / 'run-step-profile.py'))
    ns = driver['main'].__globals__
    ns['CAMPAIGN'] = ROOT.name
    ns['CONTEXT'] = 212992
    ns['BATCHED_TOKENS'] = 8192
    variant = 'candidate' if options.candidate else 'baseline'
    ns['cell_name'] = lambda cell: f'b70-grouped-split-k-{variant}'
    original_build = ns['build_launch']
    original_benchmark = ns['run_benchmark']

    def build(cell, out, args):
        if cell != 'mtp4':
            raise ValueError('this experiment is fixed MTP4 only')
        argv, metadata = original_build(cell, out, args)
        metadata['experiment'] = f'grouped-query Split-K {variant}; fixed MTP4, 212992/8192'
        if options.candidate:
            module = ROOT / 'qwen38_grouped_split_k.py'
            patch = ROOT / 'qwen38_step_timing_patch.py'
            mounts = [
                {'host': str(module), 'container': '/experiment/qwen38_step_timing_overlay.py', 'mode': 'ro', 'role': 'grouped_split_k'},
                {'host': str(patch), 'container': '/experiment/patch.py', 'mode': 'ro', 'role': 'worker_import_shim'},
            ]
            environment = ['PYTHONPATH=/experiment', 'B70_STEP_TIMING=1', 'B70_GROUPED_SPLIT_K=1', 'B70_GROUPED_SPLIT_K_SPLITS=32']
            extra = []
            for mount in mounts:
                extra.extend(['-v', f"{mount['host']}:{mount['container']}:ro"])
            for value in environment:
                extra.extend(['-e', value])
            index = argv.index('--entrypoint')
            argv[index:index] = extra
            prefix, serve = argv[-1].rsplit('; exec ', 1)
            argv[-1] = prefix + '; /opt/venv/bin/python -P /experiment/patch.py; exec ' + serve
            metadata['mounts'].extend(mounts)
            metadata['environment'].extend(environment)
            metadata['candidate_sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (module, patch)}
        return argv, metadata

    def benchmark(*args, **kwargs):
        out = Path(args[0])
        diag = runpy.run_path(str(DIAGNOSTICS / 'run-acceptance-diagnostics.py'))
        dns = diag['run'].__globals__
        probe, checks = dns['load_previous_modules']()
        probe.IMAGE = ns['DEFAULT_MTP_IMAGE']
        client = dns['DiagnosticClient'](out, request_timeout=900)
        gates = dns['run_shared_gates'](client, out, probe, checks, True)
        assert gates['status'] == 'passed'
        return original_benchmark(*args, **kwargs)

    ns['build_launch'] = build
    ns['run_benchmark'] = benchmark
    sys.argv = [sys.argv[0], '--cell', 'mtp4', *remaining]
    return driver['main']()


if __name__ == '__main__':
    sys.exit(main())
