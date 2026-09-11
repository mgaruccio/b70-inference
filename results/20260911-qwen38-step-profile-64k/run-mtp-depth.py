#!/usr/bin/env python3
"""One measured candidate: MTP2 vs MTP4, same production-capacity benchmark shape."""
import argparse
import json
import runpy
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / '20260911-qwen38-dspark-layer-norm'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--depth', type=int, choices=(2, 4), required=True)
    options, remaining = parser.parse_known_args()
    if '--profile' in remaining:
        parser.error('this candidate comparison is unprofiled')
    driver = runpy.run_path(str(ROOT / 'run-step-profile.py'))
    namespace = driver['main'].__globals__
    namespace['CONTEXT'] = 212992
    namespace['BATCHED_TOKENS'] = 8192
    namespace['cell_name'] = lambda cell: f'b70-step-depth-mtp{options.depth}'
    original_build = namespace['build_launch']
    original_bench = namespace['run_benchmark']

    def build(cell, out, args):
        if cell != 'mtp4':
            raise ValueError('depth experiment only supports the MTP bundle')
        argv, metadata = original_build(cell, out, args)
        serve = metadata['serve']
        index = serve.index('--speculative-config') + 1
        spec = json.loads(serve[index])
        assert spec == {'method': 'mtp', 'num_speculative_tokens': 4}
        spec['num_speculative_tokens'] = options.depth
        serve[index] = json.dumps(spec)
        argv[-1] = argv[-1].rsplit('; exec ', 1)[0] + '; exec ' + shlex.join(serve)
        metadata['speculation'] = spec
        metadata['experiment'] = f'MTP{options.depth}; production-capacity 212992/8192; no acceptance or quantization changes'
        return argv, metadata

    def benchmark(*args, **kwargs):
        # Use the original campaign's real-API canaries, functional sandbox,
        # finite logprob boundaries, and repeatability smoke before timing.
        out = Path(args[0])
        diag = runpy.run_path(str(PREVIOUS / 'run-acceptance-diagnostics.py'))
        ns = diag['run'].__globals__
        probe, checks = ns['load_previous_modules']()
        probe.IMAGE = namespace['DEFAULT_MTP_IMAGE']
        client = ns['DiagnosticClient'](out, request_timeout=900)
        gates = ns['run_shared_gates'](client, out, probe, checks, True)
        assert gates['status'] == 'passed'
        return original_bench(*args, **kwargs)

    namespace['build_launch'] = build
    namespace['run_benchmark'] = benchmark
    sys.argv = [sys.argv[0], *remaining]
    return driver['main']()


if __name__ == '__main__':
    sys.exit(main())
