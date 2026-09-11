#!/usr/bin/env python3
"""Reuse the existing real-API driver for annotated target eager attribution."""
import argparse
import hashlib
import runpy
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / '20260911-qwen38-step-profile-64k'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cell', choices=('mtp4', 'dspark'), required=True)
    options, remaining = parser.parse_known_args()
    driver = runpy.run_path(str(PREVIOUS / 'run-step-profile.py'))
    ns = driver['main'].__globals__
    ns['CAMPAIGN'] = ROOT.name
    ns['CONTEXT'] = 212992 if options.cell == 'mtp4' else 65664
    ns['BATCHED_TOKENS'] = 8192 if options.cell == 'mtp4' else 2048
    ns['cell_name'] = lambda cell: f'b70-target-internals-{cell}'
    original = ns['build_launch']

    def build(cell, out, args):
        argv, metadata = original(cell, out, args)
        serve = metadata['serve']
        index = serve.index('--compilation-config')
        del serve[index:index + 2]
        serve.remove('--cudagraph-metrics')
        serve.append('--enforce-eager')
        for old, new in [('--profiler-config.torch_profiler_record_shapes=false', '--profiler-config.torch_profiler_record_shapes=true')]:
            serve[serve.index(old)] = new
        metadata['profiler_config']['torch_profiler_record_shapes'] = True
        if cell == 'dspark':
            serve[serve.index('--profiler-config.ignore_frontend=true')] = '--profiler-config.ignore_frontend=false'
            metadata['profiler_config']['ignore_frontend'] = False
        argv[argv.index('VLLM_XPU_ENABLE_XPU_GRAPH=1')] = 'VLLM_XPU_ENABLE_XPU_GRAPH=0'
        module = ROOT / 'target-annotations.py'
        patch = ROOT / 'qwen38_step_timing_patch.py'
        extra = ['-v', f'{module}:/annotations/qwen38_step_timing_overlay.py:ro',
                 '-v', f'{patch}:/annotations/patch.py:ro',
                 '-e', 'PYTHONPATH=/annotations', '-e', 'B70_STEP_TIMING=1']
        index = argv.index('--entrypoint')
        argv[index:index] = extra
        prefix = argv[-1].rsplit('; exec ', 1)[0]
        argv[-1] = prefix + '; /opt/venv/bin/python -P /annotations/patch.py; exec ' + shlex.join(serve)
        metadata['environment'] = [x.replace('VLLM_XPU_ENABLE_XPU_GRAPH=1', 'VLLM_XPU_ENABLE_XPU_GRAPH=0') for x in metadata['environment']]
        metadata['environment'].extend(['PYTHONPATH=/annotations', 'B70_STEP_TIMING=1'])
        metadata['common_contract']['xpu_graphs'] = False
        metadata['annotation_sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (module, patch)}
        metadata['attribution_only'] = 'Eager target-forward module labels; kernel attribution, not graph throughput or replay timing.'
        return argv, metadata

    ns['build_launch'] = build
    sys.argv = [sys.argv[0], '--cell', options.cell, '--profile',
                '--profile-delay-iterations', '3', '--profile-stop-after-events', '24', *remaining]
    return driver['main']()


if __name__ == '__main__':
    sys.exit(main())
