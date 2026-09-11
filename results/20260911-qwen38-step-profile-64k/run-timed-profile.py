#!/usr/bin/env python3
"""Add disposable graph timing (or eager attribution) to this campaign's driver."""
import argparse
import hashlib
import runpy
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--eager-attribution', action='store_true')
    options, remaining = parser.parse_known_args()
    if '--profile' not in remaining:
        parser.error('timing wrapper requires --profile; use the plain driver for throughput')
    driver = runpy.run_path(str(ROOT / 'run-step-profile.py'))
    namespace = driver['main'].__globals__
    original = namespace['build_launch']

    def build(cell, out, args):
        argv, metadata = original(cell, out, args)
        serve = metadata['serve']
        if cell == 'dspark':
            # 73029d424 leaves AsyncLLM.profiler unset when ignore_frontend=True.
            index = serve.index('--profiler-config.ignore_frontend=true')
            serve[index] = '--profiler-config.ignore_frontend=false'
            metadata['profiler_config']['ignore_frontend'] = False
            metadata['profiler_workaround'] = 'Enable native frontend profiling to avoid AsyncLLM.profiler AttributeError; separate profiling overhead from throughput.'
        if options.eager_attribution:
            index = serve.index('--compilation-config')
            del serve[index:index + 2]
            serve.remove('--cudagraph-metrics')
            serve.append('--enforce-eager')
            index = serve.index('--profiler-config.torch_profiler_record_shapes=false')
            serve[index] = '--profiler-config.torch_profiler_record_shapes=true'
            metadata['profiler_config']['torch_profiler_record_shapes'] = True
            index = argv.index('VLLM_XPU_ENABLE_XPU_GRAPH=1')
            argv[index] = 'VLLM_XPU_ENABLE_XPU_GRAPH=0'
            metadata['environment'] = [x.replace('VLLM_XPU_ENABLE_XPU_GRAPH=1', 'VLLM_XPU_ENABLE_XPU_GRAPH=0') for x in metadata['environment']]
            metadata['common_contract']['xpu_graphs'] = False
            metadata['attribution_mode'] = 'eager-only; not graph replay or throughput evidence'
            argv[-1] = argv[-1].rsplit('; exec ', 1)[0] + '; exec ' + shlex.join(serve)
        else:
            timing = ROOT / 'timing'
            files = [timing / ('qwen38_step_timing_' + suffix + '.py') for suffix in ('overlay', 'patch')]
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            env = ['B70_STEP_TIMING=1', 'B70_STEP_TIMING_DIR=/output/step-timing', 'B70_STEP_TIMING_MAX_SAMPLES=32', 'PYTHONPATH=/timing']
            index = argv.index('--entrypoint')
            extra = ['-v', f'{timing}:/timing:ro']
            for item in env:
                extra.extend(['-e', item])
            argv[index:index] = extra
            prefix = argv[-1].rsplit('; exec ', 1)[0]
            argv[-1] = prefix + '; /opt/venv/bin/python -P /timing/qwen38_step_timing_patch.py; exec ' + shlex.join(serve)
            metadata['environment'].extend(env)
            metadata['mounts'].append({'host': str(timing), 'container': '/timing', 'mode': 'ro'})
            metadata['timing_overlay_sha256'] = hashes
            metadata['attribution_mode'] = 'current-stream graph-region XPU events; separate from native trace and unprofiled throughput'
        return argv, metadata

    namespace['build_launch'] = build
    sys.argv = [sys.argv[0], *remaining]
    return driver['main']()


if __name__ == '__main__':
    sys.exit(main())
