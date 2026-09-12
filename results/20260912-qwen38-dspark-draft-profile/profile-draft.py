"""Real corrected-DSpark HTTP profile using the existing cleanup/workload driver."""
import hashlib
from pathlib import Path
import runpy
import shlex
import sys

ROOT = Path(__file__).resolve().parent


def main():
    driver = runpy.run_path(str(ROOT.parent / '20260911-qwen38-step-profile-64k/run-step-profile.py'))
    ns = driver['main'].__globals__
    ns.update(CAMPAIGN=ROOT.name, CONTEXT=65664, BATCHED_TOKENS=2048,
              cell_name=lambda cell: 'b70-dspark-draft-profile')
    original = ns['build_launch']

    def build(cell, out, args):
        argv, metadata = original(cell, out, args)
        serve = metadata['serve']
        index = serve.index('--compilation-config')
        del serve[index:index + 2]
        serve.remove('--cudagraph-metrics')
        serve.append('--enforce-eager')
        for key, old, new in [('torch_profiler_record_shapes', 'false', 'true'),
                              ('ignore_frontend', 'true', 'false')]:
            serve[serve.index(f'--profiler-config.{key}={old}')] = f'--profiler-config.{key}={new}'
            metadata['profiler_config'][key] = new == 'true'
        setting = 'VLLM_XPU_ENABLE_XPU_GRAPH=1'
        argv[argv.index(setting)] = setting[:-1] + '0'
        module = ROOT / 'draft-annotations.py'
        patch = ROOT.parent / '20260911-qwen38-target-verification/qwen38_step_timing_patch.py'
        index = argv.index('--entrypoint')
        argv[index:index] = ['-v', f'{module}:/annotations/qwen38_step_timing_overlay.py:ro',
                             '-v', f'{patch}:/annotations/patch.py:ro',
                             '-e', 'PYTHONPATH=/annotations', '-e', 'B70_STEP_TIMING=1']
        prefix = argv[-1].rsplit('; exec ', 1)[0]
        argv[-1] = prefix + '; /opt/venv/bin/python -P /annotations/patch.py; exec ' + shlex.join(serve)
        metadata['environment'] = [x.replace(setting, setting[:-1] + '0') for x in metadata['environment']]
        metadata['environment'].extend(['PYTHONPATH=/annotations', 'B70_STEP_TIMING=1'])
        metadata['common_contract']['xpu_graphs'] = False
        metadata['annotation_sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (module, patch)}
        metadata['attribution_only'] = 'Eager draft backbone, sampling and context-KV scopes; not graph throughput.'
        return argv, metadata

    ns['build_launch'] = build
    sys.argv = [sys.argv[0], '--cell', 'dspark', '--profile',
                '--profile-delay-iterations', '3', '--profile-stop-after-events', '24', *sys.argv[1:]]
    return driver['main']()


if __name__ == '__main__':
    raise SystemExit(main())
