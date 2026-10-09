#!/usr/bin/env python3
"""Run one owned, bounded operator probe under the existing B70 host guards."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import uuid

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
GUARD = ROOT / '20261009-qwen38-gdn-locality/native-boundary64k-assets-20261009-100052-ff9f86/registered-controller.py'
GUARD_SHA = 'fafe61a614a8328f63f0a7af41fa90da46041aa6cec514a01fe47b485b669da5'
IMAGE = 'vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(GUARD.read_bytes()).hexdigest() != GUARD_SHA:
        raise RuntimeError('existing guard source changed')
    guard = runpy.run_path(str(GUARD))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('probe-gemm-padding.py').resolve()
    # Freeze the actual executed probe beside its raw output.
    frozen = out / source.name
    frozen.write_bytes(source.read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    name = 'b70gemm' + uuid.uuid4().hex[:12]
    guard['preconditions'](out, name)
    before = guard['kernel'](out, 'pre')
    if any(v['error_lines'] for v in before['commands'].values()):
        raise RuntimeError('pre-existing guarded kernel errors')
    gid = str(Path('/dev/dri/renderD128').stat().st_gid)
    argv = ['docker', 'create', '--name', name, '--network', 'none', '--ipc', 'host',
            '--device', '/dev/dri', '--group-add', gid,
            '-e', 'VLLM_TARGET_DEVICE=xpu', '-e', 'ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE',
            '-e', 'ZE_AFFINITY_MASK=0', '-v', f'{frozen}:/probe.py:ro',
            '-v', f'{out}:/output', '--entrypoint', '/opt/venv/bin/python', IMAGE,
            '-B', '/probe.py', '--out', '/output/probe.json']
    guard['put'](out / 'command.json', {'argv': argv, 'source_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest()})
    cid = None
    result = 1
    try:
        create = guard['cmd'](argv, 60)
        guard['put'](out / 'container-create.json', {'returncode': create.returncode, 'stdout': create.stdout})
        if create.returncode != 0:
            raise RuntimeError('container create failed')
        cid = create.stdout.strip()
        if not re.fullmatch('[0-9a-f]{64}', cid):
            raise RuntimeError('unexpected owned container identifier')
        try:
            process = subprocess.run(['docker', 'start', '--attach', cid], capture_output=True,
                                     text=True, timeout=1200, check=False)
            (out / 'probe.log').write_text(process.stdout + '\nSTDERR:\n' + process.stderr)
        except subprocess.TimeoutExpired as error:
            (out / 'probe.log').write_bytes((error.stdout or b'') + b'\nTIMEOUT\n' + (error.stderr or b''))
            raise
        inspect = guard['cmd'](['docker', 'inspect', '--format', '{{json .State}}', cid])
        if inspect.returncode:
            raise RuntimeError('cannot inspect owned probe exit status')
        state = json.loads(inspect.stdout)
        guard['put'](out / 'container-state.json', state)
        if state['Running']:
            raise RuntimeError('attach returned while container still running')
        result = int(state['ExitCode'])
    finally:
        if cid and re.fullmatch('[0-9a-f]{64}', cid):
            cleanup = guard['cmd'](['docker', 'rm', '-f', cid], 60)
            guard['put'](out / 'container-cleanup.json', {'returncode': cleanup.returncode, 'stdout': cleanup.stdout})
            if cleanup.returncode:
                result = 1
        after = guard['kernel'](out, 'post')
        old = {line for v in before['commands'].values() for line in v['error_lines']}
        new = sorted({line for v in after['commands'].values() for line in v['error_lines']} - old)
        post = out / 'postconditions'
        post.mkdir()
        guard['preconditions'](post, name)
        guard['put'](out / 'cell-result.json', {'driver_exit': result,
                    'new_guarded_error_lines': new, 'tier': 'development',
                    'purpose': 'synthetic operator probe, not serving throughput'})
        if new:
            raise RuntimeError('new guarded kernel errors')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
