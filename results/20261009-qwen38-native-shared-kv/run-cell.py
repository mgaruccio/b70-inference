#!/usr/bin/env python3
"""Revalidate unchanged native shared-KV assets under the current B70 guards."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import uuid

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
PRIOR = ROOT / '20260913-qwen38-native-grouped-verify'
BUILD = PRIOR / 'build-06'
GUARD = ROOT / '20261009-qwen38-gdn-locality/native-boundary64k-assets-20261009-100052-ff9f86/registered-controller.py'
LIBRARY = BUILD / 'build/libb70_grouped_verify.so'
IMAGE = 'vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
HASHES = {
    GUARD: 'fafe61a614a8328f63f0a7af41fa90da46041aa6cec514a01fe47b485b669da5',
    LIBRARY: 'e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec',
    PRIOR / 'run-serving.py': '76de6a9bafa47130b3a74c0478ca8b146c5e2f88b55baa555433baab9c99ae95',
    PRIOR / 'serving-overlay.py': '2b40ce2aa4adfcfdb7582ace330af1347c84845253fc1b8710e767dee3a760f5',
    PRIOR / 'grouped_verify.py': 'b02cfac244386358871f9834ca202acb6dc3b93ffb95faf0aa267d858bc222e6',
    BUILD / 'inputs/grouped_verify.py': 'b02cfac244386358871f9834ca202acb6dc3b93ffb95faf0aa267d858bc222e6',
    BUILD / 'inputs/probe.py': '69cd0722b8b49bf3802b826e3680cf53bd67f46c4234cecd5168daecdd9ade94',
    BUILD / 'inputs/check-grouped-split-k.py': 'b7e1213e0b3f454ca85479730aed1798a2efac7e10bce67c9410b41cad98cd20',
    BUILD / 'inputs/binding.cpp': '715731e1caa3e5daedec7700376af4eecb1dfcb68fdc4bcb361669ef734888d6',
    BUILD / 'inputs/patch-native.py': '17754c47b8ef63c44b9e850f7b242309d6a32dee588ca474925326266c6c8090',
}


def verify_sources():
    observed = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in HASHES}
    for path, expected in HASHES.items():
        if observed[str(path)] != expected:
            raise RuntimeError(f'frozen dependency changed: {path}')
    return observed


def operator(out, name, guard):
    inputs = out / 'inputs'
    inputs.mkdir()
    for source in HASHES:
        if source.parent == BUILD / 'inputs':
            (inputs / source.name).write_bytes(source.read_bytes())
    gid = str(Path('/dev/dri/renderD128').stat().st_gid)
    argv = ['docker', 'create', '--name', name, '--network', 'none',
            '--cpus=12', '--memory=24g', '--memory-swap=24g',
            '--device', '/dev/dri', '--group-add', gid,
            '-e', 'ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE', '-e', 'ZE_AFFINITY_MASK=0',
            '-v', f'{LIBRARY}:/candidate/libb70_grouped_verify.so:ro',
            '-v', f'{inputs}:/inputs:ro', '-v', f'{out}:/output',
            '--entrypoint', '/bin/bash', IMAGE, '-lc',
            'set -euo pipefail; /opt/venv/bin/python -m torch.utils.collect_env '
            '> /output/collect-env.txt 2>&1; exec /opt/venv/bin/python -u -B -P '
            '/inputs/probe.py --library /candidate/libb70_grouped_verify.so --output /output']
    guard['put'](out / 'command.json', {'argv': argv})
    cid = None
    try:
        create = guard['cmd'](argv, 60)
        guard['put'](out / 'container-create.json', {'returncode': create.returncode, 'stdout': create.stdout})
        if create.returncode:
            raise RuntimeError('container create failed')
        cid = create.stdout.strip()
        if not re.fullmatch('[0-9a-f]{64}', cid):
            raise RuntimeError('unexpected owned container identifier')
        with (out / 'probe.log').open('w') as log:
            subprocess.run(['docker', 'start', '--attach', cid], stdout=log,
                           stderr=subprocess.STDOUT, timeout=2700, check=False)
        inspect = guard['cmd'](['docker', 'inspect', '--format', '{{json .State}}', cid])
        if inspect.returncode:
            raise RuntimeError('cannot inspect owned probe exit status')
        state = json.loads(inspect.stdout)
        guard['put'](out / 'container-state.json', state)
        if state['Running']:
            raise RuntimeError('attach returned while container still running')
        return int(state['ExitCode'])
    finally:
        if cid and re.fullmatch('[0-9a-f]{64}', cid):
            cleanup = guard['cmd'](['docker', 'rm', '-f', cid], 60)
            guard['put'](out / 'container-cleanup.json', {'returncode': cleanup.returncode, 'stdout': cleanup.stdout})
            if cleanup.returncode:
                raise RuntimeError('owned probe cleanup failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['operator', 'baseline', 'candidate'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sources = verify_sources()
    guard = runpy.run_path(str(GUARD))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    name = ('b70shared' + uuid.uuid4().hex[:12] if args.arm == 'operator'
            else f'b70-grouped-serving-mtp4-{args.arm}')
    guard['put'](out / 'source-hashes.json', sources)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    guard['preconditions'](out, name)
    before = guard['kernel'](out, 'pre')
    if any(v['error_lines'] for v in before['commands'].values()):
        raise RuntimeError('pre-existing guarded kernel errors')
    result = 1
    try:
        if args.arm == 'operator':
            result = operator(out, name, guard)
        else:
            adapter = PRIOR / 'run-serving.py'
            for source in (adapter, PRIOR / 'serving-overlay.py', PRIOR / 'grouped_verify.py'):
                (out / source.name).write_bytes(source.read_bytes())
            sys.argv = [str(adapter), '--out', str(out / 'native'),
                        '--startup-timeout', '1800', '--benchmark-timeout', '3600']
            if args.arm == 'candidate':
                sys.argv.extend(['--candidate', '--library', str(LIBRARY)])
            guard['put'](out / 'command.json', {'argv': sys.argv.copy(), 'execution': 'runpy main'})
            result = int(runpy.run_path(str(adapter))['main']())
    finally:
        after = guard['kernel'](out, 'post')
        old = {line for v in before['commands'].values() for line in v['error_lines']}
        new = sorted({line for v in after['commands'].values() for line in v['error_lines']} - old)
        post = out / 'postconditions'
        post.mkdir()
        guard['preconditions'](post, name)
        verify_sources()
        guard['put'](out / 'cell-result.json', {
            'driver_exit': result, 'new_guarded_error_lines': new,
            'arm': args.arm, 'tier': 'development', 'quality_sensitive': True})
        if new:
            raise RuntimeError('new guarded kernel errors')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
