#!/usr/bin/env python3
"""Run one complete frozen quality arm, then score it in the original CPU image."""
import argparse
import hashlib
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('target', 'native', 'candidate'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'inference-host':
        raise RuntimeError('Full quality must run on inference-host, never Pi')
    frozen_path = Path(__file__).with_name('rescore-saved.py')
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != '1cbdf50ce36f46060e27c6967c981697f2ba7b618cc033a39bb60d921bcc6c75':
        raise RuntimeError('Frozen scoring wrapper changed')
    frozen = runpy.run_path(str(frozen_path))
    old, image = frozen['OLD'], frozen['IMAGE']
    previous = frozen['ROOT'] / '20261009-qwen38-native-shared-kv/run-cell.py'
    hashes = {**frozen['HASHES'],
              previous: '58fc22ce410d84c6559c07109973568078e6ac80004a2c8c0769a25dc277fd09',
              old / 'run-quality.py': '8c7af825b33da9b2c7d5717b55d9908cab23b1874dfb080e06e69d11a2720b5a',
              old / 'preparation-02/prepared/prepared.jsonl': 'd6d3b81b8f17e282f381535b5e3e6501d84ba982728e585a35a195cf68b40a4f'}
    for path, expected in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen dependency changed: {path}')
    prior = runpy.run_path(str(previous))
    sources = {**prior['verify_sources'](), **{str(p): h for p, h in hashes.items()}}
    guard = runpy.run_path(str(frozen['GUARD']))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    name = f'b70-grouped-quality-{args.arm}'
    guard['preconditions'](out, name)
    before = guard['kernel'](out, 'pre')
    if any(v['error_lines'] for v in before['commands'].values()):
        raise RuntimeError('Pre-existing guarded kernel error')
    guard['put'](out / 'source-hashes.json', sources)
    for source in (Path(__file__), old / 'quality.py', old / 'run-quality.py'):
        (out / source.name).write_bytes(source.read_bytes())
    generation = out / 'native'
    sys.argv = [str(old / 'run-quality.py'), '--arm', args.arm,
                '--data', str(old / 'preparation-02/prepared'), '--out', str(generation),
                '--startup-timeout', '1800', '--benchmark-timeout', '18000']
    guard['put'](out / 'generation-command.json', {'argv': sys.argv.copy(), 'execution': 'runpy main'})
    generation_rc, score_rc = 1, 1
    try:
        generation_rc = int(runpy.run_path(str(old / 'run-quality.py'))['main']())
        if generation_rc:
            raise RuntimeError('Full generation failed; retaining incomplete artifacts')
        dest = generation / 'scores'
        dest.mkdir(exist_ok=False)
        score_name = 'b70-full-score-' + uuid.uuid4().hex[:12]
        guard['preconditions'](dest, score_name)
        argv = ['docker', 'run', '--rm', '--name', score_name, '--network', 'none',
                '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--pids-limit', '128', '--cpus', '4', '--memory', '8g',
                '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=2g',
                '--user', f'{os.getuid()}:{os.getgid()}',
                '-v', f'{old}/preparation-02/prepared:/input/prepared:ro',
                '-v', f'{generation}:/input/generation:ro', '-v', f'{dest}:/output:rw',
                image, 'score', '--task', 'all', '--allow-code-execution',
                '--data', '/input/prepared', '--generation', '/input/generation/generation.jsonl',
                '--out', '/output/scores.json']
        guard['put'](dest / 'command.json', {'argv': argv,
                     'generation_sha256': hashlib.sha256((generation / 'generation.jsonl').read_bytes()).hexdigest()})
        try:
            with (dest / 'score.log').open('w') as log:
                score_rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=7200, check=False).returncode
        finally:
            inspect = guard['cmd'](['docker', 'container', 'inspect', score_name], 30)
            if inspect.returncode == 0:
                cleanup = guard['cmd'](['docker', 'rm', '-f', score_name], 60)
                guard['put'](dest / 'cleanup.json', {'returncode': cleanup.returncode, 'stdout': cleanup.stdout})
                if cleanup.returncode:
                    score_rc = 1
            else:
                guard['put'](dest / 'cleanup.json', {'auto_removed': True, 'inspect_returncode': inspect.returncode})
            guard['put'](dest / 'exit.json', {'returncode': score_rc})
        if score_rc:
            raise RuntimeError('Full scoring failed; see scores/score.log')
    finally:
        after = guard['kernel'](out, 'post')
        original = {s for v in before['commands'].values() for s in v['error_lines']}
        new = sorted({s for v in after['commands'].values() for s in v['error_lines']} - original)
        post = out / 'postconditions'
        post.mkdir()
        guard['preconditions'](post, name)
        prior['verify_sources']()
        for path, expected in hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Dependency changed during full quality: {path}')
        guard['put'](out / 'cell-result.json', {'arm': args.arm, 'driver_exit': generation_rc,
                     'score_exit': score_rc, 'new_guarded_error_lines': new,
                     'full_suite': True, 'quality_qualified': False})
        if new:
            raise RuntimeError('New guarded kernel errors')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
