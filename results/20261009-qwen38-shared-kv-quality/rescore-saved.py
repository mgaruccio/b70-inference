#!/usr/bin/env python3
"""Re-score immutable saved HumanEval+ samples in the existing isolated image."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import uuid

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
OLD = ROOT / '20260913-qwen38-grouped-quality'
OUT = ROOT / '20261009-qwen38-shared-kv-quality'
IMAGE = 'sha256:4638874848e6aace7d1b5b8a1f1bdeb9993eb9edffdf8035dbef6057b23949c9'
GUARD = ROOT / '20261009-qwen38-gdn-locality/native-boundary64k-assets-20261009-100052-ff9f86/registered-controller.py'
HASHES = {GUARD: 'fafe61a614a8328f63f0a7af41fa90da46041aa6cec514a01fe47b485b669da5',
          OLD / 'quality.py': '8544176f922079868d022f100de2f2e424a4617be321ae1df08dbbd2fdb11c48',
          OLD / 'preparation-02/prepared/manifest.json': 'b9cbdb3e7eeda41b93eb8e2d2224313283682f58b922b4c12e866269b7bd9517'}


def main():
    if socket.gethostname().split('.')[0] != 'inference-host':
        raise RuntimeError('Evaluator must run on inference-host, not Pi')
    for path, expected in HASHES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen source changed: {path}')
    guard = runpy.run_path(str(GUARD))
    for arm in ('target', 'native', 'candidate'):
        dest = OUT / f'rescore-saved-{arm}'
        dest.mkdir(parents=True, exist_ok=False)
        source = OLD / f'full-{arm}-01'
        name = 'b70he-rescore-' + uuid.uuid4().hex[:12]
        guard['preconditions'](dest, name)
        argv = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
                '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--pids-limit', '128', '--cpus', '4', '--memory', '8g',
                '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=2g',
                '--user', f'{os.getuid()}:{os.getgid()}',
                '-v', f'{OLD}/preparation-02/prepared:/input/prepared:ro',
                '-v', f'{source}:/input/generation:ro', '-v', f'{dest}:/output:rw',
                IMAGE, 'score', '--task', 'humanevalplus', '--allow-code-execution',
                '--data', '/input/prepared', '--generation', '/input/generation/generation.jsonl',
                '--out', '/output/scores.json']
        guard['put'](dest / 'command.json', {'argv': argv, 'arm': arm,
                     'generation_sha256': hashlib.sha256((source / 'generation.jsonl').read_bytes()).hexdigest(),
                     'source_hashes': {str(p): h for p, h in HASHES.items()}})
        (dest / 'rescore-saved.py').write_bytes(Path(__file__).read_bytes())
        rc = 1
        try:
            with (dest / 'score.log').open('w') as log:
                rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=1800, check=False).returncode
        finally:
            # Name was verified absent by preconditions; only this command owns it.
            inspect = guard['cmd'](['docker', 'container', 'inspect', name], 30)
            if inspect.returncode == 0:
                cleanup = guard['cmd'](['docker', 'rm', '-f', name], 60)
                guard['put'](dest / 'cleanup.json', {'returncode': cleanup.returncode, 'stdout': cleanup.stdout})
                if cleanup.returncode:
                    rc = 1
            else:
                guard['put'](dest / 'cleanup.json', {'auto_removed': True, 'inspect_returncode': inspect.returncode})
            post = dest / 'postconditions'
            post.mkdir()
            guard['preconditions'](post, name)
            guard['put'](dest / 'exit.json', {'returncode': rc, 'arm': arm})
        if rc:
            raise RuntimeError(f'{arm} scoring failed; see {dest}/score.log')
        print(json.dumps({'arm': arm, 'returncode': rc, 'out': str(dest)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
