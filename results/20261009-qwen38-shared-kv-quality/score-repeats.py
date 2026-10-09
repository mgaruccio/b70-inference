#!/usr/bin/env python3
"""Score all sixteen diagnostic repetitions with the unchanged frozen evaluator."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import uuid


def main():
    if socket.gethostname().split('.')[0] != 'inference-host':
        raise RuntimeError('Run the evaluator on inference-host, never Pi')
    source = Path(__file__).with_name('rescore-saved.py')
    if hashlib.sha256(source.read_bytes()).hexdigest() != '1cbdf50ce36f46060e27c6967c981697f2ba7b618cc033a39bb60d921bcc6c75':
        raise RuntimeError('Frozen scorer wrapper changed')
    frozen = runpy.run_path(str(source))
    for path, expected in frozen['HASHES'].items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen dependency changed: {path}')
    root, old, image = frozen['OUT'], frozen['OLD'], frozen['IMAGE']
    guard = runpy.run_path(str(frozen['GUARD']))
    cells = [('a1', 'native'), ('b1', 'candidate'), ('b2', 'candidate'), ('a2', 'native')]
    expected_ids = {'HumanEval/1', 'HumanEval/11', 'HumanEval/19', 'HumanEval/126', 'HumanEval/130'}
    # Validate the whole generation campaign before starting any scoring.
    for label, arm in cells:
        cell = root / f'repeat-{label}'
        status = json.loads((cell / 'cell-result.json').read_text())
        if status['driver_exit'] != 0 or status['arm'] != arm or status['new_guarded_error_lines']:
            raise RuntimeError(f'Generation cell did not pass its guards: {cell}')
        for repeat in range(1, 5):
            point = cell / 'native' / f'repeat-{repeat:02d}'
            rows = [json.loads(line) for line in (point / 'generation.jsonl').read_text().splitlines()]
            meta = json.loads((point / 'generation.jsonl.meta.json').read_text())
            if len(rows) != 5 or {r['id'] for r in rows} != expected_ids or meta.get('full_run') is not False:
                raise RuntimeError(f'Not the declared diagnostic subset: {point}')
    for label, arm in cells:
        for repeat in range(1, 5):
            point = root / f'repeat-{label}' / 'native' / f'repeat-{repeat:02d}'
            dest = point / 'scores'
            dest.mkdir(exist_ok=False)
            name = 'b70he-repeat-score-' + uuid.uuid4().hex[:12]
            guard['preconditions'](dest, name)
            argv = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
                    '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                    '--pids-limit', '128', '--cpus', '4', '--memory', '8g',
                    '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=2g',
                    '--user', f'{os.getuid()}:{os.getgid()}',
                    '-v', f'{old}/preparation-02/prepared:/input/prepared:ro',
                    '-v', f'{point}:/input/generation:ro', '-v', f'{dest}:/output:rw',
                    image, 'score', '--task', 'humanevalplus', '--allow-code-execution',
                    '--data', '/input/prepared', '--generation', '/input/generation/generation.jsonl',
                    '--out', '/output/scores.json']
            guard['put'](dest / 'command.json', {'argv': argv, 'cell': label, 'arm': arm,
                         'repeat': repeat, 'generation_sha256': hashlib.sha256((point / 'generation.jsonl').read_bytes()).hexdigest(),
                         'source_hashes': {str(p): h for p, h in frozen['HASHES'].items()}})
            (dest / 'score-repeats.py').write_bytes(Path(__file__).read_bytes())
            rc = 1
            try:
                with (dest / 'score.log').open('w') as log:
                    rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=1800, check=False).returncode
            finally:
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
                guard['put'](dest / 'exit.json', {'returncode': rc})
            if rc:
                raise RuntimeError(f'Scoring failed: {dest}')
            print(json.dumps({'cell': label, 'arm': arm, 'repeat': repeat, 'returncode': rc}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
