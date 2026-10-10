#!/usr/bin/env python3
"""Run the public community benchmark against the unchanged local serving launcher."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import time
import urllib.request

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
OUT = ROOT / '20261009-qwen38-community-protocol/current-serving-02'
LAUNCHER = Path('/home/mike/inference/launchers/start-qwen38.sh')
COOKBOOK = Path('/home/mike/inference/src/intel-arc-pro-b70-inference-cookbook')
GUARD = ROOT / '20261009-qwen38-gdn-locality/native-boundary64k-assets-20261009-100052-ff9f86/registered-controller.py'
GUARD_SHA256 = 'fafe61a614a8328f63f0a7af41fa90da46041aa6cec514a01fe47b485b669da5'
BLOBS = {'b70-realworld-context-harness.py': 'a4ca3c28c2d87436f80e93c42cb9f9712f3765e7',
         'b70-generate-exact-prompts.py': '645c021089c82d671a0c332497cfe2ec503c3f57'}
PREFILL_PROMPTS = None
CHECK_BUILD = None
CELLS = ((512, 128), (8192, 128), (8192, 1), (130944, 128))
DECODE_PROMPTS = None


def main():
    assert hashlib.sha256(GUARD.read_bytes()).hexdigest() == GUARD_SHA256
    guard = runpy.run_path(str(GUARD))
    OUT.mkdir(parents=True, exist_ok=False)
    guard['preconditions'](OUT, 'qwen38')
    before = guard['kernel'](OUT, 'pre')
    assert not any(v['error_lines'] for v in before['commands'].values())
    (OUT / 'run.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'start-qwen38.sh').write_bytes(LAUNCHER.read_bytes())
    for name, expected in BLOBS.items():
        raw = (COOKBOOK / 'benchmarks' / name).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == expected
        (OUT / name).write_bytes(raw)
    client = OUT / 'b70-realworld-context-harness.py'
    original = client.read_text()
    old = 'reasoning = delta.get("reasoning_content")'
    new = 'reasoning = delta.get("reasoning_content") or delta.get("reasoning")'
    assert original.count(old) == 1
    client.write_text(original.replace(old, new))
    guard['put'](OUT / 'client-compatibility.json', {'original_git_blobs': BLOBS,
                 'old': old, 'new': new, 'reason': 'Observed current serving SSE uses delta.reasoning',
                 'sha256': hashlib.sha256(client.read_bytes()).hexdigest()})
    commands = []
    def execute(argv, log_name, timeout=1800):
        commands.append(argv)
        guard['put'](OUT / 'commands.json', commands)
        with (OUT / log_name).open('w') as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    cid, process, success = None, None, False
    with (OUT / 'server.log').open('w') as server_log:
        try:
            process = subprocess.Popen([str(LAUNCHER)], stdout=server_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Serving launcher exited; see server.log')
                if cid is None:
                    found = subprocess.run(['docker', 'inspect', '--format', '{{.Id}}', 'qwen38'], capture_output=True, text=True)
                    if found.returncode == 0:
                        cid = found.stdout.strip()
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('Server startup timed out')
            assert cid
            execute(['docker', 'inspect', cid], 'container.json', 30)
            if CHECK_BUILD:
                CHECK_BUILD(OUT, False)
            execute(['docker', 'cp', str(OUT / 'b70-generate-exact-prompts.py'), f'{cid}:/tmp/b70-community-prompts.py'], 'copy-generator.log', 30)
            previous_prompts = DECODE_PROMPTS or OUT.parent / 'current-serving/prompts.json'
            if DECODE_PROMPTS and not previous_prompts.exists():
                targets = ','.join(str(p) for p in sorted({p for p, g in CELLS if g > 1}))
                execute(['docker', 'exec', cid, '/opt/venv/bin/python', '/tmp/b70-community-prompts.py',
                         '--model', '/model', '--output', '/tmp/b70-community-decode.json',
                         '--targets', targets, '--per-target', '6'], 'generate-decode-prompts.log', 7200)
                execute(['docker', 'cp', f'{cid}:/tmp/b70-community-decode.json', str(previous_prompts)], 'copy-decode-prompts.log', 60)
            (OUT / 'prompts.json').write_bytes(previous_prompts.read_bytes())
            # New entropy for the prefill cell: do not reuse the decode prompts
            # against the unchanged production setup's enabled prefix cache.
            if PREFILL_PROMPTS:
                (OUT / 'prefill-prompts.json').write_bytes(PREFILL_PROMPTS.read_bytes())
            else:
                execute(['docker', 'exec', cid, '/opt/venv/bin/python', '/tmp/b70-community-prompts.py',
                         '--model', '/model', '--output', '/tmp/b70-community-prefill.json',
                         '--targets', '8192', '--per-target', '6'], 'generate-prompts.log')
                execute(['docker', 'cp', f'{cid}:/tmp/b70-community-prefill.json', str(OUT / 'prefill-prompts.json')], 'copy-prompts.log', 30)
            summaries = []
            for prompt, output in CELLS:
                label = f'p{prompt}-g{output}'
                print('Starting ' + label, flush=True)
                execute(['/usr/bin/python3', '-B', str(OUT / 'b70-realworld-context-harness.py'),
                         '--mode', 'context', '--prompts', str(OUT / ('prefill-prompts.json' if output == 1 else 'prompts.json')),
                         '--target', str(prompt), '--output', str(output), '--budget', '8192',
                         '--reps', '5', '--model', 'qwen38', '--root', 'http://127.0.0.1:8000',
                         '--outdir', str(OUT / label), '--full-output-warmup', '--ignore-eos'], label + '.log', 7200)
                result = json.loads((OUT / label / 'results.json').read_text())
                assert len(result['records']) == 5
                assert all(r['prompt_tokens'] == prompt and r['completion_tokens'] == output for r in result['records'])
                summaries.append({'cell': label, 'summary': result['summary']})
                guard['put'](OUT / 'summary.json', summaries)
                print(json.dumps(summaries[-1]), flush=True)
            if CHECK_BUILD:
                CHECK_BUILD(OUT, True)
            success = True
        finally:
            if cid:
                stopped = subprocess.run(['docker', 'stop', '--time', '30', cid], capture_output=True, text=True, timeout=90)
                guard['put'](OUT / 'cleanup.json', {'container_id': cid, 'returncode': stopped.returncode, 'stdout': stopped.stdout, 'stderr': stopped.stderr})
            if process:
                process.wait(timeout=120)
            post = OUT / 'postconditions'
            post.mkdir()
            guard['preconditions'](post, 'qwen38')
            after = guard['kernel'](OUT, 'post')
            old_errors = {s for v in before['commands'].values() for s in v['error_lines']}
            new_errors = sorted({s for v in after['commands'].values() for s in v['error_lines']} - old_errors)
            guard['put'](OUT / 'exit.json', {'success': success, 'new_guarded_error_lines': new_errors})
            assert not new_errors


if __name__ == '__main__':
    main()
