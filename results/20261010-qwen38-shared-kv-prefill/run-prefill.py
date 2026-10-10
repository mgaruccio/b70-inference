#!/usr/bin/env python3
"""Host-specific development screen; reuse the established shared-KV recipe/client."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import signal
import subprocess
import time
import urllib.request

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
PRIOR = ROOT / '20261009-qwen38-community-protocol'
OUT = ROOT / '20261010-qwen38-shared-kv-prefill'
COOKBOOK = Path('/home/mike/inference/src/intel-arc-pro-b70-inference-cookbook')
PRODUCTION = Path('/home/mike/inference/launchers/start-qwen38.sh')
PRODUCTION_SHA = '63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4'
POWER = Path('/sys/class/drm/card0/device/hwmon/hwmon2/power1_cap')
LIBRARY = ROOT / '20260913-qwen38-native-grouped-verify/build-06/build/libb70_grouped_verify.so'
LIBRARY_SHA = 'e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec'
TARGETS = (8192, 32768, 65536)


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(argv, path, timeout=120):
    with path.open('w') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_power(value):
    subprocess.run(['sudo', '-n', 'tee', str(POWER)], input=value + '\n', text=True,
                   stdout=subprocess.DEVNULL, check=True, timeout=15)
    assert POWER.read_text().strip() == value


def interrupt(signum, frame):
    raise InterruptedError(f'signal {signum}')


def prepare_client(out):
    source = COOKBOOK / 'benchmarks/b70-realworld-context-harness.py'
    raw = source.read_bytes()
    assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == 'a4ca3c28c2d87436f80e93c42cb9f9712f3765e7'
    text = raw.decode()
    replacements = {
        'reasoning = delta.get("reasoning_content")': 'reasoning = delta.get("reasoning_content") or delta.get("reasoning")',
        '"values": values,': '"raw_metrics": text,\n        "values": values,',
        '    return record\n': '''    record["server_metrics_before"] = before["raw_metrics"]
    record["server_metrics_after"] = after["raw_metrics"]
    for metric in ("request_prefill_time_seconds", "request_queue_time_seconds", "request_inference_time_seconds"):
        deltas = {}
        for suffix in ("sum", "count"):
            name = "vllm:" + metric + "_" + suffix
            try:
                _, a = metric_series(before["raw_metrics"], (name,))
                _, b = metric_series(after["raw_metrics"], (name,))
                deltas[suffix] = sum(b.values()) - sum(a.values())
            except RuntimeError:
                deltas[suffix] = None
        record["server_" + metric] = deltas
    return record
''',
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    compile(text, str(out / 'client.py'), 'exec')
    (out / 'client.py').write_text(text)
    put(out / 'client-changes.json', replacements)
    return out / 'client.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, choices=(8192, 16384, 32768), required=True)
    args = parser.parse_args()
    out = OUT / f'batch-{args.budget}'
    out.mkdir(parents=True, exist_ok=False)
    assert sha(PRODUCTION) == PRODUCTION_SHA
    assert sha(LIBRARY) == LIBRARY_SHA
    guard = runpy.run_path(str(PRIOR / 'power-230w-01/guard-230w.py'))
    original_power = POWER.read_text().strip()
    assert original_power == '275000000', original_power
    # Reuse the hardware/CPU/boot/idle checks at the original power first.
    guard['preconditions'].__globals__['POWER_CAP_UW'] = original_power
    guard['preconditions'](out, 'qwen38')
    before_kernel = guard['kernel'](out, 'pre')
    assert not any(v['error_lines'] for v in before_kernel['commands'].values())
    launcher = (PRIOR / 'power-230w-01/shared-kv.sh').read_text()
    assert launcher.count('--max-num-batched-tokens 8192') == 1
    launcher = launcher.replace('--max-num-batched-tokens 8192', f'--max-num-batched-tokens {args.budget}')
    launcher = launcher.replace('0.0.0.0:${PORT:-8000}:8000', '127.0.0.1:8000:8000')
    (out / 'launcher.sh').write_text(launcher)
    run(['bash', '-n', str(out / 'launcher.sh')], out / 'bash-check.txt')
    client = prepare_client(out)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    run(['uname', '-a'], out / 'uname.txt')
    run(['lscpu'], out / 'lscpu.txt')
    commands = []
    status = {'budget': args.budget, 'tier': 'development', 'success': False,
              'original_power_uw': original_power, 'experiment_power_uw': '230000000'}
    process = cid = None
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    def execute(argv, name, timeout=7200):
        commands.append(argv)
        put(out / 'commands.json', commands)
        run(argv, out / name, timeout)
    try:
        set_power('230000000')
        guard['preconditions'].__globals__['POWER_CAP_UW'] = '230000000'
        guard['preconditions'](out, 'qwen38')
        with (out / 'server.log').open('w') as log:
            process = subprocess.Popen(['bash', str(out / 'launcher.sh')], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Server exited before health; see server.log')
            if cid is None:
                result = subprocess.run(['docker', 'inspect', '--format', '{{.Id}}', 'qwen38'], capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    cid = result.stdout.strip()
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError('Server health timeout')
        assert cid
        execute(['docker', 'inspect', cid], 'container.json', 30)
        log = (out / 'server.log').read_text(errors='replace')
        eligible = [s for s in log.splitlines() if '"event":"eligible-dispatch"' in s and LIBRARY_SHA in s]
        unsupported = [s for s in log.splitlines() if '"event":"unsupported-q5"' in s]
        put(out / 'dispatch.json', {'eligible': eligible, 'unsupported': unsupported})
        assert eligible and not unsupported, 'Shared-KV shape contract failed; do not benchmark fallback'
        generator = COOKBOOK / 'benchmarks/b70-generate-exact-prompts.py'
        prompts = OUT / 'prefill-prompts.json'
        if not prompts.exists():
            execute(['docker', 'cp', str(generator), f'{cid}:/tmp/prefill-prompts.py'], 'copy-generator.txt', 30)
            execute(['docker', 'exec', cid, '/opt/venv/bin/python', '/tmp/prefill-prompts.py', '--model', '/model', '--output', '/tmp/prefill-prompts.json', '--targets', ','.join(map(str, TARGETS)), '--per-target', '6'], 'generate-prompts.txt')
            execute(['docker', 'cp', f'{cid}:/tmp/prefill-prompts.json', str(prompts)], 'copy-prompts.txt', 60)
        # A real 128-token request verifies that decode still exercises shared-KV.
        cells = [(512, 128, PRIOR / 'current-serving/prompts.json')]
        cells.extend((n, 1, prompts) for n in TARGETS)
        summary = []
        for n, output, prompt_path in cells:
            label = f'p{n}-g{output}'
            print(f'budget={args.budget} cell={label}', flush=True)
            execute(['/usr/bin/python3', '-B', str(client), '--mode', 'context', '--prompts', str(prompt_path), '--target', str(n), '--output', str(output), '--budget', str(args.budget), '--reps', '5', '--model', 'qwen38', '--root', 'http://127.0.0.1:8000', '--outdir', str(out / label), '--full-output-warmup', '--ignore-eos'], label + '.log')
            result = json.loads((out / label / 'results.json').read_text())
            assert len(result['records']) == 5
            assert all(r['prompt_tokens'] == n and r['completion_tokens'] == output and r['prefix_cache_hits_delta'] == 0 for r in result['records'])
            summary.append({'cell': label, 'summary': result['summary']})
            put(out / 'summary.json', summary)
            print(json.dumps(summary[-1]), flush=True)
        adapter = runpy.run_path(str(ROOT / '20260913-qwen38-native-grouped-verify/run-serving.py'))
        adapter['_candidate_execution_evidence'](out)
        assert '"event":"unsupported-q5"' not in (out / 'server.log').read_text(errors='replace')
        status['success'] = True
    except BaseException as exc:
        status['error'] = repr(exc)
        raise
    finally:
        try:
            if cid:
                stopped = subprocess.run(['docker', 'stop', '--time', '30', cid], capture_output=True, text=True, timeout=90)
                put(out / 'cleanup.json', {'container_id': cid, 'returncode': stopped.returncode, 'stdout': stopped.stdout, 'stderr': stopped.stderr})
            if process:
                process.wait(timeout=120)
        finally:
            set_power(original_power)
            assert sha(PRODUCTION) == PRODUCTION_SHA
            guard['preconditions'].__globals__['POWER_CAP_UW'] = original_power
            restored = out / 'restored'
            restored.mkdir()
            guard['preconditions'](restored, 'qwen38')
            after_kernel = guard['kernel'](out, 'post')
            status['new_kernel_errors'] = sorted({s for v in after_kernel['commands'].values() for s in v['error_lines']} - {s for v in before_kernel['commands'].values() for s in v['error_lines']})
            status['restored'] = True
            put(out / 'status.json', status)
            assert not status['new_kernel_errors']


if __name__ == '__main__':
    main()
