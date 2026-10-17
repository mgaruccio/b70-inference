#!/usr/bin/env python3
"""One uncached real-API 8270-token prefill; instrumentation is diagnostic, not a speed result."""
import json
import pathlib
import runpy
import shlex
import signal
import subprocess
import time

ROOT = pathlib.Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
OUT = ROOT / 'prefill-ssm-r1-unitrace'
NAME = 'nemotron-bolt-prefill-ssm-profile'
ORIGINAL = 'nemotron-bolt-probe'


def command(*argv, timeout=180):
    print(shlex.join(argv), flush=True)
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def ready(api, name):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        assert command('docker', 'inspect', '-f', '{{.State.Running}}', name).strip() == 'true'
        try:
            return api['request']('http://127.0.0.1:18098', '/v1/models', timeout=3)
        except (OSError, ValueError):
            time.sleep(2)
    raise TimeoutError('server readiness')


def main():
    assert command('hostname').strip() == 'inference-host'
    OUT.mkdir(exist_ok=False)
    (OUT / 'controller.py').write_bytes(pathlib.Path(__file__).read_bytes())
    runner = runpy.run_path(str(ROOT / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(ROOT / 'probe-api.py'))
    before = runner['hardware']()
    save('hardware-before.json', before)
    old = json.loads(command('docker', 'inspect', ORIGINAL))[0]
    assert old['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert old['State']['Running']
    assert next(m['Source'] for m in old['Mounts'] if m['Destination'] == '/src') == str(runner['PREFILL_SSM'])
    assert 'GGML_SYCL_DEBUG=1' not in old['Config']['Env']
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {ORIGINAL}, 'Foreign workload running'
    assert NAME not in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines(), 'Profile container already exists'
    save('original-container.json', old)
    # Reuse the exact validated server argv, image, devices, and mounts.
    create = json.loads((ROOT / 'ssm-prefill-r1-confirm-candidate/create-argv.json').read_text())
    create[create.index('--name') + 1] = NAME
    image_index = create.index(runner['IMAGE'])
    create[image_index:image_index] = [
        '--stop-timeout', '90',
        '-v', str(ROOT / 'prefill-unitrace-tool/build') + ':/profiler:ro',
        '-v', str(OUT) + ':/profile',
    ]
    original_shell = old['Config']['Cmd'][-1]
    server = original_shell.rsplit('exec ', 1)[1]
    assert server.startswith('/src/build-sycl/bin/llama-server ')
    create[-1] = ('set -eo pipefail; source /opt/intel/oneapi/setvars.sh --force; '
                  'export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}; '
                  'exec /profiler/unitrace --device-timing --host-timing --kernel-submission '
                  '--chrome-kernel-logging --session boltprefill --start-paused '
                  '--output /profile/unitrace.txt ' + server)
    save('create-argv.json', create)
    # Use exactly the existing long-benchmark prompt; only output budget is 1 for attribution.
    payload = json.loads((ROOT / 'prefill-ssm-r1-baseline/long/measured-00.request.json').read_text())
    prompt = payload['messages'][0]['content']
    assert payload['cache_prompt'] is False and payload['temperature'] == 0
    summary = {'tier': 'development-diagnostic', 'status': 'prepared',
               'note': 'Profiler timings are attribution only; not comparable to uninstrumented performance.'}
    stopped_original = created = False

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped_original = True
        command('docker', 'stop', '-t', '90', ORIGINAL)
        summary['container_id'] = command(*create, timeout=60).strip()
        created = True
        command('docker', 'start', NAME)
        summary['models'] = ready(api, NAME)
        summary['warmup'] = api['chat']('http://127.0.0.1:18098', prompt, OUT / 'warmup', count=1, stream=True, ignore_eos=True)
        command('docker', 'exec', NAME, '/profiler/unitrace', '--resume', 'boltprefill')
        summary['profile_start_epoch_ns'] = time.time_ns()
        summary['profile'] = api['chat']('http://127.0.0.1:18098', prompt, OUT / 'profile', count=1, stream=True, ignore_eos=True)
        summary['profile_end_epoch_ns'] = time.time_ns()
        command('docker', 'exec', NAME, '/profiler/unitrace', '--stop', 'boltprefill')
        for key in ('warmup', 'profile'):
            result = summary[key]
            assert result['usage']['prompt_tokens'] == 8270, result['usage']
            assert result['usage']['completion_tokens'] == 1, result['usage']
            assert result['timings']['cache_n'] == 0, result['timings']
        summary['status'] = 'passed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        try:
            if created:
                command('docker', 'stop', '-t', '90', NAME)
                (OUT / 'server.log').write_text(command('docker', 'logs', NAME))
                save('profile-container.json', json.loads(command('docker', 'inspect', NAME))[0])
                command('docker', 'cp', NAME + ':/llama-server.1.json', str(OUT / 'llama-server.1.json'))
        finally:
            if stopped_original:
                command('docker', 'start', ORIGINAL)
                summary['restored_models'] = ready(api, ORIGINAL)
                summary['restored_smoke'] = api['chat']('http://127.0.0.1:18098', 'Reply with exactly this word and nothing else: READY', OUT / 'restored-smoke', count=32)
                assert summary['restored_smoke']['text'].strip() == 'READY'
            after = runner['hardware']()
            save('hardware-after.json', after)
            save('summary.json', summary)
            assert after == before, 'Hardware settings changed'
    print('PREFILL_PROFILE_PASSED', flush=True)


if __name__ == '__main__':
    main()
