#!/usr/bin/env python3
"""Host-only accepted-build baseline and one decode-warmed unitrace capture."""
import datetime
import json
from pathlib import Path
import runpy
import signal
import subprocess
import time

OLD = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
GROUPED = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')
ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-decode-profile')
ORIGINAL = 'nemotron-bolt-probe'
URL = 'http://127.0.0.1:18098'
EXPECTED = {
    'llama-server': '22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d',
    'libggml-sycl.so.0': '076b16475de805572cf7a623eecca827fd04e58903bf03a003121f60af56fd2f',
    'test-backend-ops': 'a6a3f666754fce8edfbb59d9840290a188ba2681e7065f812de3359f4e12c98d',
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_request(result, prompt_tokens):
    assert result['usage']['prompt_tokens'] == prompt_tokens, result['usage']
    assert result['usage']['completion_tokens'] == 128, result['usage']
    assert result['timings']['cache_n'] == 0, result['timings']
    assert result['timings']['predicted_n'] == 128, result['timings']
    assert result['finish_reason'] == 'length' and result['text'].strip()


def main():
    helper = runpy.run_path(str(GROUPED / 'run-grouped-diagnostic.py'))
    command, ready, binaries = (helper[k] for k in ('command', 'ready', 'binaries'))
    runner = runpy.run_path(str(OLD / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD / 'probe-api.py'))
    assert command('hostname').strip() == 'inference-host'
    source = runner['PREFILL_SSM']
    assert binaries(source) == EXPECTED
    before = runner['hardware']()
    old = json.loads(command('docker', 'inspect', ORIGINAL))[0]
    assert old['State']['Running'] and old['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in old['Mounts'] if m['Destination'] == '/src') == str(source)
    assert 'GGML_SYCL_DEBUG=1' not in old['Config']['Env']
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {ORIGINAL}
    names = {phase: 'bolt-decode-r1-' + phase for phase in ('baseline', 'profile')}
    existing = command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    assert not any(name in existing for name in names.values())
    verified = json.loads((OLD / 'model-verified.json').read_text())
    assert verified['pass_'] and Path(verified['path']).parent == runner['MODEL']
    assert Path(verified['path']).stat().st_size == verified['bytes']
    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    save(out / 'original-container.json', old)
    save(out / 'hardware-before.json', before)
    save(out / 'baseline-binaries.json', EXPECTED)
    save(out / 'model-verified.json', verified)
    (out / 'unitrace-commit.txt').write_bytes((OLD / 'prefill-unitrace-tool/pti-commit.txt').read_bytes())
    summary = {'status': 'prepared', 'tier': 'development', 'source': str(source), 'phases': {}}
    stopped = False
    created = []

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped = True
        command('docker', 'stop', '-t', '90', ORIGINAL)
        for phase, name in names.items():
            cell = out / phase
            cell.mkdir()
            create = json.loads((OLD / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
            create[create.index('--name') + 1] = name
            assert str(source) + ':/src:ro' in create
            assert create[-1] == old['Config']['Cmd'][-1], 'Recovery command differs from live accepted launch'
            prefix, server = create[-1].rsplit('exec ', 1)
            assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in prefix
            if phase == 'profile':
                index = create.index(runner['IMAGE'])
                create[index:index] = ['--stop-timeout', '90',
                    '-v', str(OLD / 'prefill-unitrace-tool/build') + ':/profiler:ro',
                    '-v', str(cell) + ':/profile']
                create[-1] = (prefix + 'exec /profiler/unitrace --device-timing --host-timing '
                              '--kernel-submission --chrome-kernel-logging '
                              '--session boltdecode --start-paused --output /profile/unitrace.txt ' + server)
            save(cell / 'create-argv.json', create)
            command(*create, timeout=60)
            created.append((name, cell, phase))
            command('docker', 'start', name)
            data = {'models': ready(api, name)}
            summary['phases'][phase] = data
            native = command('docker', 'logs', name)
            (cell / 'native-and-startup.log').write_text(native)
            assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in native
            modes = ('smoke', 'quality', 'bench', 'long') if phase == 'baseline' else ('quality',)
            for mode in modes:
                log = command('python3', '-B', str(OLD / 'probe-api.py'), '--out', str(cell / mode),
                              '--mode', mode, '--repeats', '3', timeout=2400)
                (cell / (mode + '.log')).write_text(log)
                data[mode] = json.loads((cell / mode / 'summary.json').read_text())
                assert data[mode]['pass']
                if mode == 'quality':
                    assert len(data[mode]['checks']) == 20 and all(x['pass'] for x in data[mode]['checks'])
                if mode in ('bench', 'long'):
                    assert len(data[mode]['measured']) == 3
                    for result in data[mode]['measured']:
                        check_request(result, 55 if mode == 'bench' else 8270)
                    if mode == 'long':
                        anchor = data[mode]['anchor']
                        assert anchor['usage']['prompt_tokens'] == 8238
                        assert anchor['text'].strip() == 'BOLT-7319-CEDAR'
            if phase == 'profile':
                payload = json.loads((out / 'baseline/long/measured-00.request.json').read_text())
                prompt = payload['messages'][0]['content']
                data['warmup'] = api['chat'](URL, prompt, cell / 'warmup', count=128, stream=True, ignore_eos=True)
                check_request(data['warmup'], 8270)
                assert json.loads((cell / 'warmup.request.json').read_text()) == payload
                command('docker', 'exec', name, '/profiler/unitrace', '--resume', 'boltdecode')
                data['profile_start_epoch_ns'] = time.time_ns()
                data['profile_since'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
                data['profile'] = api['chat'](URL, prompt, cell / 'profile', count=128, stream=True, ignore_eos=True)
                data['profile_end_epoch_ns'] = time.time_ns()
                command('docker', 'exec', name, '/profiler/unitrace', '--stop', 'boltdecode')
                check_request(data['profile'], 8270)
                assert json.loads((cell / 'profile.request.json').read_text()) == payload
                (cell / 'profile-window.log').write_text(command('docker', 'logs', '--timestamps', '--since', data['profile_since'], name))
            command('docker', 'stop', '-t', '90', name)
            assert binaries(source) == EXPECTED and runner['hardware']() == before
            data['status'] = 'passed'
            save(cell / 'summary.json', data)
        summary['status'] = 'passed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        raise
    finally:
        try:
            for name, cell, phase in created:
                command('docker', 'stop', '-t', '90', name)
                (cell / 'server.log').write_text(command('docker', 'logs', '--timestamps', name))
                save(cell / 'container.json', json.loads(command('docker', 'inspect', name))[0])
                if phase == 'profile':
                    command('docker', 'cp', name + ':/llama-server.1.json', str(cell / 'llama-server.1.json'))
        finally:
            try:
                if stopped:
                    assert not set(names.values()).intersection(command('docker', 'ps', '--format', '{{.Names}}').splitlines())
                    command('docker', 'start', ORIGINAL)
                    summary['restored_models'] = ready(api, ORIGINAL)
                    smoke = api['chat'](URL, 'Reply with exactly this word and nothing else: READY', out / 'restored-smoke', count=32)
                    assert smoke['text'].strip() == 'READY'
                    restored = json.loads(command('docker', 'inspect', ORIGINAL))[0]
                    assert restored['Id'] == old['Id'] and restored['Config'] == old['Config'] and restored['Mounts'] == old['Mounts']
                    save(out / 'restored-container.json', restored)
                    assert binaries(source) == EXPECTED
                after = runner['hardware']()
                save(out / 'hardware-after.json', after)
                assert after == before
                summary['restored'] = True
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                save(out / 'summary.json', summary)
    print('BOLT_DECODE_PROFILE_CAPTURE_PASSED', flush=True)


if __name__ == '__main__':
    main()
