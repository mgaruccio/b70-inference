#!/usr/bin/env python3
"""Host-only grouped geometry on real Bolt; native gates first, restore SSM in finally."""
import datetime
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import signal
import subprocess
import time

ROOT = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
ART = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')
OUT = ART / 'diagnostic-r1'
SOURCE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-grouped-diag-r1')
NAME = 'nemotron-bolt-grouped-diagnostic'
ORIGINAL = 'nemotron-bolt-probe'
URL = 'http://127.0.0.1:18098'


def command(*argv, timeout=180):
    print(shlex.join(argv), flush=True)
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ready(api, name):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        assert command('docker', 'inspect', '-f', '{{.State.Running}}', name).strip() == 'true'
        try:
            return api['request'](URL, '/v1/models', timeout=3)
        except (OSError, ValueError):
            time.sleep(2)
    raise TimeoutError('Bolt readiness timeout')


def binaries(source):
    return {name: hashlib.sha256((source / 'build-sycl/bin' / name).read_bytes()).hexdigest()
            for name in ('llama-server', 'libggml-sycl.so.0', 'test-backend-ops')}


def main():
    assert command('hostname').strip() == 'inference-host'
    assert 'BOLT_GROUPED_DIAGNOSTIC_RUNTIME_PRECHECK_PASSED' in (ART / 'diagnostic-runtime-precheck.log').read_text()
    assert (SOURCE / 'UPSTREAM_COMMIT').read_text().strip() == 'b6334e37e6ef286fc5091a41b6561b2113d915fb'
    runner = runpy.run_path(str(ROOT / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(ROOT / 'probe-api.py'))
    before = runner['hardware']()
    old = json.loads(command('docker', 'inspect', ORIGINAL))[0]
    assert old['State']['Running'] and old['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    baseline = runner['PREFILL_SSM']
    assert next(m['Source'] for m in old['Mounts'] if m['Destination'] == '/src') == str(baseline)
    assert 'GGML_SYCL_DEBUG=1' not in old['Config']['Env']
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {ORIGINAL}
    assert NAME not in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    verified = json.loads((ROOT / 'model-verified.json').read_text())
    assert verified['pass_'] and Path(verified['path']).parent == runner['MODEL']
    assert Path(verified['path']).stat().st_size == verified['bytes']
    OUT.mkdir(exist_ok=False)
    (OUT / 'controller.py').write_bytes(Path(__file__).read_bytes())
    save('hardware-before.json', before)
    save('original-container.json', old)
    baseline_hashes = binaries(baseline)
    save('baseline-binaries.json', baseline_hashes)
    save('diagnostic-binaries.json', binaries(SOURCE))
    create = json.loads((ROOT / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
    assert create[create.index('--name') + 1] == ORIGINAL
    assert str(baseline) + ':/src:ro' in create
    create[create.index('--name') + 1] = NAME
    create[create.index(str(baseline) + ':/src:ro')] = str(SOURCE) + ':/src:ro'
    create[create.index(runner['IMAGE']):create.index(runner['IMAGE'])] = ['-e', 'GGML_SYCL_DEBUG=1']
    # Preserve the complete native385 gate and server command from accepted recovery.
    assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in create[-1]
    save('create-argv.json', create)
    payload = json.loads((ROOT / 'prefill-ssm-r1-baseline/long/measured-00.request.json').read_text())
    assert payload['cache_prompt'] is False and payload['temperature'] == 0
    prompt = payload['messages'][0]['content']
    summary = {'tier': 'development-diagnostic', 'status': 'prepared', 'source': str(SOURCE),
               'note': 'Host diagnostics only. No kernel arithmetic or dispatch selection changes; no speed claim.'}
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
        native_log = command('docker', 'logs', NAME)
        assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in native_log
        (OUT / 'native-and-startup.log').write_text(native_log)
        # A full long warmup includes decode, unlike the prior output1-only profile.
        summary['warmup'] = api['chat'](URL, prompt, OUT / 'warmup', count=128, stream=True, ignore_eos=True)
        since = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
        summary['diagnostic_since'] = since
        summary['diagnostic'] = api['chat'](URL, prompt, OUT / 'diagnostic', count=1, stream=True, ignore_eos=True)
        (OUT / 'diagnostic-window.log').write_text(command('docker', 'logs', '--timestamps', '--since', since, NAME))
        for key, count in [('warmup', 128), ('diagnostic', 1)]:
            result = summary[key]
            assert result['usage']['prompt_tokens'] == 8270, result['usage']
            assert result['usage']['completion_tokens'] == count, result['usage']
            assert result['timings']['cache_n'] == 0, result['timings']
        command('python3', '-B', str(ART / 'analyze-grouped-diagnostic.py'),
                str(OUT / 'diagnostic-window.log'), '--output', str(OUT / 'analysis.json'))
        analysis = json.loads((OUT / 'analysis.json').read_text())
        assert analysis['grouped_calls'] > 0, 'No grouped IQ2 activation in diagnostic window'
        summary['path_accounting'] = {key: analysis[key] for key in
                                      ('calls', 'grouped_calls', 'fallback_calls', 'inner_fallback_reasons',
                                       'outer_fallback_reasons', 'grouped_geometry')}
        assert runner['hardware']() == before
        summary['status'] = 'passed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        try:
            if created:
                command('docker', 'stop', '-t', '90', NAME)
                (OUT / 'server.log').write_text(command('docker', 'logs', '--timestamps', NAME))
                save('diagnostic-container.json', json.loads(command('docker', 'inspect', NAME))[0])
        finally:
            try:
                if stopped_original:
                    command('docker', 'start', ORIGINAL)
                    summary['restored_models'] = ready(api, ORIGINAL)
                    summary['restored_smoke'] = api['chat'](URL, 'Reply with exactly this word and nothing else: READY', OUT / 'restored-smoke', count=32)
                    assert summary['restored_smoke']['text'].strip() == 'READY'
                    assert binaries(baseline) == baseline_hashes
                after = runner['hardware']()
                save('hardware-after.json', after)
                assert after == before
                summary['restored'] = True
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                save('summary.json', summary)
    print('BOLT_GROUPED_DIAGNOSTIC_PASSED', flush=True)


if __name__ == '__main__':
    main()
