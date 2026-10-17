#!/usr/bin/env python3
"""Remote-only fixed-count fidelity controls. No timing or promotion authorization."""
import datetime
import json
from pathlib import Path
import runpy
import shlex
import signal
import subprocess
import urllib.error

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-fidelity-controls')
OLD = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
PRIOR = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-outprod')
DRAFT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr')
PROFILE = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-decode-profile')
GROUPED = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')
ACCEPTED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-outprod-r1')
ORIGINAL = 'nemotron-bolt-probe'
URL = 'http://127.0.0.1:18098'
ORDER = ('A0', 'B0', 'C0', 'C1', 'B1', 'A1')
SPEC = ['--spec-type', 'draft-dflash', '--spec-draft-n-max', '4', '--spec-draft-n-min', '0',
        '--spec-draft-p-min', '0', '--spec-draft-sampling', 'greedy', '--no-spec-draft-backend-sampling',
        '--spec-draft-ngl', '999', '--spec-draft-model', '/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_create(base, label, image):
    assert label in ORDER
    argv = list(base)
    argv[argv.index('--name') + 1] = 'bolt-fidelity-r1-' + label.lower()
    prefix, server = argv[-1].rsplit('exec ', 1)
    assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in prefix
    assert '--spec-type' not in server
    if label[0] != 'A':
        argv[argv.index(str(ACCEPTED) + ':/src:ro')] = str(CANDIDATE) + ':/src:ro'
        index = argv.index(image)
        argv[index:index] = ['-v', str(PRIOR / 'native-outprod.sh') + ':/native-outprod.sh:ro']
        prefix += 'bash /native-outprod.sh; '
    if label[0] == 'C':
        index = argv.index(image)
        argv[index:index] = ['-v', str(DRAFT / 'draft') + ':/draft:ro']
        server += ' ' + shlex.join(SPEC)
    argv[-1] = prefix + 'unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; exec ' + server
    return argv


def diagnostic_payload(payload):
    result = dict(payload)
    result.pop('stream_options', None)
    result.update(stream=False, verbose=True, return_tokens=True, logprobs=True, top_logprobs=5, backend_sampling=False)
    return result


def main():
    helper = runpy.run_path(str(GROUPED / 'run-grouped-diagnostic.py'))
    command, ready, binaries = (helper[k] for k in ('command', 'ready', 'binaries'))
    runner = runpy.run_path(str(OLD / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD / 'probe-api.py'))
    profile = runpy.run_path(str(PROFILE / 'run-decode-profile.py'))
    assert command('hostname').strip() == 'inference-host'
    accepted_hashes = profile['EXPECTED']
    candidate_hashes = json.loads((PRIOR / 'activation-r1/candidate-binaries.json').read_text())
    assert binaries(ACCEPTED) == accepted_hashes and binaries(CANDIDATE) == candidate_hashes
    draft = json.loads((DRAFT / 'draft-verified.json').read_text())
    assert draft['pass'] and Path(draft['path']).stat().st_size == draft['bytes']
    before = runner['hardware']()
    original = json.loads(command('docker', 'inspect', ORIGINAL))[0]
    assert original['State']['Running'] and original['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in original['Mounts'] if m['Destination'] == '/src') == str(ACCEPTED)
    assert 'GGML_SYCL_DEBUG=1' not in original['Config']['Env']
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {ORIGINAL}
    names = ['bolt-fidelity-r1-' + label.lower() for label in ORDER]
    existing = command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    assert not set(names).intersection(existing)
    base = json.loads((OLD / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
    assert base[-1] == original['Config']['Cmd'][-1]
    payload = json.loads((PROFILE / 'run-r1/baseline/bench/measured-00.request.json').read_text())
    assert payload['temperature'] == 0 and payload['seed'] == 42 and payload['cache_prompt'] is False
    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    (out / 'declaration.md').write_bytes((ROOT / 'README.md').read_bytes())
    (out / 'native-outprod.sh').write_bytes((PRIOR / 'native-outprod.sh').read_bytes())
    save(out / 'original-container.json', original)
    save(out / 'accepted-binaries.json', accepted_hashes)
    save(out / 'candidate-binaries.json', candidate_hashes)
    save(out / 'draft-verified.json', draft)
    save(out / 'hardware-before.json', before)
    save(out / 'frozen-payload.json', payload)
    summary = dict(status='prepared', tier='development-diagnostic', order=ORDER, timing_permitted=False, phases={})
    stopped, created = False, []

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped = True
        command('docker', 'stop', '-t', '90', ORIGINAL)
        for label, name in zip(ORDER, names):
            cell = out / label
            cell.mkdir()
            argv = make_create(base, label, runner['IMAGE'])
            save(cell / 'create-argv.json', argv)
            command(*argv, timeout=60)
            created.append((name, cell))
            command('docker', 'start', name)
            data = dict(arm=label[0], responses=[])
            summary['phases'][label] = data
            data['models'] = ready(api, name)
            startup = command('docker', 'logs', name)
            (cell / 'native-and-startup.log').write_text(startup)
            assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in startup
            if label[0] != 'A':
                assert 'BOLT_LILICORR_OUTPROD_NATIVE_PASSED' in startup
            if label[0] == 'C':
                assert "adding speculative implementation 'draft-dflash'" in startup
            else:
                assert 'adding speculative implementation' not in startup
            quality_log = command('python3', '-B', str(OLD / 'probe-api.py'), '--out', str(cell / 'quality'),
                                  '--mode', 'quality', timeout=1200)
            (cell / 'quality.log').write_text(quality_log)
            quality = json.loads((cell / 'quality/summary.json').read_text())
            assert quality['pass'] and len(quality['checks']) == 20 and all(x['pass'] for x in quality['checks'])
            data['quality'] = quality
            data['probe_window_since'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            for index in range(-1, 3):
                stem = 'warmup' if index == -1 else f'repeat-{index:02}'
                result = api['chat'](URL, payload['messages'][0]['content'], cell / stem,
                                     count=128, stream=True, ignore_eos=True)
                assert json.loads((cell / (stem + '.request.json')).read_text()) == payload
                assert result['usage'] and result['finish_reason'], 'Incomplete stream; stop and inspect logs'
                profile['check_request'](result, 55)
                if index == -1:
                    data['warmup'] = result
                else:
                    data['responses'].append(result)
            diag = diagnostic_payload(payload)
            save(cell / 'diagnostic.request.json', diag)
            try:
                body = api['request'](URL, '/v1/chat/completions', diag)
            except urllib.error.HTTPError as exc:
                (cell / 'diagnostic.http-error.txt').write_bytes(exc.read())
                raise
            save(cell / 'diagnostic.response.json', body)
            assert body['usage']['prompt_tokens'] == 55 and body['usage']['completion_tokens'] == 128
            assert body['timings']['cache_n'] == 0 and body['choices'][0]['finish_reason'] == 'length'
            tokens = body['__verbose']['tokens']
            assert len(tokens) == 128 and all(isinstance(t, int) for t in tokens)
            data['diagnostic_token_count'] = len(tokens)
            data['status'] = 'completed'
            save(cell / 'summary.json', data)
            command('docker', 'stop', '-t', '90', name)
            assert binaries(ACCEPTED) == accepted_hashes and binaries(CANDIDATE) == candidate_hashes
            assert runner['hardware']() == before
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        raise
    finally:
        try:
            for name, cell in created:
                try:
                    command('docker', 'stop', '-t', '90', name)
                finally:
                    (cell / 'server.log').write_text(command('docker', 'logs', '--timestamps', name))
                    save(cell / 'container.json', json.loads(command('docker', 'inspect', name))[0])
        finally:
            try:
                if stopped:
                    assert not set(names).intersection(command('docker', 'ps', '--format', '{{.Names}}').splitlines())
                    command('docker', 'start', ORIGINAL)
                    summary['restored_models'] = ready(api, ORIGINAL)
                    smoke = api['chat'](URL, 'Reply with exactly this word and nothing else: READY', out / 'restored-smoke', count=32)
                    assert smoke['text'].strip() == 'READY'
                    restored = json.loads(command('docker', 'inspect', ORIGINAL))[0]
                    save(out / 'restored-container.json', restored)
                    assert restored['Id'] == original['Id'] and restored['Config'] == original['Config']
                    assert sorted(restored['Mounts'], key=lambda m:m['Destination']) == sorted(original['Mounts'], key=lambda m:m['Destination'])
                assert binaries(ACCEPTED) == accepted_hashes and binaries(CANDIDATE) == candidate_hashes
                after = runner['hardware']()
                save(out / 'hardware-after.json', after)
                assert after == before
                summary['restored'] = True
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                save(out / 'summary.json', summary)
    print('BOLT_FIDELITY_CONTROL_MATRIX_COMPLETED_NO_TIMING', flush=True)


if __name__ == '__main__':
    main()
