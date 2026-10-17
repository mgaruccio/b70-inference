#!/usr/bin/env python3
"""Fixed target-only public-API replay controls, executed on inference-host only."""
import datetime
import json
from pathlib import Path
import runpy
import signal
import subprocess
import urllib.error

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-state-controls')
PREVIOUS = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-fidelity-controls')
CELLS = (('O0V1', 0, 1), ('O1V0', 1, 0))
MARKER = 'BOLT_STATE_SERVER_START'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_create(base, label, opt, vmm):
    assert (label, opt, vmm) in CELLS
    argv = list(base)
    argv[argv.index('--name') + 1] = 'bolt-state-r1-' + label.lower()
    assert argv[-3:-1] == ['bash', '-lc']
    assert not any(a.startswith(('GGML_SYCL_ENABLE_OPT=', 'GGML_SYCL_ENABLE_VMM=')) for a in argv)
    argv[-4:-4] = ['-e', f'GGML_SYCL_ENABLE_OPT={opt}', '-e', f'GGML_SYCL_ENABLE_VMM={vmm}']
    prefix, server = argv[-1].rsplit('exec ', 1)
    assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in prefix and '--spec-type' not in server
    argv[-1] = prefix + 'unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; echo ' + MARKER + '; exec ' + server
    return argv


def native_payload(prompt_ids, count):
    return dict(prompt=list(prompt_ids), n_predict=count, temperature=0, seed=42,
                cache_prompt=False, return_tokens=True, n_probs=5, post_sampling_probs=False,
                stream=False, ignore_eos=True, backend_sampling=False)


def check_native(body, prompt_count, output_count):
    assert body['tokens_evaluated'] == prompt_count and body['tokens_predicted'] == output_count
    assert body['timings']['cache_n'] == 0 and body['stop_type'] == 'limit'
    assert len(body['tokens']) == output_count and all(type(t) is int for t in body['tokens'])
    assert body['generation_settings']['speculative.types'] == 'none'
    assert body['generation_settings']['backend_sampling'] is False
    assert body['generation_settings']['temperature'] == 0 and body['generation_settings']['seed'] == 42
    assert body['completion_probabilities']


def main():
    previous = runpy.run_path(str(PREVIOUS / 'run-controls.py'))
    helper = runpy.run_path(str(previous['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(previous['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(previous['OLD'] / 'probe-api.py'))
    profile = runpy.run_path(str(previous['PROFILE'] / 'run-decode-profile.py'))
    command, ready, binaries = (helper[k] for k in ('command', 'ready', 'binaries'))
    assert command('hostname').strip() == 'inference-host'
    url, original_name, accepted = previous['URL'], previous['ORIGINAL'], previous['ACCEPTED']
    hashes = profile['EXPECTED']
    assert binaries(accepted) == hashes
    old_run = PREVIOUS / 'run-r1'
    bodies = [json.loads((old_run / label / 'diagnostic.response.json').read_text()) for label in previous['ORDER']]
    prefix_ids = bodies[0]['__verbose']['tokens'][:39]
    rendered = bodies[0]['__verbose']['prompt']
    assert all(b['__verbose']['tokens'][:39] == prefix_ids and b['__verbose']['prompt'] == rendered for b in bodies)
    chat_payload = json.loads((old_run / 'A0/diagnostic.request.json').read_text())
    frozen = json.loads((old_run / 'frozen-payload.json').read_text())
    original = json.loads(command('docker', 'inspect', original_name))[0]
    assert original['State']['Running'] and original['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in original['Mounts'] if m['Destination'] == '/src') == str(accepted)
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {original_name}
    names = ['bolt-state-r1-' + label.lower() for label, _, _ in CELLS]
    assert not set(names).intersection(command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines())
    base = json.loads((previous['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
    assert base[-1] == original['Config']['Cmd'][-1]
    before = runner['hardware']()
    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    (out / 'declaration.md').write_bytes((ROOT / 'README.md').read_bytes())
    save(out / 'original-container.json', original)
    save(out / 'binaries.json', hashes)
    save(out / 'hardware-before.json', before)
    save(out / 'frozen-prefix.json', dict(rendered_prompt=rendered, generated_prefix_ids=prefix_ids))
    summary = dict(status='prepared', timing_permitted=False, promotion_permitted=False, cells={})
    created = []
    stopped = False

    def request(cell, stem, endpoint, payload):
        save(cell / (stem + '.request.json'), payload)
        try:
            body = api['request'](url, endpoint, payload)
        except urllib.error.HTTPError as exc:
            (cell / (stem + '.http-error.txt')).write_bytes(exc.read())
            raise
        save(cell / (stem + '.response.json'), body)
        return body

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped = True
        command('docker', 'stop', '-t', '90', original_name)
        for label, opt, vmm in CELLS:
            cell = out / label
            cell.mkdir()
            name = 'bolt-state-r1-' + label.lower()
            argv = make_create(base, label, opt, vmm)
            save(cell / 'create-argv.json', argv)
            command(*argv, timeout=60)
            created.append((name, cell))
            data = dict(status='starting', opt=opt, vmm=vmm, responses=[])
            summary['cells'][label] = data
            command('docker', 'start', name)
            data['models'] = ready(api, name)
            startup = command('docker', 'logs', name)
            (cell / 'native-and-startup.log').write_text(startup)
            assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in startup
            assert MARKER in startup
            model_log = startup.split(MARKER, 1)[1]
            assert f'GGML_SYCL_ENABLE_OPT: {opt}' in model_log
            assert f'GGML_SYCL_ENABLE_VMM: {vmm}' in model_log
            assert 'GGML_SYCL_ENABLE_FUSION: 1' in model_log and 'GGML_SYCL_ENABLE_ESIMD: 1' in model_log
            assert 'adding speculative implementation' not in model_log
            quality_log = command('python3', '-B', str(previous['OLD'] / 'probe-api.py'),
                                  '--out', str(cell / 'quality'), '--mode', 'quality', timeout=1200)
            (cell / 'quality.log').write_text(quality_log)
            quality = json.loads((cell / 'quality/summary.json').read_text())
            assert quality['pass'] and len(quality['checks']) == 20 and all(c['pass'] for c in quality['checks'])
            data['quality_passed'] = 20
            data['probe_window_since'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            warmup = api['chat'](url, frozen['messages'][0]['content'], cell / 'warmup', count=128, stream=True, ignore_eos=True)
            assert json.loads((cell / 'warmup.request.json').read_text()) == frozen
            profile['check_request'](warmup, 55)
            template = request(cell, 'template', '/apply-template', chat_payload)
            assert template['prompt'] == rendered
            plain = request(cell, 'tokenize-plain', '/tokenize', dict(content=rendered, add_special=False, parse_special=True))
            special = request(cell, 'tokenize-special', '/tokenize', dict(content=rendered, add_special=True, parse_special=True))
            prompt_ids = plain['tokens']
            assert prompt_ids == special['tokens'] and len(prompt_ids) == 55 and all(type(t) is int for t in prompt_ids)
            detokenized = request(cell, 'detokenize', '/detokenize', dict(tokens=prompt_ids))
            assert detokenized['content'] == rendered
            data['prompt_ids'] = prompt_ids
            for i in range(3):
                stem = f'chat-{i:02}'
                body = request(cell, stem, '/v1/chat/completions', chat_payload)
                assert body['usage']['prompt_tokens'] == 55 and body['usage']['completion_tokens'] == 128
                assert body['choices'][0]['finish_reason'] == 'length' and body['timings']['cache_n'] == 0
                native = body['__verbose']
                check_native(native, 55, 128)
                assert native['prompt'] == rendered
                data['responses'].append(dict(stem=stem, family='chat', tokens=native['tokens']))
            for family, ids, count, repeats in [('native', prompt_ids, 128, 3), ('replay', prompt_ids + prefix_ids, 1, 6)]:
                payload = native_payload(ids, count)
                for i in range(repeats):
                    stem = f'{family}-{i:02}'
                    body = request(cell, stem, '/completion', payload)
                    check_native(body, len(ids), count)
                    data['responses'].append(dict(stem=stem, family=family, tokens=body['tokens']))
            data['status'] = 'completed'
            save(cell / 'summary.json', data)
            command('docker', 'stop', '-t', '90', name)
            assert binaries(accepted) == hashes and runner['hardware']() == before
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        raise
    finally:
        cleanup_errors = []
        try:
            for name, cell in created:
                for action in ('stop', 'logs', 'inspect'):
                    try:
                        if action == 'stop':
                            command('docker', 'stop', '-t', '90', name)
                        elif action == 'logs':
                            (cell / 'server.log').write_text(command('docker', 'logs', '--timestamps', name))
                        else:
                            save(cell / 'container.json', json.loads(command('docker', 'inspect', name))[0])
                    except BaseException as exc:
                        cleanup_errors.append(dict(container=name, action=action, error=repr(exc)))
        finally:
            try:
                if stopped:
                    assert not set(names).intersection(command('docker', 'ps', '--format', '{{.Names}}').splitlines())
                    command('docker', 'start', original_name)
                    summary['restored_models'] = ready(api, original_name)
                    smoke = api['chat'](url, 'Reply with exactly this word and nothing else: READY', out / 'restored-smoke', count=32)
                    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
                    restored = json.loads(command('docker', 'inspect', original_name))[0]
                    save(out / 'restored-container.json', restored)
                    assert restored['Id'] == original['Id'] and restored['Config'] == original['Config']
                    assert sorted(restored['Mounts'], key=lambda m:m['Destination']) == sorted(original['Mounts'], key=lambda m:m['Destination'])
                assert binaries(accepted) == hashes
                after = runner['hardware']()
                save(out / 'hardware-after.json', after)
                assert after == before
                summary['restored'] = True
                if cleanup_errors:
                    summary['cleanup_errors'] = cleanup_errors
                    raise RuntimeError('Experiment cleanup artifacts incomplete; inspect cleanup_errors')
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                save(out / 'summary.json', summary)
    print('BOLT_STATE_CONTROL_MATRIX_COMPLETED_NO_TIMING', flush=True)


if __name__ == '__main__':
    main()
