#!/usr/bin/env python3
"""Real-server callback localization; fixed trace-on/off controls, never timing."""
import datetime
import hashlib
import json
from pathlib import Path
import runpy
import signal
import subprocess
import urllib.error

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-block-trace')
PREFIX = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefix-replay')
PREVIOUS = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-fidelity-controls')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-block-trace-r1')
CELLS = (('on', 1), ('off', 0))
MARKER = 'BOLT_TENSOR_SERVER_START'


def binaries(source):
    names = ('llama-server', 'libggml-sycl.so.0', 'test-backend-ops', 'libllama-server-impl.so')
    return {name: hashlib.sha256((Path(source) / 'build-sycl/bin' / name).read_bytes()).hexdigest() for name in names}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_create(base, label, enabled, accepted):
    assert (label, enabled) in CELLS
    argv = list(base)
    argv[argv.index('--name') + 1] = 'bolt-block-trace-r1-' + label
    argv[argv.index(str(accepted) + ':/src:ro')] = str(CANDIDATE) + ':/src:ro'
    assert argv[-3:-1] == ['bash', '-lc']
    argv[-4:-4] = ['-e', f'BOLT_TRACE_TENSORS={enabled}']
    prefix, server = argv[-1].rsplit('exec ', 1)
    assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in prefix and '--spec-type' not in server
    argv[-1] = prefix + 'unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; echo ' + MARKER + '; exec ' + server
    return argv


def main():
    previous = runpy.run_path(str(PREVIOUS / 'run-controls.py'))
    replay = runpy.run_path(str(PREFIX / 'run-replay.py'))
    helper = runpy.run_path(str(previous['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(previous['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(previous['OLD'] / 'probe-api.py'))
    profile = runpy.run_path(str(previous['PROFILE'] / 'run-decode-profile.py'))
    command, ready = (helper[k] for k in ('command', 'ready'))
    assert command('hostname').strip() == 'inference-host'
    assert 'BOLT_TENSOR_TRACE_SOURCE_AND_BINARY_SCOPE_VERIFIED' in (ROOT / 'post-build-verification.log').read_text()
    assert 'BOLT_TENSOR_TRACE_RUNTIME_PRECHECK_PASSED' in (ROOT / 'runtime-precheck.log').read_text()
    assert json.loads((ROOT / 'source-diff.json').read_text())['changed'] == ['tools/server/server-context.cpp']
    url, original_name, accepted = previous['URL'], previous['ORIGINAL'], previous['ACCEPTED']
    hashes = dict(profile['EXPECTED'], **{'libllama-server-impl.so': '67ae7c8673110b52d11924677ad662141dc93f57d7c21d67a3f0d3e4489e09ad'})
    candidate_hashes = json.loads((ROOT / 'candidate-binaries.json').read_text())
    assert binaries(accepted) == hashes and binaries(CANDIDATE) == candidate_hashes
    assert all(hashes[k] == candidate_hashes[k] for k in hashes if k != 'libllama-server-impl.so')
    assert hashes['libllama-server-impl.so'] != candidate_hashes['libllama-server-impl.so']
    frozen_prefix = json.loads((PREFIX / 'run-r1/frozen-prefix.json').read_text())
    prefix_ids, rendered = frozen_prefix['generated_prefix_ids'], frozen_prefix['rendered_prompt']
    assert len(prefix_ids) == 39
    chat_payload = json.loads((PREVIOUS / 'run-r1/A0/diagnostic.request.json').read_text())
    frozen = json.loads((PREVIOUS / 'run-r1/frozen-payload.json').read_text())
    original = json.loads(command('docker', 'inspect', original_name))[0]
    assert original['State']['Running'] and original['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in original['Mounts'] if m['Destination'] == '/src') == str(accepted)
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {original_name}
    names = ['bolt-block-trace-r1-' + label for label, _ in CELLS]
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
    save(out / 'candidate-binaries.json', candidate_hashes)
    save(out / 'hardware-before.json', before)
    save(out / 'frozen-prefix.json', frozen_prefix)
    summary = dict(status='prepared', timing_permitted=False, promotion_permitted=False, cells={})
    created, stopped = [], False

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
        for label, enabled in CELLS:
            cell = out / label
            cell.mkdir()
            name = 'bolt-block-trace-r1-' + label
            argv = make_create(base, label, enabled, accepted)
            save(cell / 'create-argv.json', argv)
            command(*argv, timeout=60)
            created.append((name, cell))
            data = dict(status='starting', tracing=bool(enabled), responses=[])
            summary['cells'][label] = data
            command('docker', 'start', name)
            data['models'] = ready(api, name)
            startup = command('docker', 'logs', name)
            (cell / 'native-and-startup.log').write_text(startup)
            assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in startup and MARKER in startup
            model_log = startup.split(MARKER, 1)[1]
            assert ('BOLT_TENSOR_TRACE_ENABLED' in model_log) == bool(enabled)
            assert 'GGML_SYCL_ENABLE_FUSION: 1' in model_log and 'GGML_SYCL_ENABLE_ESIMD: 1' in model_log
            assert 'GGML_SYCL_ENABLE_OPT: 1' in model_log and 'GGML_SYCL_ENABLE_VMM: 1' in model_log
            assert 'adding speculative implementation' not in model_log
            quality_log = command('python3', '-B', str(previous['OLD'] / 'probe-api.py'),
                                  '--out', str(cell / 'quality'), '--mode', 'quality', timeout=1200)
            (cell / 'quality.log').write_text(quality_log)
            quality = json.loads((cell / 'quality/summary.json').read_text())
            assert quality['pass'] and len(quality['checks']) == 20 and all(c['pass'] for c in quality['checks'])
            data['quality_passed'] = 20
            warm = api['chat'](url, frozen['messages'][0]['content'], cell / 'warmup', count=128, stream=True, ignore_eos=True)
            assert json.loads((cell / 'warmup.request.json').read_text()) == frozen
            profile['check_request'](warm, 55)
            template = request(cell, 'template', '/apply-template', chat_payload)
            assert template['prompt'] == rendered
            plain = request(cell, 'tokenize-plain', '/tokenize', dict(content=rendered, add_special=False, parse_special=True))
            special = request(cell, 'tokenize-special', '/tokenize', dict(content=rendered, add_special=True, parse_special=True))
            prompt_ids = plain['tokens']
            assert prompt_ids == special['tokens'] and len(prompt_ids) == 55 and all(type(t) is int for t in prompt_ids)
            detok = request(cell, 'detokenize', '/detokenize', dict(tokens=prompt_ids))
            assert detok['content'] == rendered
            assert prompt_ids == json.loads((PREFIX / 'run-r1/F1E1/summary.json').read_text())['prompt_ids']
            data['prompt_ids'] = prompt_ids
            data['replay_window_since'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            payload = replay['native_payload'](prompt_ids + prefix_ids, 1)
            for i in range(6):
                stem = f'replay-{i:02}'
                body = request(cell, stem, '/completion', payload)
                replay['check_native'](body, 94, 1)
                data['responses'].append(dict(stem=stem, tokens=body['tokens']))
            data['status'] = 'completed'
            save(cell / 'summary.json', data)
            command('docker', 'stop', '-t', '90', name)
            assert binaries(accepted) == hashes and binaries(CANDIDATE) == candidate_hashes
            assert runner['hardware']() == before
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
                assert binaries(accepted) == hashes and binaries(CANDIDATE) == candidate_hashes
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
    print('BOLT_TENSOR_TRACE_MATRIX_COMPLETED_NO_TIMING', flush=True)


if __name__ == '__main__':
    main()
