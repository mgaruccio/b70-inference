#!/usr/bin/env python3
"""Remote-only Bolt LiLiCorr fidelity controls; no timing or promotion."""
import datetime
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import signal
import subprocess
import urllib.error


ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-deterministic')
OLD = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
PRIOR = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-outprod')
DRAFT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr')
PROFILE = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-decode-profile')
GROUPED = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')
ACCEPTED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
BASELINE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-dnn-deterministic-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-deterministic-r1')
ORIGINAL = 'nemotron-bolt-probe'
URL = 'http://127.0.0.1:18098'
PREFIX = 'bolt-lilicorr-deterministic-r1-'
SERVER_MARKER = 'BOLT_LILICORR_DETERMINISTIC_SERVER_START'
ORDER = ('A0', 'B0', 'C0', 'C1', 'B1', 'A1')
ARTIFACTS = ('llama-server', 'libggml-sycl.so.0', 'test-backend-ops', 'libllama-server-impl.so')
SERVER_ENV = (
    ('BOLT_TRACE_TENSORS', '0'),
    ('ONEDNN_VERBOSE', '0'),
    ('GGML_SYCL_ENABLE_DNN', '1'),
)
FORBIDDEN_MODEL_LOG = ('BOLT_TENSOR_TRACE_ENABLED', 'common_debug_cb_eval',
                       'BOLT_PROJECTION_REPEAT', 'onednn_verbose')
SPEC = [
    '--spec-type', 'draft-dflash', '--spec-draft-n-max', '4', '--spec-draft-n-min', '0',
    '--spec-draft-p-min', '0', '--spec-draft-sampling', 'greedy',
    '--no-spec-draft-backend-sampling', '--spec-draft-ngl', '999',
    '--spec-draft-model', '/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf',
]
BASELINE_EXPECTED = {
    'llama-server': '22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d',
    'libggml-sycl.so.0': '95733d9042844ad751e82b78d8dccb6326605afdd7ddf0f9c5bb2bc40e82bf58',
    'test-backend-ops': 'a6a3f666754fce8edfbb59d9840290a188ba2681e7065f812de3359f4e12c98d',
    'libllama-server-impl.so': 'ae1d0d772b97676ef711605cb52adfcd2529517bfed43671b6e8496482cf566b',
}
ACCEPTED_IMPL = '67ae7c8673110b52d11924677ad662141dc93f57d7c21d67a3f0d3e4489e09ad'
DRAFT_SHA256 = '080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194'
DRAFT_BYTES = 998242560
SOURCE_SCOPE = [
    'ggml/src/ggml-sycl/ggml-sycl.cpp',
    'ggml/src/ggml-sycl/outprod.cpp',
    'ggml/src/ggml-sycl/outprod.hpp',
    'tests/test-backend-ops.cpp',
]


def command(*argv, timeout=180):
    print(shlex.join(argv), flush=True)
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    return json.loads(path.read_text())


def binaries(source):
    """Return the complete four-artifact receipt for a build tree."""
    source = Path(source)
    return {
        name: hashlib.sha256((source / 'build-sycl/bin' / name).read_bytes()).hexdigest()
        for name in ARTIFACTS
    }


def accepted_expected(profile):
    expected = dict(profile['EXPECTED'])
    expected['libllama-server-impl.so'] = ACCEPTED_IMPL
    assert set(expected) == set(ARTIFACTS)
    return expected


def check_stream_result(result, prompt_tokens=55, completion_tokens=128):
    assert result['usage']['prompt_tokens'] == prompt_tokens, result['usage']
    assert result['usage']['completion_tokens'] == completion_tokens, result['usage']
    assert result['timings']['cache_n'] == 0, result['timings']
    assert result['timings']['predicted_n'] == completion_tokens, result['timings']
    assert result['finish_reason'] == 'length' and result['text'].strip()
    return result['text']


def check_diagnostic_result(body, prompt_tokens=55, completion_tokens=128):
    assert body['usage']['prompt_tokens'] == prompt_tokens, body['usage']
    assert body['usage']['completion_tokens'] == completion_tokens, body['usage']
    assert body['timings']['cache_n'] == 0, body['timings']
    assert body['choices'][0]['finish_reason'] == 'length'
    tokens = body['__verbose']['tokens']
    assert len(tokens) == completion_tokens and all(type(token) is int for token in tokens)
    rows = body['choices'][0]['logprobs']['content']
    assert len(rows) == completion_tokens
    assert [row['id'] for row in rows] == tokens
    return tokens


def check_model_log(model_log, arm):
    """Check only the post-marker server phase; native debug output is allowed."""
    lowered = model_log.lower()
    assert 'ggml_sycl_enable_dnn: 1' in lowered, 'DNN1 was not reported by the server'
    for marker in FORBIDDEN_MODEL_LOG:
        assert marker.lower() not in lowered, marker
    assert 'ggml_sycl_debug=1' not in lowered
    if arm == 'C':
        assert "adding speculative implementation 'draft-dflash'" in model_log
    else:
        assert 'adding speculative implementation' not in model_log


def make_create(base, label, image):
    """Derive one arm from the immutable accepted launch argv."""
    assert label in ORDER
    argv = list(base)
    argv[argv.index('--name') + 1] = PREFIX + label.lower()
    source = BASELINE if label[0] == 'A' else CANDIDATE
    accepted_mount = str(ACCEPTED) + ':/src:ro'
    assert accepted_mount in argv
    argv[argv.index(accepted_mount)] = str(source) + ':/src:ro'

    # These are container-wide for native gates and the actual server alike.
    assert not any(any(arg.startswith(key + '=') for key, _ in SERVER_ENV) for arg in argv)
    env = []
    for key, value in SERVER_ENV:
        env.extend(['-e', f'{key}={value}'])
    index = argv.index(image)
    argv[index:index] = env

    prefix, server = argv[-1].rsplit('exec ', 1)
    assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in prefix
    assert '--spec-type' not in server
    if label[0] != 'A':
        index = argv.index(image)
        argv[index:index] = ['-v', str(PRIOR / 'native-outprod.sh') + ':/native-outprod.sh:ro']
        prefix += 'bash /native-outprod.sh; '
    if label[0] == 'C':
        index = argv.index(image)
        argv[index:index] = ['-v', str(DRAFT / 'draft') + ':/draft:ro']
        server += ' ' + shlex.join(SPEC)
    argv[-1] = (prefix + 'unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; echo '
                 + SERVER_MARKER + '; exec ' + server)
    return argv


def diagnostic_payload(payload):
    result = dict(payload)
    result.pop('stream_options', None)
    result.update(stream=False, verbose=True, return_tokens=True, logprobs=True,
                  top_logprobs=5, backend_sampling=False)
    return result


def main():
    helper = runpy.run_path(str(GROUPED / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(OLD / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD / 'probe-api.py'))
    profile = runpy.run_path(str(PROFILE / 'run-decode-profile.py'))
    command, ready = helper['command'], helper['ready']
    assert command('hostname').strip() == 'inference-host'

    assert 'BOLT_LILICORR_DETERMINISTIC_BUILD_VERIFIED' in (ROOT / 'post-build-verification.log').read_text()
    assert 'BOLT_LILICORR_DETERMINISTIC_RUNTIME_PRECHECK_PASSED' in (ROOT / 'runtime-precheck.log').read_text()
    source_diff = load(ROOT / 'source-diff.json')
    assert source_diff['changed_relative_to_baseline'] == SOURCE_SCOPE

    accepted_hashes = load(ROOT / 'accepted-binaries.json')
    baseline_hashes = load(ROOT / 'baseline-binaries.json')
    candidate_hashes = load(ROOT / 'candidate-binaries.json')
    assert accepted_hashes == accepted_expected(profile)
    assert baseline_hashes == BASELINE_EXPECTED
    assert set(candidate_hashes) == set(ARTIFACTS)
    assert binaries(ACCEPTED) == accepted_hashes
    assert binaries(BASELINE) == baseline_hashes
    assert binaries(CANDIDATE) == candidate_hashes

    draft = load(DRAFT / 'draft-verified.json')
    assert draft['pass'] and draft['bytes'] == DRAFT_BYTES and draft['sha256'] == DRAFT_SHA256
    draft_path = Path(draft['path'])
    assert draft_path.stat().st_size == DRAFT_BYTES
    assert hashlib.sha256(draft_path.read_bytes()).hexdigest() == DRAFT_SHA256

    before = runner['hardware']()
    original = json.loads(command('docker', 'inspect', ORIGINAL))[0]
    assert original['State']['Running'] and original['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in original['Mounts'] if m['Destination'] == '/src') == str(ACCEPTED)
    assert 'GGML_SYCL_DEBUG=1' not in original['Config']['Env']
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {ORIGINAL}

    names = [PREFIX + label.lower() for label in ORDER]
    existing = command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    assert not set(names).intersection(existing)
    base = load(OLD / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json')
    assert base[base.index('--name') + 1] == ORIGINAL
    assert base[-1] == original['Config']['Cmd'][-1]
    payload = load(PROFILE / 'run-r1/baseline/bench/measured-00.request.json')
    assert payload['temperature'] == 0 and payload['seed'] == 42
    assert payload['cache_prompt'] is False and payload['max_tokens'] == 128
    assert payload['stream'] is True and payload['stream_options'] == {'include_usage': True}

    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    (out / 'declaration.md').write_bytes((ROOT / 'README.md').read_bytes())
    (out / 'native-outprod.sh').write_bytes((PRIOR / 'native-outprod.sh').read_bytes())
    save(out / 'original-container.json', original)
    save(out / 'accepted-binaries.json', accepted_hashes)
    save(out / 'baseline-binaries.json', baseline_hashes)
    save(out / 'candidate-binaries.json', candidate_hashes)
    save(out / 'draft-verified.json', draft)
    save(out / 'source-diff.json', source_diff)
    save(out / 'hardware-before.json', before)
    save(out / 'frozen-payload.json', payload)
    summary = dict(status='prepared', tier='development-diagnostic', order=ORDER,
                   timing_permitted=False, promotion_permitted=False, phases={})
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
            data = dict(arm=label[0], launch=label[1], responses=[])
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
            assert SERVER_MARKER in startup
            model_log = startup.split(SERVER_MARKER, 1)[1]
            check_model_log(model_log, label[0])
            data['server_marker'] = SERVER_MARKER

            quality_log = command('python3', '-B', str(OLD / 'probe-api.py'), '--out',
                                  str(cell / 'quality'), '--mode', 'quality', timeout=1200)
            (cell / 'quality.log').write_text(quality_log)
            quality = load(cell / 'quality/summary.json')
            assert quality['pass'] and len(quality['checks']) == 20 and all(x['pass'] for x in quality['checks'])
            data['quality'] = quality
            data['probe_window_since'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')

            prompt = payload['messages'][0]['content']
            for index in range(-1, 3):
                stem = 'warmup' if index == -1 else f'repeat-{index:02}'
                result = api['chat'](URL, prompt, cell / stem, count=128, stream=True, ignore_eos=True)
                assert load(cell / (stem + '.request.json')) == payload
                profile['check_request'](result, 55)
                check_stream_result(result)
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
            diagnostic_tokens = check_diagnostic_result(body)
            data['diagnostic_token_count'] = len(diagnostic_tokens)
            data['diagnostic_logprob_count'] = len(body['choices'][0]['logprobs']['content'])
            data['status'] = 'completed'
            save(cell / 'summary.json', data)
            command('docker', 'stop', '-t', '90', name)
            assert (binaries(ACCEPTED), binaries(BASELINE), binaries(CANDIDATE)) == (
                accepted_hashes, baseline_hashes, candidate_hashes)
            assert runner['hardware']() == before
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        raise
    finally:
        # Artifact capture is best effort, but restoration is nested in its finally.
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
                    command('docker', 'start', ORIGINAL)
                    summary['restored_models'] = ready(api, ORIGINAL)
                    smoke = api['chat'](URL, 'Reply with exactly this word and nothing else: READY',
                                        out / 'restored-smoke', count=32)
                    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
                    save(out / 'restored-smoke.result.json', smoke)
                    restored = json.loads(command('docker', 'inspect', ORIGINAL))[0]
                    save(out / 'restored-container.json', restored)
                    assert restored['Id'] == original['Id'] and restored['Config'] == original['Config']
                    assert sorted(restored['Mounts'], key=lambda item: item['Destination']) == sorted(
                        original['Mounts'], key=lambda item: item['Destination'])
                assert (binaries(ACCEPTED), binaries(BASELINE), binaries(CANDIDATE)) == (
                    accepted_hashes, baseline_hashes, candidate_hashes)
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
    print('BOLT_LILICORR_DETERMINISTIC_FIDELITY_CONTROL_MATRIX_COMPLETED_NO_TIMING', flush=True)


if __name__ == '__main__':
    main()
