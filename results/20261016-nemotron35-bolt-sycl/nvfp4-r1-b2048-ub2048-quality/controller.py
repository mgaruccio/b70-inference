#!/usr/bin/env python3
"""One approved host-only Bolt compatibility/baseline cell; always restore Qwen."""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import signal
import socket
import subprocess
import time

ROOT = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
SRC = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37')
MODEL = Path('/home/mike/inference/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF')
IMAGE = 'intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04'
NAME = 'nemotron-bolt-probe'
HELPER = '/home/mike/b70-evals/20261012-qwen38-prefill-profiling/run-pilot.py'


def absent(command, name, timeout=90):
    """Require two consecutive absent samples, including stopped containers."""
    deadline, consecutive = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        names = command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
        consecutive = consecutive + 1 if name not in names else 0
        if consecutive == 2:
            return
        time.sleep(2)
    raise TimeoutError(f'Container name did not settle absent: {name}')


def canary(h):
    result = h['call']('/v1/completions', {'model': 'qwen38', 'prompt': 'The sum of two and three is',
                      'max_tokens': 1, 'temperature': 0, 'logprobs': 1}, timeout=180)
    body = result['body']
    assert body['choices'][0]['text'].strip() in ('five', '5'), result
    assert body['usage']['completion_tokens'] == 1 and body['usage']['prompt_tokens'] > 0, result
    values = body['choices'][0]['logprobs']['token_logprobs']
    assert values and all(math.isfinite(x) for x in values), result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--ubatch', type=int, default=128)
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--long-bench', action='store_true')
    parser.add_argument('--repeatability', action='store_true')
    parser.add_argument('--esimd', type=int, choices=(0, 1))
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--source', type=Path, default=SRC)
    parser.add_argument('--test-nvfp4', action='store_true')
    parser.add_argument('--dnnl-verbose', action='store_true', help='Diagnostic dispatch logs; not a comparable timing run.')
    parser.add_argument('--weights', choices=('original', 'ssm-q8_0', 'ssm-q6_k'), default='original')
    args = parser.parse_args()
    source = args.source.resolve()
    assert source in (SRC, SRC.with_name(SRC.name + '-nvfp4-r1'))
    assert not args.test_nvfp4 or source != SRC
    assert args.weights == 'original' or source == SRC
    model_dir = MODEL if args.weights == 'original' else MODEL.with_name(MODEL.name + '-' + args.weights)
    assert args.name.replace('-', '').replace('_', '').isalnum()
    assert 1024 <= args.context <= 262144 and 1 <= args.ubatch <= args.batch <= 4096
    out = ROOT / args.name
    out.mkdir(exist_ok=False)
    h = runpy.run_path(HELPER)
    api = runpy.run_path(str(ROOT / 'probe-api.py'))
    command, save = h['command'], h['save']
    before = h['snapshot']()
    save(out / 'preflight.json', before)
    h['kernel'](out / 'kernel-pre.log')
    assert command('docker', 'ps', '--format', '{{.Names}}').splitlines() == ['qwen38']
    assert NAME not in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    assert command('docker', 'inspect', '--format', '{{.Image}}', 'qwen38').strip() == 'sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
    save(out / 'qwen-before.json', canary(h))
    manifest_name = 'model-verified.json' if args.weights == 'original' else f'model-verified-{args.weights}.json'
    verified = json.loads((ROOT / manifest_name).read_text())
    assert Path(verified['path']) == model_dir / 'NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf'
    assert verified['pass_'] and Path(verified['path']).stat().st_size == verified['bytes']
    assert (source / 'UPSTREAM_COMMIT').read_text().strip() == 'b6334e37e6ef286fc5091a41b6561b2113d915fb'
    assert (source / 'build-sycl/bin/llama-server').is_file()
    assert (ROOT / 'runtime-precheck.log').is_file()
    if source != SRC:
        assert (ROOT / 'nvfp4-r1-runtime-precheck.log').is_file()
    if args.test_nvfp4:
        assert (source / 'build-sycl/bin/test-backend-ops').is_file()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 18098))
    production = h['PRODUCTION']
    (out / 'production-launcher.sh').write_bytes(production.read_bytes())
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    (out / 'base-helper.py').write_bytes(Path(HELPER).read_bytes())
    server = ['/src/build-sycl/bin/llama-server', '--verbosity', '4', '--model', '/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf',
              '--alias', 'nemotron-bolt', '-ngl', '999', '--ctx-size', str(args.context),
              '--parallel', '1', '--batch-size', str(args.batch), '--ubatch-size', str(args.ubatch),
              '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--flash-attn', 'on',
              '--jinja', '--host', '0.0.0.0', '--port', '18098', '--temp', '0', '--load-mode', 'none']
    setup = ('set -eo pipefail; source /opt/intel/oneapi/setvars.sh --force; '
             'export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}; ')
    numerical = ''
    if args.test_nvfp4:
        test = shlex.join(['/src/build-sycl/bin/test-backend-ops', 'test', '-b', 'SYCL0',
                           '-o', 'MUL_MAT', '-p', 'nvfp4_reorder=1'])
        numerical = ('echo NVFP4_NUMERICAL_OPT1; GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_OPT=1 ' + test +
                     ' 2>&1 | tee /tmp/nvfp4-opt1.log; '
                     "grep -Fx '  56/56 tests passed' /tmp/nvfp4-opt1.log; "
                     "grep -F 'Calling reorder_mul_mat_vec_nvfp4_q8_1_sycl' /tmp/nvfp4-opt1.log > /dev/null; "
                     'echo NVFP4_NUMERICAL_OPT0; GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_OPT=0 ' + test +
                     ' 2>&1 | tee /tmp/nvfp4-opt0.log; '
                     "grep -Fx '  56/56 tests passed' /tmp/nvfp4-opt0.log; "
                     "if grep -F 'Calling reorder_mul_mat_vec_nvfp4_q8_1_sycl' /tmp/nvfp4-opt0.log; then exit 1; fi; "
                     'echo NVFP4_NUMERICAL_PASSED; ')
    shell = setup + numerical + 'exec ' + shlex.join(server)
    # Create while Qwen is still serving: timeout cannot cause a late GPU start.
    create = ['docker', 'create', '--name', NAME, '--ipc=host',
              '-p', '127.0.0.1:18098:18098', '--device', '/dev/dri', '--group-add', str(os.stat('/dev/dri/renderD128').st_gid),
              '-v', '/dev/dri:/dev/dri:ro', '-v', str(source) + ':/src:ro', '-v', str(model_dir) + ':/models:ro',
              '-v', '/home/mike/inference/src/oneDNN/install:/opt/onednn:ro',
              '-e', 'ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE', '-e', 'ZE_AFFINITY_MASK=0',
              '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:0', IMAGE, 'bash', '-lc', shell]
    if args.esimd is not None:
        create[2:2] = ['-e', f'GGML_SYCL_ENABLE_ESIMD={args.esimd}']
    if args.dnnl_verbose:
        create[2:2] = ['-e', 'ONEDNN_VERBOSE=1']
    save(out / 'create-argv.json', create)
    summary = {'tier': 'development', 'configuration': {**vars(args), 'source': str(source)}, 'status': 'prepared'}
    save(out / 'summary.json', summary)
    disrupted, error = False, None
    cleanup = {'warnings': []}

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')

    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # Preserve nohup's SSH-loss protection.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)

    def wait_api():
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if command('docker', 'inspect', '--format', '{{.State.Running}}', NAME).strip() != 'true':
                raise RuntimeError('Bolt stopped before readiness')
            try:
                api['request']('http://127.0.0.1:18098', '/health', timeout=3)
                return api['request']('http://127.0.0.1:18098', '/v1/models', timeout=3)
            except (OSError, ValueError):
                time.sleep(2)
        raise TimeoutError('Bolt startup deadline')

    try:
        summary['container_id'] = command(*create, timeout=60).strip()
        assert command('docker', 'inspect', '--format', '{{.State.Running}}', NAME).strip() == 'false'
        disrupted = True
        command('docker', 'stop', '--time', '30', 'qwen38')
        absent(command, 'qwen38')
        assert not command('docker', 'ps', '--format', '{{.Names}}').splitlines()
        command('docker', 'start', NAME, timeout=60)
        summary['models'] = wait_api()
        command('python3', '-B', str(ROOT / 'probe-api.py'), '--out', str(out / 'smoke'), timeout=1200)
        summary['smoke'] = json.loads((out / 'smoke/summary.json').read_text())
        if args.quality:
            command('python3', '-B', str(ROOT / 'probe-api.py'), '--out', str(out / 'quality'), '--mode', 'quality', timeout=1800)
            summary['quality'] = json.loads((out / 'quality/summary.json').read_text())
        if args.bench:
            command('python3', '-B', str(ROOT / 'probe-api.py'), '--out', str(out / 'bench'), '--mode', 'bench', timeout=1800)
            summary['bench'] = json.loads((out / 'bench/summary.json').read_text())
        if args.long_bench:
            command('python3', '-B', str(ROOT / 'probe-api.py'), '--out', str(out / 'long'), '--mode', 'long', timeout=2400)
            summary['long'] = json.loads((out / 'long/summary.json').read_text())
        if args.repeatability:
            command('python3', '-B', str(ROOT / 'probe-api.py'), '--out', str(out / 'repeat'), '--mode', 'repeat', timeout=1200)
            summary['repeat'] = json.loads((out / 'repeat/summary.json').read_text())
        summary['status'] = 'passed'
    except BaseException as exc:
        error = exc
        summary.update(status='failed', error=repr(exc))
    finally:
        # Ignore additional termination requests only while completing bounded restoration.
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        try:
            (out / 'server.log').write_text(command('docker', 'logs', NAME))
        except BaseException as exc:
            cleanup['warnings'].append(f'logs: {exc!r}')
        try:
            # The container was created before disruption; no pending create can launch a GPU owner.
            # Removing this known ID/name also prevents an interrupted start from arriving late.
            if NAME in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines():
                try:
                    command('docker', 'rm', '-f', NAME, timeout=90)
                except BaseException as exc:
                    cleanup['warnings'].append(f'remove: {exc!r}')
            absent(command, NAME)
            if disrupted:
                names = command('docker', 'ps', '--format', '{{.Names}}').splitlines()
                assert names in ([], ['qwen38']), f'Unknown GPU owner blocks restore: {names}'
                # Finish a possibly interrupted original stop before restarting the immutable launcher.
                if 'qwen38' in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines():
                    command('docker', 'stop', '--time', '30', 'qwen38')
                absent(command, 'qwen38')
                assert not command('docker', 'ps', '--format', '{{.Names}}').splitlines()
                deadline, process, attempts = time.monotonic() + 600, None, 0
                while time.monotonic() < deadline:
                    if process is None or process.poll() is not None:
                        # A failed launcher (e.g. delayed name removal) is rechecked, not blindly retried.
                        running = command('docker', 'ps', '--format', '{{.Names}}').splitlines()
                        assert running in ([], ['qwen38']), running
                        if not running:
                            assert attempts < 3, 'Original Qwen launcher repeatedly exited'
                            attempts += 1
                            absent(command, 'qwen38')
                            with (out / 'production-restored.log').open('a') as log:
                                process = subprocess.Popen(['bash', str(production)], stdin=subprocess.DEVNULL,
                                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        models = h['call']('/v1/models', timeout=3)
                        assert any(m['id'] == 'qwen38' and m['max_model_len'] == 212992 for m in models['body']['data'])
                        cleanup['canary'] = canary(h)
                        break
                    except (OSError, ValueError, RuntimeError):
                        time.sleep(2)
                else:
                    raise TimeoutError('Original Qwen readiness/canary deadline')
                cleanup['restore_attempts'] = attempts
                assert h['snapshot']() == before, 'Host invariants changed'
                h['kernel'](out / 'kernel-post.log')
                cleanup['production_restored'] = True
        except BaseException as exc:
            cleanup['error'] = repr(exc)
            error = error or exc
        summary['cleanup'] = cleanup
        save(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    if error:
        raise error


if __name__ == '__main__':
    main()
