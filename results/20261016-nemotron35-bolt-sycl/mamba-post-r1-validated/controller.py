#!/usr/bin/env python3
"""Bolt-only experiment cell. Leave Bolt serving; never launch another model."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import signal
import subprocess
import time

ROOT = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
BASE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1')
MODEL = Path('/home/mike/inference/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF')
IMAGE = 'intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04'
NAME = 'nemotron-bolt-probe'
LABEL = 'b70.experiment=bolt-fusion'
SSM = BASE.with_name('llama.cpp-bolt-b6334e37-nvfp4-ssm-cache-r1')
MOE = BASE.with_name('llama.cpp-bolt-b6334e37-nvfp4-moe-reduce-r1')
COMBINED = BASE.with_name('llama.cpp-bolt-b6334e37-nvfp4-ssm-moe-r1')
POST = BASE.with_name('llama.cpp-bolt-b6334e37-nvfp4-mamba-post-r1')


def native_legacy_commands():
    test = '/src/build-sycl/bin/test-backend-ops test -b SYCL0'
    commands = []
    for opt in (1, 0):
        log = f'/tmp/legacy-opt{opt}.log'
        commands += [f'GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_OPT={opt} {test} -o MUL_MAT -p nvfp4_reorder=1 2>&1 | tee {log}',
                     f"grep -Fx '  56/56 tests passed' {log}",
                     f"if grep -Ei 'not supported|skipping large tensors' {log}; then exit 1; fi"]
        marker = f"grep -F 'Calling reorder_mul_mat_vec_nvfp4_q8_1_sycl' {log}"
        commands += [marker + ' > /dev/null' if opt else f'if {marker}; then exit 1; fi']
    return commands


def native_ssm_gate():
    test = '/src/build-sycl/bin/test-backend-ops test -b SYCL0'
    commands = native_legacy_commands()
    for fusion, expected in ((0, 0), (1, 11)):
        log = f'/tmp/ssm-cache-fusion{fusion}.log'
        commands += [f'echo SSM_CACHE_NATIVE_FUSION{fusion}',
                     f'GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_FUSION={fusion} {test} -o SSM_SCAN_CACHE_FUSION -p bolt_ssm_cache= 2>&1 | tee {log}',
                     f"grep -Fx '  26/26 tests passed' {log}",
                     f"if grep -Ei 'not supported|skipping large tensors' {log}; then exit 1; fi",
                     f"count=$(grep -Fc 'call ggml_sycl_ssm_scan_fused_cache:' {log} || true)",
                     f'test "$count" -eq {expected}']
    return '; '.join(commands + ['echo SSM_CACHE_NUMERICAL_PASSED']) + '; '


def native_moe_gate(legacy=True):
    test = '/src/build-sycl/bin/test-backend-ops test -b SYCL0'
    commands = native_legacy_commands() if legacy else []
    for fusion, expected in ((0, 0), (1, 8)):
        log = f'/tmp/moe-reduce-fusion{fusion}.log'
        commands += [f'echo MOE_REDUCE_NATIVE_FUSION{fusion}',
                     f"GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_FUSION={fusion} {test} -o '^MOE_REDUCE_9$' -j 1 2>&1 | tee {log}",
                     f"grep -Fx '  20/20 tests passed' {log}",
                     f"if grep -Ei 'not supported|skipping large tensors' {log}; then exit 1; fi",
                     f"count=$(grep -Fc 'MOE_REDUCE_9 full-array: mismatches=0 max_abs=0 ' {log} || true)",
                     'test "$count" -eq 20',
                     f"count=$(grep -Fc 'call ggml_sycl_op_moe_weighted_reduce:' {log} || true)",
                     f'test "$count" -eq {expected}']
    return '; '.join(commands + ['echo MOE_REDUCE_NUMERICAL_PASSED']) + '; '


def native_post_gate():
    test = '/src/build-sycl/bin/test-backend-ops test -b SYCL0'
    commands = native_legacy_commands()
    for fusion, expected in ((0, 0), (1, 16)):
        log = f'/tmp/mamba-post-fusion{fusion}.log'
        commands += [f'echo MAMBA_POST_NATIVE_FUSION{fusion}',
                     f"GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_FUSION={fusion} {test} -o '^MAMBA_POST$' -j 1 2>&1 | tee {log}",
                     f"grep -Fx '  37/37 tests passed' {log}",
                     f"if grep -Ei 'not supported|skipping large tensors' {log}; then exit 1; fi",
                     f"count=$(grep -Fc 'MAMBA_POST full-array: mismatches=0 max_abs=0 ' {log} || true)",
                     'test "$count" -eq 37',
                     f"count=$(grep -Fc 'call ggml_sycl_op_mamba_post:' {log} || true)",
                     f'test "$count" -eq {expected}']
    return '; '.join(commands + ['echo MAMBA_POST_NUMERICAL_PASSED']) + '; '


# Registration requires a fixed native gate; no arbitrary source/command injection.
CANDIDATES = {BASE: '', SSM: native_ssm_gate(), MOE: native_moe_gate(),
              COMBINED: native_ssm_gate() + native_moe_gate(legacy=False), POST: native_post_gate()}


def command(*argv, timeout=90):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def hardware():
    boost = Path('/sys/devices/system/cpu/cpufreq/boost').read_text().strip()
    caps = [p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_max_freq')]
    power = [p.read_text().strip() for p in Path('/sys/class/drm').glob('card*/device/hwmon/hwmon*/power1_cap')]
    assert boost == '0' and len(caps) == 16 and set(caps) == {'3801000'}
    assert power and set(power) == {'275000000'}
    return {'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'boost': boost, 'cpu_caps': caps, 'power_caps': power}


def replace_owned_container():
    running = command('docker', 'ps', '--format', '{{.Names}}').splitlines()
    assert set(running) <= {NAME}, f'GPU is occupied: {running}'
    if NAME in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines():
        label = command('docker', 'inspect', '--format', '{{index .Config.Labels "b70.experiment"}}', NAME).strip()
        assert label == 'bolt-fusion', 'Refusing to remove an unowned container'
        command('docker', 'rm', '-f', NAME)
        assert NAME not in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', type=Path, default=BASE)
    parser.add_argument('--measure', action='store_true', help='Fixed quality gate, then short/8K timings and retrieval.')
    parser.add_argument('--debug', action='store_true', help='Smoke-only dispatch trace, never timing.')
    args = parser.parse_args()
    source = args.source.resolve()
    assert source in CANDIDATES, 'Unregistered candidate/native gate'
    assert args.name.replace('-', '').replace('_', '').isalnum()
    assert not (args.measure and args.debug), 'Debug traces are not timing runs'
    assert (source / 'UPSTREAM_COMMIT').read_text().strip() == 'b6334e37e6ef286fc5091a41b6561b2113d915fb'
    assert (source / 'build-sycl/bin/llama-server').is_file()
    if source == SSM:
        assert (source / 'build-sycl/bin/test-backend-ops').is_file()
        assert 'SSM_CACHE_RUNTIME_PRECHECK_PASSED' in (ROOT / 'ssm-cache-r1-runtime-precheck.log').read_text()
    if source == MOE:
        assert (source / 'build-sycl/bin/test-backend-ops').is_file()
        assert 'MOE_REDUCE_RUNTIME_PRECHECK_PASSED' in (ROOT / 'moe-reduce-r1-runtime-precheck.log').read_text()
    if source == COMBINED:
        assert (source / 'build-sycl/bin/test-backend-ops').is_file()
        assert 'SSM_MOE_RUNTIME_PRECHECK_PASSED' in (ROOT / 'ssm-moe-r1-runtime-precheck.log').read_text()
    if source == POST:
        assert (source / 'build-sycl/bin/test-backend-ops').is_file()
        assert 'MAMBA_POST_RUNTIME_PRECHECK_PASSED' in (ROOT / 'mamba-post-r1-runtime-precheck.log').read_text()
    verified = json.loads((ROOT / 'model-verified.json').read_text())
    assert verified['pass_'] and Path(verified['path']) == MODEL / 'NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf'
    assert Path(verified['path']).stat().st_size == verified['bytes']
    before = hardware()
    out = ROOT / args.name
    out.mkdir(exist_ok=False)
    save(out / 'preflight.json', before)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    api = runpy.run_path(str(ROOT / 'probe-api.py'))
    server = ['/src/build-sycl/bin/llama-server', '--verbosity', '4', '--model',
              '/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf', '--alias', 'nemotron-bolt',
              '-ngl', '999', '--ctx-size', '65536', '--parallel', '1', '--batch-size', '1024',
              '--ubatch-size', '1024', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
              '--flash-attn', 'on', '--jinja', '--host', '0.0.0.0', '--port', '18098',
              '--temp', '0', '--load-mode', 'none']
    shell = ('set -eo pipefail; source /opt/intel/oneapi/setvars.sh --force; '
             'export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}; '
             + CANDIDATES[source] + 'exec ' + shlex.join(server))
    create = ['docker', 'create', '--name', NAME, '--label', LABEL, '--ipc=host',
              '-p', '127.0.0.1:18098:18098', '--device', '/dev/dri',
              '--group-add', str(os.stat('/dev/dri/renderD128').st_gid),
              '-v', '/dev/dri:/dev/dri:ro', '-v', str(source) + ':/src:ro',
              '-v', str(MODEL) + ':/models:ro',
              '-v', '/home/mike/inference/src/oneDNN/install:/opt/onednn:ro',
              '-e', 'ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE', '-e', 'ZE_AFFINITY_MASK=0',
              '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:0', '-e', 'GGML_SYCL_ENABLE_ESIMD=1',
              '-e', 'GGML_SYCL_ENABLE_FUSION=1', '-e', 'GGML_SYCL_ENABLE_GRAPH=0']
    if args.debug:
        create += ['-e', 'GGML_SYCL_DEBUG=1']
    create += [IMAGE, 'bash', '-lc', shell]
    save(out / 'create-argv.json', create)
    summary = {'tier': 'development', 'source': str(source), 'measure': args.measure,
               'debug': args.debug, 'status': 'prepared'}
    created = False
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        replace_owned_container()
        summary['container_id'] = command(*create, timeout=60).strip()
        created = True
        command('docker', 'start', NAME, timeout=60)
        deadline = time.monotonic() + 600
        while True:
            assert command('docker', 'inspect', '--format', '{{.State.Running}}', NAME).strip() == 'true', 'Bolt/native gate stopped'
            try:
                summary['models'] = api['request']('http://127.0.0.1:18098', '/v1/models', timeout=3)
                break
            except (OSError, ValueError):
                assert time.monotonic() < deadline, 'Bolt readiness timeout'
                time.sleep(2)
        for mode in (['smoke', 'quality', 'bench', 'long'] if args.measure else ['smoke']):
            command('python3', '-B', str(ROOT / 'probe-api.py'), '--out', str(out / mode), '--mode', mode, timeout=2400)
            summary[mode] = json.loads((out / mode / 'summary.json').read_text())
        assert hardware() == before, 'Hardware settings changed'
        summary['status'] = 'passed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        if created:
            try:
                (out / 'server.log').write_text(command('docker', 'logs', NAME))
                summary['bolt_running'] = command('docker', 'inspect', '--format', '{{.State.Running}}', NAME).strip() == 'true'
            except Exception as exc:
                summary['log_error'] = repr(exc)
        save(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
