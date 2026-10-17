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
MOE_POST = BASE.with_name('llama.cpp-bolt-b6334e37-nvfp4-moe-post-r1')
ROUTER = BASE.with_name('llama.cpp-bolt-b6334e37-nvfp4-router-r1')
PREFILL_SSM = BASE.with_name('llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
PREFILL_NVFP4 = BASE.with_name('llama.cpp-bolt-b6334e37-nvfp4-prefill-pack-r2')


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


def native_post_gate(legacy=True):
    test = '/src/build-sycl/bin/test-backend-ops test -b SYCL0'
    commands = native_legacy_commands() if legacy else []
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


def native_router_gate():
    test = '/src/build-sycl/bin/test-backend-ops test -b SYCL0'
    commands = []
    for fusion, expected in ((0, 0), (1, 67)):
        log = f'/tmp/bolt-router-fusion{fusion}.log'
        commands += [f'echo BOLT_ROUTER_NATIVE_FUSION{fusion}',
                     f"GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_FUSION={fusion} {test} -o '^BOLT_ROUTER$' -j 1 2>&1 | tee {log}",
                     f"grep -Fx '  82/82 tests passed' {log}",
                     f"if grep -Ei 'not supported|skipping large tensors' {log}; then exit 1; fi"]
        for marker in ('BOLT_ROUTER floats: mismatches=0 ', 'BOLT_ROUTER indices: mismatches=0 '):
            commands += [f"count=$(grep -Fc '{marker}' {log} || true)", 'test "$count" -eq 82']
        commands += [f"count=$(grep -Fc 'BOLT_ROUTER allocator: logits=probs exact (2 graphs)' {log} || true)",
                     'test "$count" -eq 5',
                     f"count=$(grep -Fc 'BOLT_ROUTER consumer bytes: mismatches=0 ' {log} || true)",
                     'test "$count" -eq 2',
                     f"count=$(grep -Fc 'call ggml_sycl_op_router_fused:' {log} || true)",
                     f'test "$count" -eq {expected}']
    return '; '.join(commands + ['echo BOLT_ROUTER_NUMERICAL_PASSED']) + '; '


def native_prefill_gate():
    commands = []
    for fusion, expected in [(0, 0), (1, 23)]:
        log = f'/tmp/ssm-prefill-fusion{fusion}.log'
        commands += [f'echo BOLT_SSM_PREFILL_NATIVE_FUSION{fusion}',
                     f'GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_VMM=1 GGML_SYCL_ENABLE_FUSION={fusion} '
                     "/src/build-sycl/bin/test-backend-ops test -b SYCL0 -o '^BOLT_SSM_PREFILL$' -j 1 "
                     f'2>&1 | tee {log}',
                     f"grep -Fx '  34/34 tests passed' {log}",
                     f"if grep -Ei 'not supported|skipping large tensors|BOLT_SSM_PREFILL_MISMATCH' {log}; then exit 1; fi"]
        for backend in ['under_test', 'CPU']:
            commands += [f"count=$(grep -Fc '[BOLT_SSM_PREFILL] backend={backend} ' {log} || true)",
                         'test "$count" -eq 34']
        commands += [f"count=$(grep -Fc 'y_mismatches=0 state_mismatches=0 canary_mismatches=0 ' {log} || true)",
                     'test "$count" -eq 68',
                     f"count=$(grep -Fc 'call ggml_sycl_ssm_scan_prefill:' {log} || true)",
                     f'test "$count" -eq {expected}']
    log = '/tmp/ssm-prefill-vmm-off.log'
    commands += ['echo BOLT_SSM_PREFILL_NATIVE_VMM0',
                 'GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_VMM=0 GGML_SYCL_ENABLE_FUSION=1 '
                 "/src/build-sycl/bin/test-backend-ops test -b SYCL0 -o '^BOLT_SSM_PREFILL$' -j 1 "
                 "-p 'n_seq_tokens=2,n_head=128,n_group=8,n_seqs=1,K=1,' "
                 f'2>&1 | tee {log}',
                 f"grep -Fx '  1/1 tests passed' {log}",
                 f"if grep -Ei 'not supported|skipping large tensors|BOLT_SSM_PREFILL_MISMATCH|call ggml_sycl_ssm_scan_prefill:' {log}; then exit 1; fi",
                 f"count=$(grep -Fc 'y_mismatches=0 state_mismatches=0 canary_mismatches=0 ' {log} || true)",
                 'test "$count" -eq 2']
    return '; '.join(commands + ['echo BOLT_SSM_PREFILL_NUMERICAL_PASSED']) + '; '


def native_nvfp4_gate():
    log = '/tmp/nvfp4-prefill-r2-exactness.log'
    commands = [
        f'GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-nvfp4-sycl-r2 2>&1 | tee {log}',
        f"grep -Fx 'NVFP4 native exactness PASS' {log}",
        f"if grep -Ei 'ineligible|failure|not supported|skipping' {log}; then exit 1; fi",
    ]
    for marker, expected in [
        ('path=packed reordered=0 dst=f32', 10),
        ('path=packed reordered=1 dst=f32', 10),
        ('path=original reordered=0 dst=f32', 13),
        ('path=original reordered=1 dst=f32', 14),
        ('path=original reordered=0 dst=f16', 1),
    ]:
        commands += [f"count=$(grep -Fc '[SYCL][NVFP4] {marker} ' {log} || true)",
                     f'test "$count" -eq {expected}']
    for layout in ('original', 'reordered'):
        commands += [f"count=$(grep -Ec '^NVFP4 case layout={layout} blocks=.* exact=1$' {log} || true)",
                     'test "$count" -eq 11']
    return '; '.join(commands + ['echo BOLT_NVFP4_PREFILL_NUMERICAL_PASSED']) + '; '

# Registration requires a fixed native gate; no arbitrary source/command injection.
CANDIDATES = {BASE: '', SSM: native_ssm_gate(), MOE: native_moe_gate(),
              COMBINED: native_ssm_gate() + native_moe_gate(legacy=False), POST: native_post_gate(),
              MOE_POST: native_moe_gate() + native_post_gate(legacy=False),
              ROUTER: native_moe_gate() + native_router_gate(),
              PREFILL_SSM: native_moe_gate() + native_router_gate() + native_prefill_gate(),
              PREFILL_NVFP4: native_moe_gate() + native_router_gate() + native_prefill_gate() + native_nvfp4_gate()}


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
    if source == MOE_POST:
        assert (source / 'build-sycl/bin/test-backend-ops').is_file()
        assert 'MOE_POST_RUNTIME_PRECHECK_PASSED' in (ROOT / 'moe-post-r1-runtime-precheck.log').read_text()
    if source == ROUTER:
        assert (source / 'build-sycl/bin/test-backend-ops').is_file()
        assert 'BOLT_ROUTER_RUNTIME_PRECHECK_PASSED' in (ROOT / 'router-r1-runtime-precheck.log').read_text()
    if source == PREFILL_NVFP4:
        assert (source / 'build-sycl/bin/test-nvfp4-sycl-r2').is_file()
        assert 'BOLT_NVFP4_PREFILL_RUNTIME_PRECHECK_PASSED' in (ROOT / 'nvfp4-prefill-r2-runtime-precheck.log').read_text()
        if args.measure:
            debug = json.loads((ROOT / 'nvfp4-prefill-r2-dispatch/summary.json').read_text())
            assert debug['status'] == 'passed' and debug['debug'] and debug['source'] == str(source)
    if source == PREFILL_SSM:
        assert (source / 'build-sycl/bin/test-backend-ops').is_file()
        assert 'BOLT_SSM_PREFILL_RUNTIME_PRECHECK_PASSED' in (ROOT / 'ssm-prefill-r1-runtime-precheck.log').read_text()
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
        if args.debug and source in (PREFILL_SSM, PREFILL_NVFP4):
            # Real prefill activation only: one output, never used as a timing result.
            payload = json.loads((ROOT / 'prefill-router-r1-baseline/long/measured-00.request.json').read_text())
            activation = api['chat']('http://127.0.0.1:18098', payload['messages'][0]['content'],
                                     out / 'prefill-activation', count=1, stream=True, ignore_eos=True)
            assert activation['usage']['prompt_tokens'] == 8270
            assert activation['usage']['completion_tokens'] == 1
            assert activation['timings']['cache_n'] == 0
            summary['prefill_activation'] = activation
            marker = 'BOLT_NVFP4_PREFILL_NUMERICAL_PASSED' if source == PREFILL_NVFP4 else 'BOLT_SSM_PREFILL_NUMERICAL_PASSED'
            model_log = command('docker', 'logs', NAME).split(marker, 1)[1]
            assert 'call ggml_sycl_ssm_scan_prefill:' in model_log, 'Staged scan never activated on the model'
            if source == PREFILL_NVFP4:
                packed = [line for line in model_log.splitlines() if '[SYCL][NVFP4] path=packed ' in line]
                reordered = [line for line in packed if 'reordered=1 dst=f32 ' in line]
                assert len(reordered) >= 720, 'Packed reordered conversion did not activate on warmed long prefill'
                summary['packed_model_calls'] = len(packed)
                summary['packed_reordered_model_calls'] = len(reordered)
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
