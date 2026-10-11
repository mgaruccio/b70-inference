#!/usr/bin/env python3
"""One bounded, host-specific supplemental GPU screen; not a serving benchmark."""
import json
from pathlib import Path
import runpy
import signal
import subprocess

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
CAMPAIGN = ROOT / '20261010-qwen38-shared-kv-prefill'
PRIOR = ROOT / '20261009-qwen38-community-protocol/power-230w-01'
IMAGE = 'vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL = '/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16'
NAME = 'qwen38-gemm-screen'


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def interrupt(signum, frame):
    raise RuntimeError(f'interrupted by signal {signum}')


def main():
    out = CAMPAIGN / 'gemm-screen-04'
    out.mkdir(exist_ok=False)
    serving = runpy.run_path(str(CAMPAIGN / 'run-prefill-gemm.py'))
    guard = runpy.run_path(str(PRIOR / 'guard-230w.py'))
    power = serving['POWER'].read_text().strip()
    assert power == '275000000'
    assert serving['sha'](serving['PRODUCTION']) == serving['PRODUCTION_SHA']
    assert serving['sha'](serving['LIBRARY']) == serving['LIBRARY_SHA']
    guard['preconditions'].__globals__['POWER_CAP_UW'] = power
    guard['preconditions'](out, NAME)
    before = guard['kernel'](out, 'pre')
    assert not any(v['error_lines'] for v in before['commands'].values())
    cidfile = out / 'container-id'
    render_gid = str(Path('/dev/dri/renderD128').stat().st_gid)
    command = [
        'docker', 'run', '--rm', '--name', NAME, '--cidfile', str(cidfile),
        '--network', 'none', '--device', '/dev/dri', '--group-add', render_gid,
        '--ipc=host', '--shm-size=16g',
        '-v', f'{MODEL}:/model:ro', '-v', f'{CAMPAIGN}:/experiment:ro',
        '-v', f'{out}:/results', '-e', 'VLLM_TARGET_DEVICE=xpu',
        '-e', 'ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE', '-e', 'ZE_AFFINITY_MASK=0',
        '-e', 'ONEDNN_VERBOSE=none', '--entrypoint', '/opt/venv/bin/python', IMAGE,
        '/experiment/bench-prefill-int4.py', '--model', '/model',
        '--out', '/results/bench.json', '--max-projections', '4',
        '--reps', '5', '--warmup', '2', '--reference-columns', '8',
        '--configs', 'tiled-fp16',
    ]
    put(out / 'command.json', command)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    for name in ('bench-prefill-int4.py', 'prefill_int4.py'):
        (out / name).write_bytes((CAMPAIGN / name).read_bytes())
    status = {'tier': 'development', 'kind': 'supplemental real-weight GEMM screen',
              'original_power_uw': power, 'experiment_power_uw': '230000000',
              'executed': False, 'restored': False}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    try:
        serving['set_power']('230000000')
        guard['preconditions'].__globals__['POWER_CAP_UW'] = '230000000'
        guard['preconditions'](out, NAME)
        with (out / 'bench.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        status['executed'] = True
        status['benchmark_returncode'] = result.returncode
        if (out / 'bench.json').is_file():
            report = json.loads((out / 'bench.json').read_text())
            status['benchmark_status'] = report['status']
            print(json.dumps({'status': report['status'], 'cases': len(report['cases']),
                              'errors': report['errors']}), flush=True)
        else:
            raise RuntimeError('No benchmark report; inspect bench.log')
    except BaseException as exc:
        status['error'] = repr(exc)
        raise
    finally:
        try:
            if cidfile.is_file():
                cid = cidfile.read_text().strip()
                # Stop only this run's container ID; --rm may already have removed it.
                inspected = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', cid],
                                           capture_output=True, text=True, timeout=15)
                if inspected.returncode == 0 and inspected.stdout.strip() == 'true':
                    stopped = subprocess.run(['docker', 'stop', '--time', '30', cid],
                                             capture_output=True, text=True, timeout=90)
                    put(out / 'cleanup.json', {'id': cid, 'returncode': stopped.returncode,
                                              'stdout': stopped.stdout, 'stderr': stopped.stderr})
        finally:
            serving['set_power'](power)
            assert serving['sha'](serving['PRODUCTION']) == serving['PRODUCTION_SHA']
            guard['preconditions'].__globals__['POWER_CAP_UW'] = power
            restored = out / 'restored'
            restored.mkdir()
            guard['preconditions'](restored, NAME)
            after = guard['kernel'](out, 'post')
            status['new_kernel_errors'] = sorted(
                {s for v in after['commands'].values() for s in v['error_lines']} -
                {s for v in before['commands'].values() for s in v['error_lines']})
            status['restored'] = True
            put(out / 'status.json', status)
            assert not status['new_kernel_errors']


if __name__ == '__main__':
    main()
