#!/usr/bin/env python3
"""One approved 32K INT4 shape/dispatch capture; always restore unchanged Qwen."""
import argparse
import json
from pathlib import Path
import re
import runpy
import signal
import subprocess
import time

ROOT = Path('/home/mike/b70-evals/20261013-qwen38-int4-dispatch')
HELPER = Path('/home/mike/b70-evals/20261012-qwen38-prefill-profiling/run-pilot.py')
NAME = 'qwen38-int4-shape-dispatch-20261013'
IMAGE = 'sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'


def interrupted(signum, frame):
    raise RuntimeError(f'Interrupted by signal {signum}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.parent == ROOT and re.fullmatch(r'[A-Za-z0-9_-]+', out.name)
    compatibility = json.loads((ROOT / 'compatibility/compatibility.json').read_text())
    assert compatibility['pass'] is True and compatibility['graphs_enabled'] is True
    h = runpy.run_path(str(HELPER))
    command, save, call = h['command'], h['save'], h['call']
    production = h['PRODUCTION']
    before = h['snapshot']()
    assert command('docker', 'ps', '--format', '{{.Names}}').splitlines() == ['qwen38']
    assert command('docker', 'inspect', '--format', '{{.Image}}', 'qwen38').strip() == IMAGE
    out.mkdir(exist_ok=False)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    (out / 'base-helper.py').write_bytes(HELPER.read_bytes())
    save(out / 'preflight.json', before)
    h['kernel'](out / 'kernel-pre.log')
    original = production.read_text()
    (out / 'production-launcher.sh').write_text(original)
    (out / 'production-image.txt').write_text(IMAGE + '\n')
    installed = """import importlib.metadata as m,pathlib,json
for name in ('vllm','torch','vllm-xpu-kernels'):
 d=m.distribution(name);print('DIST',name,d.version)
 if name=='vllm':
  for rel in ('config/profiler.py','profiler/wrapper.py','v1/worker/xpu_worker.py','model_executor/kernels/linear/mixed_precision/xpu.py'):
   p=pathlib.Path(d.locate_file('vllm'))/rel
   print('FILE',p);print(p.read_text() if p.exists() else 'ABSENT')
"""
    (out / 'installed-profiler-source.txt').write_text(command('docker', 'exec', 'qwen38', 'python3', '-c', installed))
    trace_dir = out / 'torch'
    trace_dir.mkdir()
    flags = ('--enable-logging-iteration-details --profiler-config.profiler=torch '
             '--profiler-config.torch_profiler_dir=/profile '
             '--profiler-config.torch_profiler_record_shapes=true '
             '--profiler-config.torch_profiler_with_stack=false '
             '--profiler-config.torch_profiler_with_memory=false '
             '--profiler-config.torch_profiler_with_flops=false '
             '--profiler-config.torch_profiler_use_gzip=true '
             '--profiler-config.ignore_frontend=true --profiler-config.max_iterations=16 ')
    replacements = {
        '--name qwen38 ': '--name ' + NAME + ' ',
        '0.0.0.0:${PORT:-8000}:8000': '127.0.0.1:8000:8000',
        '  --entrypoint bash': (f'  -v {trace_dir}:/profile \\\n'
                              '  -e ONEDNN_VERBOSE=profile_create,dispatch -e ONEDNN_VERBOSE_TIMESTAMP=1 \\\n'
                              '  --entrypoint bash'),
        '--served-model-name qwen38': flags + '--served-model-name qwen38',
    }
    candidate = original
    for old, new in replacements.items():
        assert candidate.count(old) == 1, old
        candidate = candidate.replace(old, new)
    launcher = out / 'diagnostic-launcher.sh'
    launcher.write_text(candidate)
    command('bash', '-n', str(launcher))
    summary = {'tier': 'development diagnostic, not a performance comparison',
               'length': 32768, 'budget': 8192, 'prefix_cache': 'on / Mamba align',
               'changes': ['record_shapes', 'create/dispatch verbosity', 'iteration logging'],
               'status': 'prepared', 'compatibility': compatibility}
    save(out / 'summary.json', summary)
    process = None
    log = None
    disrupted = False
    error = None
    profile_started = False

    def start(path, logpath):
        nonlocal process, log
        log = logpath.open('w')
        process = subprocess.Popen(['bash', str(path)], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Launcher exited {process.returncode}; see {logpath}')
            try:
                models = call('/v1/models', timeout=3)
                assert any(m['id'] == 'qwen38' and m['max_model_len'] == 212992 for m in models['body']['data'])
                save(logpath.with_suffix('.models.json'), models)
                return
            except (OSError, ValueError):
                time.sleep(2)
        raise TimeoutError(f'Startup deadline: {logpath}')

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        disrupted = True
        summary['status'] = 'starting-diagnostic'
        save(out / 'summary.json', summary)
        command('docker', 'stop', '--time', '30', 'qwen38')
        start(launcher, out / 'server.log')
        assert 'B70_FIX_UNIFORM_DECODE_PREFILL:' in (out / 'server.log').read_text()
        paths = call('/openapi.json', timeout=10)['body']['paths']
        assert '/start_profile' in paths and '/stop_profile' in paths
        summary['warmup'] = h['complete'](32768, 13100, out / 'warmup-response.json')
        (out / 'metrics-before.txt').write_text(call('/metrics', timeout=10)['raw_text'])
        offset = (out / 'server.log').stat().st_size
        summary['start_profile'] = call('/start_profile', {}, timeout=120)
        profile_started = True
        try:
            save(out / 'profiled-request.json', {'model': 'qwen38', 'prompt': [13101] + [15] * 32767,
                 'temperature': 0, 'seed': 42, 'max_tokens': 1, 'ignore_eos': True, 'logprobs': 1})
            summary['profiled'] = h['complete'](32768, 13101, out / 'profiled-response.json')
        finally:
            summary['stop_profile'] = call('/stop_profile', {}, timeout=180)
            profile_started = False
            save(out / 'summary.json', summary)
        (out / 'metrics-after.txt').write_text(call('/metrics', timeout=10)['raw_text'])
        deadline = time.monotonic() + 5
        while True:
            with (out / 'server.log').open('rb') as stream:
                stream.seek(offset)
                window = stream.read().decode('utf-8', 'replace')
            rows = h['parse_iterations'](window)
            if sum(r['context_tokens'] for r in rows) >= 32768 or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        (out / 'profile-window.log').write_text(window)
        save(out / 'iterations.json', rows)
        assert sum(r['context_tokens'] for r in rows) == 32768, 'Incomplete scheduled-token ledger'
        summary['device_analysis'] = h['analyze_trace'](trace_dir)
        assert summary['device_analysis']['device_event_count'] > 0
        summary['status'] = 'captured-needs-shape-analysis'
    except BaseException as exc:
        error = exc
        summary.update(status='failed', error=repr(exc))
    finally:
        cleanup = {}
        cleanup_errors = []
        if profile_started:
            cleanup['profile_flush_not_confirmed'] = True
        # Cleanup failures must not skip a fresh container-state check/restoration.
        try:
            if NAME in command('docker', 'ps', '--format', '{{.Names}}').splitlines():
                command('docker', 'stop', '--time', '30', NAME)
        except BaseException as exc:
            cleanup_errors.append(repr(exc))
        if process is not None:
            try:
                process.wait(timeout=90)
            except BaseException as exc:
                cleanup_errors.append(repr(exc))
                # Prevent a not-yet-visible diagnostic launch appearing after restore.
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except BaseException as exc:
                    cleanup_errors.append(repr(exc))
                    try:
                        process.kill()
                        process.wait(timeout=10)
                    except BaseException as exc:
                        cleanup_errors.append(repr(exc))
        if log is not None:
            try:
                log.close()
            except BaseException as exc:
                cleanup_errors.append(repr(exc))
        try:
            if disrupted:
                # Unknown state or a surviving diagnostic blocks a second GPU owner.
                names = command('docker', 'ps', '--format', '{{.Names}}').splitlines()
                if not names:
                    start(production, out / 'production-restored.log')
                    cleanup['restoration'] = 'original-restarted'
                else:
                    assert names == ['qwen38'], 'Unexpected/surviving container blocks restoration'
                    cleanup['restoration'] = 'original-retained'
                cleanup['probe'] = h['complete'](5, 13102, out / 'restored-response.json')
                cleanup['production_running'] = True
                assert h['snapshot']() == before, 'Host invariants changed'
                h['kernel'](out / 'kernel-post.log')
                process = None  # Original detached production deliberately stays running.
                if log is not None:
                    log.close()
                log = None
            cleanup['invariants_pass'] = True
        except BaseException as exc:
            cleanup['error'] = repr(exc)
            error = error or exc
        if cleanup_errors:
            cleanup['warnings'] = cleanup_errors
        summary['cleanup'] = cleanup
        save(out / 'summary.json', summary)
        print(json.dumps({'status': summary['status'], 'cleanup': cleanup, 'output': str(out)}), flush=True)
    if error:
        raise error


if __name__ == '__main__':
    main()
