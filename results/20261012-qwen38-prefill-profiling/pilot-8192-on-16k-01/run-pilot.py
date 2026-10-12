#!/usr/bin/env python3
"""One guarded, native-Torch prefill diagnostic cell; always restore production."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

PRODUCTION = Path('/home/mike/inference/launchers/start-qwen38.sh')
LIBRARY = PRODUCTION.parent / 'libattn_kernels_xe_2-head256-k64.so'
GUARD = Path('/home/mike/inference/src/intel-arc-pro-b70-inference-cookbook/patches/patch_uniform_decode_prefill.py')
EXPECTED = {
    PRODUCTION: '5701b40e355e99b65c776dd6c9e499980ccc3be758f17288c30f840f86e99a1d',
    LIBRARY: 'f5195ace0074610d266e38322060dc4b71bf39c30dd4b7ab984fa830fa05b019',
    GUARD: 'd26eb73c90a61c9f4f0056c7009cbe8c5a758d22a4db7c3ab8bf3ec28d73e66c',
}
NAME = 'qwen38-native-prefill-profile'
BASE = 'http://127.0.0.1:8000'
FAULTS = 'Hardware Error|Machine Check|GPU HANG|wedged|Xid|data fabric|soft lockup|blocked for more|GT.*reset|GuC.*(failed|error)'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def command(*argv, timeout=90):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def snapshot():
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in EXPECTED}
    assert all(hashes[str(p)] == value for p, value in EXPECTED.items()), hashes
    policies = list(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_max_freq'))
    caps = {str(p): p.read_text().strip() for p in policies}
    boost = Path('/sys/devices/system/cpu/cpufreq/boost').read_text().strip()
    assert boost == '0' and len(caps) == 16 and set(caps.values()) == {'3801000'}, (boost, caps)
    power = {str(p): p.read_text().strip() for p in Path('/sys/class/drm').glob('card*/device/hwmon/hwmon*/power1_cap')}
    assert power and set(power.values()) == {'275000000'}, power
    return {'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'boost': boost, 'caps': caps, 'power': power, 'hashes': hashes}


def kernel(path):
    result = subprocess.run(['sudo', '-n', 'journalctl', '-k', '-b', '--no-pager', '--grep=' + FAULTS],
                            capture_output=True, text=True, timeout=30)
    path.write_text(result.stdout + result.stderr)
    assert result.returncode == 1 and not result.stderr.strip(), 'Kernel fault or failed health check: ' + str(path)


def call(path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode('utf-8', 'replace')
        raise RuntimeError(f'{path}: HTTP {exc.code}: {raw}') from exc
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = {'text': raw.decode('utf-8', 'replace')}
    return {'http_status': status, 'elapsed_s': time.monotonic() - started, 'body': body, 'raw_text': raw.decode('utf-8', 'replace')}


def complete(length, first_token, record_path=None):
    payload = {'model': 'qwen38', 'prompt': [first_token] + [15] * (length - 1),
               'temperature': 0, 'seed': 42, 'max_tokens': 1, 'ignore_eos': True, 'logprobs': 1}
    result = call('/v1/completions', payload)
    result['length'] = length
    result['first_token'] = first_token
    if record_path is not None:
        record_path.write_text(result['raw_text'])
    body = result['body']
    usage = body['usage']
    values = body['choices'][0]['logprobs']['token_logprobs']
    assert usage['prompt_tokens'] == length and usage['completion_tokens'] == 1, usage
    assert values and all(math.isfinite(v) for v in values), values
    assert (usage.get('prompt_tokens_details') or {}).get('cached_tokens') in (None, 0), usage
    return result


def parse_iterations(text):
    rows = []
    for line in text.splitlines():
        if 'Iteration(' not in line:
            continue
        context = re.search(r'([\d,]+) context tokens', line)
        generation = re.search(r'([\d,]+) generation tokens', line)
        iteration = re.search(r'Iteration\((\d+)\)', line)
        if context and generation and iteration:
            rows.append({'iteration': int(iteration[1]),
                         'context_tokens': int(context[1].replace(',', '')),
                         'generation_tokens': int(generation[1].replace(',', '')),
                         'raw': line})
    return rows


def analyze_trace(directory):
    files = sorted(directory.glob('*.pt.trace.json*'))
    assert files, 'No Torch trace exported'
    grouped = collections.defaultdict(lambda: {'calls': 0, 'device_ms': 0.0})
    categories = collections.Counter()
    count = 0
    begin, end = math.inf, -math.inf
    for path in files:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            trace = json.load(stream)
        for event in trace.get('traceEvents', []):
            category = str(event.get('cat', '')).lower()
            categories[category] += 1
            # CPU scopes and runtime launches are explicitly excluded.
            if event.get('ph') != 'X' or category not in {'kernel', 'gpu_op', 'xpu_kernel'}:
                continue
            duration = float(event.get('dur', 0))
            if duration <= 0:
                continue
            count += 1
            name = event['name']
            grouped[name]['calls'] += 1
            grouped[name]['device_ms'] += duration / 1000
            begin = min(begin, float(event['ts']))
            end = max(end, float(event['ts']) + duration)
    result = {'trace_files': [str(p) for p in files], 'categories': dict(categories),
              'device_event_count': count,
              'sum_device_activity_ms': sum(v['device_ms'] for v in grouped.values()),
              'device_first_to_last_span_ms': (end - begin) / 1000 if count else None,
              'kernels': sorted([{'name': n, **v} for n, v in grouped.items()],
                                key=lambda v: v['device_ms'], reverse=True),
              'caveat': 'Device activity sum is not elapsed time; overlap and graph opacity require review. CPU scopes are not GPU durations.'}
    return result


def interrupted(signum, frame):
    raise InterruptedError('signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, choices=(8192, 8320, 6656), default=8192)
    parser.add_argument('--prefix-cache', choices=('on', 'off'), default='on')
    parser.add_argument('--length', type=int, choices=(16384, 32768), default=16384)
    args = parser.parse_args()
    out = args.output.resolve()
    assert re.fullmatch(r'/[A-Za-z0-9_./-]+', str(out)), 'Unsafe mount path'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'run-pilot.py').write_bytes(Path(__file__).read_bytes())
    original = PRODUCTION.read_text()
    (out / 'production-launcher.sh').write_text(original)
    before = snapshot()
    save(out / 'preflight.json', before)
    kernel(out / 'kernel-pre.log')
    assert command('docker', 'ps', '--format', '{{.Names}}').splitlines() == ['qwen38']
    summary = {'tier': 'development', 'budget': args.budget, 'prefix_cache': args.prefix_cache,
               'length': args.length, 'status': 'starting', 'unprofiled': []}
    save(out / 'summary.json', summary)
    trace_dir = out / 'torch'
    trace_dir.mkdir()
    flags = ('--enable-logging-iteration-details --profiler-config.profiler=torch '
             '--profiler-config.torch_profiler_dir=/profile '
             '--profiler-config.torch_profiler_with_stack=false '
             '--profiler-config.torch_profiler_record_shapes=false '
             '--profiler-config.torch_profiler_with_memory=false '
             '--profiler-config.ignore_frontend=true --profiler-config.max_iterations=16 ')
    text = original
    replacements = {
        '--name qwen38 ': '--name ' + NAME + ' ',
        '0.0.0.0:${PORT:-8000}:8000': '127.0.0.1:8000:8000',
        '--max-num-batched-tokens 8192': '--max-num-batched-tokens ' + str(args.budget),
        '  --entrypoint bash': f'  -v {trace_dir}:/profile \\\n  --entrypoint bash',
        '--served-model-name qwen38': flags + '--served-model-name qwen38',
    }
    if args.prefix_cache == 'off':
        replacements['--enable-prefix-caching'] = '--no-enable-prefix-caching'
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    candidate = out / 'diagnostic-launcher.sh'
    candidate.write_text(text)
    command('bash', '-n', str(candidate))
    process = None
    log = None
    disrupted = False
    error = None

    def start(launcher, logpath):
        nonlocal process, log
        log = logpath.open('w')
        process = subprocess.Popen(['bash', str(launcher)], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Launcher exited: ' + str(process.returncode))
            try:
                models = call('/v1/models', timeout=3)['body']
                assert any(m['id'] == 'qwen38' and m['max_model_len'] == 212992 for m in models['data'])
                save(logpath.with_suffix('.models.json'), models)
                return
            except (OSError, ValueError):
                time.sleep(2)
        raise TimeoutError('startup')

    try:
        disrupted = True
        command('docker', 'stop', '--time', '30', 'qwen38')
        start(candidate, out / 'server.log')
        assert 'B70_FIX_UNIFORM_DECODE_PREFILL:' in (out / 'server.log').read_text()
        paths = call('/openapi.json', timeout=10)['body']['paths']
        assert '/start_profile' in paths and '/stop_profile' in paths, 'Native profiling APIs absent'
        summary['warmup'] = complete(args.length, 1100, out / 'warmup.json')
        save(out / 'summary.json', summary)
        for repetition in range(3):
            summary['unprofiled'].append(complete(args.length, 1101 + repetition, out / f'unprofiled-{repetition}.json'))
            save(out / 'summary.json', summary)
        (out / 'metrics-before.txt').write_text(call('/metrics', timeout=10)['body']['text'])
        offset = (out / 'server.log').stat().st_size
        summary['start_profile'] = call('/start_profile', {}, timeout=120)
        try:
            summary['profiled'] = complete(args.length, 1109, out / 'profiled.json')
        finally:
            summary['stop_profile'] = call('/stop_profile', {}, timeout=180)
            save(out / 'summary.json', summary)
        (out / 'metrics-after.txt').write_text(call('/metrics', timeout=10)['body']['text'])
        # Application log delivery, not background-task completion polling.
        deadline = time.monotonic() + 5
        while True:
            with (out / 'server.log').open('rb') as stream:
                stream.seek(offset)
                window = stream.read().decode('utf-8', 'replace')
            rows = parse_iterations(window)
            if sum(r['context_tokens'] for r in rows) >= args.length or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        (out / 'profile-window.log').write_text(window)
        save(out / 'iterations.json', rows)
        assert sum(r['context_tokens'] for r in rows) == args.length, 'Scheduled-token ledger incomplete'
        analysis = analyze_trace(trace_dir)
        save(out / 'device-analysis.json', analysis)
        assert analysis['device_event_count'] > 0, 'No real device activity; CPU timings cannot substitute'
        summary['status'] = 'captured-needs-review'
    except BaseException as exc:
        summary.update(status='failed', error=str(exc))
        error = exc
    finally:
        cleanup = {}
        try:
            if NAME in command('docker', 'ps', '--format', '{{.Names}}').splitlines():
                command('docker', 'stop', '--time', '30', NAME)
            if process is not None:
                process.wait(timeout=90)
            if log is not None:
                log.close()
            assert snapshot() == before, 'Host invariants changed'
            kernel(out / 'kernel-post.log')
            if disrupted:
                names = command('docker', 'ps', '--format', '{{.Names}}').splitlines()
                if not names:
                    start(PRODUCTION, out / 'production-restored.log')
                    cleanup['restoration'] = 'restarted_original'
                else:
                    assert names == ['qwen38'], 'Unexpected container; restoration blocked'
                    cleanup['restoration'] = 'original_retained'
                cleanup['probe'] = complete(5, 1234, out / 'restored-probe.json')
                assert snapshot() == before
                kernel(out / 'kernel-restored.log')
                cleanup['production_running'] = True
                # Deliberately retain the restored detached production launcher.
                process = None
                if log is not None:
                    log.close()
                log = None
        except BaseException as exc:
            cleanup.update(production_running=False, error=str(exc))
            error = error or exc
        summary['cleanup'] = cleanup
        save(out / 'summary.json', summary)
        print(json.dumps({'status': summary['status'], 'cleanup': cleanup,
                          'directory': str(out)}), flush=True)
    if error:
        raise error


if __name__ == '__main__':
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    main()
