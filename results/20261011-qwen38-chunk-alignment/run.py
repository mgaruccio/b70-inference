#!/usr/bin/env python3
"""Host-specific development screen; guarded K64, no production budget promotion."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request

PRODUCTION = Path('/home/mike/inference/launchers/start-qwen38.sh')
LIBRARY = Path('/home/mike/inference/launchers/libattn_kernels_xe_2-head256-k64.so')
LIBRARY_SHA = 'f5195ace0074610d266e38322060dc4b71bf39c30dd4b7ab984fa830fa05b019'
PATCH = Path('/home/mike/inference/src/intel-arc-pro-b70-inference-cookbook/patches/patch_uniform_decode_prefill.py')
PATCH_SHA = 'd26eb73c90a61c9f4f0056c7009cbe8c5a758d22a4db7c3ab8bf3ec28d73e66c'
BASE = 'http://127.0.0.1:8000'
NAME = 'qwen38-chunk-screen'
PRIMARY = (16384, 32768)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=60)


def power():
    return {str(p): p.read_text().strip() for p in Path('/sys/class/drm').glob('card*/device/hwmon/hwmon*/power1_cap')}


def post(path, payload, timeout=600):
    request = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')[:4000]
        raise RuntimeError(f'{path}: HTTP {exc.code}: {detail}') from exc
    return body, time.perf_counter() - started


def ids(length, repetition):
    # Same exact prompts in every arm; prefix caching is off for timed cells.
    return [1000 + repetition] + [15] * (length - 1)


class Campaign:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        (root / 'run.py').write_bytes(Path(__file__).read_bytes())
        self.original = PRODUCTION.read_text()
        self.original_sha = hashlib.sha256(PRODUCTION.read_bytes()).hexdigest()
        self.initial_power = power()
        self.process = None
        self.log = None
        self.summary = {'tier': 'development', 'production_sha': self.original_sha,
                        'library_sha': hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
                        'power': self.initial_power, 'cells': [], 'status': 'starting'}
        self.checkpoint()
        assert self.summary['library_sha'] == LIBRARY_SHA
        self.summary['patch_sha'] = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        assert self.summary['patch_sha'] == PATCH_SHA
        assert self.initial_power and set(self.initial_power.values()) == {'275000000'}
        assert not command('docker', 'ps', '--format', '{{.Names}}').strip(), 'Other containers are running'
        assert 'uniform_decode_prefill' in self.original, 'Maintained guard not installed'
        assert '--max-num-batched-tokens 8192' in self.original
        assert '--max-model-len 212992' in self.original
        (root / 'production-launcher.sh').write_text(self.original)
        (root / 'environment.txt').write_text(command('uname', '-a') + command('lscpu'))

    def checkpoint(self):
        write_json(self.root / 'summary.json', self.summary)

    def start(self, label, budget, cached=False, production=False):
        assert power() == self.initial_power, 'Power changed'
        assert hashlib.sha256(PRODUCTION.read_bytes()).hexdigest() == self.original_sha, 'Launcher changed'
        assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == PATCH_SHA, 'Guard changed'
        text = self.original
        if not production:
            assert text.count('--name qwen38 ') == 1
            text = text.replace('--name qwen38 ', '--name ' + NAME + ' ')
            assert text.count('0.0.0.0:${PORT:-8000}:8000') == 1
            text = text.replace('0.0.0.0:${PORT:-8000}:8000', '127.0.0.1:8000:8000')
            assert text.count('--max-num-batched-tokens 8192') == 1
            text = text.replace('--max-num-batched-tokens 8192', '--max-num-batched-tokens ' + str(budget))
            if not cached:
                assert text.count('--enable-prefix-caching') == 1
                text = text.replace('--enable-prefix-caching', '--no-enable-prefix-caching')
        directory = self.root / label
        directory.mkdir()
        launcher = directory / 'launcher.sh'
        launcher.write_text(text)
        command('bash', '-n', str(launcher))
        self.log = (directory / 'server.log').open('w')
        self.process = subprocess.Popen(['bash', str(launcher)], stdin=subprocess.DEVNULL,
                                        stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(label + ' launcher exited: ' + str(self.process.returncode))
            try:
                with urllib.request.urlopen(BASE + '/v1/models', timeout=5) as response:
                    models = json.load(response)
                assert any(m['id'] == 'qwen38' and m['max_model_len'] == 212992 for m in models['data'])
                write_json(directory / 'models.json', models)
                break
            except (OSError, ValueError):
                time.sleep(3)
        else:
            raise TimeoutError(label + ' startup')
        assert 'B70_FIX_UNIFORM_DECODE_PREFILL:' in (directory / 'server.log').read_text(), 'Guard did not report applying/replaying'
        print('READY', label, flush=True)
        cell = {'label': label, 'budget': budget, 'prefix_cache': cached, 'rows': [], 'status': 'running'}
        cell['effective_mamba_cache_mode'] = 'align' if cached else 'none'
        self.summary['cells'].append(cell)
        self.checkpoint()
        return directory, cell

    def stop(self):
        names = command('docker', 'ps', '--format', '{{.Names}}').splitlines()
        if NAME in names:
            command('docker', 'stop', '--time', '30', NAME)
        if self.process is not None:
            self.process.wait(timeout=60)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        assert power() == self.initial_power, 'Power changed'

    def raw(self, cell, length, repetition, kind='ttft'):
        payload = {'model': 'qwen38', 'prompt': ids(length, repetition), 'temperature': 0,
                   'max_tokens': 1, 'ignore_eos': True, 'logprobs': 1}
        body, elapsed = post('/v1/completions', payload, timeout=900)
        usage = body.get('usage') or {}
        choices = body.get('choices') or []
        logprobs = ((choices[0].get('logprobs') or {}).get('token_logprobs') or []) if choices else []
        finite = bool(logprobs) and all(isinstance(v, (float, int)) and math.isfinite(v) for v in logprobs)
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        row = {'kind': kind, 'length': length, 'rep': repetition, 'ttft_s': elapsed,
               'finite': finite, 'response': body, 'cached_tokens': cached}
        cell['rows'].append(row)
        self.checkpoint()
        assert finite, 'Nonfinite or missing logprobs at ' + str(length)
        assert usage.get('prompt_tokens') == length and usage.get('completion_tokens') == 1, 'Wrong token counts'
        if not cell['prefix_cache'] or cell.get('require_cold'):
            assert cached in (None, 0), 'Unexpected prefix hit'
        print(cell['label'], kind, length, repetition, round(elapsed, 4), flush=True)
        return elapsed

    def canaries(self, cell):
        prompts = [('arithmetic', 'What is 17+25? Reply with digits only.'),
                   ('json', 'Return exactly this JSON object, with no other text: {"ok":true}'),
                   ('code', 'Return only Python code, no Markdown, defining add(a,b) to return a+b.')]
        for label, prompt in prompts:
            body, elapsed = post('/v1/chat/completions', {
                'model': 'qwen38', 'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0, 'seed': 42, 'max_tokens': 256,
                'chat_template_kwargs': {'enable_thinking': False}})
            row = {'kind': 'canary', 'label': label, 'passed': False, 'elapsed_s': elapsed, 'response': body}
            cell['rows'].append(row)
            self.checkpoint()
            text = body['choices'][0]['message'].get('content') or ''
            if label == 'arithmetic':
                passed = text.strip() == '42'
            elif label == 'json':
                passed = json.loads(text) == {'ok': True}
            else:
                tree = ast.parse(text)
                functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'add']
                passed = len(functions) == 1 and [a.arg for a in functions[0].args.args] == ['a', 'b']
                returns = [n.value for n in ast.walk(functions[0]) if isinstance(n, ast.Return)] if passed else []
                passed = passed and len(returns) == 1 and isinstance(returns[0], ast.BinOp) and isinstance(returns[0].op, ast.Add) and isinstance(returns[0].left, ast.Name) and returns[0].left.id == 'a' and isinstance(returns[0].right, ast.Name) and returns[0].right.id == 'b'
            row['passed'] = passed
            self.checkpoint()
            assert passed, label + ' canary failed'

    def screen(self, label, budget):
        directory, cell = self.start(label, budget)
        for n in (5, 69, 133):
            self.raw(cell, n, 900, 'boundary')
        self.canaries(cell)
        for n in PRIMARY:
            self.raw(cell, n, 800, 'warmup')
        for repetition in range(5):
            for n in PRIMARY:
                self.raw(cell, n, repetition)
        cell['median_ttft_s'] = {str(n): statistics.median(r['ttft_s'] for r in cell['rows'] if r['kind'] == 'ttft' and r['length'] == n) for n in PRIMARY}
        cell['status'] = 'completed'
        self.checkpoint()
        self.stop()
        return cell

    def decode(self, directory, cell, repetition):
        payload = {'model': 'qwen38', 'prompt': ids(128, 42), 'temperature': 0, 'max_tokens': 128,
                   'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True}}
        request = urllib.request.Request(BASE + '/v1/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        first = None
        usage = None
        done = False
        with (directory / ('decode-' + str(repetition) + '.jsonl')).open('w') as log, urllib.request.urlopen(request, timeout=180) as response:
            for raw in response:
                now = time.perf_counter()
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                log.write(json.dumps({'elapsed_s': now - start, 'data': data}) + '\n')
                if data == '[DONE]':
                    done = True
                    break
                obj = json.loads(data)
                assert not obj.get('error'), str(obj.get('error'))
                if obj.get('usage'):
                    usage = obj['usage']
                if first is None and any(c.get('text') for c in obj.get('choices', [])):
                    first = now
        end = time.perf_counter()
        assert done and first is not None and usage and usage['completion_tokens'] == 128
        row = {'kind': 'decode128', 'rep': repetition, 'usage': usage,
               'ttft_s': first - start, 'decode_tok_s': 127 / (end - first), 'elapsed_s': end - start}
        cell['rows'].append(row)
        self.checkpoint()

    def run(self):
        error = None
        try:
            _, guard = self.start('guard-prefix-on', 8192, cached=True)
            guard['require_cold'] = True
            for n in list(range(1, 129)) + [133, 197, 261, 49925]:
                self.raw(guard, n, 700 + n % 4000, 'boundary')
            self.canaries(guard)
            guard['status'] = 'passed'
            self.checkpoint()
            self.stop()
            baseline = self.screen('baseline-before', 8192)
            candidates = [self.screen('candidate-' + str(b), b) for b in (8320, 6656)]
            possible = [c for c in candidates if all(c['median_ttft_s'][str(n)] < baseline['median_ttft_s'][str(n)] for n in PRIMARY)]
            if possible:
                after = self.screen('baseline-after', 8192)
                qualified = [c for c in possible if all(c['median_ttft_s'][str(n)] < after['median_ttft_s'][str(n)] for n in PRIMARY)]
                # Extend only candidates that beat BOTH bracketing baselines.
                for budget in ([8192] + [c['budget'] for c in qualified]) if qualified else []:
                    directory, cell = self.start('extended-' + str(budget), budget)
                    self.raw(cell, 98304, 800, 'warmup')
                    for repetition in range(5):
                        self.raw(cell, 98304, repetition)
                    for repetition in range(3):
                        self.decode(directory, cell, repetition)
                    self.raw(cell, 212991, 700, 'capacity')
                    cell['status'] = 'completed'
                    self.checkpoint()
                    self.stop()
            else:
                qualified = []
            self.summary['primary_pass_budgets'] = [c['budget'] for c in qualified]
            self.summary['status'] = 'completed-no-promotion'
        except BaseException as exc:
            self.summary['status'] = 'failed'
            self.summary['error'] = repr(exc)
            error = exc
        finally:
            cleanup = {}
            try:
                self.stop()
            except BaseException as exc:
                cleanup['stop_error'] = repr(exc)
                error = error or exc
            try:
                assert not command('docker', 'ps', '--format', '{{.Names}}').strip(), 'Container remains; cannot safely restore'
                if self.process is not None:
                    self.process.terminate()
                    self.process.wait(timeout=10)
                    self.process = None
                if self.log is not None:
                    self.log.close()
                    self.log = None
                _, final = self.start('production-restored', 8192, cached=True, production=True)
                self.canaries(final)
                final['status'] = 'running'
                cleanup.update(production_running=True, power=power(), containers=command('docker', 'ps', '--format', '{{.Names}}').splitlines())
                # The explicitly restored production container stays up.
                self.process = None
                self.log.close()
                self.log = None
            except BaseException as exc:
                cleanup.update(production_running=False, error=repr(exc))
                error = error or exc
            self.summary['cleanup'] = cleanup
            self.checkpoint()
        if error:
            raise error


def interrupted(signum, frame):
    raise InterruptedError('signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    Campaign(args.output).run()


if __name__ == '__main__':
    main()
