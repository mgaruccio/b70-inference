#!/usr/bin/env python3
"""Development-only matched stock MTP4/no-spec ABBA through existing API client."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))
import qwen38_lossy_probe as probe
import qwen38_mtp_reference as reference
import qwen38_mtp_native_corpus as native
import qwen38_mtp_tune_probe as tune

LAUNCHER = Path('/home/mike/inference/launchers/start-qwen38.sh')
POWER = Path('/sys/class/drm/card0/device/hwmon/hwmon2/power1_cap')
GUARD = ROOT / 'patch_uniform_decode_prefill.py'
PORT = 8001
probe.BASE = f'http://127.0.0.1:{PORT}'
PROMPTS = [
 ('coding-merge', 'coding', 'Implement a Python function that merges two sorted integer lists without sorting their concatenation. Include edge-case examples and explain time and space complexity.'),
 ('coding-cache', 'coding', 'Explain and implement a bounded LRU cache in Python using collections.OrderedDict. Include a complete class and a worked example.'),
 ('prose-rain', 'prose', 'Write a detailed accessible explanation of how rain forms, from evaporation through clouds to precipitation. Distinguish condensation from freezing.'),
 ('prose-library', 'prose', 'Write a thoughtful account of a fictional village opening its first public library. Describe practical challenges, volunteers and the opening day.'),
 ('reasoning-crossing', 'reasoning', 'Four people take 1, 2, 5 and 10 minutes to cross a bridge at night. At most two cross together and they must carry a single torch. Explain a minimum-time crossing schedule and verify its total.'),
 ('reasoning-bayes', 'reasoning', 'A condition affects 1 percent of people. A test has 90 percent sensitivity and 95 percent specificity. Explain the probability a positive result indicates the condition, using a population of 10000.'),
 ('json-records', 'structured', 'Return only valid JSON: an object with key records holding six objects. Each object must have id from 1 through 6, name as item followed by its id, and active true for odd ids and false for even ids.'),
 ('json-nesting', 'structured', 'Return only valid JSON representing a project named Orchard, with three milestones named Plan, Build and Test. Each milestone has a positive integer order and an empty tasks array.'),
 ('repeat-table', 'repetitive', 'Produce a Markdown table of the integers 1 through 40 with columns n, n squared, and n cubed. Do not skip rows.'),
 ('repeat-items', 'repetitive', 'Write 40 numbered lines. Each line must have exactly the form Item N: ready, with N running consecutively from 1 to 40.'),
 ('entropy-words', 'high_entropy', 'Invent 30 distinct unusual compound words. For each give a short creative definition involving a different everyday object. Avoid conventional words and repeated endings.'),
 ('entropy-scenes', 'high_entropy', 'Write a sequence of 12 very different imaginary scenes. In each combine an unexpected location, a surprising object, and a different emotion. Avoid repeating themes or objects.')
]

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def check_host():
    if probe.command('docker', 'ps', '-q').strip():
        raise RuntimeError('Host not idle; will not stop unrelated containers')
    if hashlib.sha256(LAUNCHER.read_bytes()).hexdigest() != probe.LAUNCHER_SHA:
        raise RuntimeError('Persistent launcher changed')
    if POWER.read_text().strip() != '275000000':
        raise RuntimeError('Power cap changed')
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError('Disk reserve too small')
    busy = subprocess.run(['fuser', '/dev/dri/renderD128'], capture_output=True)
    if busy.returncode == 0:
        raise RuntimeError('GPU already held by another process')
    if busy.returncode != 1:
        raise RuntimeError('Unable to verify GPU ownership')


def stop_owned(name, cell, proc):
    inspect = subprocess.run(['docker', 'inspect', '-f', '{{range .Mounts}}{{if eq .Destination "/profile"}}{{.Source}}{{end}}{{end}}', name], capture_output=True, text=True)
    if inspect.returncode == 0:
        if inspect.stdout.strip() != str(cell):
            raise RuntimeError('Container ownership mismatch; refusing cleanup')
        probe.command('docker', 'stop', '--time', '30', name, timeout=60)
    if proc is not None:
        proc.wait(timeout=90)


def check_tokens(row):
    rendered = probe.post('/tokenize', {'model': 'qwen38', 'messages': [{'role': 'user', 'content': row['prompt']}], 'chat_template_kwargs': {'enable_thinking': False}})
    if rendered['tokens'] != row['prompt_token_ids']:
        raise RuntimeError('Rendered prompt/token identity mismatch')


def request(client, label, text, max_tokens):
    before = native.metric_snapshot(probe.BASE, 30)
    row = probe.Cell.chat(client, label, text, max_tokens, False)
    # Metrics export is asynchronous: wait for exact accounting, not a fixed sleep.
    deadline = time.monotonic() + 15
    while True:
        after = native.metric_snapshot(probe.BASE, 30)
        delta = {key: after.get(key, 0) - before.get(key, 0) for key in before.keys() | after.keys()}
        count = native.metric_total(delta, 'request_generation_tokens_sum')
        if count == row['usage']['completion_tokens']:
            break
        if count > row['usage']['completion_tokens'] or time.monotonic() >= deadline:
            raise RuntimeError('Counters did not account for exactly this request')
        time.sleep(.1)
    if any(not math.isfinite(x) or x < -1e-9 for x in delta.values()):
        raise RuntimeError('Counter reset or invalid metric')
    if native.metric_total(delta, 'prefix_cache_hits_total') != 0:
        raise RuntimeError('Unexpected prefix cache hits')
    row.update(metric_deltas=delta, counters_before=before, counters_after=after, prompt=text)
    check_tokens(row)
    save(client.out / f'{label}-result.json', row)
    return row


def run_cell(label, speculative):
    check_host()
    cell = ROOT / label
    cell.mkdir(mode=0o700)
    name = 'qwen38-mtp-baseline-' + label.lower()
    original = LAUNCHER.read_bytes()
    text = reference.launcher_text(original, cell)
    text = native.replace_once(text, '--name qwen38 ', f'--name {name} ')
    text = native.replace_once(text, '-p "127.0.0.1:8000:8000"', f'-p "127.0.0.1:{PORT}:8000"')
    if not speculative:
        text = native.replace_once(text, '--speculative-config "{\\"method\\":\\"mtp\\",\\"num_speculative_tokens\\":4}"', '')
    patches = cell / 'reference-source/patches'
    patches.mkdir(parents=True)
    for filename in reference.PATCHES:
        shutil.copy2(reference.PATCH_ROOT / filename, patches / filename)
    shutil.copy2(GUARD, cell / GUARD.name)
    (cell / 'launcher.sh').write_text(text)
    subprocess.run(['bash', '-n', str(cell / 'launcher.sh')], check=True)
    save(cell / 'launch-config.json', dict(label=label, speculative_tokens=4 if speculative else 0, image=reference.IMAGE, launcher_sha256=hashlib.sha256(text.encode()).hexdigest(), patch_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in patches.iterdir()}, context=212992, prefix_caching=False, thinking=False, temperature=0, seed=42, max_output_tokens=512, concurrency=1, power_watts=275, public_api=probe.BASE))
    print('CELL_START=' + label, flush=True)
    proc = None
    report = dict(status='starting', label=label)
    try:
        with (cell / 'server.log').open('w') as log:
            proc = subprocess.Popen(['bash', str(cell / 'launcher.sh')], stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'Server startup failed ({proc.returncode}); see {cell}/server.log')
                try:
                    probe.get('/health')
                    break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Server readiness exceeded 900 seconds')
                time.sleep(3)
            models = json.loads(probe.get('/v1/models'))
            save(cell / 'models.json', models)
            if not any(m['id'] == 'qwen38' and m['max_model_len'] == 212992 for m in models['data']):
                raise RuntimeError('Wrong model/context exposed by API')
            print('CELL_READY=' + label, flush=True)
            save(cell / 'runtime.json', dict(container=probe.command('docker', 'inspect', name), versions=probe.command('docker', 'exec', name, 'python', '-c', 'import torch,vllm; print(torch.__version__); print(vllm.__version__); print(torch.xpu.get_device_properties(0))', timeout=60)))
            client = SimpleNamespace(out=cell, rows=[])
            checks = [
                ('arithmetic', 'What is 19 + 23? Reply with only the integer.', lambda x: x.strip() == '42'),
                ('json', 'Return only JSON, no markdown, with the single key answer and integer value 42.', lambda x: json.loads(x) == {'answer':42})
            ]
            for key, prompt, predicate in checks:
                row = request(client, 'canary-' + key, prompt, 64)
                if row['finish_reason'] != 'stop' or not predicate(row['content']):
                    raise RuntimeError('Functional canary failed: ' + key)
            request(client, 'warmup', PROMPTS[0][2], 128)
            rows = []
            for key, category, prompt in PROMPTS:
                row = request(client, key, prompt, 512)
                row['category'] = category
                save(cell / f'{key}-result.json', row)
                rows.append(row)
            accepted = tune.aggregate(rows) if speculative else None
            if not speculative and any(native.metric_total(r['metric_deltas'], 'spec_decode_num_drafts_total') for r in rows):
                raise RuntimeError('No-spec control actually drafted tokens')
            if speculative and accepted['accepted'] > accepted['proposed']:
                raise RuntimeError('Invalid speculative accounting')
            structured = {r['label']: {'finish_reason': r['finish_reason'], 'json_valid': is_json(r['content']) if r['finish_reason'] == 'stop' else None} for r in rows if r['category'] == 'structured'}
            report.update(status='completed', requests=len(rows), accepted=accepted, median_decode_tps=statistics.median(r['decode_tps'] for r in rows), median_e2e_tps=statistics.median(r['usage']['completion_tokens'] / r['elapsed_s'] for r in rows), generated_tokens=sum(r['usage']['completion_tokens'] for r in rows), structured=structured)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        save(cell / 'summary.json', report)
        stop_owned(name, cell, proc)
        check_host()
    print('CELL_COMPLETED=' + json.dumps(report), flush=True)
    return report


def is_json(text):
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}; preserve artifacts and clean up owned server')


def main():
    os.umask(0o077)
    signal.signal(signal.SIGTERM, interrupted)
    check_host()
    if hashlib.sha256(GUARD.read_bytes()).hexdigest() != probe.GUARD_SHA:
        raise RuntimeError('Prefill guard changed')
    save(ROOT / 'protocol.json', dict(tier='development', baseline='stock checkpoint/native MTP4; no-spec target-only control', cells=['A1-MTP4', 'B1-no-spec', 'B2-no-spec', 'A2-MTP4'], prompts=PROMPTS, natural_eos=True, max_output_tokens=512, sampling='greedy seed42; thinking disabled; no tool execution', differences='Only --speculative-config removed in no-spec; same target, patches, fp16/GPTQ+FP8, C1, graphs and cold prefix cache. Temporary loopback binding and balanced mode versus production.', metrics='Native accepted draft tokens / native draft passes, excludes bonus. Not all base-model calls are inferred from this counter. Exact API token IDs, SSE, latency, counter snapshots retained.', limitations='Synthetic development prompts, short contexts; not a held-out generalization claim or production quality qualification. Variable natural-EOS lengths; separately report repeated-output and cross-arm differences.', cleanup='Stop only container matching owned /profile mount; persistent launcher and power cap unchanged', sources=['https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json', 'https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/models/qwen3_5_mtp.py', 'https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/step3p5.py'], conclusions='Checkpoint has one native MTP layer; physical layer is reused across speculative steps with state, token and native KV-cache machinery. Reuse native verifier, not Glimmer KV cropping.'))
    reports = []
    try:
        for label, spec in [('A1-MTP4', True), ('B1-no-spec', False), ('B2-no-spec', False), ('A2-MTP4', True)]:
            reports.append(run_cell(label, spec))
        matches = {}
        for first, second in [('A1-MTP4', 'A2-MTP4'), ('B1-no-spec', 'B2-no-spec'), ('A1-MTP4', 'B1-no-spec'), ('A2-MTP4', 'B2-no-spec')]:
            values = []
            for key, category, _ in PROMPTS:
                a = json.loads((ROOT / first / f'{key}-result.json').read_text())
                b = json.loads((ROOT / second / f'{key}-result.json').read_text())
                assert a['prompt_token_ids'] == b['prompt_token_ids']
                values.append(dict(prompt=key, category=category, exact=a['token_ids'] == b['token_ids'], first_tokens=len(a['token_ids']), second_tokens=len(b['token_ids'])))
            matches[first + ':' + second] = values
        result = dict(status='completed', tier='development', cells=reports, token_matches=matches, production_promotion=False, historical_results_not_pooled=True)
        save(ROOT / 'comparison.json', result)
        print('BENCHMARK_COMPLETE=' + json.dumps(result), flush=True)
    except BaseException as exc:
        save(ROOT / 'failure.json', dict(status='failed', error=repr(exc), completed_cells=reports))
        raise
    finally:
        save(ROOT / 'invariants.json', dict(persistent_launcher_sha256=hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(), expected=probe.LAUNCHER_SHA, power_watts=int(POWER.read_text()) / 1e6, containers=probe.command('docker', 'ps', '--format', '{{.Names}}').splitlines()))

if __name__ == '__main__':
    main()
