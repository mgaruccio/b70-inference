#!/usr/bin/env python3
"""Disposable quality-only server lifecycle; execute only on inference-host."""
import argparse
import json
import runpy
import shlex
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPED = ROOT.parent / '20260913-qwen38-native-grouped-verify'
PROFILE = ROOT.parent / '20260911-qwen38-step-profile-64k'


def configure_launch(adapter, original, arm, cell, out, args):
    argv, meta = original(cell, out, args)
    meta['experiment'] = 'grouped quality validation; no production promotion'
    meta['comparison_arm'] = arm
    meta['intentional_comparison_difference'] = {
        'target': 'same target/runtime; speculative-config removed',
        'native': 'native MTP4 baseline',
        'candidate': 'native MTP4 plus qualified build06 grouped attention',
    }[arm]
    if arm == 'target':
        serve = meta['serve']
        index = serve.index('--speculative-config')
        del serve[index:index + 2]
        prefix, _ = argv[-1].rsplit('; exec ', 1)
        argv[-1] = prefix + '; exec ' + shlex.join(serve)
        meta['speculation'] = {'enabled': False, 'method': None, 'num_speculative_tokens': 0}
    elif arm == 'candidate':
        module = GROUPED / 'serving-overlay.py'
        seam = GROUPED / 'grouped_verify.py'
        patch = adapter['CANONICAL_PATCH']
        lib = adapter['DEFAULT_LIBRARY']
        for path in (module, seam, patch, lib):
            if not path.is_file():
                raise RuntimeError(f'Missing candidate asset: {path}')
        if adapter['sha256_file'](lib) != adapter['EXPECTED_LIBRARY_SHA256']:
            raise RuntimeError('Candidate library hash mismatch')
        if adapter['sha256_file'](patch) != adapter['EXPECTED_CANONICAL_PATCH_SHA256']:
            raise RuntimeError('Import patch hash mismatch')
        mounts = adapter['_candidate_mounts'](module, patch, seam, lib)
        env = ['PYTHONPATH=/experiment', 'B70_STEP_TIMING=1', 'B70_GROUPED_SERVING=1',
               'B70_GROUPED_SERVING_LIBRARY=/candidate/libb70_grouped_verify.so']
        extra = []
        for mount in mounts:
            extra.extend(['-v', f"{mount['host']}:{mount['container']}:ro"])
        for value in env:
            extra.extend(['-e', value])
        index = argv.index('--entrypoint')
        argv[index:index] = extra
        prefix, serve = argv[-1].rsplit('; exec ', 1)
        argv[-1] = prefix + '; /opt/venv/bin/python -P /experiment/qwen38_step_timing_patch.py; exec ' + serve
        meta['mounts'].extend(mounts)
        meta['environment'].extend(env)
        meta['candidate_sources'] = {str(p): adapter['sha256_file'](p) for p in (module, seam, patch)}
        meta['candidate_library'] = {'host': str(lib), 'sha256': adapter['sha256_file'](lib)}
    return argv, meta


def long_diagnostic(out, base_url, smoke):
    """Real 64K input class, frozen confirmation prompts, two repeats, token IDs."""
    dest = out / 'long-divergence'
    dest.mkdir()
    rows = []
    for trial in range(1, 2 if smoke else 7):
        source = GROUPED / f'confirm-a1/length-65536/long-context/points/length-65536/measured-{trial:02}/request.json'
        payload = json.loads(source.read_text())
        payload.pop('stream_options')
        payload.update(stream=False, return_token_ids=True)
        for repeat in range(1, 3):
            stem = dest / f'prompt-{trial:02}-repeat-{repeat}'
            stem.with_suffix('.request.json').write_text(json.dumps(payload))
            request = urllib.request.Request(base_url.rstrip('/') + '/v1/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=900) as response:
                raw = response.read()
            stem.with_suffix('.response.json').write_bytes(raw)
            data = json.loads(raw)
            choice = data['choices'][0]
            ids = choice.get('token_ids')
            prompt_ids = data.get('prompt_token_ids')
            if prompt_ids is None:
                prompt_ids = choice.get('prompt_token_ids')
            if not isinstance(ids, list) or len(ids) != 128 or prompt_ids != payload['prompt']:
                raise RuntimeError('Long diagnostic missing/mismatched token IDs')
            rows.append({'trial': trial, 'repeat': repeat, 'token_ids': ids,
                         'text': choice.get('text'), 'finish_reason': choice.get('finish_reason')})
    (dest / 'summary.json').write_text(json.dumps({'rows': rows, 'prompt_tokens': 65536,
        'output_tokens': 128, 'ignore_eos': True, 'smoke': smoke}, indent=2))
    return {'path': str(dest), 'requests': len(rows)}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arm', choices=('target', 'native', 'candidate'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    options, remaining = parser.parse_known_args()
    if socket.gethostname().split('.')[0] != 'inference-host':
        parser.error('Quality serving must run on inference-host, never the Pi host')
    adapter = runpy.run_path(str(GROUPED / 'run-serving.py'))
    adapter['_reject_profile_mode'](parser, remaining)
    driver = runpy.run_path(str(PROFILE / 'run-step-profile.py'))
    ns = driver['main'].__globals__
    ns['CAMPAIGN'] = ROOT.name
    ns['CONTEXT'] = 212992
    ns['BATCHED_TOKENS'] = 8192
    ns['cell_name'] = lambda cell: f'b70-grouped-quality-{options.arm}'
    original_build = ns['build_launch']
    original_write = ns['write_json']
    ns['build_launch'] = lambda cell, out, args: configure_launch(adapter, original_build, options.arm, cell, out, args)

    def write_json(path, value):
        if path.name == 'summary.json' and isinstance(value, dict) and value.get('campaign') == ROOT.name:
            value['purpose'] = 'Quality-sensitive three-arm evaluation; not a performance or equivalence claim'
            value['workload_contract'] = {'prepared_data': str(options.data.resolve()),
                'temperature': 0, 'seed': 42, 'max_tokens': 4096, 'ignore_eos': False,
                'prefix_cache': False, 'limit': options.limit, 'arm': options.arm}
        original_write(path, value)
    ns['write_json'] = write_json

    def benchmark(out, args, _long_client):
        command = [sys.executable, '-u', str(ROOT / 'quality.py'), 'generate',
                   '--arm', options.arm, '--timeout', '900',
                   '--data', str(options.data.resolve()), '--out', str(out / 'generation.jsonl'),
                   '--base-url', args.base_url.rstrip('/') + '/v1']
        if options.limit is not None:
            command.extend(['--limit', str(options.limit)])
        original_write(out / 'quality-command.json', command)
        with (out / 'quality-client.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.benchmark_timeout)
        if completed.returncode:
            raise RuntimeError(f'Quality generation exited {completed.returncode}; see quality-client.log')
        evidence = {'generation': str(out / 'generation.jsonl'), 'returncode': completed.returncode}
        evidence['long_divergence'] = long_diagnostic(out, args.base_url, options.limit is not None)
        if options.arm == 'candidate':
            evidence['candidate_execution'] = adapter['_candidate_execution_evidence'](out)
        return evidence
    ns['run_benchmark'] = benchmark
    sys.argv = [sys.argv[0], '--cell', 'mtp4', *remaining]
    return driver['main']()


if __name__ == '__main__':
    raise SystemExit(main())
