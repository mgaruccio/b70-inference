#!/usr/bin/env python3
"""Four repetitions of five frozen diagnostic tasks through the existing API path."""
import argparse
import hashlib
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
CAMPAIGN = ROOT / '20261009-qwen38-shared-kv-quality'
OLD = ROOT / '20260913-qwen38-grouped-quality'
PREVIOUS = ROOT / '20261009-qwen38-native-shared-kv/run-cell.py'
DATA = CAMPAIGN / 'diagnostic-five.jsonl'
HASHES = {
    PREVIOUS: '58fc22ce410d84c6559c07109973568078e6ac80004a2c8c0769a25dc277fd09',
    DATA: 'f82d262c69e7fb2d63ecf8dedcbcdb28cfee241d3ab6538db94d3b0327e296f2',
    OLD / 'quality.py': '8544176f922079868d022f100de2f2e424a4617be321ae1df08dbbd2fdb11c48',
    OLD / 'run-quality.py': '8c7af825b33da9b2c7d5717b55d9908cab23b1874dfb080e06e69d11a2720b5a',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('native', 'candidate'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for path, expected in HASHES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen dependency changed: {path}')
    previous = runpy.run_path(str(PREVIOUS))
    sources = previous['verify_sources']()
    sources.update({str(p): h for p, h in HASHES.items()})
    guard = runpy.run_path(str(previous['GUARD']))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    name = f'b70-he-repeat-{args.arm}'
    guard['preconditions'](out, name)
    before = guard['kernel'](out, 'pre')
    if any(v['error_lines'] for v in before['commands'].values()):
        raise RuntimeError('Pre-existing guarded kernel error')
    guard['put'](out / 'source-hashes.json', sources)
    for source in (Path(__file__), DATA, OLD / 'quality.py', OLD / 'run-quality.py'):
        (out / source.name).write_bytes(source.read_bytes())
    quality = runpy.run_path(str(OLD / 'run-quality.py'))
    adapter = runpy.run_path(str(previous['PRIOR'] / 'run-serving.py'))
    driver = runpy.run_path(str(guard['DRIVER']))
    ns = driver['main'].__globals__
    ns.update(CAMPAIGN=CAMPAIGN.name, CONTEXT=212992, BATCHED_TOKENS=8192,
              cell_name=lambda cell: name)
    original_build = ns['build_launch']
    def build(cell, dest, config):
        argv, metadata = quality['configure_launch'](
            adapter, original_build, args.arm, cell, dest, config)
        metadata['diagnostic_workload'] = {
            'data': str(DATA), 'data_sha256': HASHES[DATA],
            'repetitions': 4, 'tasks_per_repetition': 5,
            'max_output_tokens': 4096, 'seed': 42, 'temperature': 0,
            'full_quality_run': False, 'long_diagnostic_prompt_tokens': 65536,
        }
        return argv, metadata

    ns['build_launch'] = build

    def benchmark(dest, config, _client):
        records = []
        for repeat in range(1, 5):
            point = dest / f'repeat-{repeat:02d}'
            point.mkdir()
            command = [sys.executable, '-B', '-u', str(OLD / 'quality.py'), 'generate',
                       '--arm', args.arm, '--timeout', '900', '--data', str(DATA),
                       '--limit', '5', '--out', str(point / 'generation.jsonl'),
                       '--base-url', config.base_url.rstrip('/') + '/v1']
            guard['put'](point / 'command.json', {'argv': command})
            with (point / 'client.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=1800, check=False)
            guard['put'](point / 'exit.json', {'returncode': result.returncode})
            if result.returncode:
                raise RuntimeError(f'Diagnostic generation failed: {point}')
            records.append({'repeat': repeat, 'generation': str(point / 'generation.jsonl')})
        evidence = {'repeats': records, 'scope': 'diagnostic subset, not full quality qualification'}
        evidence['long_divergence'] = quality['long_diagnostic'](dest, config.base_url, True)
        if args.arm == 'candidate':
            evidence['candidate_execution'] = adapter['_candidate_execution_evidence'](dest)
        return evidence

    ns['run_benchmark'] = benchmark
    sys.argv = [str(guard['DRIVER']), '--cell', 'mtp4', '--out', str(out / 'native'),
                '--startup-timeout', '1800', '--benchmark-timeout', '7200']
    rc = 1
    try:
        rc = int(driver['main']())
    finally:
        after = guard['kernel'](out, 'post')
        old = {s for v in before['commands'].values() for s in v['error_lines']}
        new = sorted({s for v in after['commands'].values() for s in v['error_lines']} - old)
        post = out / 'postconditions'
        post.mkdir()
        guard['preconditions'](post, name)
        previous['verify_sources']()
        for path, expected in HASHES.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Dependency changed during run: {path}')
        guard['put'](out / 'cell-result.json', {'driver_exit': rc, 'arm': args.arm,
                     'new_guarded_error_lines': new, 'tier': 'development',
                     'quality_qualified': False})
        if new:
            raise RuntimeError('New guarded kernel errors')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
