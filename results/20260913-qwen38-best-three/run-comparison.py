#!/usr/bin/env python3
"""Replay recorded best bundles through existing public API gates and cold client."""
import argparse
import json
from pathlib import Path
import runpy
import shlex
import sys

ROOT = Path(__file__).resolve().parent
LENGTHS = (512, 8192, 32768, 65536)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=('mtp4', 'dspark', 'dflash'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    driver = runpy.run_path(str(ROOT.parent / '20260911-qwen38-step-profile-64k/run-step-profile.py'))
    ns = driver['run'].__globals__
    reference = ROOT / f'{options.cell}-reference-launch.json'
    archived = json.loads(reference.read_text())
    serve = shlex.split(archived[-1].rsplit('; exec ', 1)[1])
    setting = lambda flag: serve[serve.index(flag) + 1]
    context = int(setting('--max-model-len'))
    ns.update(CAMPAIGN=ROOT.name, CONTEXT=context,
              BATCHED_TOKENS=int(setting('--max-num-batched-tokens')),
              cell_name=lambda cell: f'b70-best-three-{cell}')
    args = ns['parse_args'](['--cell', 'mtp4' if options.cell == 'mtp4' else 'dspark',
                             '--out', str(options.out)])
    args.cell = options.cell
    original_benchmark = ns['run_benchmark']

    def launch(cell, out, args):
        argv = list(archived)
        argv[argv.index('--name') + 1] = ns['cell_name'](cell)
        argv[argv.index('--group-add') + 1] = str(Path('/dev/dri/renderD128').stat().st_gid)
        mounts, environment = [], []
        for index, value in enumerate(argv):
            if value == '-v':
                parts = argv[index + 1].split(':')
                if parts[1] == '/output':
                    parts = [str(out), '/output']
                elif parts[1] == '/profile':
                    # Historical source is input, never a writable artifact destination.
                    parts = [parts[0], parts[1], 'ro']
                argv[index + 1] = ':'.join(parts)
                mounts.append({'host': parts[0], 'container': parts[1],
                               'mode': parts[2] if len(parts) > 2 else 'rw'})
            elif value == '-e':
                environment.append(argv[index + 1])
        return argv, {'serve': serve, 'mounts': mounts, 'environment': environment,
                      'speculation': json.loads(setting('--speculative-config')),
                      'common_contract': {'target_quantization': 'gptq', 'dtype': 'float16',
                          'target_kv': 'fp8', 'concurrency': 1, 'prefix_cache': False,
                          'max_model_len': context, 'prompt_lengths': LENGTHS},
                      'reference_launch': str(reference),
                      'intentional_changes': ['unique container name', 'new output directory',
                                               'historical profile source mounted read-only'],
                      'comparison': 'user-selected best configuration bundles, not isolated algorithms'}

    def assets(cell, args):
        _, metadata = launch(cell, args.out.resolve(), args)
        paths = {'reference_launch': reference, 'campaign_driver': Path(__file__).resolve()}
        mapping = {}
        for index, mount in enumerate(metadata['mounts']):
            if mount['container'] != '/output':
                path = Path(mount['host'])
                paths[f'mount_{index}'] = path
                mapping[mount['container']] = path
                if path.is_dir() and (path / 'config.json').is_file():
                    paths[f'mount_{index}_config'] = path / 'config.json'
        # Capture scripts invoked through directory mounts, not just individual mounts.
        for token in shlex.split(archived[-1].split('; exec ', 1)[0]):
            token = token.rstrip(';')
            if token.startswith('/') and token.endswith('.py'):
                for container in sorted(mapping, key=len, reverse=True):
                    if token == container or token.startswith(container + '/'):
                        paths[f'patch_{len(paths)}'] = Path(str(mapping[container]) + token[len(container):])
                        break
        for label, path in paths.items():
            if not path.exists():
                raise RuntimeError(f'Missing archived input {label}: {path}')
        if cell == 'dspark':
            module = next(m['host'] for m in metadata['mounts']
                          if m['container'] == '/noncausal/qwen38_step_timing_overlay.py')
            if ns['sha256_file'](Path(module)) != 'e19d3fa94a11942aeb9db4ca4552ab4357373c14d7095f7612761a05b1d924ff':
                raise RuntimeError('DSpark Split-K source differs from final qualified module')
        return paths

    def benchmark(out, args, long_client):
        if args.cell == 'dspark':
            if '[B70_DSPARK_NONCAUSAL_SPLIT_K] eligible C1/Q7 BF16 dispatch' not in (out / 'server.log').read_text():
                raise RuntimeError('DSpark Split-K dispatch was not observed')
        diag = runpy.run_path(str(ROOT.parent / '20260911-qwen38-dspark-layer-norm/run-acceptance-diagnostics.py'))
        dns = diag['run'].__globals__
        probe, checks = dns['load_previous_modules']()
        probe.IMAGE = ns['DEFAULT_MTP_IMAGE'] if args.cell == 'mtp4' else ns['DEFAULT_DSPARK_IMAGE']
        client = dns['DiagnosticClient'](out, request_timeout=900)
        gates = dns['run_shared_gates'](client, out, probe, checks, True)
        if gates['status'] != 'passed':
            raise RuntimeError(f'Public API gates failed: {gates}')
        results = {}
        try:
            for length in LENGTHS:
                ns['PROMPT_TOKENS'] = length
                point = out / f'length-{length}'
                point.mkdir()
                print(f'BEGIN {args.cell} input={length}', flush=True)
                results[str(length)] = original_benchmark(point, args, long_client)
                print(f'PASS {args.cell} input={length}', flush=True)
        finally:
            ns['PROMPT_TOKENS'] = 65536
        return {'gates': gates, 'lengths': results}

    ns.update(build_launch=launch, required_assets=assets, run_benchmark=benchmark)
    result = driver['run'](args)
    summary_path = args.out / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['purpose'] = 'Best MTP4 / DSpark Split-K / DFlash2 bundle comparison; development only'
    summary['workload_contract']['prompt_tokens'] = list(LENGTHS)
    ns['write_json'](summary_path, summary)
    return result


if __name__ == '__main__':
    sys.exit(main())
