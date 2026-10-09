#!/usr/bin/env python3
"""Development-only split-KV cell using the existing native serving lifecycle."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import sys
import uuid

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
GUARD = ROOT / '20261009-qwen38-gdn-locality/native-boundary64k-assets-20261009-100052-ff9f86/registered-controller.py'
GUARD_SHA = 'fafe61a614a8328f63f0a7af41fa90da46041aa6cec514a01fe47b485b669da5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', choices=['auto', '8', '16', '32'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(GUARD.read_bytes()).hexdigest() != GUARD_SHA:
        raise RuntimeError('existing guard/controller changed')
    guard = runpy.run_path(str(GUARD))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    name = 'b70splits' + uuid.uuid4().hex[:12]
    guard['preconditions'](out, name)
    before = guard['kernel'](out, 'pre')
    if any(v['error_lines'] for v in before['commands'].values()):
        raise RuntimeError('guarded pre-kernel errors present')
    driver = runpy.run_path(str(guard['DRIVER']))
    ns = driver['main'].__globals__
    ns.update(CAMPAIGN='20261009-qwen38-gemm-attention', CONTEXT=212992,
              BATCHED_TOKENS=8192, PROMPT_TOKENS=65536,
              cell_name=lambda _cell: name)
    original = ns['build_launch']
    patch = Path(__file__).with_name('patch-spec-splits.py').resolve()

    def build(cell, cell_out, config):
        argv, metadata = original(cell, cell_out, config)
        metadata['development_experiment'] = {
            'split_count': args.splits,
            'scope': 'only batch1 five-row (5,24,256) native speculative attention',
            'baseline': 'unchanged automatic split heuristic',
            'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
            'precision_context_acceptance_unchanged': True,
            'profiler_enabled': False,
        }
        # Both arms run the same source-checking adapter; auto leaves bytes intact.
        idx = argv.index('--entrypoint')
        argv[idx:idx] = ['-v', f'{patch}:/split-patch.py:ro']
        metadata['mounts'].append({'host': str(patch), 'container': '/split-patch.py',
                                   'mode': 'ro', 'role': 'development_split_override'})
        prefix, serve = argv[-1].rsplit('; exec ', 1)
        argv[-1] = prefix + '; python /split-patch.py ' + shlex.quote(args.splits) + '; exec ' + serve
        return argv, metadata

    ns['build_launch'] = build
    sys.argv = [str(guard['DRIVER']), '--cell', 'mtp4', '--out', str(out / 'native'),
                '--startup-timeout', '1800', '--benchmark-timeout', '3600']
    result = 1
    try:
        result = int(driver['main']())
    finally:
        after = guard['kernel'](out, 'post')
        old_errors = {line for v in before['commands'].values() for line in v['error_lines']}
        new_errors = sorted({line for v in after['commands'].values() for line in v['error_lines']} - old_errors)
        post = out / 'postconditions'
        post.mkdir()
        guard['preconditions'](post, name)
        guard['put'](out / 'cell-result.json', {
            'driver_exit': result, 'new_guarded_error_lines': new_errors,
            'splits': args.splits, 'tier': 'development',
            'test': '65536->128, one warmup + six measured public-API requests',
        })
        if new_errors:
            raise RuntimeError('new guarded kernel errors')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
