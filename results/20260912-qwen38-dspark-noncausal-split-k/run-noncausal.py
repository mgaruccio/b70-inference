"""Development DSpark A/B using existing real-API workload, gates and cleanup."""
import argparse
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--candidate', action='store_true')
    options, remaining = parser.parse_known_args()
    driver = runpy.run_path(str(ROOT.parent / '20260911-qwen38-step-profile-64k/run-step-profile.py'))
    ns = driver['main'].__globals__
    variant = 'candidate' if options.candidate else 'baseline'
    ns.update(CAMPAIGN=ROOT.name, CONTEXT=65664, BATCHED_TOKENS=2048,
              cell_name=lambda cell: f'b70-dspark-noncausal-{variant}')
    original_build, original_benchmark = ns['build_launch'], ns['run_benchmark']

    def build(cell, out, args):
        if cell != 'dspark' or args.profile_mode:
            raise ValueError('This experiment requires unprofiled DSpark')
        argv, metadata = original_build(cell, out, args)
        metadata['experiment'] = f'native noncausal Split-K {variant}; fixed DSpark7, 65664/2048'
        if options.candidate:
            module = ROOT / 'qwen38_noncausal_split_k.py'
            patch = ROOT.parent / '20260911-qwen38-target-verification/qwen38_step_timing_patch.py'
            mounts = [
                {'host': str(module), 'container': '/noncausal/qwen38_step_timing_overlay.py', 'mode': 'ro', 'role': 'noncausal_split_k'},
                {'host': str(patch), 'container': '/noncausal/patch.py', 'mode': 'ro', 'role': 'worker_import_shim'},
            ]
            environment = ['PYTHONPATH=/noncausal', 'B70_STEP_TIMING=1', 'B70_DSPARK_NONCAUSAL_SPLIT_K=1']
            extra = []
            for mount in mounts:
                extra.extend(['-v', f"{mount['host']}:{mount['container']}:ro"])
            for value in environment:
                extra.extend(['-e', value])
            index = argv.index('--entrypoint')
            argv[index:index] = extra
            prefix, serve = argv[-1].rsplit('; exec ', 1)
            argv[-1] = prefix + '; /opt/venv/bin/python -P /noncausal/patch.py; exec ' + serve
            metadata['mounts'].extend(mounts)
            metadata['environment'].extend(environment)
            metadata['candidate_sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (module, patch)}
        return argv, metadata

    def benchmark(*args, **kwargs):
        out = Path(args[0])
        if options.candidate:
            log = (out / 'server.log').read_text()
            if '[B70_DSPARK_NONCAUSAL_SPLIT_K] eligible C1/Q7 BF16 dispatch' not in log:
                raise RuntimeError('No candidate dispatch observed during graph capture/startup')
        diag = runpy.run_path(str(ROOT.parent / '20260911-qwen38-dspark-layer-norm/run-acceptance-diagnostics.py'))
        dns = diag['run'].__globals__
        probe, checks = dns['load_previous_modules']()
        probe.IMAGE = ns['DEFAULT_DSPARK_IMAGE']
        client = dns['DiagnosticClient'](out, request_timeout=900)
        gates = dns['run_shared_gates'](client, out, probe, checks, True)
        assert gates['status'] == 'passed'
        return original_benchmark(*args, **kwargs)

    ns['build_launch'], ns['run_benchmark'] = build, benchmark
    sys.argv = [sys.argv[0], '--cell', 'dspark', *remaining]
    return driver['main']()


if __name__ == '__main__':
    raise SystemExit(main())
