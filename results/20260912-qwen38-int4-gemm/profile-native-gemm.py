"""Reuse the annotated real-request profile with oneDNN matmul diagnostics."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / '20260911-qwen38-target-verification'


def main():
    target = runpy.run_path(str(PREVIOUS / 'run-target-internals.py'))
    ns = target['main'].__globals__

    def load_driver(path):
        driver = runpy.run_path(path)
        dns = driver['main'].__globals__
        original = dns['build_launch']

        def build(cell, out, args):
            if cell != 'mtp4':
                raise ValueError('native GEMM investigation is fixed MTP4')
            argv, metadata = original(cell, out, args)
            setting = 'ONEDNN_VERBOSE=profile,filter=matmul'
            index = argv.index('--entrypoint')
            argv[index:index] = ['-e', setting]
            metadata['environment'].append(setting)
            metadata['gemm_profile_only'] = 'oneDNN verbosity and eager profiler affect timing; no throughput claim'
            return argv, metadata

        dns['build_launch'] = build
        return driver

    # Reuse its eager annotations, original capacity, real HTTP request and
    # cleanup. Only this module's runpy binding is replaced, not global runpy.
    ns['ROOT'] = ROOT
    ns['runpy'] = SimpleNamespace(run_path=load_driver)
    sys.argv = [sys.argv[0], '--cell', 'mtp4', *sys.argv[1:]]
    return target['main']()


if __name__ == '__main__':
    raise SystemExit(main())
