#!/usr/bin/env python3
"""Disposable-container-only native split-count override; never patch the host."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SOURCE_SHA = '2a8ce07e2839232bc9f0e9cc9969a4410099c0c75ce616e72d4583e950864942'
ANCHOR = '        None,  # num_splits (let the kernel pick via get_num_splits)'


def patched_source(source, choice):
    if choice not in ('auto', '8', '16', '32'):
        raise ValueError('unsupported split choice')
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise RuntimeError('pinned installed flash attention interface changed')
    if source.count(ANCHOR) != 1:
        raise RuntimeError('expected exactly one speculative helper split argument')
    if choice == 'auto':
        return source
    return source.replace(ANCHOR,
        f'        {choice} if batch == 1 and tuple(q.shape) == (5, 24, 256) else None,  # B70 development split override', 1)


def main():
    choice = sys.argv[1]
    spec = importlib.util.find_spec('vllm_xpu_kernels')
    if spec is None or spec.origin is None:
        raise RuntimeError('installed XPU kernel package missing')
    path = Path(spec.origin).with_name('flash_attn_interface.py')
    source = path.read_text()
    changed = patched_source(source, choice)
    compile(changed, str(path), 'exec')
    if changed != source:
        path.write_text(changed)
    record = {'split_choice': choice, 'path': str(path), 'before_sha256': SOURCE_SHA,
              'after_sha256': hashlib.sha256(changed.encode()).hexdigest(),
              'scope': 'batch1 five-row native speculative helper only'}
    print(json.dumps(record), flush=True)
    Path('/output/split-patch.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
