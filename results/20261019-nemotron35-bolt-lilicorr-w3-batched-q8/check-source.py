#!/usr/bin/env python3
"""Read-only exact-r2/patch/candidate gate; no native or public fidelity claim."""
import argparse
import hashlib
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError('do not disable assertions in this experimental gate')
ROOT = Path(__file__).resolve().parent
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
SOURCE_SCOPE = ['ggml/src/ggml-sycl/dmmv.cpp', 'ggml/src/ggml-sycl/dmmv.hpp',
                'ggml/src/ggml-sycl/ggml-sycl.cpp']
PATCH_SHA256 = 'a1b926f9d3a15613b49dcceea96a5dc647a78d87270ec8485899f187e696face'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check(parent=PARENT, candidate=None, patch=None):
    parent = Path(parent)
    if patch is None:
        patch = (ROOT / 'batched-q8.patch').read_bytes()
    pins = json.loads((ROOT / 'source-hashes.json').read_text())
    assert sorted(pins) == SOURCE_SCOPE
    assert sha256(patch) == PATCH_SHA256, 'not the reviewed bounded patch'
    assert [line.split(' b/', 1)[1] for line in patch.decode().splitlines()
            if line.startswith('diff --git ')] == SOURCE_SCOPE
    for name, hashes in pins.items():
        assert sha256((parent / name).read_bytes()) == hashes['parent'], 'unexpected r2: ' + name
        if candidate is not None:
            assert sha256((Path(candidate) / name).read_bytes()) == hashes['candidate'], 'unexpected candidate: ' + name
    return dict(source_scope=SOURCE_SCOPE, patch_sha256=PATCH_SHA256, source_hashes=pins,
                candidate_checked=candidate is not None, native_and_public_fidelity='NOT_TESTED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.parent, args.candidate), indent=2))
