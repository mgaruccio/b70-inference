#!/usr/bin/env python3
"""Case-local build helpers and a native-command PRINTER. No service/GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError('do not disable assertions in this experimental gate')

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-columns8')
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-columns8-r1')
ACCEPTED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
SOURCE_SCOPE = ['ggml/src/ggml-sycl/ggml-sycl.cpp', 'tests/test-backend-ops.cpp']
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so')
# Existing widths-single2 candidate-binaries.json / run-r1/accepted-binaries.json.
PARENT_EXPECTED = {
    'llama-server': '22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d',
    'libllama-server-impl.so': '43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede',
    'libggml-sycl.so.0': '64d6a58f17cb92ea9fe6354d9c553f74724565a58fc018d2f4bbd013d090a72e',
    'test-backend-ops': 'bdd86349fac5ab43fa32d465aad27cec7f31dcad0651f7611403d8a783f10c7d',
    'libllama-common.so': '934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812',
}
ACCEPTED_EXPECTED = {
    'llama-server': '22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d',
    'libllama-server-impl.so': '67ae7c8673110b52d11924677ad662141dc93f57d7c21d67a3f0d3e4489e09ad',
    'libggml-sycl.so.0': '076b16475de805572cf7a623eecca827fd04e58903bf03a003121f60af56fd2f',
    'test-backend-ops': 'a6a3f666754fce8edfbb59d9840290a188ba2681e7065f812de3359f4e12c98d',
    'libllama-common.so': '731100ef8c379050fd5fd99f95fb1e6dc1b1fa3498b057ca200f02c8d9758cca',
}


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binaries(source):
    return {name: sha256(source / 'build-sycl/bin' / name) for name in ARTIFACTS}


def source_hashes(root):
    # Same full non-build source-map convention as the parent experiment.
    return {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
            if p.is_file() and p.relative_to(root).parts[0] != 'build-sycl'}


def check_consumed_sources(source, label):
    expected = load(ROOT / 'source-hashes.json')
    assert set(expected[label]) == set(SOURCE_SCOPE)
    assert sha256(ROOT / 'router-columns8.patch') == expected['patch_sha256']
    assert {name: sha256(source / name) for name in SOURCE_SCOPE} == expected[label], label


def check_protected_sources(candidate=True):
    expected = load(ROOT / 'protected-source-hashes.json')
    assert source_hashes(ACCEPTED) == expected['accepted'], 'accepted source changed'
    assert source_hashes(PARENT) == expected['parent'], 'parent source changed'
    check_consumed_sources(PARENT, 'parent')
    if candidate:
        actual = source_hashes(CANDIDATE)
        changed = sorted(k for k in actual.keys() | expected['parent'].keys()
                         if actual.get(k) != expected['parent'].get(k))
        assert changed == SOURCE_SCOPE, changed
        check_consumed_sources(CANDIDATE, 'candidate')


def check_protected_binaries():
    for label, source, expected in (('accepted', ACCEPTED, ACCEPTED_EXPECTED),
                                    ('parent', PARENT, PARENT_EXPECTED)):
        assert binaries(source) == load(ROOT / (label + '-binaries.json')) == expected, label


# INNER command only: the lead owns the stopped-original/exclusive-DRM window,
# container creation, raw log retention, and restoration. Never execute on import.
# Mount candidate at /src:ro, oneDNN at /opt/onednn:ro, this case at
# /router-columns8:ro, and a NEW native-r1 directory at /capture (read/write).
NATIVE_COMMAND = r'''set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
export GGML_SYCL_ENABLE_DNN=1 GGML_SYCL_ENABLE_OPT=1 GGML_SYCL_ENABLE_VMM=1
export GGML_SYCL_ENABLE_ESIMD=1 GGML_SYCL_ENABLE_FUSION=1 GGML_SYCL_ENABLE_GRAPH=0
export BOLT_TRACE_TENSORS=0 ONEDNN_VERBOSE=0
export BOLT_VERIFY_CAPTURE_DIR=/capture
for tag in router_single2 router_columns8; do
    test ! -e "/capture/$tag.log"
    GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT \
        -p "^$tag=1," 2>&1 | tee "/capture/$tag.log"
    python3 -B /router-columns8/analyze-trace.py --tag "$tag" "/capture/$tag.log"
done
echo BOLT_VERIFY_ROUTER_COLUMNS8_NATIVE_PASSED'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-native-command', action='store_true')
    args = parser.parse_args()
    if args.print_native_command:
        print(NATIVE_COMMAND)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
