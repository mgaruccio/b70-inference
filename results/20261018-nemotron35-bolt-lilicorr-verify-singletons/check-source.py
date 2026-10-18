#!/usr/bin/env python3
"""Read-only exact-parent/patch/candidate gate. Not a native or fidelity test."""
import argparse
import hashlib
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError('do not disable assertions in this experimental gate')
ROOT = Path(__file__).resolve().parent
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1')
SOURCE_SCOPE = ['ggml/src/ggml-sycl/ggml-sycl.cpp', 'src/llama-graph.cpp',
                'src/models/nemotron-h.cpp', 'tests/test-backend-ops.cpp']
PATCH_SHA256 = '4362d830523cfc12498655c4a818fb33b570ead82bf774db0f488b5ab456b6ea'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check(parent=PARENT, candidate=None, patch=None):
    parent = Path(parent)
    if patch is None:
        patch = (ROOT / 'verify-singletons.patch').read_text()
    pins = json.loads((ROOT / 'source-hashes.json').read_text())
    assert sorted(pins) == SOURCE_SCOPE
    assert sha256(patch.encode()) == PATCH_SHA256, 'not the reviewed bounded patch'
    assert [line.split(' b/', 1)[1] for line in patch.splitlines() if line.startswith('diff --git ')] == SOURCE_SCOPE
    for name, hashes in pins.items():
        assert sha256((parent / name).read_bytes()) == hashes['parent'], 'unexpected parent: ' + name
        if candidate is not None:
            assert sha256((Path(candidate) / name).read_bytes()) == hashes['candidate'], 'unexpected candidate: ' + name
    return dict(source_scope=SOURCE_SCOPE, patch_sha256=PATCH_SHA256, source_hashes=pins,
                candidate_checked=candidate is not None, native_and_public_fidelity='NOT_TESTED')


def regression(parent, candidate=None, patch=None):
    if patch is None:
        patch = (ROOT / 'verify-singletons.patch').read_text()
    result = check(parent, candidate, patch)
    controls = {
        'width9': ('n > 8', 'n > 9'),
        'width1': ('n < 3', 'n < 1'),
        'mark': ('!= GGML_PREC_F32', '== GGML_PREC_F32'),
        'scope-marker': ('ggml_get_op_params_i32(dst, 15) != 0x4e535631', 'ggml_get_op_params_i32(dst, 15) == 0x4e535631'),
        'producer-marker': ('res->op_params[15] = 0x4e535631;', 'res->op_params[15] = 0;'),
        'graph-count': ('== this->n_tokens', '<= this->n_tokens'),
        'output-count': ('this->n_outputs == this->n_tokens', 'this->n_outputs <= this->n_tokens'),
        'architecture': ('LLM_ARCH_NEMOTRON_H_MOE', 'LLM_ARCH_DFLASH'),
        'identity': ('res->src[1] == input', 'res->src[1] != input'),
        'mtp': ('digit >= n_layer', 'digit > n_layer'),
        'all-column-preflight': ('ggml_tensor src1_token[8]', 'ggml_tensor src1_token[2]'),
        'stride-overflow': ('SIZE_MAX / n', 'SIZE_MAX'),
        'alias': ('ggml_sycl_verify_singletons_disjoint(ctx, t, dst)', 'true'),
    }
    for name, (old, new) in controls.items():
        assert old in patch, name
        try:
            check(parent, patch=patch.replace(old, new))
        except AssertionError:
            continue
        raise AssertionError('accepted unreviewed patch: ' + name)
    result['rejected_patch_mutations'] = list(controls)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(regression(args.parent, args.candidate), indent=2))
