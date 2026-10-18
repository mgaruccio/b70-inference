#!/usr/bin/env python3
"""Read-only CPU/source regression: exact target insertion and inherited predicate.

No model, backend, compiler, service, source writes, or runtime route simulation.
"""
import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

if not __debug__:
    raise RuntimeError('do not disable assertions in this experimental gate')

PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-verify-phase-r1')
TARGET = 'src/models/nemotron-h.cpp'
PARENT_SHA256 = '6eec4bbf1b636a91a2a618545ff22877f1a94645923794ed29ccfe8c87b27ed0'
GRAPH_SHA256 = 'c28d647b7c9b552089259431eb7f671e9d1409cdeaf64f66d47f33de981a4abc'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check(parent=PARENT, candidate=None, patch=None):
    before = (parent / TARGET).read_bytes()
    graph = (parent / 'src/llama-graph.cpp').read_bytes()
    assert sha256(before) == PARENT_SHA256, 'unexpected target parent source'
    assert sha256(graph) == GRAPH_SHA256, 'unexpected inherited generic guard source'
    if patch is None:
        patch = Path(__file__).with_name('router-callsite.patch').read_text()
    added = ''.join(line[1:] for line in patch.splitlines(keepends=True)
                    if line.startswith('+') and not line.startswith('+++'))
    site = ('        ggml_tensor * router_logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur);\n'
            '        cb(router_logits, "ffn_moe_logits", il);\n')
    source = before.decode()
    assert source.count(site) == 1, 'target callsite must be unique and immediately precede cb'
    call, callback = site.splitlines(keepends=True)
    after = source.replace(site, call + added + callback)
    exact_patch = f'diff --git a/{TARGET} b/{TARGET}\n' + ''.join(difflib.unified_diff(
        source.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile='a/' + TARGET, tofile='b/' + TARGET))
    assert patch == exact_patch, 'patch must only insert at the target callsite (not MTP)'

    # Compare the entire if/setter block, not just selected positive controls.
    # Only the local tensor names and local column-count spelling may differ.
    inherited = re.search(r'        if \(arch == LLM_ARCH_NEMOTRON_H_MOE &&.*?\n        }\n',
                          graph.decode(), re.S).group()
    inherited = re.sub(r'(?<!->)\bn_tokens\b', 'cur->ne[1]', inherited)
    inherited = re.sub(r'\bgate_inp\b', 'model.layers[il].ffn_gate_inp', inherited)
    inherited = re.sub(r'\blogits\b', 'router_logits', inherited)
    inserted = '\n'.join(line for line in added.splitlines()
                         if not line.lstrip().startswith('//'))
    assert re.sub(r'\s+', '', inserted) == re.sub(r'\s+', '', inherited), 'phase predicate changed'
    assert 'build_moe_ffn(inp_latent,' in after and 'router_logits, nullptr,' in after
    if candidate is not None:
        assert (candidate / TARGET).read_bytes() == after.encode(), 'candidate differs from exact insertion'
        assert (candidate / 'src/llama-graph.cpp').read_bytes() == graph, 'old generic guard changed'
    return {'target': TARGET, 'parent_sha256': sha256(before),
            'patched_sha256': sha256(after.encode()), 'patch_sha256': sha256(patch.encode()),
            'checks': ['exact target insertion before cb', 'entire inherited phase predicate',
                       'precomputed router passed to build_moe_ffn',
                       'candidate bytes' if candidate else 'candidate bytes pending build']}


def regression(parent, candidate=None, patch=None):
    if patch is None:
        patch = Path(__file__).with_name('router-callsite.patch').read_text()
    result = check(parent, candidate, patch)
    controls = {
        'width9': ('cur->ne[1] <= 8', 'cur->ne[1] <= 9'),
        'width1': ('cur->ne[1] >= 3', 'cur->ne[1] >= 1'),
        'graph-count': ('cur->ne[1] == this->n_tokens', 'cur->ne[1] <= this->n_tokens'),
        'output-count': ('this->n_outputs == this->n_tokens', 'this->n_outputs <= this->n_tokens'),
        'architecture': ('LLM_ARCH_NEMOTRON_H_MOE', 'LLM_ARCH_NEMOTRON_H'),
        'type': ('cur->type == GGML_TYPE_F32', 'cur->type == GGML_TYPE_F16'),
        'input-identity': ('router_logits->src[1] == cur', 'router_logits->src[1] == inp_latent'),
        'weight-identity': ('router_logits->src[0] == model.layers[il].ffn_gate_inp',
                            'router_logits->src[0] != model.layers[il].ffn_gate_inp'),
        'mtp-path': ('src/models/nemotron-h.cpp', 'src/models/nemotron-h-moe.cpp'),
    }
    for name, (old, new) in controls.items():
        assert old in patch, name
        try:
            check(parent, patch=patch.replace(old, new))
        except AssertionError:
            continue
        raise AssertionError('accepted invalid patch: ' + name)
    result['rejected_controls'] = list(controls)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(regression(args.parent, args.candidate), indent=2))
