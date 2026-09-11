#!/usr/bin/env python3
"""Temporary hook, layered AFTER (not instead of) the exact canonical overlay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

from parity_common import OVERLAY_SHA, sha

MODEL = "model_executor/models/qwen3_dflash.py"
SPECULATOR = "v1/worker/gpu/spec_decode/dspark/speculator.py"
EDITS = {
    MODEL: [
        ('logger = init_logger(__name__)\n',
         'logger = init_logger(__name__)\nfrom parity_capture import tensor as _parity_tensor\n'),
        ('        all_kv_flat = F.linear(\n',
         '        _parity_tensor("context_norm", normed_context_states)\n        all_kv_flat = F.linear(\n'),
        ('        return all_k, all_v\n',
         '        _parity_tensor("context_k_raw", all_k)\n        _parity_tensor("context_v", all_v)\n        return all_k, all_v\n'),
        ('        return all_k_normed\n',
         '        _parity_tensor("context_k_norm", all_k_normed)\n        return all_k_normed\n'),
        ('        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)\n',
         '        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)\n        _parity_tensor("context_k_rope", all_k_final)\n'),
    ],
    SPECULATOR: [
        ('        self._sample_sequential(num_reqs, head_hidden)\n',
         '        self._sample_sequential(num_reqs, head_hidden)\n\n'
         '# Experiment-only: no replacement proposal/sampler implementation.\n'
         'from parity_capture import install as _install_parity_capture\n'
         '_install_parity_capture(DSparkSpeculator)\n'),
    ],
}


def transform(sources, reverse=False):
    result = dict(sources)
    for name, edits in EDITS.items():
        for old, new in (reversed(edits) if reverse else edits):
            a, b = (new, old) if reverse else (old, new)
            if result[name].count(a) != 1:
                raise RuntimeError(f"capture hook anchor missing/ambiguous: {name}")
            result[name] = result[name].replace(a, b, 1)
        compile(result[name], name, "exec")
    return result


def prepare(sources, canonical):
    hooked = "_install_parity_capture(DSparkSpeculator)" in sources[SPECULATOR]
    base = transform(sources, reverse=True) if hooked else sources
    if canonical["prepare"](base) != base:
        raise RuntimeError("apply the corrected canonical BF16 overlay first")
    result = transform(base)
    if hooked and result != sources:
        raise RuntimeError("not an exact diagnostic-hook replay")
    return result, base


def verify(root, overlay, boundary):
    if sha(overlay) != OVERLAY_SHA:
        raise RuntimeError("canonical overlay is not commit 8a19b58's layer-specific norm overlay")
    canonical = runpy.run_path(str(overlay))
    paths = {p: root / p for p in canonical["PINNED_SHA256"]}
    for path in paths.values():
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"source outside installed root: {path}")
    sources = {p: path.read_bytes().decode() for p, path in paths.items()}
    result, base = prepare(sources, canonical)
    boundary_module = runpy.run_path(str(boundary))
    xpu = (root / "_xpu_ops.py").read_text()
    if boundary_module["patch_text"](xpu) != xpu:
        raise RuntimeError("original installed-source XPU boundary check failed")
    if sha(root / "v1/attention/backends/gdn_attn.py") != "5173f3394c1385d215bd99f0d12290e8336da844b96da612954da01d62a0b062":
        raise RuntimeError("prefill dependency changed")
    record = {
        "root": str(root), "canonical_exact_replay_after_removing_only_hook": True,
        "canonical_overlay_sha256": sha(overlay),
        "base_sha256": {p: hashlib.sha256(s.encode()).hexdigest() for p, s in base.items()},
        "hooked_sha256": {p: hashlib.sha256(s.encode()).hexdigest() for p, s in result.items()},
        "xpu_boundary_sha256": hashlib.sha256(xpu.encode()).hexdigest(),
        "hook_files": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "input_batch_sha256": sha(root / "v1/worker/gpu/input_batch.py"),
    }
    return paths, sources, result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/venv/lib/python3.12/site-packages/vllm"))
    parser.add_argument("--overlay", type=Path, default=Path("/experiment/patch_dspark_bf16.py"))
    parser.add_argument("--boundary", type=Path, default=Path("/experiment/patch-vllm-qwen38-xpu-boundary.py"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if os.environ.get("B70_DSPARK_BF16") != "1":
        parser.error("requires the opt-in corrected overlay environment")
    paths, sources, result, record = verify(args.root, args.overlay, args.boundary)
    if args.check:
        if sources != result:
            raise RuntimeError("capture overlay is not installed")
    else:
        for name in EDITS:
            paths[name].write_bytes(result[name].encode())
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
