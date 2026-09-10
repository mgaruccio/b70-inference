#!/usr/bin/env python3
"""Pinned opt-in Qwen 3.8 DFlash2 KV-cache group-size overlay.

This overlay is for the pinned vLLM 73029d424 research image only.  It changes
only the group-size selection in ``_get_kv_cache_groups_uniform_page_size``.
The patched function accepts exactly the Qwen 3.8 target layout (48 Mamba,
16 full-attention, and 5 sliding-window layers), keeps the upstream grouping
and striding algorithm, and forces its group size to eight.  Any other layout
fails closed rather than becoming a generic group-size setting.

Run this script explicitly for the cache-group-size=8 experiment.  It does not
read an environment variable and it does not alter scheduler, cache dtype,
rollback, or native-kernel code.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path


MARKER = "B70_QWEN38_DFLASH2_CACHE_GROUPS"
START = "def _get_kv_cache_groups_uniform_page_size(\n"
END = "\ndef _get_per_layer_spec(\n"
ORIGINAL_FUNCTION_SHA256 = "e310cba8a00be7b9f349deff63ea6a9ac5d1b80956a7028ec0d8d4f7d3518206"
PATCHED_FUNCTION_SHA256 = "7cf30bf6160cf1fe0189a527fcf57c20ea62ecce8763fa26b1d851568c3116ac"

# The complete selection block is pinned so a nearby upstream change cannot be
# silently patched as though it were the reviewed function.
ORIGINAL_SELECTION = '''    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in layer_buckets])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
'''

REPLACEMENT_SELECTION = '''    # B70_QWEN38_DFLASH2_CACHE_GROUPS: opt-in Qwen 3.8 DFlash2 geometry.
    # This is a pinned layout guard, not a general group-size knob.
    expected_bucket_types = {
        MambaSpec: 48,
        FullAttentionSpec: 16,
        SlidingWindowSpec: 5,
    }
    bucket_layer_names = [
        layer_name for layers in layer_buckets for layer_name in layers
    ]
    actual_bucket_types = {
        type(kv_cache_spec[layers[0]]): len(layers)
        for layers in layer_buckets
    }
    if (
        len(kv_cache_spec) != 69
        or len(layer_buckets) != 3
        or len(spec_buckets) != 3
        or actual_bucket_types != expected_bucket_types
        or len(bucket_layer_names) != len(kv_cache_spec)
        or len(set(bucket_layer_names)) != len(bucket_layer_names)
        or set(bucket_layer_names) != set(kv_cache_spec)
        or any(
            type(kv_cache_spec[layer_name]) is not type(kv_cache_spec[layers[0]])
            for layers in layer_buckets
            for layer_name in layers
        )
        or len({spec.page_size_bytes for spec in kv_cache_spec.values()}) != 1
    ):
        raise RuntimeError(
            "B70 DFlash2 cache-group overlay requires the pinned "
            "48 Mamba / 16 FullAttention / 5 SlidingWindow layout "
            "with uniform page bytes"
        )
    group_size = 8
'''


def patch_text(source: str) -> str:
    """Apply once, or validate an exact replay; refuse drift before any write."""
    for anchor in (START, END):
        if source.count(anchor) != 1:
            raise RuntimeError("pinned KV-cache group function anchors changed")

    start = source.index(START)
    end = source.index(END, start)
    function = source[start:end]

    if MARKER in source:
        if source.count(MARKER) != 1 or MARKER not in function:
            raise RuntimeError("partial/changed DFlash2 cache-group patch")
        if hashlib.sha256(function.encode()).hexdigest() != PATCHED_FUNCTION_SHA256:
            raise RuntimeError("moved/changed DFlash2 cache-group patch")
        compile(source, "kv_cache_utils.py", "exec")
        return source

    if hashlib.sha256(function.encode()).hexdigest() != ORIGINAL_FUNCTION_SHA256:
        raise RuntimeError("pinned KV-cache group function changed")
    if function.count(ORIGINAL_SELECTION) != 1:
        raise RuntimeError("pinned KV-cache group-size selection changed")

    patched_function = function.replace(ORIGINAL_SELECTION, REPLACEMENT_SELECTION, 1)
    result = source[:start] + patched_function + source[end:]
    if hashlib.sha256(patched_function.encode()).hexdigest() != PATCHED_FUNCTION_SHA256:
        raise RuntimeError("internal DFlash2 cache-group patch hash mismatch")
    compile(result, "kv_cache_utils.py", "exec")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="vllm package root; omit inside serving image")
    args = parser.parse_args()

    root = args.root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("vllm package not found")
        root = Path(spec.origin).parent

    path = root / "v1/core/kv_cache_utils.py"
    before = path.read_text(encoding="utf-8")
    after = patch_text(before)
    if after != before:
        path.write_text(after, encoding="utf-8")
        print(f"{MARKER}: patched {path}", flush=True)
    else:
        print(f"{MARKER}: already patched {path}", flush=True)


if __name__ == "__main__":
    main()
